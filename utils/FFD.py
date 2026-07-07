"""
First Fit Decreasing (FFD) 慢层部署算法
算法流程：
1. 使用FFD算法部署传统微服务到传统服务器
2. 部署AI微服务到AI服务器（物理部署）

算法逻辑：
1. 为每个请求流选择确定性的到达边缘节点
2. 使用M/M/C排队论模型计算微服务资源需求
3. 根据与到达节点的网络距离对传统服务器进行排序（升序，最近的优先）
4. 按顺序尝试部署微服务，资源不足时选择下一个最近的服务器
5. 部署AI微服务到AI服务器

【修复说明】
- 正确创建和使用DecisionVariables对象
- 确保所有部署函数都能接收到正确的decision_vars参数
- 保持与Next Fit部署算法的一致性
"""

import random
import numpy as np
import hashlib
from fractions import Fraction
from typing import List, Dict, Tuple, Optional
from Constant import SystemState, MicroserviceInstance, ServerType, DecisionVariables
from Myopic_optimization import create_myopic_optimizer
from Deployment import get_servers_sorted_by_communication_delay


def select_deterministic_arrival_node(flow_id: str, traditional_servers: List,
                                      seed: Optional[int] = None):
    """根据flow_id和seed选择稳定到达节点"""
    if not traditional_servers:
        return None
    key = f"{flow_id}:{seed if seed is not None else 0}".encode("utf-8")
    digest = hashlib.md5(key).hexdigest()
    index = int(digest[:8], 16) % len(traditional_servers)
    return traditional_servers[index]


def sort_servers_resource_first(servers: List) -> List:
    """FFD正式基线使用资源优先固定顺序，不看链路延迟或HAPA readiness。"""
    return sorted(
        servers,
        key=lambda server: (
            -float(getattr(server, "cpu_cores", 0)),
            -float(getattr(server, "memory_capacity", 0.0)),
            str(getattr(server, "server_id", "")),
        ),
    )


def sort_microservices_decreasing(request_flow, microservices: List) -> List:
    """按资源需求降序处理微服务，保持First-Fit Decreasing边界。"""
    chain = request_flow.service_chain
    return sorted(
        microservices,
        key=lambda ms: (
            -float(chain.core_allocations.get(ms.ms_id, 1)),
            -float(chain.memory_allocations.get(ms.ms_id, 1.2)),
            str(ms.ms_id),
        ),
    )


def run_ffd_resource_first_ai_placement(system_state: SystemState) -> bool:
    """FFD专用AI资源优先放置，不使用delay-spare评分。"""
    from Deployment import _ai_reservation_feasible, _consume_ai_reservation
    from Deployment import _create_ai_instance, _estimate_ai_reservation

    ai_servers = sort_servers_resource_first([
        server for server in system_state.edge_servers.values()
        if server.server_type.value == "ai_capable"
    ])
    if not ai_servers:
        return False
    residual = {
        server.server_id: {
            "gpu_units": float(server.gpu_units),
            "gpu_memory": float(server.gpu_memory),
            "model_storage": float(server.model_storage),
            "context_units": float(getattr(server, "max_concurrent_contexts", 8)),
        }
        for server in ai_servers
    }
    server_model_cache = {server.server_id: set() for server in ai_servers}
    for flow_id, request_flow in sorted(system_state.request_flows.items()):
        ai_microservice = request_flow.service_chain.ai_microservice
        if not ai_microservice:
            continue
        placed = False
        for server in ai_servers:
            if server.available_cpu < 1:
                continue
            reservation = _estimate_ai_reservation(request_flow, ai_microservice, server, system_state)
            if ai_microservice.ms_id in server_model_cache.get(server.server_id, set()):
                # 同一AI模型在同节点复用模型缓存，FFD不能重复扣model storage。
                reservation = dict(reservation)
                reservation["model_storage"] = 0.0
            if not _ai_reservation_feasible(server.server_id, residual, reservation):
                continue
            _consume_ai_reservation(server.server_id, residual, reservation)
            server_model_cache[server.server_id].add(ai_microservice.ms_id)
            _create_ai_instance(flow_id, request_flow, ai_microservice, server, system_state, reservation, "FFD")
            placed = True
            break
        if not placed:
            return False
    system_state.ffd_ai_residual_after = residual
    return True


def build_ffd_first_fit_routing_context(system_state: SystemState, context: Dict) -> bool:
    """FFD专用确定性路由：按server_id选择第一个可行后继，不看延迟或spare。"""
    system_state.stream_transfer_probabilities.clear()
    for request_flow in system_state.request_flows.values():
        request_flow.routing_probabilities = {}
    routing_rows = {}
    for flow_id, request_flow in system_state.request_flows.items():
        chain = request_flow.service_chain.microservices
        flow_probs = {}
        for idx in range(1, len(chain)):
            origin_ms = chain[idx - 1]
            dest_ms = chain[idx]
            origin_placements = system_state.stream_allocated_resources.get(flow_id, {}).get(origin_ms.ms_id, {})
            if dest_ms.service_type == "ai":
                dest_placements = system_state.stream_allocated_resources.get(flow_id, {}).get(dest_ms.ms_id, {})
            else:
                dest_placements = {}
                for _, resources in system_state.stream_allocated_resources.items():
                    for server_id, allocated in resources.get(dest_ms.ms_id, {}).items():
                        dest_placements[server_id] = dest_placements.get(server_id, 0) + allocated
            if not origin_placements or not dest_placements:
                continue
            dest_ids = sorted(dest_placements.keys())
            selected_dest = dest_ids[0]
            for origin_server_id in sorted(origin_placements.keys()):
                row_key = f"{flow_id}:{origin_ms.ms_id}@{origin_server_id}->{dest_ms.ms_id}"
                routing_rows[row_key] = {selected_dest: 1.0}
                transfer_key = (origin_server_id, origin_ms.ms_id, selected_dest, dest_ms.ms_id)
                flow_probs[transfer_key] = Fraction(1, 1)
                request_flow.add_routing_probability(origin_server_id, selected_dest, 1.0)
        system_state.stream_transfer_probabilities[flow_id] = flow_probs
    context["deterministic_routing_matrix"] = routing_rows
    context["routing_policy"] = "FFD-deterministic"
    context["routing_selection_policy"] = "first_fit_successor"
    context["routing_entropy"] = 0.0
    return True


def ffd_deployment(system_state: SystemState, seed: Optional[int] = None) -> bool:
    """
    FFD部署算法旧兼容主函数
    【修复】添加完整的DecisionVariables管理
    Args:
        system_state: 系统状态对象
        seed: 随机种子，用于确保可复现性
    Returns:
        bool: 是否所有微服务都部署成功
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    print("\n=== 开始FFD legacy部署算法 ===")

    # 【修复】创建全局决策变量对象，用于整个FFD部署过程
    decision_vars = DecisionVariables(system_state.time_frame)

    # 获取传统服务器列表
    traditional_servers = [server for server in system_state.edge_servers.values()
                          if server.server_type.value == "traditional"]

    if not traditional_servers:
        print("没有传统服务器可用于FFD部署")
        return False

    print(f"可用传统服务器: {len(traditional_servers)} 台")

    # ===== 第一阶段：传统微服务部署 =====
    print(f"\n--- 第一阶段：传统微服务FFD部署 ---")

    # 使用M/M/C模型计算所有请求流的资源需求
    for flow_id, request_flow in system_state.request_flows.items():
        from Deployment import calculate_ms_resource
        total_cores = calculate_ms_resource(request_flow)
        print(f"{flow_id}: 总核心需求 {total_cores}")

    # 统计部署结果
    successful_deployments = 0
    total_microservices = 0

    # 为每个请求流执行FFD部署
    for flow_id, request_flow in system_state.request_flows.items():
        print(f"\n--- 处理请求流 {flow_id} ---")

        # 获取传统微服务列表
        traditional_microservices = request_flow.service_chain.get_traditional_microservices()
        if not traditional_microservices:
            print(f"请求流 {flow_id} 没有传统微服务需要部署")
            continue

        total_microservices += len(traditional_microservices)

        # 为请求流选择一个稳定到达边缘节点
        arrival_node = select_deterministic_arrival_node(flow_id, traditional_servers, seed)
        print(f"选择到达节点: {arrival_node.server_id}")

        # 根据与到达节点的距离对所有传统服务器排序
        sorted_servers = get_servers_sorted_by_communication_delay(arrival_node.server_id, traditional_servers,system_state)
        print(f"服务器距离排序: {[s.server_id for s in sorted_servers[:5]]}...")  # 只显示前5个

        # 为每个传统微服务选择最近且资源充足的服务器
        flow_success = deploy_traditional_microservices_ffd(
            request_flow, traditional_microservices, sorted_servers, system_state, decision_vars)

        if flow_success:
            successful_deployments += len(traditional_microservices)
            print(f"✓ 请求流 {flow_id} 的 {len(traditional_microservices)} 个传统微服务部署成功")
        else:
            print(f"✗ 请求流 {flow_id} 部分微服务部署失败")

    # 计算并更新路由概率矩阵
    print(f"\n--- 计算路由概率矩阵 ---")
    try:
        system_state.calculate_stream_transfer_probabilities()
        validation_errors = system_state.validate_transfer_probabilities()
        if validation_errors:
            print("⚠️ 路由概率验证发现错误:")
            for flow_id, errors in validation_errors.items():
                print(f"  {flow_id}: {errors}")
        else:
            print("✓ 路由概率验证通过")
    except Exception as e:
        print(f"⚠️ 路由概率矩阵计算失败: {e}")

    # ===== 第二阶段：AI微服务物理部署 =====
    print(f"\n--- 第二阶段：AI微服务物理部署 ---")
    from Deployment import deploy_ai_microservices
    ai_deployment_success = deploy_ai_microservices(system_state)

    if ai_deployment_success:
        print(f"✓ 所有AI微服务物理部署成功")
    else:
        print(f"⚠️ 部分AI微服务物理部署失败")
        return False

    # ===== 第三阶段：Myopic Optimization卸载决策 =====
    print(f"\n--- 第三阶段：Myopic Optimization AI卸载决策 ---")

    # 获取AI服务器性能因子
    env_manager = system_state.environment_manager
    if env_manager is None:
        print("⚠️ 环境状态管理器未初始化，跳过卸载决策")
        return False

    # 更新环境状态以获取最新的性能因子
    env_manager.update_all_ai_server_states()
    SH, SQ, SZ = env_manager.get_state_components()

    print(f"AI服务器性能因子 (SH): {SH}")

    # 创建Myopic优化器并做出卸载决策
    myopic_optimizer = create_myopic_optimizer()
    offloading_decisions = myopic_optimizer.make_offloading_decision(SH, system_state)

    print(f"Myopic卸载决策: {offloading_decisions}")
    print(f"决策解释: {['本地处理' if d==0 else '云端卸载' for d in offloading_decisions]}")

    # ===== 第四阶段：应用卸载决策 =====
    print(f"\n--- 第四阶段：应用卸载决策 ---")

    apply_success = apply_offloading_decisions_to_ai_microservices(
        system_state, offloading_decisions, SH, SQ, SZ)

    if not apply_success:
        print("⚠️ 卸载决策应用失败")
        return False

    # ===== 第五阶段：计算最终系统性能指标 =====
    print(f"\n--- 第五阶段：计算最终系统性能指标 ---")

    # 计算所有请求流的端到端延迟
    import Waitingtime
    system_state.total_latency = 0.0
    for flow_id in system_state.request_flows.keys():
        flow_delay = Waitingtime.calculate_flow_end_to_end_delay_global(flow_id, system_state)
        system_state.request_flows[flow_id].ca_latency = flow_delay
        system_state.total_latency += flow_delay

    # 计算平均延迟
    num_flows = len(system_state.request_flows)
    avg_latency = system_state.total_latency / max(num_flows, 1)

    # 更新所有服务器的虚拟能耗队列
    system_state.update_all_energy_queues()

    # 计算系统总能耗
    import EnergyConsumption
    system_state.total_energy_consumption = EnergyConsumption.calculate_system_total_energy_consumption(system_state)

    # 显示最终结果
    print(f"\n=== FFD legacy部署算法完成 ===")
    print(f"传统微服务部署成功率: {successful_deployments}/{total_microservices}")
    print(f"AI微服务物理部署: {'成功' if ai_deployment_success else '失败'}")
    print(f"AI卸载决策: {np.sum(offloading_decisions == 0)}个本地处理, {np.sum(offloading_decisions == 1)}个云端卸载")
    print(f"系统总延迟: {system_state.total_latency:.2f} ms")
    print(f"请求流平均延迟: {avg_latency:.2f} ms")
    print(f"系统总能耗: {system_state.total_energy_consumption:.3f} J")

    # 【修复】保存决策变量的验证结果
    violations = decision_vars.validate_decisions(system_state)
    if violations:
        print(f"⚠️ 发现 {len(violations)} 个约束违反:")
        for violation in violations[:3]:  # 只显示前3个
            print(f"  {violation}")
        if len(violations) > 3:
            print(f"  ... 以及其他 {len(violations) - 3} 个违反")
    else:
        print("✓ 所有约束条件满足")

    # 返回传统微服务和AI微服务都部署成功
    return successful_deployments == total_microservices and ai_deployment_success


def run_FFD_slow_context(system_state: SystemState, seed: Optional[int] = None) -> bool:
    """
    FFD慢层部署入口
    仅部署传统微服务和AI微服务物理位置，不执行快层卸载决策。
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    from Deployment import reset_slow_deployments
    reset_slow_deployments(system_state)

    print("\n=== 开始FFD慢层部署 ===")
    decision_vars = DecisionVariables(system_state.time_frame)
    traditional_servers = [server for server in system_state.edge_servers.values()
                          if server.server_type.value == "traditional"]
    if not traditional_servers:
        print("没有传统服务器可用于FFD部署")
        return False

    successful_deployments = 0
    total_microservices = 0
    for flow_id, request_flow in system_state.request_flows.items():
        traditional_microservices = request_flow.service_chain.get_traditional_microservices()
        if not traditional_microservices:
            continue

        total_microservices += len(traditional_microservices)
        # FFD正式基线只按资源容量做first-fit，不引入链路延迟或spare评分。
        sorted_servers = sort_servers_resource_first(traditional_servers)
        ordered_microservices = sort_microservices_decreasing(request_flow, traditional_microservices)
        flow_success = deploy_traditional_microservices_ffd(
            request_flow, ordered_microservices, sorted_servers, system_state, decision_vars)

        if flow_success:
            successful_deployments += len(traditional_microservices)
        else:
            print(f"FFD慢层请求流 {flow_id} 部分微服务部署失败")

    ai_deployment_success = run_ffd_resource_first_ai_placement(system_state)
    if not ai_deployment_success:
        system_state.ffd_context = {
            "failed": True,
            "failure_reason": "FFD AI placement failed",
            "seed": seed,
            "server_order_policy": "resource_first_ffd",
            "ai_placement_policy": "resource_first_ai",
            "uses_delay_sorted_servers": False,
            "uses_spare_delay_ai_score": False,
        }
        return False

    routing_context = {"policy": "FFD"}
    route_ok = build_ffd_first_fit_routing_context(system_state, routing_context)
    system_state.ffd_context = {
        "traditional_successful": successful_deployments,
        "traditional_total": total_microservices,
        "ai_deployment_success": ai_deployment_success,
        "routing_policy": "FFD-deterministic",
        "routing_selection_policy": routing_context.get("routing_selection_policy"),
        "routing_entropy": routing_context.get("routing_entropy", 0.0),
        "server_order_policy": "resource_first_ffd",
        "microservice_order_policy": "decreasing_resource_demand",
        "ai_placement_policy": "resource_first_ai",
        "uses_delay_sorted_servers": False,
        "uses_spare_delay_ai_score": False,
        "seed": seed,
    }
    return successful_deployments == total_microservices and route_ok


def deploy_traditional_microservices_ffd(request_flow: 'RequestFlow',
                                       traditional_microservices: List['Microservice'],
                                       sorted_servers: List['EdgeServer'],
                                       system_state: SystemState,
                                       decision_vars: DecisionVariables) -> bool:
    """
    使用FFD策略部署传统微服务
    【修复】正确接收和使用DecisionVariables对象
    Args:
        request_flow: 请求流对象
        traditional_microservices: 传统微服务列表
        sorted_servers: 按距离排序的服务器列表
        system_state: 系统状态对象
        decision_vars: 决策变量对象
    Returns:
        bool: 是否部署成功
    """
    deployment_success = True

    for microservice in traditional_microservices:
        ms_id = microservice.ms_id

        # 从服务链中获取M/M/C模型计算的资源需求
        required_cores = request_flow.service_chain.core_allocations.get(ms_id, 1)
        required_memory = request_flow.service_chain.memory_allocations.get(ms_id, 1.2)

        print(f"  部署微服务 {ms_id}: 需要 {required_cores} 核心, {required_memory:.2f} GB内存")

        # 尝试在最近的服务器上部署
        deployed = False
        for server in sorted_servers:
            from Deployment import deploy_single_microservice_to_server

            requirements = {'cores': required_cores, 'memory': required_memory}
            # 传递decision_vars对象
            if deploy_single_microservice_to_server(
                request_flow, microservice, requirements, server, system_state, decision_vars):
                deployed = True
                print(f"    ✓ 部署到 {server.server_id}")
                break

        if not deployed:
            print(f"    ✗ {ms_id} 部署失败：所有服务器资源不足")
            deployment_success = False

    return deployment_success


def apply_offloading_decisions_to_ai_microservices(
    system_state: SystemState,
    offloading_decisions: np.ndarray,
    SH: np.ndarray,
    SQ: np.ndarray,
    SZ: np.ndarray
) -> bool:
    """
    将Myopic Optimization的卸载决策应用到AI微服务

    Args:
        system_state: 系统状态对象
        offloading_decisions: N维二进制决策向量 (0=本地, 1=云端)
        SH: AI服务器性能因子
        SQ: 虚拟能耗队列状态
        SZ: 虚拟延迟队列状态

    Returns:
        bool: 是否应用成功
    """
    print(f"应用卸载决策到AI微服务...")

    # 获取AI服务器列表
    ai_servers = [server for server in system_state.edge_servers.values()
                  if server.server_type.value == "ai_capable"]
    ai_server_ids = sorted([server.server_id for server in ai_servers])

    if len(offloading_decisions) != len(ai_server_ids):
        print(f"⚠️ 卸载决策维度({len(offloading_decisions)}) != AI服务器数量({len(ai_server_ids)})")
        return False

    success_count = 0

    # 为每个AI服务器应用卸载决策
    for i, server_id in enumerate(ai_server_ids):
        decision = offloading_decisions[i]
        server = system_state.edge_servers[server_id]

        print(f"  AI服务器 {server_id}: {'云端卸载' if decision == 1 else '本地处理'}")

        # 查找部署在该服务器上的AI微服务实例
        ai_instance = None
        request_flow = None

        for instance_id, instance in system_state.microservice_instances.items():
            if (instance.server_id == server_id and
                instance.microservice.service_type == "ai"):
                ai_instance = instance

                # 从instance_id中提取flow_id
                parts = instance_id.split('_')
                if len(parts) >= 2:
                    flow_id = f"{parts[0]}_{parts[1]}"
                    request_flow = system_state.request_flows.get(flow_id)
                break

        if ai_instance is None or request_flow is None:
            print(f"    ⚠️ 未找到AI服务器 {server_id} 上的AI微服务实例")
            continue

        # 应用卸载决策
        if decision == 0:  # 本地处理
            success = apply_local_processing_decision(
                ai_instance, request_flow, server, system_state, SH[i])
        else:  # 云端卸载
            success = apply_cloud_offloading_decision(
                ai_instance, request_flow, server, system_state, SH[i])

        if success:
            success_count += 1
            print(f"    ✓ 决策应用成功")
        else:
            print(f"    ✗ 决策应用失败")

    print(f"卸载决策应用完成: {success_count}/{len(ai_server_ids)} 个成功")
    return success_count == len(ai_server_ids)


def apply_local_processing_decision(
    ai_instance: 'MicroserviceInstance',
    request_flow: 'RequestFlow',
    server: 'EdgeServer',
    system_state: 'SystemState',
    performance_factor: float
) -> bool:
    """
    应用本地处理决策
    """
    try:
        from Deployment import (calculate_local_ai_inference_latency,
                              calculate_required_gpu_memory,
                              calculate_required_model_storage)
        from EnergyConsumption import calculate_local_ai_processing_energy

        # 计算本地处理所需资源
        total_latency, queue_delay, processing_delay, required_gpu_units = \
            calculate_local_ai_inference_latency(
                request_flow, ai_instance.microservice, server, system_state, False)

        if np.isinf(total_latency):
            print(f"      本地处理不可行：服务强度过载")
            return False

        required_gpu_memory = calculate_required_gpu_memory(
            request_flow, ai_instance.microservice, required_gpu_units, False)
        required_model_storage = calculate_required_model_storage(ai_instance.microservice)

        # 检查资源充足性
        if (server.available_gpu_units < required_gpu_units or
            server.available_gpu_memory < required_gpu_memory or
            server.available_model_storage < required_model_storage):
            print(f"      本地资源不足，无法应用本地处理决策")
            return False

        # 分配资源
        server.available_gpu_units -= required_gpu_units
        server.available_gpu_memory -= required_gpu_memory
        server.available_model_storage -= required_model_storage
        server.activate_gpu_for_inference()

        # 更新AI微服务实例状态
        ai_instance.processing_mode = "local_processing"
        ai_instance.gpu_memory_allocated = required_gpu_memory
        ai_instance.model_storage_allocated = required_model_storage
        ai_instance.inference_latency = total_latency
        ai_instance.cloud_latency = 0.0
        ai_instance.queue_delay = queue_delay
        ai_instance.processing_delay = processing_delay

        # 设置GPU单元分配（如果实例对象支持）
        if hasattr(ai_instance, 'gpu_units_allocated'):
            ai_instance.gpu_units_allocated = required_gpu_units

        print(f"      本地处理配置: {required_gpu_units}GPU单元, {required_gpu_memory:.1f}GB内存, 延迟{total_latency:.2f}ms")
        return True

    except Exception as e:
        print(f"      本地处理决策应用失败: {e}")
        return False


def apply_cloud_offloading_decision(
    ai_instance: 'MicroserviceInstance',
    request_flow: 'RequestFlow',
    server: 'EdgeServer',
    system_state: 'SystemState',
    performance_factor: float
) -> bool:
    """
    应用云端卸载决策
    """
    try:
        from Deployment import evaluate_cloud_deployment
        from EnergyConsumption import (calculate_cloud_processing_energy,
                                     calculate_optimized_communication_energy)

        # 评估云端部署
        cloud_evaluation = evaluate_cloud_deployment(
            request_flow, ai_instance.microservice, system_state)

        if not cloud_evaluation['feasible']:
            print(f"      云端卸载不可行")
            return False

        # 计算云端延迟（考虑性能因子调整）
        base_cloud_latency = cloud_evaluation['total_latency']
        performance_adjustment = 0.9 + 0.2 * performance_factor
        adjusted_cloud_latency = base_cloud_latency * performance_adjustment

        # 计算云端能耗
        cloud_processing_energy = calculate_cloud_processing_energy(server)
        communication_energy = calculate_optimized_communication_energy(
            request_flow, compression_ratio=1.0, server=server)

        # 更新AI微服务实例状态
        ai_instance.processing_mode = "cloud_offloaded"
        ai_instance.gpu_memory_allocated = 0.0  # 云端卸载不占用GPU资源
        ai_instance.model_storage_allocated = 0.0
        ai_instance.inference_latency = 0.0
        ai_instance.cloud_latency = adjusted_cloud_latency
        ai_instance.queue_delay = 0.0  # 云端处理无本地队列延迟
        ai_instance.processing_delay = adjusted_cloud_latency

        # 云端卸载不需要占用本地GPU资源
        # 但需要少量CPU用于协调
        if server.available_cpu > 0:
            server.available_cpu -= 1  # 预留1个CPU用于云端协调

        print(f"      云端卸载配置: 延迟{adjusted_cloud_latency:.2f}ms, 无GPU资源占用")
        return True

    except Exception as e:
        print(f"      云端卸载决策应用失败: {e}")
        return False


def get_ffd_deployment_summary(system_state: SystemState) -> Dict:
    """
    获取FFD部署结果摘要（包含AI卸载决策信息）
    """
    summary = {
        'total_traditional_instances': 0,
        'total_ai_instances': 0,
        'servers_used': set(),
        'ai_servers_used': set(),
        'resource_utilization': {},
        'flow_deployment_status': {},
        'ai_offloading_summary': {},
        'queue_performance': {}
    }

    # 统计传统微服务实例
    total_queue_delay = 0.0
    total_processing_delay = 0.0
    valid_instances = 0

    # 统计AI微服务卸载决策
    local_processing_count = 0
    cloud_offloading_count = 0

    for instance_id, instance in system_state.microservice_instances.items():
        if instance.microservice.service_type == "traditional":
            summary['total_traditional_instances'] += 1
            summary['servers_used'].add(instance.server_id)

            # 统计队列性能
            if not np.isinf(instance.queue_delay) and not np.isinf(instance.processing_delay):
                total_queue_delay += instance.queue_delay
                total_processing_delay += instance.processing_delay
                valid_instances += 1

        elif instance.microservice.service_type == "ai":
            summary['total_ai_instances'] += 1
            summary['ai_servers_used'].add(instance.server_id)

            # 统计AI卸载决策
            processing_mode = getattr(instance, 'processing_mode', 'unknown')
            if processing_mode == "local_processing":
                local_processing_count += 1
            elif processing_mode == "cloud_offloaded":
                cloud_offloading_count += 1

    # 计算平均队列性能
    if valid_instances > 0:
        summary['queue_performance'] = {
            'avg_queue_delay': total_queue_delay / valid_instances,
            'avg_processing_delay': total_processing_delay / valid_instances,
            'avg_total_delay': (total_queue_delay + total_processing_delay) / valid_instances
        }

    # AI卸载决策摘要
    summary['ai_offloading_summary'] = {
        'local_processing': local_processing_count,
        'cloud_offloading': cloud_offloading_count,
        'local_ratio': local_processing_count / max(local_processing_count + cloud_offloading_count, 1),
        'cloud_ratio': cloud_offloading_count / max(local_processing_count + cloud_offloading_count, 1)
    }

    # 统计服务器资源利用率
    for server_id, server in system_state.edge_servers.items():
        if server.server_type.value == "traditional":
            cpu_utilization = (server.cpu_cores - server.available_cpu) / server.cpu_cores
            memory_utilization = (server.memory_capacity - server.available_memory) / server.memory_capacity
            summary['resource_utilization'][server_id] = {
                'cpu_utilization': cpu_utilization,
                'memory_utilization': memory_utilization,
                'server_type': 'traditional'
            }
        elif server.server_type.value == "ai_capable":
            cpu_utilization = (server.cpu_cores - server.available_cpu) / server.cpu_cores
            gpu_utilization = (server.gpu_units - server.available_gpu_units) / max(server.gpu_units, 1)
            gpu_memory_utilization = (server.gpu_memory - server.available_gpu_memory) / max(server.gpu_memory, 1)
            summary['resource_utilization'][server_id] = {
                'cpu_utilization': cpu_utilization,
                'gpu_utilization': gpu_utilization,
                'gpu_memory_utilization': gpu_memory_utilization,
                'server_type': 'ai_capable'
            }

    # 统计请求流部署状态
    for flow_id, request_flow in system_state.request_flows.items():
        traditional_count = len(request_flow.service_chain.get_traditional_microservices())
        ai_count = 1 if request_flow.service_chain.ai_microservice else 0

        deployed_traditional = sum(1 for instance in system_state.microservice_instances.values()
                                 if flow_id in instance.instance_id and instance.microservice.service_type == "traditional")
        deployed_ai = sum(1 for instance in system_state.microservice_instances.values()
                        if flow_id in instance.instance_id and instance.microservice.service_type == "ai")

        summary['flow_deployment_status'][flow_id] = {
            'total_traditional': traditional_count,
            'deployed_traditional': deployed_traditional,
            'total_ai': ai_count,
            'deployed_ai': deployed_ai,
            'traditional_success_rate': deployed_traditional / max(traditional_count, 1),
            'ai_success_rate': deployed_ai / max(ai_count, 1)
        }

    summary['servers_used'] = len(summary['servers_used'])
    summary['ai_servers_used'] = len(summary['ai_servers_used'])
    return summary


def print_ffd_deployment_summary(system_state: SystemState):
    """
    打印FFD部署结果摘要
    """
    summary = get_ffd_deployment_summary(system_state)

    print(f"\n=== FFD部署结果摘要 ===")
    print(f"传统微服务实例总数: {summary['total_traditional_instances']}")
    print(f"AI微服务实例总数: {summary['total_ai_instances']}")
    print(f"使用的传统服务器数量: {summary['servers_used']}")
    print(f"使用的AI服务器数量: {summary['ai_servers_used']}")

    # 显示AI卸载决策统计
    ai_summary = summary['ai_offloading_summary']
    print(f"\nAI微服务卸载决策统计:")
    print(f"  本地处理: {ai_summary['local_processing']} 个 ({ai_summary['local_ratio']:.1%})")
    print(f"  云端卸载: {ai_summary['cloud_offloading']} 个 ({ai_summary['cloud_ratio']:.1%})")

    # 显示队列性能（M/M/C模型结果）
    if 'queue_performance' in summary and summary['queue_performance']:
        perf = summary['queue_performance']
        print(f"\nM/M/C队列性能指标:")
        print(f"  平均队列延迟: {perf['avg_queue_delay']:.3f} ms")
        print(f"  平均处理延迟: {perf['avg_processing_delay']:.3f} ms")
        print(f"  平均总延迟: {perf['avg_total_delay']:.3f} ms")

    print(f"\n服务器资源利用率:")
    for server_id, utilization in summary['resource_utilization'].items():
        if utilization['cpu_utilization'] > 0:  # 只显示有部署的服务器
            if utilization['server_type'] == 'traditional':
                print(f"  {server_id}(传统): CPU {utilization['cpu_utilization']:.1%}, "
                      f"内存 {utilization['memory_utilization']:.1%}")
            else:
                print(f"  {server_id}(AI): CPU {utilization['cpu_utilization']:.1%}, "
                      f"GPU {utilization['gpu_utilization']:.1%}, "
                      f"GPU内存 {utilization['gpu_memory_utilization']:.1%}")

    print(f"\n请求流部署状态:")
    for flow_id, status in summary['flow_deployment_status'].items():
        print(f"  {flow_id}: 传统{status['deployed_traditional']}/{status['total_traditional']} "
              f"({status['traditional_success_rate']:.1%}), "
              f"AI{status['deployed_ai']}/{status['total_ai']} "
              f"({status['ai_success_rate']:.1%})")
