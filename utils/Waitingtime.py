import numpy as np
from typing import Dict, List, Optional, TYPE_CHECKING

# 使用TYPE_CHECKING避免循环导入
if TYPE_CHECKING:
    from Constant import RequestFlow, MicroserviceInstance, SystemState, ServiceChain, Microservice

def calculate_chain_communication_delay(service_chain: 'ServiceChain',system_state: 'SystemState') -> float:
    """
    计算服务链的通信延迟
    Args:
        service_chain: 服务链对象
        system_state: 系统状态对象
    Returns:
        float: 服务链的通信延迟
    """
    if not system_state.network_topology:
        return 0.0

    chain_delay = 0.0
    microservice_sequence = service_chain.get_microservice_sequence()

    for i in range(len(microservice_sequence) - 1):
        current_ms_id = microservice_sequence[i]
        next_ms_id = microservice_sequence[i + 1]
        # 找到微服务的部署服务器
        current_server = find_microservice_server(current_ms_id, system_state)
        next_server = find_microservice_server(next_ms_id, system_state)

        if current_server and next_server and current_server != next_server:
            delay = system_state.network_topology.get_communication_delay(
                current_server, next_server)
            chain_delay += delay

    return chain_delay


def find_microservice_server(microservice_id: str, system_state: 'SystemState') -> Optional[str]:
    """
    查找微服务部署的服务器
    Args:
        microservice_id: 微服务ID
        system_state: 系统状态对象

    Returns:
        Optional[str]: 服务器ID，如果未找到返回None
    """
    for instance in system_state.microservice_instances.values():
        if instance.microservice.ms_id == microservice_id:
            return instance.server_id
    return None

def calculate_microservice_instance_delay(instance: 'MicroserviceInstance') -> float:
    """
    计算单个微服务实例的总延迟
    Args:
        instance: 微服务实例对象
    Returns:
        float: 微服务实例的总延迟（队列延迟 + 处理延迟）
    """
    if np.isinf(instance.queue_delay) or np.isinf(instance.processing_delay):
        return float('inf')
    return instance.queue_delay + instance.processing_delay


def calculate_service_chain_delay(service_chain: 'ServiceChain',
                                  system_state: 'SystemState') -> float:
    """
    计算服务链的总延迟（包括所有微服务实例延迟和通信延迟）
    Args:
        service_chain: 服务链对象
        system_state: 系统状态对象
    Returns:
        float: 服务链的总延迟
    """
    total_chain_delay = 0.0

    # 计算链中每个微服务的延迟
    for microservice in service_chain.microservices:
        ms_id = microservice.ms_id
        # 找到该微服务的实例
        for instance in system_state.microservice_instances.values():
            if instance.microservice.ms_id == ms_id:
                instance_delay = calculate_microservice_instance_delay(instance)
                if not np.isinf(instance_delay):
                    total_chain_delay += instance_delay
                break

    # 加上通信延迟
    communication_delay = calculate_chain_communication_delay(service_chain, system_state)
    total_chain_delay += communication_delay

    return total_chain_delay


def calculate_ms_tolerable_time_fairly(service_chain: 'ServiceChain', target_ms_id: str,
                                          flow_max_latency_ms: float) -> float:
    """
    计算请求流中传统微服务的均分容忍时延大小（毫秒）
    基于各传统微服务处理率的倒数进行公平分配
    Args:
        service_chain: 服务链对象，包含微服务序列
        target_ms_id: 目标微服务ID，要计算容忍时延的微服务（必须是传统微服务）
        flow_max_latency_ms: 请求流的最大容忍时延 (ms)

    Returns:
        float: 目标微服务的分配容忍时延 (ms) # 修改单位
    """
    # 找到目标微服务
    target_microservice = None
    for ms in service_chain.microservices:
        if ms.ms_id == target_ms_id:
            target_microservice = ms
            break

    if target_microservice is None:
        return 0.0

    # 计算所有传统微服务处理率的倒数之和
    reciprocal_sum = 0.0
    for microservice in service_chain.microservices:
        if microservice.service_rate > 0:
            reciprocal_sum += 1.0 / microservice.service_rate

    # 计算目标微服务的处理率倒数
    if target_microservice.service_rate > 0:
        target_reciprocal = 1.0 / target_microservice.service_rate
    else:
        print(f"Warning: Target microservice {target_ms_id} has zero service rate, using default value 0.001")
        target_reciprocal = 1.0 / 0.001

    # 按比例分配容忍时延 (毫秒)
    allocated_latency_ms = (target_reciprocal / reciprocal_sum) * flow_max_latency_ms
    return allocated_latency_ms


def calculate_ai_microservice_delay_in_path(last_traditional_server: str,
                                            ai_deployment_info: Dict,
                                            system_state: 'SystemState',
                                            path_idx: int = 0,
                                            show_details: bool = True) -> float:
    """
    计算路径中AI微服务相关的延迟
    正确处理云端卸载时的通信延迟
    延迟组成：
    1. 传统服务器 → AI服务器的通信延迟（两种模式都需要）
    2. AI服务器的处理延迟（本地处理或云端卸载）
    """
    ai_total_delay = 0.0

    # 第一步：计算传统微服务链最后一个节点到AI服务器的通信延迟
    # 无论是本地处理还是云端卸载，都需要这个通信延迟
    ai_server_id = ai_deployment_info['server_id']

    if last_traditional_server != ai_server_id and system_state.network_topology:
        comm_to_ai_delay = system_state.network_topology.get_communication_delay(
            last_traditional_server, ai_server_id)
        ai_total_delay += comm_to_ai_delay
        '''
        if show_details and path_idx < 3:
            print(f"  通信延迟 {last_traditional_server} → {ai_server_id}(AI服务器): {comm_to_ai_delay:.2f}ms")
    elif show_details and path_idx < 3:
        print(f"  {last_traditional_server} → {ai_server_id}: 同服务器，无通信延迟")
        '''
    # 第二步：加上AI服务器的处理延迟
    processing_mode = ai_deployment_info.get('location', 'unknown')
    ai_processing_delay = ai_deployment_info['latency']
    ai_total_delay += ai_processing_delay
    '''
    if show_details and path_idx < 3:
        if processing_mode == 'local_processing':
            print(f"  AI本地处理延迟: {ai_processing_delay:.2f}ms")
        elif processing_mode == 'cloud_offloaded':
            print(f"  AI云端卸载延迟: {ai_processing_delay:.2f}ms (包含AI服务器→云端→AI服务器)")
        else:
            print(f"  AI处理延迟: {ai_processing_delay:.2f}ms")

        print(f"  AI微服务总延迟: {ai_total_delay:.2f}ms")
    '''
    return ai_total_delay


def calculate_routing_probability_weighted_delay(flow_id: str, system_state: 'SystemState') -> Optional[float]:
    """
    使用慢层routing概率计算请求级期望端到端延迟
    没有routing表时返回None，由旧路径枚举逻辑兜底。
    """
    route_table = getattr(system_state, "stream_transfer_probabilities", {}) or {}
    transfer_probs = route_table.get(flow_id, {})
    if not transfer_probs or flow_id not in system_state.request_flows:
        return None

    request_flow = system_state.request_flows[flow_id]
    service_chain = request_flow.service_chain
    traditional_microservices = service_chain.get_traditional_microservices()

    processing_delay = 0.0
    for microservice in traditional_microservices:
        processing_delay += _weighted_processing_delay(flow_id, microservice.ms_id, system_state)

    ai_info = find_ai_microservice_deployment(flow_id, service_chain.ai_microservice, system_state)
    ai_delay = float(ai_info.get("latency", 0.0)) if ai_info else 0.0

    communication_delay = 0.0
    probability_mass = 0.0
    for transfer_key, raw_prob in transfer_probs.items():
        if len(transfer_key) != 4:
            continue
        origin_server, _, dest_server, _ = transfer_key
        prob = float(raw_prob)
        if prob <= 0:
            continue
        probability_mass += prob
        if origin_server != dest_server and system_state.network_topology:
            communication_delay += prob * system_state.network_topology.get_communication_delay(
                origin_server, dest_server
            )

    if probability_mass <= 0:
        return None
    system_state._routing_delay_consumed = True
    system_state._routing_probability_mass_for_delay = probability_mass
    return float(processing_delay + communication_delay + ai_delay)


def _weighted_processing_delay(flow_id: str, ms_id: str, system_state: 'SystemState') -> float:
    """按当前流资源分配计算微服务处理延迟期望"""
    placements = dict(system_state.stream_allocated_resources.get(flow_id, {}).get(ms_id, {}))
    if not placements:
        for _, resources in system_state.stream_allocated_resources.items():
            for server_id, allocated in resources.get(ms_id, {}).items():
                placements[server_id] = placements.get(server_id, 0) + allocated
    total_resource = sum(value for value in placements.values() if value > 0)
    if total_resource <= 0:
        return 0.0

    expected_delay = 0.0
    for server_id, allocated in placements.items():
        if allocated <= 0:
            continue
        expected_delay += (float(allocated) / float(total_resource)) * _find_instance_delay(
            ms_id, server_id, system_state
        )
    return float(expected_delay)


def _find_instance_delay(ms_id: str, server_id: str, system_state: 'SystemState') -> float:
    """查找指定服务实例延迟"""
    for instance in system_state.microservice_instances.values():
        if instance.microservice.ms_id == ms_id and instance.server_id == server_id:
            delay = calculate_microservice_instance_delay(instance)
            return 0.0 if np.isinf(delay) else float(delay)
    return 0.0


def calculate_flow_end_to_end_delay_global(flow_id: str, system_state: 'SystemState') -> float:
    """
    计算请求流的平均端到端延迟
    正确计算笛卡尔积路径数量
    Args:
        flow_id: 请求流ID
        system_state: 系统状态对象
    Returns:
        float: 平均端到端延迟 (ms)
    """
    routing_delay = calculate_routing_probability_weighted_delay(flow_id, system_state)
    if routing_delay is not None:
        return routing_delay

    system_state._routing_delay_consumed = False
    system_state._routing_probability_mass_for_delay = 0.0
    #print(f"\n=== 计算请求流 {flow_id} 的端到端延迟===")
    MAX_PATHS_TO_CALCULATE = 5000  # 最多计算100条路径
    # 获取请求流和服务链
    if flow_id not in system_state.request_flows:
        print(f"请求流 {flow_id} 不存在")
        return 0.0

    request_flow = system_state.request_flows[flow_id]
    service_chain = request_flow.service_chain
    traditional_microservices = service_chain.get_traditional_microservices()
    ai_microservice = service_chain.ai_microservice

    if len(traditional_microservices) <= 1:
        print(f"请求流 {flow_id} 传统微服务数量不足，无法计算路径延迟")
        return 0.0

    microservice_sequence = [ms.ms_id for ms in traditional_microservices]
    #print(f"微服务序列: {' → '.join(microservice_sequence)}")

    # 构建每个微服务的部署位置列表
    microservice_deployments = []  # 每个微服务的部署位置列表

    for ms_id in microservice_sequence:
        deployments = []  # 当前微服务的所有部署位置

        # 遍历所有请求流的资源分配，找到该微服务的所有部署位置
        for curr_flow_id, flow_resources in system_state.stream_allocated_resources.items():
            if ms_id in flow_resources:
                for server_id, allocated_cores in flow_resources[ms_id].items():
                    if allocated_cores > 0:
                        # 检查是否已经记录了这个服务器
                        found = False
                        for i, (srv_id, cores) in enumerate(deployments):
                            if srv_id == server_id:
                                deployments[i] = (srv_id, cores + allocated_cores)
                                found = True
                                break

                        if not found:
                            deployments.append((server_id, allocated_cores))

        microservice_deployments.append(deployments)

    # 显示部署情况
    #print(f"\n=== 微服务部署情况 ===")
    #for i, ms_id in enumerate(microservice_sequence):
        #deployments = microservice_deployments[i]
        #print(f"{ms_id} 部署在 {len(deployments)} 个服务器:")
        #for server_id, total_cores in deployments:
            #print(f"  {server_id}: {total_cores} 核心")

    ai_deployment_info = find_ai_microservice_deployment(flow_id, ai_microservice, system_state)
    '''
    print(f"\n=== AI微服务部署情况 ===")

    if ai_deployment_info['location'] == 'local_processing':
        print(f"{ai_microservice.ms_id} 部署位置: {ai_deployment_info['server_id']} (本地处理)")
        print(f"  AI延迟: {ai_deployment_info['latency']:.2f} ms")
    elif ai_deployment_info['location'] == 'cloud_offloaded':
        print(f"{ai_microservice.ms_id} 部署位置: {ai_deployment_info['server_id']} (云端卸载)")
        print(f"  云端延迟: {ai_deployment_info['latency']:.2f} ms")
    elif ai_deployment_info['location'] == 'pending_decision':
        print(f"{ai_microservice.ms_id} 部署位置: {ai_deployment_info['server_id']} (待RL决策)")
        print(f"  物理部署: 已完成")
        print(f"  处理模式: 待RL算法决定本地处理或云端卸载")
        print(f"  当前延迟: {ai_deployment_info['latency']:.2f} ms (临时值)")
    else:
        print(f"{ai_microservice.ms_id} 部署位置: {ai_deployment_info['location']}")
    '''
    # 计算正确的路径总数
    path_counts = [len(deployments) for deployments in microservice_deployments]
    total_paths = 1
    for count in path_counts:
        total_paths *= count

    #print(f"\n路径计算: {' × '.join(map(str, path_counts))} = {total_paths} 条可能路径")

        # *** 关键修改：限制路径计算数量 ***
    if total_paths > MAX_PATHS_TO_CALCULATE:
        print(f"路径数量过多({total_paths})，采用采样策略，最多计算{MAX_PATHS_TO_CALCULATE}条路径")
        # 使用智能采样策略
        all_complete_paths = generate_sampled_paths(microservice_deployments, MAX_PATHS_TO_CALCULATE)
    else:
            # 路径数量合理，计算所有路径
        import itertools
        server_lists = []
        for deployments in microservice_deployments:
            servers = [server_id for server_id, _ in deployments]
            server_lists.append(servers)

        all_complete_paths = list(itertools.product(*server_lists))

    if not all_complete_paths:
        print("没有完整的路径可用")
        return 0.0

    #print(f"实际生成路径数: {len(all_complete_paths)} 条")

    # 计算每条路径的端到端延迟
    total_delay = 0.0

    for path_idx, server_sequence in enumerate(all_complete_paths):
        path_delay = 0.0

        if path_idx < 0:  # 只显示前3条路径的详细信息
            print()
            #print(f"\n路径 {path_idx + 1}: {' → '.join(server_sequence)}")

        # 计算路径延迟
        for i, ms in enumerate(traditional_microservices):
            server_id = server_sequence[i]
            ms_id = ms.ms_id

            # 查找该微服务在该服务器上的实例延迟
            instance_delay = 0.0
            found_instance = False

            for instance in system_state.microservice_instances.values():
                if (instance.microservice.ms_id == ms_id and
                        instance.server_id == server_id):
                    instance_delay = calculate_microservice_instance_delay(instance)
                    if not np.isinf(instance_delay):
                        path_delay += instance_delay
                        if path_idx < 0:
                            print()
                            #print(f"  {ms_id}@{server_id}: 实例延迟 {instance_delay:.2f}ms")
                        found_instance = True
                        break

            if not found_instance and path_idx < 0:
                print(f"  警告: 未找到 {ms_id}@{server_id} 的实例")

            # 添加通信延迟（除了最后一个微服务）
            if i < len(traditional_microservices) - 1:
                current_server = server_sequence[i]
                next_server = server_sequence[i + 1]
                if current_server != next_server and system_state.network_topology:
                    comm_delay = system_state.network_topology.get_communication_delay(current_server, next_server)
                    path_delay += comm_delay


        # 使用分离出的AI微服务延迟计算函数
        last_traditional_server = server_sequence[-1]  # 传统微服务链的最后一个服务器
        ai_delay = calculate_ai_microservice_delay_in_path(
            last_traditional_server=last_traditional_server,
            ai_deployment_info=ai_deployment_info,
            system_state=system_state,
            path_idx=path_idx,
            show_details=True
        )

        path_delay += ai_delay
        '''
        if path_idx < 3:
            print(f"  路径总延迟: {path_delay:.2f}ms")
        elif path_idx == 0:
            print("  ... (其余路径详情省略)")
        '''
        total_delay += path_delay

    # 计算平均延迟
    average_delay = total_delay / len(all_complete_paths)
    #print(f"\n所有路径延迟总和: {total_delay:.2f}ms")
    #print(f"平均端到端延迟: {average_delay:.2f}ms")

    return average_delay


def generate_sampled_paths(microservice_deployments, max_paths):
    """
    智能采样路径生成策略
    优先选择资源占用最大的部署位置，避免完全随机采样
    """
    import random
    import itertools

    # 为每个微服务选择最重要的部署位置
    selected_deployments = []

    for deployments in microservice_deployments:
        if len(deployments) <= 5:
            # 部署位置较少，全部保留
            servers = [server_id for server_id, _ in deployments]
            selected_deployments.append(servers)
        else:
            # 部署位置较多，选择资源占用最大的前2个
            sorted_deployments = sorted(deployments, key=lambda x: x[1], reverse=True)
            top_servers = [server_id for server_id, _ in sorted_deployments[:2]]
            selected_deployments.append(top_servers)

    # 生成采样路径
    try:
        # 首先尝试生成所有可能的路径组合
        all_combinations = list(itertools.product(*selected_deployments))

        if len(all_combinations) <= max_paths:
            # 如果组合数量仍然合理，返回所有组合
            return all_combinations
        else:
            # 如果组合数量仍然过多，进行随机采样
            random.shuffle(all_combinations)
            return all_combinations[:max_paths]

    except MemoryError:
        # 如果内存不足，使用更保守的策略
        print("内存不足，使用保守采样策略")
        sampled_paths = []

        for _ in range(min(max_paths, 50)):  # 最多50条路径
            path = []
            for servers in selected_deployments:
                path.append(random.choice(servers))
            sampled_paths.append(tuple(path))

        # 去重
        return list(set(sampled_paths))

def find_ai_microservice_deployment(flow_id: str, ai_microservice: 'Microservice',
                                    system_state: 'SystemState') -> Dict:
    """
    查找AI微服务的部署位置和处理模式
    修正版本：正确处理pending_decision状态
    """
    for instance_id, instance in system_state.microservice_instances.items():
        if (instance.microservice.ms_id == ai_microservice.ms_id and
                flow_id in instance_id):

            processing_mode = getattr(instance, 'processing_mode', 'unknown')

            if processing_mode == "local_processing":
                inference_latency = getattr(instance, 'inference_latency', 0.0)
                return {
                    'location': 'local_processing',
                    'server_id': instance.server_id,  # AI服务器ID
                    'latency': inference_latency
                }
            elif processing_mode == "cloud_offloaded":
                cloud_latency = getattr(instance, 'cloud_latency', 0.0)
                return {
                    'location': 'cloud_offloaded',
                    'server_id': instance.server_id,  # AI服务器ID
                    'latency': cloud_latency
                }
            elif processing_mode == "pending_decision":
                # 新增：处理待RL决策状态
                return {
                    'location': 'pending_decision',
                    'server_id': instance.server_id,  # AI服务器ID已确定
                    'latency': 0.0  # 延迟待RL决策后确定
                }

    return {
        'location': 'unknown',
        'server_id': 'unknown',
        'latency': 0.0
    }

