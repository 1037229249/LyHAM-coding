import random
import numpy as np

np.set_printoptions(linewidth=200, suppress=True, precision=3)
from typing import List, Set, Optional, Tuple
from Constant import EdgeServer, ServerType, SystemState, NetworkTopology, Microservice, ServiceChain, RequestFlow,Link
import Deployment

def initialize_traditional_servers(system_state: SystemState, server_count: int,
                                   seed: Optional[int] = None) -> List[EdgeServer]:
    """
    初始化传统边缘服务器并添加到系统状态中
    Args:
        system_state (SystemState): 系统状态对象
        server_count (int): 传统服务器数量
    Returns:
        List[EdgeServer]: 传统边缘服务器列表
    """
    if seed is not None:
        random.seed(seed)
    traditional_servers = []

    for i in range(1, server_count+1):
        cpu_cores = random.randint(6, 8)
        memory_per_core = random.uniform(1.6, 2)
        memory_capacity = cpu_cores * memory_per_core

        server = EdgeServer(
            server_id=f"tra_v{i}",
            server_type=ServerType.TRADITIONAL,
            cpu_cores = cpu_cores,
            memory_capacity= memory_capacity,
            gpu_units=0,  # 传统服务器无GPU
            gpu_memory=0.0,
            model_storage=0.0,
            base_power=85.0,
            cpu_power_coeff=5.0,
            gpu_power_coeff=0.0,  # 传统服务器无GPU功耗
            energy_threshold=0.0, # 传统服务器不参与队列
            energy_queue_factor=50.0,
            delay_queue_factor= 10.0
        )
        system_state.add_edge_server(server)
        traditional_servers.append(server)
    return traditional_servers


def initialize_ai_servers(system_state: SystemState, server_count: int,
                          seed: Optional[int] = None,
                          gpu_units_range: Optional[Tuple[int, int]] = None,
                          max_batch_size: int = 64) -> List[EdgeServer]:
    """
    初始化AI边缘服务器并添加到系统状态中
    """
    if seed is not None:
        random.seed(seed)
    ai_server = []
    for i in range(1, server_count + 1):
        gpu_units = random.randint(gpu_units_range[0], gpu_units_range[1]) if gpu_units_range else 40
        server = EdgeServer(
            server_id=f"ai_v{i}",
            server_type=ServerType.AI_CAPABLE,
            cpu_cores=40,
            memory_capacity=64.0,
            gpu_units=gpu_units,
            gpu_memory=256.0,
            model_storage=300.0,
            base_power=70.0,
            cpu_power_coeff=5.0,
            gpu_power_coeff=150.0,
            energy_threshold=0.098,
            energy_queue_factor=5.0,
            delay_threshold=16,  # 延迟阈值 15ms
            delay_queue_factor=0.1,  #延迟队列缩放因子
            model_name="llama-3-8b",
            max_batch_size=max_batch_size,
            prefill_speed_tokens_per_sec=500000.0,
            decode_speed_tokens_per_sec=30000.0,
        )
        system_state.add_edge_server(server)
        ai_server.append(server)
    return ai_server


def initialize_network_topology(system_state: SystemState, seed: Optional[int] = None):
    """
    初始化网络拓扑，建立全连接模式：
    - 所有服务器之间都建立连接（全连接）
    - 涉及AI服务器的连接具有更大的延迟
    - 同时创建对应的网络链路对象

    延迟设置策略：
    - 传统服务器之间：低延迟 (1-10ms)
    - 传统-AI服务器之间：中等延迟 (10-25ms)
    - AI服务器之间：高延迟 (25-40ms)

    带宽设置策略：
    - 传统服务器之间：低带宽 (200-600 Mbps)
    - 传统-AI服务器之间：中等带宽 (400-800 Mbps)
    - AI服务器之间：高带宽 (800-1200 Mbps)

    Args:
        system_state (SystemState): 系统状态对象
        seed (Optional[int]): 随机种子，用于确保延迟值可复现。如果为None则使用随机种子
    """
    if not system_state.edge_servers:
        raise ValueError("No edge servers defined for network topology")

    # 设置随机种子
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    server_ids = list(system_state.edge_servers.keys())
    system_state.network_topology = NetworkTopology(server_ids)

    # 分离传统服务器和AI服务器
    traditional_servers = []
    ai_servers = []

    for server_id, server in system_state.edge_servers.items():
        if server.server_type == ServerType.TRADITIONAL:
            traditional_servers.append(server_id)
        else:
            ai_servers.append(server_id)

    # 记录已建立的连接
    established_connections = set()
    connection_count = {server_id: 0 for server_id in server_ids}
    created_links = []

    def establish_connection_with_link(server1: str, server2: str, delay: float, bandwidth: float):
        """建立双向连接并创建对应的链路对象"""
        if server1 == server2:
            return False

        connection_key = tuple(sorted([server1, server2]))
        if connection_key in established_connections:
            return False

        established_connections.add(connection_key)

        # 更新连接计数
        connection_count[server1] += 1
        connection_count[server2] += 1

        # 设置双向延迟
        system_state.network_topology.set_communication_delay(server1, server2, delay)
        system_state.network_topology.set_communication_delay(server2, server1, delay)

        # 创建链路对象（双向链路创建为一个对象，因为是无向图）
        link_id = f"link_{server1}_{server2}"
        link = Link(
            link_id=link_id,
            source_server_id=server1,
            target_server_id=server2,
            bandwidth=bandwidth,
            transmission_delay=delay
        )

        # 初始化可用带宽
        link.available_bandwidth = bandwidth

        # 添加到系统状态
        system_state.add_link(link)
        created_links.append(link)

        return True

    # 1. 传统服务器之间建立全连接（低延迟，低带宽）
    traditional_to_traditional_count = 0
    for i, server1 in enumerate(traditional_servers):
        for j, server2 in enumerate(traditional_servers):
            if i < j:  # 避免重复连接
                delay = random.uniform(2.0, 6.0)  # 传统服务器间低延迟
                bandwidth = random.uniform(200.0, 600.0)  # 传统服务器间低带宽
                if establish_connection_with_link(server1, server2, delay, bandwidth):
                    traditional_to_traditional_count += 1

    # 2. 传统服务器与AI服务器之间建立全连接（中等延迟，中等带宽）
    traditional_to_ai_count = 0
    for traditional_server in traditional_servers:
        for ai_server in ai_servers:
            delay = random.uniform(12.0, 20.0)  # 传统-AI服务器间中等延迟
            bandwidth = random.uniform(400.0, 800.0)  # 传统-AI服务器间中等带宽
            if establish_connection_with_link(traditional_server, ai_server, delay, bandwidth):
                traditional_to_ai_count += 1

    # 3. AI服务器之间建立全连接（高延迟，高带宽）
    ai_to_ai_count = 0
    for i, server1 in enumerate(ai_servers):
        for j, server2 in enumerate(ai_servers):
            if i < j:  # 避免重复连接
                delay = random.uniform(20.0, 30.0)  # AI服务器间高延迟
                bandwidth = random.uniform(800.0, 1200.0)  # AI服务器间高带宽
                if establish_connection_with_link(server1, server2, delay, bandwidth):
                    ai_to_ai_count += 1

    # 更新标准化延迟矩阵
    system_state.network_topology.update_normalized_delay_matrix()

    # 输出统计信息
    print(f"网络拓扑初始化完成（全连接模式），建立了 {len(established_connections)} 个连接")
    print(f"创建了 {len(created_links)} 个网络链路")
    print(f"传统服务器: {len(traditional_servers)} 台，AI服务器: {len(ai_servers)} 台")

    # 输出链路类型统计
    print(f"\n=== 链路类型统计 ===")
    print(f"传统-传统链路: {traditional_to_traditional_count} 个")
    print(f"传统-AI链路: {traditional_to_ai_count} 个")
    print(f"AI-AI链路: {ai_to_ai_count} 个")

    '''
    # 输出链路带宽和延迟统计
    delays = [link.transmission_delay for link in created_links]
    bandwidths = [link.bandwidth for link in created_links]
    print(f"\n=== 链路性能统计 ===")
    print(f"传输延迟范围: {min(delays):.2f} - {max(delays):.2f} ms")
    print(f"平均传输延迟: {np.mean(delays):.2f} ms")
    print(f"带宽范围: {min(bandwidths):.2f} - {max(bandwidths):.2f} Mbps")
    print(f"平均带宽: {np.mean(bandwidths):.2f} Mbps")
    print(f"总带宽容量: {sum(bandwidths):.2f} Mbps")
    '''
    # ================ 输出标准化传输延迟矩阵================
    print(f"\n=== 标准化传输延迟矩阵===")
    SE = system_state.network_topology.get_normalized_delay_matrix()

    print(f"SE矩阵形状: {SE.shape}")
    print(f"非零元素个数: {np.count_nonzero(SE)}")
    print(f"最大标准化延迟: {np.max(SE):.4f}")
    if np.count_nonzero(SE) > 0:
        print(f"平均标准化延迟 (非零): {np.mean(SE[SE > 0]):.4f}")
    '''
    print("\nSE矩阵 (标准化传输延迟矩阵):")
    print("服务器ID顺序:", server_ids)
    print(SE)
    '''
    # 显示原始延迟矩阵作为对比
    print(f"\n=== 原始延迟矩阵 (ms) ===")
    raw_delay_matrix = system_state.network_topology.raw_delay_matrix
    print("原始延迟矩阵 (单位: ms):")
    print(raw_delay_matrix)
    print(f"最大原始延迟: {np.max(raw_delay_matrix):.2f} ms")
    if np.count_nonzero(raw_delay_matrix) > 0:
        print(f"平均原始延迟 (非零): {np.mean(raw_delay_matrix[raw_delay_matrix > 0]):.2f} ms")

    '''
    # 输出链路详细信息（前10个链路作为示例）
    print(f"\n=== 链路详细信息  ===")
    for i, link in enumerate(created_links[:]):
        server1_type = system_state.edge_servers[link.source_server_id].server_type.value
        server2_type = system_state.edge_servers[link.target_server_id].server_type.value
        print(f"{link.link_id}: {link.source_server_id}({server1_type}) ↔ {link.target_server_id}({server2_type})")
        print(f"  延迟: {link.transmission_delay:.2f} ms, 带宽: {link.bandwidth:.2f} Mbps")
    '''


    return created_links


def generate_unique_microservice_ids(chain_length: int, seed: Optional[int] = None) -> tuple[List[int], int]:
    """
    为单条服务链生成唯一的微服务ID
    Args:
        chain_length (int): 服务链长度
        seed (Optional[int]): 随机种子
    Returns:
        tuple[List[int], int]: (传统微服务ID列表, AI微服务ID)
    """
    # 可用的ID范围：1-8
    t_available_ids = list(range(1, 9))
    # ai可用的ID范围：1-4
    a_available_ids = random.randint(1, 10)
    if seed is not None:
        random.seed(seed)

    # 随机打乱可用ID
    random.shuffle(t_available_ids)

    # 检查链长度是否超过可用ID数量
    if chain_length > len(t_available_ids):
        raise ValueError(f"服务链长度 {chain_length} 超过了可用微服务ID数量 {len(t_available_ids)}")

    # 分配ID：前chain_length-1个给传统微服务，最后一个给AI微服务
    traditional_ids = t_available_ids[:chain_length - 1]
    ai_id = a_available_ids

    return traditional_ids, ai_id


def initialize_request_flows(system_state: SystemState, flow_count: int, seed: Optional[int] = None,
                             chain_length_range: tuple = (6, 8),
                             fixed_arrival_rate: float = None,
                             arrival_range_req_s: Tuple[float, float] = (4.0, 8.0),
                             input_tokens_range: Tuple[int, int] = (100, 2000),
                             output_tokens_range: Tuple[int, int] = (50, 500)) -> List[RequestFlow]:
    """
    初始化请求流，确保每条链内微服务ID唯一且在1-8范围内
    Args:
        system_state (SystemState): 系统状态对象
        flow_count (int): 请求流数量
        seed (Optional[int]): 随机种子，用于确保可复现。如果为None则使用随机种子
        chain_length_range: 服务链长度范围
        fixed_arrival_rate: 固定到达率（如果指定，则所有请求流使用相同的到达率）
    Returns:
        List[RequestFlow]: 创建的请求流列表
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    created_flows = []

    for flow_idx in range(1, flow_count + 1):
        # 1. 随机确定服务链长度（4-8个微服务，确保不超过可用ID数量）
        chain_length = random.randint(chain_length_range[0], chain_length_range[1])

        # 2. 为当前链生成唯一的微服务ID
        traditional_ids, ai_id = generate_unique_microservice_ids(
            chain_length, seed=seed + flow_idx if seed is not None else None)

        # 3. 创建传统微服务列表
        traditional_microservices = []
        for ms_id in traditional_ids:
            traditional_ms = Microservice(
                ms_id=f"m{ms_id}",
                service_type="traditional",
                service_rate=random.uniform(2.0, 8.0)
            )
            traditional_microservices.append(traditional_ms)
            # 使用全局唯一键：flow_id + ms_id
            global_ms_key = f"{flow_idx}_{traditional_ms.ms_id}"
            system_state.microservices[global_ms_key] = traditional_ms

        # 4. 对传统微服务进行随机排序
        random.shuffle(traditional_microservices)

        # 5. 创建AI微服务（链的最后一个）
        ai_ms = Microservice(
            ms_id=f"a{ai_id}",
            service_type="ai",
            service_rate=random.uniform(2.0, 8.0)
        )
        # 使用全局唯一键
        global_ai_key = f"{flow_idx}_{ai_ms.ms_id}"
        system_state.microservices[global_ai_key] = ai_ms

        # 6. 组合微服务列表：随机排序的传统微服务 + AI微服务
        microservices = traditional_microservices + [ai_ms]

        # 7. 创建服务链
        chain_id = f"chain_{flow_idx}"
        service_chain = ServiceChain(chain_id, microservices)
        system_state.service_chains[chain_id] = service_chain

        # 8. 随机确定优先级（1-5）
        priority = random.randint(1, 5)

        # 9. 根据链长确定最大延迟阈值（不含ai版）
        base_latency = 50.0
        chain_length_bonus = (chain_length - 2) * 8.0  # 调整基准，因为最小链长现在是4
        random_variation = random.uniform(-5.0, 10.0)
        max_latency = base_latency + chain_length_bonus + random_variation
        max_latency = max(50.0, min(max_latency, 90.0))

        # 10. 确定初始到达率（如果指定了固定到达率则使用固定值，否则随机生成）
        if fixed_arrival_rate is not None:
            initial_arrival_rate = fixed_arrival_rate
        else:
            initial_arrival_rate = random.uniform(arrival_range_req_s[0], arrival_range_req_s[1])

        # 11. 初始化资源需求属性
        # 网络资源需求
        bandwidth_requirement = random.uniform(5.0, 50.0)  # 5-50 Mbps

        # 传统计算资源需求
        traditional_ms_count = len(traditional_microservices)
        data_memory_requirement = traditional_ms_count * random.uniform(1.0, 2.0)  # 每个传统微服务1-2GB

        # AI计算资源需求（预估的数据传输量）
        r_input_data_size = random.uniform(input_tokens_range[0], input_tokens_range[1])
        r_output_data_size = random.uniform(output_tokens_range[0], output_tokens_range[1])

        # 云端处理参数
        cloud_latency_base = random.uniform(13.0, 15.0)  # 云端基础延迟40-80ms
        cloud_bandwidth = random.uniform(120.0, 200.0)  # 云端带宽80-200Mbps

        # 12. 创建请求流（每个请求流都包含AI微服务在结尾）
        flow_id = f"flow_{flow_idx}"
        request_flow = RequestFlow(
            flow_id=flow_id,
            service_chain=service_chain,
            arrival_rate=initial_arrival_rate,
            max_latency=max_latency,
            priority=priority,
            ca_latency= 0,
            # 网络资源需求
            bandwidth_requirement=bandwidth_requirement,

            # 传统计算资源需求
            data_memory_requirement=data_memory_requirement,

            # AI计算资源需求
            r_input_data_size=r_input_data_size,
            r_output_data_size=r_output_data_size,

            # AI微服务决策（默认本地处理）
            ai_processing_choice="local",

            # 路由概率（初始化为空字典，后续由算法计算）
            routing_probabilities={},

            # 云端参数
            cloud_latency_base=cloud_latency_base,
            cloud_bandwidth=cloud_bandwidth
        )

        # 13. 添加到系统状态
        system_state.request_flows[flow_id] = request_flow
        Deployment.calculate_ms_resource(request_flow)
        created_flows.append(request_flow)

    print(f"\n=== 请求流初始化完成 ===")
    print(f"创建了 {len(created_flows)} 个请求流")
    print(f"微服务总数: {len([ms for ms in system_state.microservices.values()])} 个")
    print(
        f"传统微服务: {len([ms for ms in system_state.microservices.values() if ms.service_type == 'traditional'])} 个")
    print(f"AI微服务: {len([ms for ms in system_state.microservices.values() if ms.service_type == 'ai'])} 个")

    # 显示每个请求流的详细信息和ID唯一性验证
    #print(f"\n=== 请求流详细信息及ID唯一性验证 ===")
    for flow in created_flows:
        chain_sequence = " → ".join([ms.ms_id for ms in flow.service_chain.microservices])
        ms_ids_in_chain = [ms.ms_id for ms in flow.service_chain.microservices]

        # 验证链内ID唯一性
        unique_ids = set(ms_ids_in_chain)
        is_unique = len(ms_ids_in_chain) == len(unique_ids)

        # 验证ID范围
        id_numbers = []
        for ms_id in ms_ids_in_chain:
            if ms_id.startswith('m') or ms_id.startswith('a'):
                try:
                    id_num = int(ms_id[1:])
                    id_numbers.append(id_num)
                except ValueError:
                    pass

        in_range = all(1 <= id_num <= 8 for id_num in id_numbers)
        '''
        print(f"{flow.flow_id}:")
        print(f"  微服务链: {chain_sequence}")
        print(f"  链内ID唯一性: {'✓' if is_unique else '✗'}")
        print(f"  ID范围(1-8): {'✓' if in_range else '✗'}")
        print(f"  ID数值: {id_numbers}")
        print(f"  带宽需求: {flow.bandwidth_requirement:.2f} Mbps")
        print(f"  数据内存: {flow.data_memory_requirement:.2f} GB")
        print(f"  延迟阈值: {flow.max_latency:.2f} ms")
        print()
        '''
    # 统计信息
    chain_lengths = [len(flow.service_chain.microservices) for flow in created_flows]
    arrival_rates = [flow.arrival_rate for flow in created_flows]
    bandwidth_reqs = [flow.bandwidth_requirement for flow in created_flows]
    memory_reqs = [flow.data_memory_requirement for flow in created_flows]

    '''
    print(f"=== 资源需求统计 ===")
    print(f"服务链长度: {min(chain_lengths)} - {max(chain_lengths)} (平均: {np.mean(chain_lengths):.1f})")
    print(f"到达率: {min(arrival_rates):.2f} - {max(arrival_rates):.2f} req/s (平均: {np.mean(arrival_rates):.2f})")
    print(f"带宽需求: {min(bandwidth_reqs):.2f} - {max(bandwidth_reqs):.2f} Mbps (平均: {np.mean(bandwidth_reqs):.2f})")
    print(f"数据内存需求: {min(memory_reqs):.2f} - {max(memory_reqs):.2f} GB (平均: {np.mean(memory_reqs):.2f})")
    '''

    return created_flows
