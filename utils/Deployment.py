import numpy as np
import random
import math
import copy
from fractions import Fraction
from typing import List, Dict, Tuple, Optional
from Constant import RequestFlow, ServiceChain, Microservice, SystemState, MicroserviceInstance, ServerType, EdgeServer


def normalize_arrival_rate_for_queue(arrival_rate: float) -> float:
    """
    将论文场景的req/s转换为排队模型内部使用的req/ms
    旧代码历史上混用了两个单位，这里只在资源估算内部转换。
    """
    return arrival_rate / 1000.0 if arrival_rate > 1.0 else arrival_rate


def calculate_single_required_cores(arrival_rate: float, processing_rate: float,
                                       tolerable_time: float, max_cores: int = 20) -> int:
    """
    计算单个微服务满足延迟条件下所需的核心数
    使用M/M/C排队论模型
    Args:
        arrival_rate: 微服务的到达率 (requests/ms)
        processing_rate: 单核心处理率 (requests/ms/core)
        tolerable_time: 容忍延迟 (ms)
        max_cores: 最大可用核心数
    Returns:
        int: 所需的核心数
    """
    if arrival_rate <= 0 or processing_rate <= 0 or tolerable_time <= 0:
        return 0

    Service_intensity = arrival_rate / processing_rate  # ρ = λ/μ
    ms_resource = 0
    # 遍历核心数，找到满足延迟要求的最小核心数
    for i in range(max_cores):
        C = i + 1  # 服务器数量(核心数)
        CService_intensity = arrival_rate / (processing_rate  * C)  # ρ/c
        # 排队论稳定性条件：服务强度必须小于1
        if 0 < CService_intensity < 1:
            try:
                # 计算P0 (系统空闲概率)
                # P0 = [Σ(k=0 to c-1) (ρ^k/k!) + (ρ^c/c!) * 1/(1-ρ/c)]^(-1)
                # 第一部分：Σ(k=0 to c-1) (ρ^k/k!)
                v1 = 0.0
                for k in range(C):
                    v2 = math.factorial(k)
                    v3 = math.pow(Service_intensity, k) / v2
                    v1 += v3

                # 第二部分：(ρ^c/c!) * 1/(1-ρ/c)
                v4 = math.factorial(C)
                second_part = math.pow(Service_intensity, C) / (v4 * (1 - CService_intensity))
                # 计算P0
                p0 = 1.0 / (v1 + second_part)
                # 计算平均等待时间 (M/M/C队列延迟公式)
                # W = (ρ/c)/(1-ρ/c) * (ρ^c/c!) * P0 / λ + 1/μ
                numerator = CService_intensity * math.pow(Service_intensity, C) * p0
                denominator = arrival_rate * math.pow(1 - CService_intensity, 2) * v4
                queue_delay = numerator / denominator
                # 总延迟 = 队列延迟 + 处理延迟
                total_delay = queue_delay + (1.0 / processing_rate)
                #print(f"总延迟计算为  {total_delay}  ms")
                # 检查是否满足延迟要求
                if total_delay <= tolerable_time:
                    ms_resource = C
                    break

            except (OverflowError, ZeroDivisionError):
                # 数值计算溢出或除零，跳过此配置
                continue

    return ms_resource


def calculate_single_required_memory(arrival_rate: float, allocated_cores: int,
                                     base_memory: float = 1.2) -> float:
    """
    计算单个微服务所需的内存量
    基于初始化条件：每个传统微服务1-2GB，考虑负载和核心数
    Args:
        arrival_rate: 微服务的到达率 (requests/ms)
        allocated_cores: 分配的核心数
        base_memory: 基础内存需求 (GB) - 对应InitSystem中的1-2GB范围中位数

    Returns:
        float: 所需的内存量 (GB)
    """
    if arrival_rate <= 0 or allocated_cores <= 0:
        return max(1.0, base_memory)

    # 基于核心数的内存需求：核心数越多，说明负载越重，需要更多内存缓存数据
    # 每个核心额外需要0.15GB内存用于缓存和并发处理
    core_memory = allocated_cores * 0.15

    # 基于到达率的动态调整：高到达率需要更多内存用于请求缓冲
    # 将到达率标准化到0-1范围（假设最大到达率为10 req/ms）
    arrival_factor = min(arrival_rate / 10.0, 1.0)
    arrival_memory = arrival_factor * 0.5  # 最多额外0.5GB

    # 计算总内存需求
    total_memory = base_memory + core_memory + arrival_memory

    # 下限1GB保证基本运行，上限2GB避免过度分配
    required_memory = max(1.0, min(total_memory, 2.0))

    return round(required_memory, 2)

def calculate_ms_resource(flow: RequestFlow) -> int:
    import Waitingtime
    """
    计算一条微服务链除去AI微服务之外需要的所有核心数总和
    Args:
        flow: 请求流对象，包含服务链和相关参数
    Returns:
        int: 所有传统微服务所需的总核心数
    """
    total_cores_required = 0
    total_memory_required = 0
    # 获取请求流参数
    arrival_rate = normalize_arrival_rate_for_queue(flow.arrival_rate)  # 排队模型内部到达率(req/ms)
    chain = flow.service_chain

    # 计算去除通信延迟后的可用延迟时间
    # 减去最大通信时延，极端情况: (链长-2) * 10ms + 25ms的处理开销
    extra_latency = (chain.chain_length - 2) * 10 + 25

    # 可用于微服务处理的延迟时间
    available_latency = flow.max_latency - extra_latency

    if available_latency <= 0:
        print(f"Warning: 可用延迟时间为负值 ({available_latency:.3f}s)，无法满足延迟要求")
        return 0

    # 获取传统微服务列表 (排除AI微服务)
    traditional_microservices = chain.get_traditional_microservices()
    traditional_chain = chain.get_traditional_microservice_chain()

    '''
    print(f"\n=== 计算请求流 {flow.flow_id} 的资源需求 ===")
    print(f"到达率: {flow.arrival_rate:.2f} req/s, 内部换算 {arrival_rate:.4f} req/ms")
    print(f"总延迟阈值: {flow.max_latency:.2f} ms")
    print(f"预留通信延迟: {extra_latency:.2f} ms")
    print(f"可用处理延迟: {available_latency:.3f} ms")
    print(f"传统微服务数量: {len(traditional_microservices)}")
    '''

    # 为每个传统微服务计算所需核心数
    for ms in traditional_microservices:
        # 计算该微服务的公平分配容忍时延
        single_latency = Waitingtime.calculate_ms_tolerable_time_fairly(traditional_chain, ms.ms_id, available_latency)

        '''
        print(f"\n微服务 {ms.ms_id}:")
        print(f"  处理速率: {ms.service_rate:.2f} req/ms/core")
        print(f"  分配延迟: {single_latency:.3f} ms ")
        '''

        if single_latency > 0:
            # 计算满足延迟要求的所需核心数
            required_cores = calculate_single_required_cores(
                arrival_rate=arrival_rate,
                processing_rate=ms.service_rate,
                tolerable_time=single_latency,
                max_cores=20  # 单台服务器最多20核心
            )
            # 计算所需内存
            required_memory = calculate_single_required_memory(
                arrival_rate=arrival_rate,
                allocated_cores=required_cores,
                base_memory=1.2  # 使用1-4GB范围的中位数作为基础内存
            )
            chain.set_core_allocation(ms.ms_id , required_cores)
            chain.set_memory_allocation(ms.ms_id ,required_memory)

            total_cores_required += required_cores
            total_memory_required += required_memory
            #print(f"  所需核心数: {required_cores}")

            # 验证计算结果
            if required_cores > 0:
                actual_service_rate = required_cores * ms.service_rate
                utilization = arrival_rate / actual_service_rate
                #print(f"  服务强度: {utilization:.3f}")
            else:
                print(f"  Warning: 无法为微服务 {ms.ms_id} 找到满足延迟要求的核心配置")
        else:
            print(f"  Warning: 微服务 {ms.ms_id} 分配到的延迟时间为0或负值")
    '''
    print(f"\n=== 汇总结果 ===")
    print(f"传统微服务总核心需求: {total_cores_required}")
    print(f"平均每微服务核心数: {total_cores_required / len(traditional_microservices) if traditional_microservices else 0:.2f}")
    print(f"传统微服务总内存需求: {total_memory_required:.2f} GB")
    '''
    return total_cores_required


def get_stream_vector(request_flow: RequestFlow, system_state: SystemState) -> np.ndarray:
    """
    计算请求流的4维特征向量

    特征向量组成：
    [0] 容忍时延 (ms)
    [1] 总资源需求 (CPU+内存综合分数)
    [2] 到达率 (req/ms)
    [3] 带宽需求 (Mbps)
    Args:
        request_flow: 请求流对象
        system_state: 系统状态对象

    Returns:
        np.ndarray: 4维特征向量
    """
    vector = np.zeros(4, dtype=np.float64)

    # vector[0]: 流的最大容忍时延
    vector[0] = request_flow.max_latency

    # vector[1]: 流需要的总资源需求（CPU+内存综合）
    # 直接从服务链的已计算数据中读取
    vector[1] = calculate_total_resource_requirement(request_flow, system_state)

    # vector[2]: 流的到达率
    vector[2] = request_flow.arrival_rate

    # vector[3]: 流的带宽需求
    vector[3] = request_flow.bandwidth_requirement

    return vector


def calculate_total_resource_requirement(request_flow: RequestFlow, system_state: SystemState) -> float:
    """
    计算请求流的总资源需求（CPU + 内存的综合指标）
    直接从服务链的已计算数据中读取
    Args:
        request_flow: 请求流对象
        system_state: 系统状态对象
    Returns:
        float: 总计算资源需求（标准化后的综合分数）
    """
    service_chain = request_flow.service_chain.get_traditional_microservice_chain()
    total_cpu_cores = 0.0
    total_memory_gb = 0.0

    # 从服务链的存储数据中读取资源需求
    for microservice in service_chain.microservices:
        ms_id = microservice.ms_id

        # 读取已计算的核心分配数
        cpu_cores = service_chain.core_allocations.get(ms_id, 0)
        total_cpu_cores += cpu_cores

        # 读取已计算的内存分配数
        memory_gb = service_chain.memory_allocations.get(ms_id, 0.0)
        total_memory_gb += memory_gb

        #print(f"微服务 {ms_id}: {cpu_cores} 核心, {memory_gb:.2f} GB 内存")

    # 计算综合资源分数
    # CPU权重0.7，内存权重0.3
    # 内存标准化：假设每核心对应2GB内存作为基准
    cpu_score = total_cpu_cores
    memory_score = total_memory_gb / 2.0  # 标准化内存分数

    total_resource_score = cpu_score * 0.7 + memory_score * 0.3
    '''
    print(f"请求流 {request_flow.flow_id} 资源汇总:")
    print(f"  总CPU核心: {total_cpu_cores}")
    print(f"  总内存: {total_memory_gb:.2f} GB")
    print(f"  综合资源分数: {total_resource_score:.2f}")
    '''
    return total_resource_score


def calculate_stream_distance_matrix(request_flows: List[RequestFlow], system_state: SystemState) -> np.ndarray:
    """
    计算所有请求流之间的差异度矩阵
    Args:
        request_flows: 请求流列表
        system_state: 系统状态对象
    Returns:
        np.ndarray: 差异度矩阵
    """
    num_flows = len(request_flows)
    distance_matrix = np.zeros((num_flows, num_flows))

    # 计算所有请求流的特征向量
    vectors = []
    for flow in request_flows:
        vector = get_stream_vector(flow, system_state)
        vectors.append(vector)
    '''
    print("所有请求流的特征向量:")
    for i, (flow, vector) in enumerate(zip(request_flows, vectors)):
        print(f"{flow.flow_id}: {vector}")
    '''
    # 计算两两之间的差异度
    for i in range(num_flows):
        for j in range(num_flows):
            if i == j:
                distance_matrix[i][j] = -1  # 自己与自己的距离设为-1
            else:
                vector_i = vectors[i]
                vector_j = vectors[j]

                # 计算欧几里得距离
                euclidean_distance = np.sqrt(np.sum((vector_i - vector_j) ** 2))

                # 计算皮尔逊相关系数
                pearson_score = calculate_pearson_correlation(vector_i, vector_j)

                # 计算最终差异度
                if abs(pearson_score) > 1e-6:  # 避免除零错误
                    distance_matrix[i][j] = abs(euclidean_distance / pearson_score)
                else:
                    distance_matrix[i][j] = euclidean_distance  # 当相关系数接近0时使用欧几里得距离

    return distance_matrix


def calculate_pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """
    计算两个向量之间的皮尔逊相关系数
    Args:
        x: 第一个向量
        y: 第二个向量
    Returns:
        float: 皮尔逊相关系数
    """
    if len(x) != len(y):
        raise ValueError("向量长度不一致")

    if len(x) == 0:
        return 0.0

    # 计算均值
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # 计算分子和分母
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator_x = np.sqrt(np.sum((x - x_mean) ** 2))
    denominator_y = np.sqrt(np.sum((y - y_mean) ** 2))

    # 避免除零错误
    if denominator_x == 0 or denominator_y == 0:
        return 0.0

    correlation = numerator / (denominator_x * denominator_y)
    return correlation


def next_fit_deployment(system_state):
    """
    Next Fit部署算法
    只部署传统微服务到传统服务器上
    算法逻辑：
    1. 利用已计算的差异度矩阵选择流对
    2. 利用已计算的资源需求进行部署决策（核心数已通过M/M/C模型满足延迟要求）
    3. 使用DecisionVariables管理部署决策
    4. 使用NetworkTopology计算通信延迟（处理延迟已在资源计算阶段解决）
    Args:
        system_state: 系统状态对象，包含服务器、请求流、网络拓扑等信息
    注意：微服务的核心分配已在calculate_ms_resource()中通过排队论计算，
         确保满足延迟要求，部署阶段主要考虑通信延迟和资源分配。
    """
    from Constant import DecisionVariables, ServerType, MicroserviceInstance
    #print("=== 开始Next Fit部署算法 ===")
    # 获取所有请求流
    request_flows = list(system_state.request_flows.values())
    if not request_flows:
        #print("没有请求流需要部署")
        return False

    # 利用已计算的差异度矩阵
    if system_state.stream_distance_matrix is None:
        #print("差异度矩阵未初始化，重新计算...")
        system_state.stream_distance_matrix = calculate_stream_distance_matrix(
            request_flows, system_state)

    distance_matrix = system_state.stream_distance_matrix.copy()

    # 获取传统服务器列表
    traditional_servers = [server for server in system_state.edge_servers.values()
                           if server.server_type.value == "traditional"]

    if not traditional_servers:
        #print("没有传统服务器可用于部署")
        return False

    # 创建决策变量对象来管理部署决策
    decision_vars = DecisionVariables(system_state.time_frame)

    # 创建processed_streams标记
    processed_streams = [False] * len(request_flows)

    # 部署结果统计
    successful_deployments = 0
    '''
    print(f"初始服务器资源状况:")
    for server in sorted(traditional_servers, key=calculate_server_resource_score, reverse=True):
        score = calculate_server_resource_score(server)
        print(f"  {server.server_id}: {server.available_cpu}/{server.cpu_cores} 核心")
        print(f"    可用内存: {server.available_memory:.2f}/{server.memory_capacity:.2f} GB")
        print(f"    综合得分: {score:.2f}")
    '''
    # 主循环
    while not all(processed_streams):
        # 1. 找到剩余资源最多的服务器（综合考虑CPU和内存）
        best_server = max(traditional_servers,
                          key=lambda s: calculate_server_resource_score(s))

        # 2. 找到差异度最大的两个未处理请求流
        max_distance = -1
        selected_ri = -1
        selected_rj = -1

        for i in range(len(request_flows)):
            for j in range(len(request_flows)):
                if (not processed_streams[i] and not processed_streams[j] and
                        i != j and distance_matrix[i][j] > max_distance):
                    max_distance = distance_matrix[i][j]
                    selected_ri = i
                    selected_rj = j

        # 如果没有找到流对，处理剩余的单个流
        if selected_ri == -1:
            for i in range(len(request_flows)):
                if not processed_streams[i]:
                    selected_ri = i
                    selected_rj = -1
                    break
            if selected_ri == -1:
                break
        '''
        print(f"\n=== 处理请求流对 ===")
        print(f"最大差异度: {max_distance:.4f}")
        print(f"选中请求流: {selected_ri + 1} 和 {selected_rj + 1 if selected_rj != -1 else '(单个流)'}")
        '''
        # 3. 获取选中的请求流并按资源需求排序
        flows_to_process = []

        flow_ri = request_flows[selected_ri]
        # 利用get_stream_vector获取资源需求
        ri_vector = get_stream_vector(flow_ri, system_state)
        ri_resource_need = ri_vector[1]  # vector[1]是总资源需求
        flows_to_process.append((flow_ri, ri_resource_need, selected_ri))

        if selected_rj != -1:
            flow_rj = request_flows[selected_rj]
            rj_vector = get_stream_vector(flow_rj, system_state)
            rj_resource_need = rj_vector[1]
            flows_to_process.append((flow_rj, rj_resource_need, selected_rj))

        # 按资源需求降序排序
        flows_to_process.sort(key=lambda x: x[1], reverse=True)


        # 4. 处理每个请求流
        for flow, resource_need, flow_idx in flows_to_process:
            #print(f"\n正在为请求流 {flow.flow_id} 分配到服务器 {best_server.server_id}")
            #print(f"请求流 {flow.flow_id} 需要资源: {resource_need:.2f}")

            success = handle_single_flow_deployment(
                flow, best_server, traditional_servers, system_state, decision_vars)

            if success:
                successful_deployments += 1
                processed_streams[flow_idx] = True

                # 将对应的距离矩阵行列设为-1
                for k in range(len(request_flows)):
                    distance_matrix[flow_idx][k] = -1
                    distance_matrix[k][flow_idx] = -1

                #print(f"请求流 {flow.flow_id} 部署成功")
            else:
                print(f"请求流 {flow.flow_id} 部署失败")
                processed_streams[flow_idx] = True
        '''
        print(f"\n更新后的服务器资源排序:")
        for server in sorted(traditional_servers, key=calculate_server_resource_score, reverse=True):
            score = calculate_server_resource_score(server)
            print(f"  {server.server_id}: {server.available_cpu} 核心, {server.available_memory:.2f} GB (综合得分: {score:.2f})")
    print(f"\n=== 部署完成，开始计算路由概率矩阵 ===")
    '''
    # *** 部署完成后计算路由概率矩阵 ***
    try:
        # 计算所有请求流的路由转移概率
        system_state.calculate_stream_transfer_probabilities()

        # 验证路由概率的合法性
        validation_errors = system_state.validate_transfer_probabilities()
        if validation_errors:
            print("⚠️  路由概率验证发现错误:")
            for flow_id, errors in validation_errors.items():
                print(f"  {flow_id}: {errors}")
        else:
            print("✓ 路由概率验证通过")

        # 显示路由概率矩阵概要
        summary = system_state.get_routing_summary()
        '''
        print(f"\n=== 路由概率矩阵概要 ===")
        for flow_id, stats in summary.items():
            print(f"{flow_id}:")
            print(f"  路径数量: {stats['total_paths']}")
            print(f"  涉及服务器: {stats['servers_count']} 台")
            print(f"  涉及微服务: {stats['microservices_count']} 个")
        '''
    except Exception as e:
        print(f"⚠️  路由概率矩阵计算失败: {e}")
        import traceback
        traceback.print_exc()

    #print(f"=== Next Fit部署算法完成 ===")
    return successful_deployments == len(request_flows)

def calculate_server_resource_score(server) -> float:
    """
    计算服务器的综合资源得分（CPU + 内存）
    用于服务器选择优先级排序
    Args:
        server: 边缘服务器对象
    Returns:
        float: 综合资源得分（CPU权重0.7 + 内存权重0.3）
    """
    # CPU得分：可用核心数
    cpu_score = float(server.available_cpu)

    # 内存得分：可用内存GB数，标准化到与CPU相近的量级
    memory_score = server.available_memory / 2.0  # 假设每核心对应2GB内存

    # 综合得分：CPU权重0.7，内存权重0.3
    comprehensive_score = cpu_score * 0.95 + memory_score * 0.05

    return comprehensive_score

def handle_single_flow_deployment(flow, preferred_server, all_servers, system_state, decision_vars) -> bool:
    """
    处理单个请求流的部署，充分利用现有代码
    Args:
        flow: 请求流对象
        preferred_server: 首选服务器
        all_servers: 所有传统服务器列表
        system_state: 系统状态
        decision_vars: 决策变量对象

    Returns:
        bool: 是否部署成功
    """
    from Constant import ServerType
    # 获取传统微服务列表
    traditional_microservices = flow.service_chain.get_traditional_microservices()

    if not traditional_microservices:
        print(f"请求流 {flow.flow_id} 没有传统微服务需要部署")
        return True

    # 利用ServiceChain中已计算的资源分配信息
    total_cores_needed = 0
    total_memory_needed = 0
    ms_requirements = {}

    for ms in traditional_microservices:
        # 从ServiceChain的core_allocations中获取已计算的核心需求
        required_cores = flow.service_chain.core_allocations.get(ms.ms_id, 1)
        required_memory = flow.service_chain.memory_allocations.get(ms.ms_id, 1.2)

        ms_requirements[ms.ms_id] = {
            'cores': required_cores,
            'memory': required_memory
        }

        total_cores_needed += required_cores
        total_memory_needed += required_memory
        '''
        print(f"  微服务 {ms.ms_id}: 需要 {required_cores} 核心, {required_memory:.2f} GB内存")
        print(f"    服务率: {ms.service_rate:.2f} req/ms/core")
        
    print(f"总需求: {total_cores_needed} 核心, {total_memory_needed:.2f} GB内存")
    '''
    # 检查首选服务器是否有足够资源
    if (preferred_server.available_cpu >= total_cores_needed and
            preferred_server.available_memory >= total_memory_needed):
        #print(f"请求流 {flow.flow_id} 可以全部部署到 {preferred_server.server_id}")
        return deploy_all_microservices_to_server(
            flow, traditional_microservices, ms_requirements,
            preferred_server, system_state, decision_vars)
    else:
        #print(f"请求流 {flow.flow_id} 无法全部部署到 {preferred_server.server_id}")
        return deploy_microservices_distributed(
            flow, traditional_microservices, ms_requirements,
            preferred_server, all_servers, system_state, decision_vars)


def deploy_all_microservices_to_server(flow, microservices, ms_requirements, server, system_state,
                                       decision_vars) -> bool:
    """
    将所有微服务部署到单个服务器，使用微服务的实际service_rate
    Args:
        flow: 请求流对象
        microservices: 微服务列表
        ms_requirements: 微服务资源需求字典
        server: 目标服务器
        system_state: 系统状态
        decision_vars: 决策变量对象
    Returns:
        bool: 是否部署成功
    """
    from Constant import MicroserviceInstance

    total_cores = sum(req['cores'] for req in ms_requirements.values())
    total_memory = sum(req['memory'] for req in ms_requirements.values())

    # 执行部署
    for ms in microservices:
        req = ms_requirements[ms.ms_id]

        # 使用现有的MicroserviceInstance类创建实例
        instance_id = f"{flow.flow_id}_{ms.ms_id}_{server.server_id}"
        instance = MicroserviceInstance(
            instance_id=instance_id,
            microservice=ms,
            server_id=server.server_id,
            allocated_cores=req['cores'],
            arrival_rate=normalize_arrival_rate_for_queue(flow.arrival_rate)
        )

        # 使用微服务的实际service_rate计算服务指标
        # 延迟要求已经在calculate_ms_resource()中通过M/M/C模型保证
        if ms.service_rate > 0:
            total_service_rate = req['cores'] * ms.service_rate
            instance.service_intensity = normalize_arrival_rate_for_queue(flow.arrival_rate) / total_service_rate
            instance.processing_delay = 1.0 / ms.service_rate

            # 理论队列延迟（实际延迟已在资源分配阶段满足要求）
            if instance.service_intensity < 1:
                instance.queue_delay = instance.service_intensity / (1 - instance.service_intensity) / ms.service_rate
            else:
                instance.queue_delay = 0.0  # 已分配足够核心

        # 更新系统状态
        system_state.microservice_instances[instance_id] = instance

        # 更新决策变量
        decision_vars.set_microservice_deployment(ms.ms_id, server.server_id, 1)
        decision_vars.set_core_allocation((ms.ms_id, server.server_id), req['cores'])

        # *** 设置路由概率矩阵所需的资源分配信息 ***
        system_state.set_stream_allocated_resource(
            flow_id=flow.flow_id,
            ms_id=ms.ms_id,
            server_id=server.server_id,
            allocated_resource=req['cores']
        )
        '''
        print(f"    部署 {ms.ms_id} 到 {server.server_id}: {req['cores']} 核心, {req['memory']:.2f} GB")
        print(f"      服务率: {ms.service_rate:.2f} req/ms/core, 服务强度: {instance.service_intensity:.3f}")
        '''
    # 更新服务器资源
    server.available_cpu -= total_cores
    server.available_memory -= total_memory

    #print(f"部署后 {server.server_id} 剩余 {server.available_cpu} 核心, {server.available_memory:.2f} GB")
    return True


def backup_slow_deployment_state(system_state: SystemState) -> Dict:
    """备份慢层部署状态，用于cluster回退和失败恢复"""
    return {
        "server_states": {
            server_id: {
                "available_cpu": server.available_cpu,
                "available_memory": server.available_memory,
                "available_gpu_units": getattr(server, "available_gpu_units", 0),
                "available_gpu_memory": getattr(server, "available_gpu_memory", 0.0),
                "available_model_storage": getattr(server, "available_model_storage", 0.0),
            }
            for server_id, server in system_state.edge_servers.items()
        },
        "microservice_instances": copy.deepcopy(system_state.microservice_instances),
        "stream_allocated_resources": copy.deepcopy(system_state.stream_allocated_resources),
        "stream_transfer_probabilities": copy.deepcopy(system_state.stream_transfer_probabilities),
        "routing_probabilities": {
            flow_id: dict(flow.routing_probabilities)
            for flow_id, flow in system_state.request_flows.items()
        },
    }


def restore_slow_deployment_state(system_state: SystemState, backup: Dict):
    """恢复慢层部署状态"""
    for server_id, state in backup.get("server_states", {}).items():
        if server_id in system_state.edge_servers:
            server = system_state.edge_servers[server_id]
            server.available_cpu = state["available_cpu"]
            server.available_memory = state["available_memory"]
            server.available_gpu_units = state["available_gpu_units"]
            server.available_gpu_memory = state["available_gpu_memory"]
            server.available_model_storage = state["available_model_storage"]
    system_state.microservice_instances = copy.deepcopy(backup.get("microservice_instances", {}))
    system_state.stream_allocated_resources = copy.deepcopy(backup.get("stream_allocated_resources", {}))
    system_state.stream_transfer_probabilities = copy.deepcopy(backup.get("stream_transfer_probabilities", {}))
    for flow_id, routing in backup.get("routing_probabilities", {}).items():
        if flow_id in system_state.request_flows:
            system_state.request_flows[flow_id].routing_probabilities = dict(routing)


def reset_slow_deployments(system_state: SystemState):
    """清空服务实例并恢复服务器可用资源，虚拟队列不重置"""
    system_state.microservice_instances.clear()
    system_state.stream_allocated_resources.clear()
    system_state.stream_transfer_probabilities.clear()
    for flow in system_state.request_flows.values():
        flow.routing_probabilities.clear()
        flow.ai_processing_choice = "pending"
    for server in system_state.edge_servers.values():
        server.available_cpu = server.cpu_cores
        server.available_memory = server.memory_capacity
        if server.server_type.value == "ai_capable":
            server.available_gpu_units = server.gpu_units
            server.available_gpu_memory = server.gpu_memory
            server.available_model_storage = server.model_storage


def _get_flow_id_from_instance_id(instance_id: str) -> str:
    """从旧实例ID中恢复flow_id"""
    parts = instance_id.split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return parts[0] if parts else ""


def _collect_traditional_units(system_state: SystemState) -> Tuple[List[str], Dict[str, Dict]]:
    """收集RAPPA使用的传统微服务粒度单元"""
    units = []
    unit_meta = {}
    for flow_id in sorted(system_state.request_flows.keys()):
        request_flow = system_state.request_flows[flow_id]
        calculate_ms_resource(request_flow)
        traditional_microservices = request_flow.service_chain.get_traditional_microservices()
        for position, microservice in enumerate(traditional_microservices):
            unit_id = f"{flow_id}|{microservice.ms_id}"
            cores = request_flow.service_chain.core_allocations.get(microservice.ms_id, 1)
            memory = request_flow.service_chain.memory_allocations.get(microservice.ms_id, 1.2)
            units.append(unit_id)
            unit_meta[unit_id] = {
                "flow_id": flow_id,
                "flow": request_flow,
                "microservice": microservice,
                "position": position,
                "cores": int(max(1, cores)),
                "memory": float(max(0.1, memory)),
            }
    return units, unit_meta


def build_service_affinity_matrix(system_state: SystemState) -> Tuple[List[str], np.ndarray]:
    """
    构造RAPPA使用的服务亲和矩阵
    工程实现按同链通信、服务功能相似度和资源差异构造谱聚类输入。
    """
    units, unit_meta = _collect_traditional_units(system_state)
    n = len(units)
    if n == 0:
        return [], np.zeros((0, 0))
    affinity = np.zeros((n, n), dtype=float)
    for i, unit_i in enumerate(units):
        meta_i = unit_meta[unit_i]
        for j, unit_j in enumerate(units):
            if i == j:
                affinity[i, j] = 1.0
                continue
            meta_j = unit_meta[unit_j]
            same_flow = meta_i["flow_id"] == meta_j["flow_id"]
            same_func = meta_i["microservice"].ms_id == meta_j["microservice"].ms_id
            chain_distance = abs(meta_i["position"] - meta_j["position"])
            comm_score = 1.0 / (1.0 + chain_distance) if same_flow else 0.0
            func_score = 1.0 if same_func else 0.0
            cpu_span = max(meta_i["cores"], meta_j["cores"], 1)
            mem_span = max(meta_i["memory"], meta_j["memory"], 1.0)
            resource_gap = (
                abs(meta_i["cores"] - meta_j["cores"]) / cpu_span +
                abs(meta_i["memory"] - meta_j["memory"]) / mem_span
            ) / 2.0
            topo_score = 1.0 if same_flow and chain_distance == 1 else 0.0
            affinity[i, j] = max(0.0, 0.35 * func_score + 0.45 * comm_score + 0.20 * topo_score - 0.15 * resource_gap)
    np.fill_diagonal(affinity, 1.0)
    return units, affinity


def _deterministic_weighted_kmeans(embedding: np.ndarray, k: int, max_iter: int = 20) -> np.ndarray:
    """本地确定性K-Means，避免新增sklearn依赖"""
    n = len(embedding)
    if n == 0:
        return np.array([], dtype=int)
    k = max(1, min(k, n))
    centers = embedding[np.linspace(0, n - 1, k, dtype=int)].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        distances = np.linalg.norm(embedding[:, None, :] - centers[None, :, :], axis=2)
        new_labels = np.argmin(distances, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for cluster_id in range(k):
            members = embedding[labels == cluster_id]
            if len(members):
                centers[cluster_id] = np.mean(members, axis=0)
    return labels


def _cluster_resource_need(cluster_units: List[str], unit_meta: Dict[str, Dict]) -> Tuple[int, float]:
    """计算cluster CPU和内存需求"""
    cores = sum(unit_meta[unit]["cores"] for unit in cluster_units)
    memory = sum(unit_meta[unit]["memory"] for unit in cluster_units)
    return int(cores), float(memory)


def _fragmentation_after(server: EdgeServer, add_cores: int, add_memory: float) -> float:
    """估算放置后的CPU/内存碎片方差"""
    cpu_left = max(server.available_cpu - add_cores, 0) / max(server.cpu_cores, 1)
    mem_left = max(server.available_memory - add_memory, 0.0) / max(server.memory_capacity, 1.0)
    return float(np.var([cpu_left, mem_left]))


def _cluster_locality_cost(cluster_units: List[str], unit_meta: Dict[str, Dict],
                           server: EdgeServer, system_state: SystemState) -> float:
    """估算cluster放到某台服务器后的通信局部性成本"""
    if not system_state.network_topology:
        return 0.0
    costs = []
    for unit_id in cluster_units:
        meta = unit_meta[unit_id]
        flow_id = meta["flow_id"]
        position = meta["position"]
        chain = meta["flow"].service_chain.get_traditional_microservices()
        neighbor_ids = []
        if position > 0:
            neighbor_ids.append(chain[position - 1].ms_id)
        if position + 1 < len(chain):
            neighbor_ids.append(chain[position + 1].ms_id)
        for ms_id in neighbor_ids:
            placements = system_state.stream_allocated_resources.get(flow_id, {}).get(ms_id, {})
            for placed_server in placements:
                costs.append(system_state.network_topology.get_communication_delay(server.server_id, placed_server))
    return float(np.mean(costs)) if costs else 0.0


def _place_cluster_on_best_server(cluster_units: List[str], unit_meta: Dict[str, Dict],
                                  system_state: SystemState, decision_vars) -> bool:
    """按RAPPA-restricted score放置一个cluster"""
    traditional_servers = [
        server for server in system_state.edge_servers.values()
        if server.server_type.value == "traditional"
    ]
    need_cores, need_memory = _cluster_resource_need(cluster_units, unit_meta)
    candidates = [
        server for server in traditional_servers
        if server.available_cpu >= need_cores and server.available_memory >= need_memory
    ]
    if not candidates:
        return False

    def score(server):
        locality = _cluster_locality_cost(cluster_units, unit_meta, server, system_state)
        frag = _fragmentation_after(server, need_cores, need_memory)
        spare = (server.available_cpu / max(server.cpu_cores, 1) +
                 server.available_memory / max(server.memory_capacity, 1.0)) / 2.0
        return locality + 25.0 * frag - 0.1 * spare

    selected_server = min(candidates, key=score)
    for unit_id in sorted(cluster_units, key=lambda item: unit_meta[item]["cores"], reverse=True):
        meta = unit_meta[unit_id]
        requirements = {"cores": meta["cores"], "memory": meta["memory"]}
        ok = deploy_single_microservice_to_server(
            meta["flow"], meta["microservice"], requirements,
            selected_server, system_state, decision_vars
        )
        if not ok:
            return False
    return True


def run_RAPPA_traditional_placement(system_state: SystemState, context: Dict) -> bool:
    """
    RAPPA传统微服务放置入口
    谱聚类产生cluster，cluster放置直接落到当前系统实例和资源矩阵。
    """
    from Constant import DecisionVariables

    units, affinity = build_service_affinity_matrix(system_state)
    _, unit_meta = _collect_traditional_units(system_state)
    if not units:
        context["rappa_clusters"] = {}
        return False

    degree = np.sum(affinity, axis=1)
    try:
        inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(degree, 1e-9)))
        laplacian = np.eye(len(units)) - inv_sqrt @ affinity @ inv_sqrt
        _, eigvecs = np.linalg.eigh(laplacian)
        k = max(1, min(int(round(math.sqrt(len(units)))) or 1, len(units)))
        embedding = eigvecs[:, :k]
    except Exception:
        k = 1
        embedding = affinity

    labels = _deterministic_weighted_kmeans(embedding, k)
    clusters = {}
    for unit_id, label in zip(units, labels):
        clusters.setdefault(int(label), []).append(unit_id)

    queue = sorted(clusters.values(), key=lambda c: _cluster_resource_need(c, unit_meta)[0], reverse=True)
    decision_vars = DecisionVariables(system_state.time_frame)
    placed_clusters = {}
    split_count = 0
    backtrack_count = 0

    while queue:
        cluster_units = queue.pop(0)
        cluster_backup = backup_slow_deployment_state(system_state)
        if _place_cluster_on_best_server(cluster_units, unit_meta, system_state, decision_vars):
            placed_clusters[f"c{len(placed_clusters)}"] = list(cluster_units)
            continue
        backtrack_rappa_placement(context, system_state, cluster_backup)
        backtrack_count += 1
        if len(cluster_units) <= 1:
            context["rappa_failed_unit"] = cluster_units[0] if cluster_units else ""
            context["rappa_backtracked"] = True
            return False
        splits = split_cluster_karger_style(cluster_units, affinity, units)
        split_count += max(len(splits) - 1, 0)
        queue = splits + queue

    context["rappa_clusters"] = placed_clusters
    context["rappa_affinity_shape"] = tuple(affinity.shape)
    context["rappa_split_count"] = split_count
    context["rappa_backtrack_count"] = backtrack_count
    context["rappa_backtracked"] = backtrack_count > 0
    local_refine_rappa_placement(system_state, context)
    return True


def split_cluster_karger_style(cluster_members: List[str], affinity: np.ndarray,
                               unit_ids: List[str]) -> List[List[str]]:
    """
    Karger-style拆分的工程版
    用稳定随机收缩的思想近似割边，失败时按弱亲和成员确定性拆分。
    """
    if len(cluster_members) <= 1:
        return [list(cluster_members)]
    member_index = {unit_id: idx for idx, unit_id in enumerate(unit_ids)}
    members = list(cluster_members)
    stable_seed = sum((idx + 1) * sum(ord(ch) for ch in unit_id) for idx, unit_id in enumerate(sorted(members)))
    trials = max(12, min(96, len(members) * len(members)))
    best_cut = None
    best_parts = None

    def edge_weight(a: str, b: str) -> float:
        return max(float(affinity[member_index[a], member_index[b]]), 1e-9)

    def cut_weight(left_part: List[str], right_part: List[str]) -> float:
        return sum(edge_weight(a, b) for a in left_part for b in right_part)

    for trial in range(trials):
        rng = random.Random(stable_seed + trial * 7919)
        components = {unit_id: {unit_id} for unit_id in members}
        while len(components) > 2:
            comp_ids = list(components.keys())
            edges = []
            total_weight = 0.0
            for i, comp_a in enumerate(comp_ids):
                for comp_b in comp_ids[i + 1:]:
                    weight = sum(edge_weight(a, b) for a in components[comp_a] for b in components[comp_b])
                    if weight <= 0:
                        continue
                    edges.append((comp_a, comp_b, weight))
                    total_weight += weight
            if not edges:
                break
            pick = rng.random() * total_weight
            cursor = 0.0
            selected_a, selected_b = edges[-1][0], edges[-1][1]
            for comp_a, comp_b, weight in edges:
                cursor += weight
                if cursor >= pick:
                    selected_a, selected_b = comp_a, comp_b
                    break
            components[selected_a].update(components[selected_b])
            del components[selected_b]

        parts = [sorted(list(part)) for part in components.values()]
        if len(parts) != 2 or not parts[0] or not parts[1]:
            continue
        balance_penalty = abs(len(parts[0]) - len(parts[1])) / max(len(members), 1)
        current_cut = cut_weight(parts[0], parts[1]) + 0.05 * balance_penalty
        if best_cut is None or current_cut < best_cut:
            best_cut = current_cut
            best_parts = parts

    if best_parts:
        return best_parts

    # 兜底：弱亲和成员优先拆出
    scored = []
    for unit_id in members:
        idx = member_index.get(unit_id, 0)
        intra = sum(float(affinity[idx, member_index[other]]) for other in members if other != unit_id)
        scored.append((intra, unit_id))
    scored.sort()
    half = max(1, len(scored) // 2)
    left = [unit_id for _, unit_id in scored[:half]]
    right = [unit_id for _, unit_id in scored[half:]]
    return [left, right] if right else [left]


def backtrack_rappa_placement(context: Dict, system_state: SystemState = None,
                              snapshot: Dict = None) -> bool:
    """恢复最近可行RAPPA快照"""
    context["rappa_backtracked"] = True
    if system_state is not None and snapshot is not None:
        restore_slow_deployment_state(system_state, snapshot)
        context["rappa_backtrack_restore"] = True
    return True


def _traditional_instance_score(instance: MicroserviceInstance, server: EdgeServer,
                                system_state: SystemState) -> float:
    """计算单实例局部调整分数，越低越好"""
    flow_id = _get_flow_id_from_instance_id(instance.instance_id)
    flow = system_state.request_flows.get(flow_id)
    locality = 0.0
    comm_cost_proxy = 0.0
    if flow and system_state.network_topology:
        chain = flow.service_chain.get_traditional_microservices()
        ms_ids = [ms.ms_id for ms in chain]
        if instance.microservice.ms_id in ms_ids:
            idx = ms_ids.index(instance.microservice.ms_id)
            neighbor_ms = []
            if idx > 0:
                neighbor_ms.append(ms_ids[idx - 1])
            if idx + 1 < len(ms_ids):
                neighbor_ms.append(ms_ids[idx + 1])
            delays = []
            for ms_id in neighbor_ms:
                placements = system_state.stream_allocated_resources.get(flow_id, {}).get(ms_id, {})
                for neighbor_server in placements:
                    delays.append(system_state.network_topology.get_communication_delay(server.server_id, neighbor_server))
            locality = float(np.mean(delays)) if delays else 0.0
            if delays:
                # 按请求到达率和token规模估计通信成本压力，避免只优化平均距离。
                token_mb = (
                    float(getattr(flow, "r_input_data_size", 0.0)) +
                    float(getattr(flow, "r_output_data_size", 0.0))
                ) / 1000.0
                arrival_scale = min(max(float(getattr(flow, "arrival_rate", 1.0)), 1.0), 24.0) / 8.0
                traffic_weight = 1.0 + token_mb * arrival_scale
                comm_cost_proxy = float(np.mean([delay * traffic_weight for delay in delays]))
    # 当前实例已经占用了所在服务器资源；评估迁移候选时要扣除将要新增的CPU/内存。
    add_cores = 0 if server.server_id == instance.server_id else int(instance.allocated_cores)
    add_memory = 0.0 if server.server_id == instance.server_id else float(getattr(instance, "allocated_memory", 1.2))
    frag = _fragmentation_after(server, add_cores, add_memory)
    return 0.45 * locality + 0.35 * comm_cost_proxy + 18.0 * frag


def _move_traditional_instance(instance_id: str, target_server: EdgeServer,
                               system_state: SystemState):
    """移动一个传统微服务实例"""
    instance = system_state.microservice_instances[instance_id]
    source_server = system_state.edge_servers[instance.server_id]
    flow_id = _get_flow_id_from_instance_id(instance_id)
    memory = getattr(instance, "allocated_memory", 1.2)
    source_server.available_cpu += instance.allocated_cores
    source_server.available_memory += memory
    target_server.available_cpu -= instance.allocated_cores
    target_server.available_memory -= memory
    old_server_id = instance.server_id
    instance.server_id = target_server.server_id
    new_instance_id = f"{flow_id}_{instance.microservice.ms_id}_{target_server.server_id}"
    system_state.microservice_instances[new_instance_id] = instance
    if new_instance_id != instance_id:
        del system_state.microservice_instances[instance_id]
        instance.instance_id = new_instance_id
    alloc = system_state.stream_allocated_resources.get(flow_id, {}).get(instance.microservice.ms_id, {})
    alloc.pop(old_server_id, None)
    alloc[target_server.server_id] = instance.allocated_cores


def _ai_instance_reservation(instance: MicroserviceInstance) -> Dict[str, float]:
    return {
        "gpu_units": float(getattr(instance, "gpu_units_reserved", 1.0)),
        "gpu_memory": float(getattr(instance, "gpu_memory_reserved", 0.0)),
        "model_storage": float(getattr(instance, "model_storage_reserved", 0.0)),
    }


def _ai_instance_move_feasible(instance: MicroserviceInstance, target_server: EdgeServer) -> bool:
    if target_server.server_id == instance.server_id:
        return True
    reservation = _ai_instance_reservation(instance)
    return (
        target_server.available_cpu >= int(getattr(instance, "allocated_cores", 1)) and
        float(getattr(target_server, "available_gpu_units", target_server.gpu_units)) >= reservation["gpu_units"] and
        float(getattr(target_server, "available_gpu_memory", target_server.gpu_memory)) >= reservation["gpu_memory"] and
        float(getattr(target_server, "available_model_storage", target_server.model_storage)) >= reservation["model_storage"]
    )


def _move_ai_instance(instance_id: str, target_server: EdgeServer,
                      system_state: SystemState, context: Dict):
    """移动一个HAPA AI端点实例，不改变快层local/cloud动作。"""
    instance = system_state.microservice_instances[instance_id]
    source_server = system_state.edge_servers[instance.server_id]
    flow_id = _get_flow_id_from_instance_id(instance_id)
    old_server_id = instance.server_id
    allocated_cores = int(getattr(instance, "allocated_cores", 1))

    source_server.available_cpu += allocated_cores
    target_server.available_cpu -= allocated_cores
    instance.server_id = target_server.server_id
    new_instance_id = f"{flow_id}_{instance.microservice.ms_id}_{target_server.server_id}"
    if new_instance_id in system_state.microservice_instances and new_instance_id != instance_id:
        replica_idx = 2
        candidate_id = f"{new_instance_id}_rep{replica_idx}"
        while candidate_id in system_state.microservice_instances:
            replica_idx += 1
            candidate_id = f"{new_instance_id}_rep{replica_idx}"
        new_instance_id = candidate_id

    system_state.microservice_instances[new_instance_id] = instance
    if new_instance_id != instance_id:
        del system_state.microservice_instances[instance_id]
        instance.instance_id = new_instance_id

    alloc = system_state.stream_allocated_resources.get(flow_id, {}).get(instance.microservice.ms_id, {})
    previous_count = int(alloc.get(old_server_id, 0))
    if previous_count <= 1:
        alloc.pop(old_server_id, None)
    else:
        alloc[old_server_id] = previous_count - 1
    alloc[target_server.server_id] = int(alloc.get(target_server.server_id, 0)) + 1

    profile = context.get("hapa_profile")
    if isinstance(profile, dict):
        item = profile.pop(instance_id, None)
        if isinstance(item, dict):
            item = dict(item)
            item["server_id"] = target_server.server_id
            profile[new_instance_id] = item


def local_refine_rappa_placement(system_state: SystemState, context: Dict) -> bool:
    """执行轻量2-opt局部调整，只接受分数下降的移动"""
    weights = {
        "delay": 0.45,
        "cost": 0.35,
        "fragmentation": 0.12,
        "queue_dpp_proxy": 0.08,
    }
    accepted = 0
    accepted_swaps = 0
    traditional_servers = [
        server for server in system_state.edge_servers.values()
        if server.server_type.value == "traditional"
    ]
    for instance_id, instance in list(system_state.microservice_instances.items()):
        if instance.microservice.service_type != "traditional":
            continue
        current_server = system_state.edge_servers.get(instance.server_id)
        if not current_server:
            continue
        memory = getattr(instance, "allocated_memory", 1.2)
        current_score = _traditional_instance_score(instance, current_server, system_state)
        best_server = current_server
        best_score = current_score
        for candidate in traditional_servers:
            if candidate.server_id == current_server.server_id:
                continue
            if candidate.available_cpu < instance.allocated_cores or candidate.available_memory < memory:
                continue
            score = _traditional_instance_score(instance, candidate, system_state)
            if score + 1e-6 < best_score:
                best_score = score
                best_server = candidate
        if best_server.server_id != current_server.server_id:
            _move_traditional_instance(instance_id, best_server, system_state)
            accepted += 1
        if accepted >= 10:
            break

    traditional_instances = [
        (instance_id, instance) for instance_id, instance in system_state.microservice_instances.items()
        if instance.microservice.service_type == "traditional"
    ]
    checked = 0
    for idx, (left_id, left_instance) in enumerate(traditional_instances):
        if checked >= 60 or accepted_swaps >= 5:
            break
        for right_id, right_instance in traditional_instances[idx + 1:]:
            if checked >= 60 or accepted_swaps >= 5:
                break
            checked += 1
            if left_instance.server_id == right_instance.server_id:
                continue
            left_flow = _get_flow_id_from_instance_id(left_id)
            right_flow = _get_flow_id_from_instance_id(right_id)
            if left_flow == right_flow and left_instance.microservice.ms_id == right_instance.microservice.ms_id:
                continue
            left_server = system_state.edge_servers.get(left_instance.server_id)
            right_server = system_state.edge_servers.get(right_instance.server_id)
            if not left_server or not right_server:
                continue
            left_memory = getattr(left_instance, "allocated_memory", 1.2)
            right_memory = getattr(right_instance, "allocated_memory", 1.2)
            if left_server.available_cpu + left_instance.allocated_cores < right_instance.allocated_cores:
                continue
            if right_server.available_cpu + right_instance.allocated_cores < left_instance.allocated_cores:
                continue
            if left_server.available_memory + left_memory < right_memory:
                continue
            if right_server.available_memory + right_memory < left_memory:
                continue
            before = (
                _traditional_instance_score(left_instance, left_server, system_state) +
                _traditional_instance_score(right_instance, right_server, system_state)
            )
            snapshot = backup_slow_deployment_state(system_state)
            _move_traditional_instance(left_id, right_server, system_state)
            _move_traditional_instance(right_id, left_server, system_state)
            new_left_id = f"{left_flow}_{left_instance.microservice.ms_id}_{right_server.server_id}"
            new_right_id = f"{right_flow}_{right_instance.microservice.ms_id}_{left_server.server_id}"
            if new_left_id not in system_state.microservice_instances or new_right_id not in system_state.microservice_instances:
                restore_slow_deployment_state(system_state, snapshot)
                continue
            after = (
                _traditional_instance_score(system_state.microservice_instances[new_left_id], right_server, system_state) +
                _traditional_instance_score(system_state.microservice_instances[new_right_id], left_server, system_state)
            )
            if after + 1e-6 < before:
                accepted_swaps += 1
                checked = 60
                break
            else:
                restore_slow_deployment_state(system_state, snapshot)
    refine_summary = {
        "accepted_moves": accepted,
        "accepted_swaps": accepted_swaps,
        "routing_paths": sum(len(v) for v in getattr(system_state, "stream_transfer_probabilities", {}).values()),
        "weights": weights,
        "objective": "expected delay + routing cost + fragmentation + DPP proxy",
    }
    context["local_refinement"] = refine_summary
    context["local_refine"] = refine_summary
    return True


def calculate_hapa_priority(request_flow: RequestFlow, ai_microservice: Microservice,
                            resource_requirement: float) -> float:
    """计算HAPA中的W(s)优先级"""
    arrival = float(getattr(request_flow, "arrival_rate", 0.0))
    token_load = (
        float(getattr(request_flow, "r_input_data_size", 0.0)) +
        float(getattr(request_flow, "r_output_data_size", 0.0))
    )
    chain_degree = len(request_flow.service_chain.microservices)
    latency_sensitivity = 1.0 / max(float(getattr(request_flow, "max_latency", 1.0)), 1.0)
    return resource_requirement * (1.0 + arrival) + token_load / 1024.0 + 0.5 * chain_degree + 10.0 * latency_sensitivity


def estimate_hapa_effective_demand(request_flow: RequestFlow, previous_demand: float,
                                   alpha: float = 0.35) -> Dict[str, float]:
    """
    估算HAPA覆盖需求
    将req/s、token负载和延迟敏感度合成慢层demand_hat。
    """
    arrival = float(getattr(request_flow, "arrival_rate", 0.0))
    input_tokens = float(getattr(request_flow, "r_input_data_size", 128.0))
    output_tokens = float(getattr(request_flow, "r_output_data_size", 32.0))
    token_load = max(input_tokens + output_tokens, 1.0)
    token_factor = min(max(token_load / 512.0, 0.75), 3.0)
    max_latency = max(float(getattr(request_flow, "max_latency", 100.0)), 1.0)
    latency_factor = min(max(100.0 / max_latency, 0.75), 1.75)
    base_ewma = (1 - alpha) * float(previous_demand) + alpha * arrival
    demand_hat = base_ewma * (0.65 + 0.35 * token_factor) * (0.85 + 0.15 * latency_factor)
    return {
        "arrival_req_s": float(arrival),
        "base_ewma_req_s": float(base_ewma),
        "token_factor": float(token_factor),
        "latency_factor": float(latency_factor),
        "demand_hat_req_s": float(max(demand_hat, 1e-6)),
    }


def _estimate_ai_reservation(request_flow: RequestFlow, ai_microservice: Microservice,
                             server: EdgeServer, system_state: SystemState = None) -> Dict[str, float]:
    """估算HAPA慢层AI资源包络"""
    resource_requirement = calculate_ai_microservice_resource_requirement(request_flow, ai_microservice, system_state)
    gpu_units = int(max(1, min(server.gpu_units, math.ceil(resource_requirement))))
    gpu_memory = calculate_required_gpu_memory(request_flow, ai_microservice, gpu_units, False)
    model_storage = calculate_required_model_storage(ai_microservice)
    try:
        from ablation_resource_models import select_local_ai_config
        local_config = select_local_ai_config(
            request_flow, ai_microservice, server, system_state,
            SQ_value=0.0, SZ_value=0.0, performance_factor=1.0, V=20.0
        ) if system_state is not None else None
        if local_config:
            # 慢层包络只保证实例可放置。
            # 快层(g,b,f_GPU)枚举可继续选择更高GPU数，不能反向把首副本包络抬到高GPU配置。
            gpu_memory = max(float(gpu_memory), float(calculate_required_gpu_memory(
                request_flow, ai_microservice, gpu_units, False
            )))
            model_storage = max(float(model_storage), float(local_config.get("model_storage", model_storage)))
    except Exception:
        pass
    context_units = max(1, int(math.ceil((request_flow.r_input_data_size + request_flow.r_output_data_size) / 512.0)))
    return {
        "gpu_units": float(gpu_units),
        "gpu_memory": float(gpu_memory),
        "model_storage": float(model_storage),
        "context_units": float(context_units),
    }


def _estimate_ai_instance_capacity(request_flow: RequestFlow, server: EdgeServer) -> float:
    """估算一个AI实例在慢层包络下可覆盖的req/s需求"""
    input_tokens = max(float(getattr(request_flow, "r_input_data_size", 128.0)), 1.0)
    output_tokens = max(float(getattr(request_flow, "r_output_data_size", 32.0)), 1.0)
    prefill = max(float(getattr(server, "prefill_speed_tokens_per_sec", 1.0)), 1.0)
    decode = max(float(getattr(server, "decode_speed_tokens_per_sec", 1.0)), 1.0)
    service_time_s = input_tokens / prefill + output_tokens / decode
    batch_gain = max(1.0, min(float(getattr(server, "max_batch_size", 1)), 32.0) * 0.65)
    return batch_gain / max(service_time_s, 1e-6)


def calculate_hapa_match_score(server: EdgeServer, request_flow: RequestFlow,
                               ai_microservice: Microservice, resource_requirement: float,
                               system_state: SystemState) -> float:
    """计算HAPA中的GPU/VRAM匹配分和数据局部性分"""
    reservation = _estimate_ai_reservation(request_flow, ai_microservice, server, system_state)
    gpu_score = min(server.gpu_units / max(reservation["gpu_units"], 1.0), 2.0)
    mem_score = min(server.gpu_memory / max(reservation["gpu_memory"], 1.0), 2.0)
    storage_score = min(server.model_storage / max(reservation["model_storage"], 1.0), 1.0)
    d_loc = calculate_data_locality_score(server, request_flow, system_state)
    load_penalty = (
        server.available_cpu / max(server.cpu_cores, 1) +
        server.available_memory / max(server.memory_capacity, 1.0)
    ) / 2.0
    psi = 0.45 * gpu_score + 0.35 * mem_score + 0.20 * storage_score
    return 0.55 * psi + 0.30 * d_loc + 0.15 * load_penalty


def calculate_data_locality_score(server: EdgeServer, request_flow: RequestFlow,
                                  system_state: SystemState) -> float:
    """计算AI服务与前驱传统服务的数据局部性"""
    chain = request_flow.service_chain
    traditional_microservices = chain.get_traditional_microservices()
    if not traditional_microservices:
        return 0.0
    predecessor = traditional_microservices[-1].ms_id
    placements = system_state.stream_allocated_resources.get(request_flow.flow_id, {}).get(predecessor, {})
    if not placements:
        return 0.0
    if system_state.network_topology:
        scores = []
        for pred_server, resource in placements.items():
            delay = system_state.network_topology.get_communication_delay(pred_server, server.server_id)
            scores.append(float(resource) / (1.0 + delay))
        return float(sum(scores) / max(sum(placements.values()), 1))
    same_node_resource = float(placements.get(server.server_id, 0))
    total_resource = float(sum(placements.values())) or 1.0
    return same_node_resource / total_resource


def _ai_reservation_feasible(server_id: str, residual: Dict[str, Dict[str, float]],
                             reservation: Dict[str, float]) -> bool:
    """检查AI慢层包络是否可放置"""
    cap = residual[server_id]
    return (
        cap["gpu_units"] >= reservation["gpu_units"] and
        cap["gpu_memory"] >= reservation["gpu_memory"] and
        cap["model_storage"] >= reservation["model_storage"] and
        cap["context_units"] >= reservation["context_units"]
    )


def _consume_ai_reservation(server_id: str, residual: Dict[str, Dict[str, float]],
                            reservation: Dict[str, float]):
    """扣减HAPA内部残量，不直接扣GPU运行时资源"""
    for key in ["gpu_units", "gpu_memory", "model_storage", "context_units"]:
        residual[server_id][key] -= reservation[key]
    if "cpu_slots" in residual[server_id]:
        residual[server_id]["cpu_slots"] -= 1.0


def _create_ai_instance(flow_id: str, request_flow: RequestFlow, ai_microservice: Microservice,
                        server: EdgeServer, system_state: SystemState,
                        reservation: Dict[str, float], policy_tag: str) -> str:
    """创建AI物理实例，处理模式留给快层控制器"""
    base_instance_id = f"{flow_id}_{ai_microservice.ms_id}_{server.server_id}"
    current_allocated = system_state.get_stream_allocated_resource(
        flow_id, ai_microservice.ms_id, server.server_id
    )
    instance_id = base_instance_id
    if instance_id in system_state.microservice_instances:
        replica_idx = int(current_allocated) + 1
        instance_id = f"{base_instance_id}_rep{replica_idx}"
        while instance_id in system_state.microservice_instances:
            replica_idx += 1
            instance_id = f"{base_instance_id}_rep{replica_idx}"
    instance = MicroserviceInstance(
        instance_id=instance_id,
        microservice=ai_microservice,
        server_id=server.server_id,
        allocated_cores=1,
        arrival_rate=request_flow.arrival_rate
    )
    instance.processing_mode = "pending_decision"
    instance.gpu_units_reserved = reservation.get("gpu_units", 0.0)
    instance.gpu_memory_reserved = reservation.get("gpu_memory", 0.0)
    instance.model_storage_reserved = reservation.get("model_storage", 0.0)
    instance.context_units_reserved = reservation.get("context_units", 0.0)
    instance.placement_policy = policy_tag
    server.available_cpu -= 1
    system_state.microservice_instances[instance_id] = instance
    system_state.set_stream_allocated_resource(
        flow_id, ai_microservice.ms_id, server.server_id, int(current_allocated) + 1
    )
    request_flow.ai_processing_choice = "pending"
    return instance_id


def _clear_ai_deployments(system_state: SystemState):
    """只清理AI实例和AI资源分配"""
    for instance_id, instance in list(system_state.microservice_instances.items()):
        if instance.microservice.service_type == "ai":
            server = system_state.edge_servers.get(instance.server_id)
            if server:
                server.available_cpu = min(server.cpu_cores, server.available_cpu + instance.allocated_cores)
            del system_state.microservice_instances[instance_id]
    for flow_id, flow_resources in list(system_state.stream_allocated_resources.items()):
        for ms_id in list(flow_resources.keys()):
            if any(
                flow.service_chain.ai_microservice and flow.service_chain.ai_microservice.ms_id == ms_id
                for flow in system_state.request_flows.values()
            ):
                del flow_resources[ms_id]


def run_HAPA_ai_placement(system_state: SystemState, context: Dict) -> bool:
    """
    HAPA异构AI放置入口
    按W(s)、psi(s,v)、D_loc和EWMA需求执行多实例AI放置。
    """
    ai_servers = [
        server for server in system_state.edge_servers.values()
        if server.server_type.value == "ai_capable"
    ]
    if not ai_servers:
        context["hapa_failure_reason"] = "没有AI服务器"
        return retain_previous_ai_profile(system_state, context)

    residual = {
        server.server_id: {
            "cpu_slots": float(max(getattr(server, "available_cpu", server.cpu_cores), 0)),
            "gpu_units": float(getattr(server, "available_gpu_units", server.gpu_units)),
            "gpu_memory": float(getattr(server, "available_gpu_memory", server.gpu_memory)),
            "model_storage": float(getattr(server, "available_model_storage", server.model_storage)),
            "context_units": float(max(server.max_batch_size * 3, 1)),
        }
        for server in ai_servers
    }
    demand_ewma = copy.deepcopy(getattr(system_state, "hapa_demand_ewma", {}))
    alpha = 0.35
    hapa_items = []
    for flow_id, request_flow in system_state.request_flows.items():
        ai_microservice = request_flow.service_chain.ai_microservice
        if not ai_microservice:
            continue
        resource_requirement = calculate_ai_microservice_resource_requirement(
            request_flow, ai_microservice, system_state
        )
        previous_demand = float(demand_ewma.get(flow_id, request_flow.arrival_rate))
        demand_info = estimate_hapa_effective_demand(request_flow, previous_demand, alpha=alpha)
        estimated_demand = demand_info["demand_hat_req_s"]
        demand_ewma[flow_id] = demand_info["base_ewma_req_s"]
        priority = calculate_hapa_priority(request_flow, ai_microservice, resource_requirement)
        hapa_items.append({
            "flow_id": flow_id,
            "request_flow": request_flow,
            "ai_microservice": ai_microservice,
            "W": float(priority),
            "resource_requirement": float(resource_requirement),
            "estimated_demand": float(max(estimated_demand, 1e-6)),
            "demand_info": demand_info,
        })
    hapa_items.sort(key=lambda item: item["W"], reverse=True)
    context["hapa_priority"] = [
        {key: value for key, value in item.items() if key not in {"request_flow", "ai_microservice"}}
        for item in hapa_items
    ]

    profile = {}
    coverage = {}
    covering_quality = {}
    server_instance_counts = {server.server_id: 0 for server in ai_servers}
    server_model_cache = {server.server_id: set() for server in ai_servers}
    mu_hat = {
        server.server_id: _estimate_ai_instance_capacity(
            next(iter(system_state.request_flows.values())), server
        ) if system_state.request_flows else 0.0
        for server in ai_servers
    }
    demand_states = {}

    def effective_reservation(server: EdgeServer, item_info: Dict) -> Dict[str, float]:
        """考虑同节点模型缓存后的HAPA资源包络"""
        reservation = _estimate_ai_reservation(
            item_info["request_flow"], item_info["ai_microservice"], server, system_state
        )
        if item_info["ai_microservice"].ms_id in server_model_cache.get(server.server_id, set()):
            reservation = dict(reservation)
            reservation["model_storage"] = 0.0
        return reservation

    def remaining_items_feasible(remaining_items: List[Dict], residual_after: Dict[str, Dict[str, float]]) -> bool:
        """检查剩余flow是否至少还能放一个首副本"""
        for rem_item in remaining_items:
            feasible = False
            for rem_server in ai_servers:
                if residual_after[rem_server.server_id].get("cpu_slots", 0.0) < 1:
                    continue
                rem_reservation = effective_reservation(rem_server, rem_item)
                if _ai_reservation_feasible(rem_server.server_id, residual_after, rem_reservation):
                    feasible = True
                    break
            if not feasible:
                return False
        return True

    def place_hapa_replica(item_info: Dict, state: Dict, remaining_items: List[Dict],
                           require_lookahead: bool, prefer_new_server: bool = False) -> bool:
        """放置一个HAPA副本，并更新需求覆盖状态"""
        request_flow = item_info["request_flow"]
        ai_microservice = item_info["ai_microservice"]
        selected_server_counts = state["selected_server_counts"]
        candidates = []
        fallback_candidates = []
        for server in ai_servers:
            if residual[server.server_id].get("cpu_slots", 0.0) < 1:
                continue
            reservation = effective_reservation(server, item_info)
            if not _ai_reservation_feasible(server.server_id, residual, reservation):
                continue
            psi = calculate_hapa_match_score(server, request_flow, ai_microservice,
                                             item_info["resource_requirement"], system_state)
            d_loc = calculate_data_locality_score(server, request_flow, system_state)
            spare = (
                residual[server.server_id]["gpu_units"] / max(server.gpu_units, 1) +
                residual[server.server_id]["gpu_memory"] / max(server.gpu_memory, 1.0) +
                residual[server.server_id]["model_storage"] / max(server.model_storage, 1.0)
            ) / 3.0
            reservation_pressure = (
                reservation["gpu_units"] / max(server.gpu_units, 1) +
                reservation["gpu_memory"] / max(server.gpu_memory, 1.0) +
                reservation["model_storage"] / max(server.model_storage, 1.0)
            ) / 3.0
            global_spread = 1.0 / (1.0 + float(server_instance_counts.get(server.server_id, 0)))
            flow_reuse = 1.0 / (1.0 + float(selected_server_counts.get(server.server_id, 0)))
            spread_score = 0.7 * global_spread + 0.3 * flow_reuse
            score = (
                0.42 * psi +
                0.23 * d_loc +
                0.18 * spare +
                0.12 * spread_score -
                0.15 * reservation_pressure
            )
            if prefer_new_server and selected_server_counts.get(server.server_id, 0) > 0:
                # diversity副本优先放到不同AI节点，给UAC endpoint选择空间。
                score -= 0.45
            lookahead_ok = True
            if require_lookahead and remaining_items:
                residual_after = copy.deepcopy(residual)
                _consume_ai_reservation(server.server_id, residual_after, reservation)
                lookahead_ok = remaining_items_feasible(remaining_items, residual_after)
            row = (score, lookahead_ok, server, reservation, psi, d_loc, spare, reservation_pressure)
            fallback_candidates.append(row)
            if psi > 0.25:
                candidates.append(row)
        candidate_pool = [row for row in candidates if row[1]] or candidates
        if not candidate_pool:
            candidate_pool = [row for row in fallback_candidates if row[1]] or fallback_candidates
        if prefer_new_server and candidate_pool:
            new_server_pool = [row for row in candidate_pool if selected_server_counts.get(row[2].server_id, 0) == 0]
            if new_server_pool:
                candidate_pool = new_server_pool
        if not candidate_pool:
            return False
        _, _, selected_server, reservation, psi, d_loc, spare, reservation_pressure = max(
            candidate_pool, key=lambda row: (row[0], row[2].server_id)
        )

        instance_id = _create_ai_instance(
            item_info["flow_id"], request_flow, ai_microservice,
            selected_server, system_state, reservation, "HAPA"
        )
        _consume_ai_reservation(selected_server.server_id, residual, reservation)
        server_model_cache[selected_server.server_id].add(ai_microservice.ms_id)
        selected_server_counts[selected_server.server_id] = selected_server_counts.get(selected_server.server_id, 0) + 1
        server_instance_counts[selected_server.server_id] = server_instance_counts.get(selected_server.server_id, 0) + 1
        capacity = _estimate_ai_instance_capacity(request_flow, selected_server)
        state["covered_capacity"] += capacity
        state["replica_count"] += 1
        state["demand_left"] = max(0.0, state["demand_left"] - capacity)
        profile[instance_id] = {
            "flow_id": item_info["flow_id"],
            "server_id": selected_server.server_id,
            "microservice": ai_microservice.ms_id,
            "reserved_gpu_units": reservation["gpu_units"],
            "reserved_gpu_memory": reservation["gpu_memory"],
            "reserved_model_storage": reservation["model_storage"],
            "reserved_context_units": reservation["context_units"],
            "capacity_req_s": capacity,
            "demand_hat_req_s": item_info["estimated_demand"],
            "server_instance_count_after": server_instance_counts[selected_server.server_id],
            "flow_server_replica_index": selected_server_counts[selected_server.server_id],
            "hapa_psi": float(psi),
            "hapa_d_loc": float(d_loc),
            "hapa_spare_score": float(spare),
            "hapa_reservation_pressure": float(reservation_pressure),
            "demand_components": item_info["demand_info"],
        }
        return True

    # 第一阶段：先给每个flow放一个首副本，避免高优先级flow抢占导致后续饥饿
    for item_index, item in enumerate(hapa_items):
        flow_id = item["flow_id"]
        state = {
            "original_demand": float(item["estimated_demand"]),
            "demand_left": float(item["estimated_demand"]),
            "selected_server_counts": {},
            "covered_capacity": 0.0,
            "replica_count": 0,
        }
        demand_states[flow_id] = state
        if not place_hapa_replica(item, state, hapa_items[item_index + 1:], require_lookahead=True):
            context["hapa_failure_reason"] = f"{flow_id} AI首副本无法放置"
            context["hapa_partial_profile"] = copy.deepcopy(profile)
            retained = retain_previous_ai_profile(system_state, context)
            context["slow_profile_reused"] = retained
            return retained

    # 第二阶段：在首副本都可行后，再按W(s)补副本覆盖demand
    for item in hapa_items:
        state = demand_states[item["flow_id"]]
        while state["demand_left"] > 0:
            if not place_hapa_replica(item, state, [], require_lookahead=False):
                break

    # normal-main复现场景下，为高优先级AI流补少量不同节点副本。
    # 该阶段不改变覆盖门槛，只提供routing/UAC可消费的endpoint多样性。
    diversity_added = 0
    if getattr(system_state, "scenario_profile", "") == "heterogeneous_burst_main":
        diversity_budget = max(1, min(len(hapa_items) // 3, len(ai_servers) // 3))
        diversity_items = sorted(
            hapa_items,
            key=lambda item: (item["W"], item["estimated_demand"], item["flow_id"]),
            reverse=True,
        )
        for item in diversity_items:
            if diversity_added >= diversity_budget:
                break
            state = demand_states[item["flow_id"]]
            if state["replica_count"] >= 2:
                continue
            before_count = int(state["replica_count"])
            if place_hapa_replica(item, state, [], require_lookahead=False, prefer_new_server=True):
                if int(state["replica_count"]) > before_count:
                    diversity_added += 1

    context["hapa_latency_diversity_replicas"] = int(diversity_added)

    for item in hapa_items:
        flow_id = item["flow_id"]
        state = demand_states[flow_id]
        original_demand = state["original_demand"]
        demand_left = state["demand_left"]
        covered_capacity = state["covered_capacity"]
        coverage[flow_id] = original_demand - demand_left
        covering_quality[flow_id] = {
            "demand_hat_req_s": float(original_demand),
            "covered_capacity_req_s": float(covered_capacity),
            "covered_demand_req_s": float(original_demand - demand_left),
            "uncovered_demand_req_s": float(demand_left),
            "coverage_ratio": float(min(covered_capacity / max(original_demand, 1e-9), 1.0)),
            "replica_count": int(state["replica_count"]),
            "demand_components": item["demand_info"],
        }

    min_coverage_ratio = min(
        (item["coverage_ratio"] for item in covering_quality.values()),
        default=0.0
    )
    if min_coverage_ratio < 0.95:
        context["hapa_failure_reason"] = f"AI需求覆盖不足: min_coverage={min_coverage_ratio:.3f}"
        context["hapa_partial_profile"] = copy.deepcopy(profile)
        context["hapa_demand_covering"] = copy.deepcopy(covering_quality)
        retained = retain_previous_ai_profile(system_state, context)
        context["slow_profile_reused"] = retained
        return retained

    context["hapa_profile"] = profile
    context["hapa_coverage"] = coverage
    context["hapa_demand_covering"] = covering_quality
    context["hapa_min_coverage_ratio"] = min(
        (item["coverage_ratio"] for item in covering_quality.values()),
        default=0.0
    )
    context["hapa_mu_hat"] = mu_hat
    context["hapa_residual_after"] = copy.deepcopy(residual)
    context["hapa_server_instance_counts"] = copy.deepcopy(server_instance_counts)
    total_capacity_by_server = {}
    for profile_item in profile.values():
        server_id = profile_item.get("server_id")
        total_capacity_by_server[server_id] = total_capacity_by_server.get(server_id, 0.0) + float(
            profile_item.get("capacity_req_s", 0.0)
        )
    max_capacity = max(total_capacity_by_server.values(), default=1.0)
    context["hapa_replica_readiness"] = {
        server_id: float(total_capacity_by_server.get(server_id, 0.0) / max(max_capacity, 1e-9))
        for server_id in server_instance_counts.keys()
    }
    context["slow_profile_reused"] = False
    system_state.previous_ai_profile = copy.deepcopy(profile)
    system_state.hapa_demand_ewma = demand_ewma
    return True


def retain_previous_ai_profile(system_state: SystemState, context: Dict) -> bool:
    """非首轮慢层失败时保留上一轮可行AI profile"""
    previous = copy.deepcopy(getattr(system_state, "previous_ai_profile", None))
    if not previous:
        context["hapa_profile"] = {}
        return False
    _clear_ai_deployments(system_state)
    restored = {}
    for instance_id, item in previous.items():
        flow_id = item.get("flow_id") or _get_flow_id_from_instance_id(instance_id)
        request_flow = system_state.request_flows.get(flow_id)
        server = system_state.edge_servers.get(item.get("server_id"))
        if not request_flow or not server or server.available_cpu < 1:
            return False
        ai_microservice = request_flow.service_chain.ai_microservice
        reservation = {
            "gpu_units": float(item.get("reserved_gpu_units", 1.0)),
            "gpu_memory": float(item.get("reserved_gpu_memory", 0.0)),
            "model_storage": float(item.get("reserved_model_storage", 0.0)),
            "context_units": float(item.get("reserved_context_units", 1.0)),
        }
        new_id = _create_ai_instance(flow_id, request_flow, ai_microservice, server,
                                     system_state, reservation, "HAPA-reused")
        restored[new_id] = dict(item)
    context["hapa_profile"] = restored
    context["slow_profile_reused"] = True
    previous_routing = copy.deepcopy(getattr(system_state, "previous_ai_routing_state", None))
    if previous_routing:
        system_state.stream_transfer_probabilities = copy.deepcopy(previous_routing.get("stream_transfer_probabilities", {}))
        for flow_id, probabilities in previous_routing.get("routing_probabilities", {}).items():
            if flow_id in system_state.request_flows:
                system_state.request_flows[flow_id].routing_probabilities = copy.deepcopy(probabilities)
    return True


def _spare_capacity_score(server: EdgeServer, service_type: str) -> float:
    """计算路由spare capacity分"""
    if service_type == "ai":
        return (
            server.available_gpu_units / max(server.gpu_units, 1) +
            server.available_gpu_memory / max(server.gpu_memory, 1.0) +
            server.available_model_storage / max(server.model_storage, 1.0)
        ) / 3.0
    return (
        server.available_cpu / max(server.cpu_cores, 1) +
        server.available_memory / max(server.memory_capacity, 1.0)
    ) / 2.0


def _softmax(values: List[float]) -> List[float]:
    """稳定Softmax"""
    if not values:
        return []
    arr = np.asarray(values, dtype=float)
    arr = arr - np.max(arr)
    exp = np.exp(arr)
    total = float(np.sum(exp)) or 1.0
    return [float(v / total) for v in exp]


def _routing_queue_score(system_state: SystemState, server_id: str) -> float:
    """计算路由队列压力分，越高表示越适合接收流量"""
    energy_queue = getattr(system_state.virtual_energy_queues.get(server_id), "queue_state", 0.0) \
        if hasattr(system_state, "virtual_energy_queues") else 0.0
    delay_queue = getattr(system_state.virtual_delay_queues.get(server_id), "queue_state", 0.0) \
        if hasattr(system_state, "virtual_delay_queues") else 0.0
    pressure = max(float(energy_queue) + float(delay_queue), 0.0)
    return 1.0 / (1.0 + pressure)


def _routing_cost_score(server: EdgeServer, dest_ms: Microservice) -> float:
    """估算路由目的节点成本分，越高表示成本越低"""
    if dest_ms.service_type == "ai":
        used_gpu = max(float(server.gpu_units - getattr(server, "available_gpu_units", server.gpu_units)), 0.0)
        used_memory = max(float(server.gpu_memory - getattr(server, "available_gpu_memory", server.gpu_memory)), 0.0)
        used_storage = max(float(server.model_storage - getattr(server, "available_model_storage", server.model_storage)), 0.0)
        cost_proxy = 1.0 + 8.0 * used_gpu + 0.15 * used_memory + 0.03 * used_storage
    else:
        used_cpu = max(float(server.cpu_cores - server.available_cpu), 0.0)
        used_memory = max(float(server.memory_capacity - server.available_memory), 0.0)
        cost_proxy = 1.0 + 0.5 * used_cpu + 0.05 * used_memory
    return 1.0 / max(cost_proxy, 1e-9)


def _routing_path_cost_score(request_flow: RequestFlow, delay_ms: float,
                             server: EdgeServer, dest_ms: Microservice) -> float:
    """估算单条routing路径的通信成本分，越高表示期望通信成本越低"""
    token_mb = (
        float(getattr(request_flow, "r_input_data_size", 0.0)) +
        float(getattr(request_flow, "r_output_data_size", 0.0))
    ) / 1000.0
    arrival_scale = min(max(float(getattr(request_flow, "arrival_rate", 1.0)), 1.0), 24.0) / 8.0
    path_comm_proxy = max(float(delay_ms), 0.0) * max(token_mb, 0.05) * arrival_scale
    resource_score = _routing_cost_score(server, dest_ms)
    path_score = 1.0 / (1.0 + path_comm_proxy)
    return 0.75 * path_score + 0.25 * resource_score


def _routing_replica_ready_score(context: Dict, server_id: str, dest_ms: Microservice) -> float:
    """读取HAPA副本就绪度，传统服务默认就绪"""
    if dest_ms.service_type != "ai":
        return 1.0
    readiness = context.get("hapa_replica_readiness", {})
    if server_id in readiness:
        return float(readiness.get(server_id, 0.0))
    for item in context.get("hapa_profile", {}).values():
        if item.get("server_id") == server_id and item.get("microservice") == dest_ms.ms_id:
            return 1.0
    return 0.0


def build_softmax_routing_context(system_state: SystemState, context: Dict) -> bool:
    """
    构建基于delay/cost/spare capacity/queue的Softmax routing上下文
    直接写入stream_transfer_probabilities，供后续指标和成本计算使用。
    """
    system_state.stream_transfer_probabilities.clear()
    for request_flow in system_state.request_flows.values():
        request_flow.routing_probabilities = {}
    routing_rows = {}
    routing_components = {}
    entropy_terms = []
    for flow_id, request_flow in system_state.request_flows.items():
        chain = request_flow.service_chain.microservices
        flow_probs = {}
        for idx in range(1, len(chain)):
            origin_ms = chain[idx - 1]
            dest_ms = chain[idx]
            origin_placements = system_state.stream_allocated_resources.get(flow_id, {}).get(origin_ms.ms_id, {})
            dest_placements = _get_successor_dest_placements(system_state, flow_id, dest_ms)
            if not origin_placements or not dest_placements:
                continue
            for origin_server_id in origin_placements:
                scores = []
                dest_server_ids = list(dest_placements.keys())
                spare_values = [
                    _spare_capacity_score(system_state.edge_servers[server_id], dest_ms.service_type)
                    for server_id in dest_server_ids
                ]
                spare_total = sum(spare_values) + 1e-9
                component_row = {}
                for dest_server_id, spare in zip(dest_server_ids, spare_values):
                    delay = 1.0
                    if system_state.network_topology:
                        delay = system_state.network_topology.get_communication_delay(origin_server_id, dest_server_id)
                    dest_server = system_state.edge_servers[dest_server_id]
                    delay_score = 1.0 / (delay + 1.0)
                    spare_score = spare / spare_total
                    cost_score = _routing_path_cost_score(request_flow, delay, dest_server, dest_ms)
                    queue_score = _routing_queue_score(system_state, dest_server_id)
                    replica_ready_score = _routing_replica_ready_score(context, dest_server_id, dest_ms)
                    # GSLA主实验优先兑现低时延和低通信成本；资源余量/队列/副本ready作为次级约束。
                    final_score = (
                        0.42 * delay_score +
                        0.05 * spare_score +
                        0.03 * queue_score +
                        0.08 * replica_ready_score +
                        0.42 * cost_score
                    )
                    scores.append(final_score)
                    component_row[dest_server_id] = {
                        "delay_ms": float(delay),
                        "delay_score": float(delay_score),
                        "spare_score": float(spare_score),
                        "cost_score": float(cost_score),
                        "queue_score": float(queue_score),
                        "replica_ready_score": float(replica_ready_score),
                        "final_score": float(final_score),
                    }
                if len(scores) <= 1:
                    probs = [1.0] if scores else []
                else:
                    # 延迟/成本敏感场景下保留Softmax概率路由，但只在top-k端点上分流，避免接近均匀导致时延优势被稀释
                    routing_top_k = min(2, len(scores))
                    routing_temperature = 120.0
                    top_indices = sorted(
                        range(len(scores)),
                        key=lambda score_idx: (-scores[score_idx], dest_server_ids[score_idx])
                    )[:routing_top_k]
                    top_probs = _softmax([scores[score_idx] * routing_temperature for score_idx in top_indices])
                    probs = [0.0] * len(scores)
                    for score_idx, prob in zip(top_indices, top_probs):
                        probs[score_idx] = prob
                row_key = f"{flow_id}:{origin_ms.ms_id}@{origin_server_id}->{dest_ms.ms_id}"
                routing_rows[row_key] = dict(zip(dest_server_ids, probs))
                routing_components[row_key] = component_row
                for dest_server_id, prob in zip(dest_server_ids, probs):
                    transfer_key = (origin_server_id, origin_ms.ms_id, dest_server_id, dest_ms.ms_id)
                    flow_probs[transfer_key] = Fraction(prob).limit_denominator(1000000)
                    request_flow.add_routing_probability(origin_server_id, dest_server_id, prob)
                    if prob > 0:
                        entropy_terms.append(-prob * math.log(prob + 1e-12))
        system_state.stream_transfer_probabilities[flow_id] = flow_probs
    context["softmax_routing_matrix"] = routing_rows
    context["routing_score_components"] = routing_components
    context["softmax_routing"] = {
        "row_count": len(routing_rows),
        "path_count": sum(len(row) for row in routing_rows.values()),
        "probability_rows_valid": all(abs(sum(row.values()) - 1.0) < 1e-6 for row in routing_rows.values()),
        "top_k": 2,
        "temperature": 120.0,
        "weights": {
            "link_latency": 0.42,
            "spare_capacity": 0.05,
            "queue_pressure": 0.03,
            "replica_readiness": 0.08,
            "cost": 0.42,
        },
    }
    context["routing_entropy"] = float(max(0.0, sum(entropy_terms)))
    context["routing_policy"] = "GSLA-cost-delay-softmax"
    return True



def _flow_traffic_weight(request_flow: RequestFlow) -> float:
    """按到达率和token规模估计一条flow的通信权重。"""
    token_mb = (
        float(getattr(request_flow, "r_input_data_size", 0.0)) +
        float(getattr(request_flow, "r_output_data_size", 0.0))
    ) / 1000.0
    arrival_scale = min(max(float(getattr(request_flow, "arrival_rate", 1.0)), 1.0), 24.0) / 8.0
    return 1.0 + token_mb * arrival_scale


def _routing_weighted_instance_score(instance: MicroserviceInstance, candidate_server: EdgeServer,
                                     system_state: SystemState) -> Tuple[float, int]:
    """
    估计实例迁移到候选服务器后的路由加权通信代价。
    这里只用于GSLA二次局部调整，避免预路由proxy与最终routing脱节。
    """
    flow_id = _get_flow_id_from_instance_id(instance.instance_id)
    request_flow = system_state.request_flows.get(flow_id)
    routing = getattr(system_state, "stream_transfer_probabilities", {}).get(flow_id, {})
    if not request_flow or not routing or not system_state.network_topology:
        return _traditional_instance_score(instance, candidate_server, system_state), 0

    total_cost = 0.0
    consumed_paths = 0
    traffic_weight = _flow_traffic_weight(request_flow)
    current_server_id = instance.server_id
    candidate_server_id = candidate_server.server_id
    ms_id = instance.microservice.ms_id
    for transfer_key, probability in routing.items():
        origin_server_id, origin_ms_id, dest_server_id, dest_ms_id = transfer_key
        prob = float(probability)
        if prob <= 0:
            continue
        next_origin_server = origin_server_id
        next_dest_server = dest_server_id
        touched = False
        if origin_ms_id == ms_id and origin_server_id == current_server_id:
            next_origin_server = candidate_server_id
            touched = True
        if dest_ms_id == ms_id and dest_server_id == current_server_id:
            next_dest_server = candidate_server_id
            touched = True
        if not touched:
            continue
        delay = system_state.network_topology.get_communication_delay(next_origin_server, next_dest_server)
        total_cost += prob * traffic_weight * float(delay)
        consumed_paths += 1

    add_cores = 0 if candidate_server.server_id == instance.server_id else int(instance.allocated_cores)
    add_memory = 0.0 if candidate_server.server_id == instance.server_id else float(getattr(instance, "allocated_memory", 1.2))
    frag = _fragmentation_after(candidate_server, add_cores, add_memory)
    if consumed_paths == 0:
        return _traditional_instance_score(instance, candidate_server, system_state), 0
    return 0.62 * total_cost + 18.0 * frag, consumed_paths


def post_refine_gsla_after_routing(system_state: SystemState, context: Dict,
                                   max_moves: int = 8) -> bool:
    """
    在Softmax routing之后做一次受限局部调整。
    只接受降低routing-weighted通信代价的移动；该步骤是工程近似，不声明全局最优。
    """
    routing_paths = sum(len(v) for v in getattr(system_state, "stream_transfer_probabilities", {}).values())
    accepted = 0
    considered = 0
    traditional_accepted = 0
    traditional_considered = 0
    traditional_servers = [
        server for server in system_state.edge_servers.values()
        if server.server_type.value == "traditional"
    ]
    for instance_id in list(system_state.microservice_instances.keys()):
        if accepted >= max_moves:
            break
        instance = system_state.microservice_instances.get(instance_id)
        if not instance or instance.microservice.service_type != "traditional":
            continue
        current_server = system_state.edge_servers.get(instance.server_id)
        if not current_server:
            continue
        current_score, current_paths = _routing_weighted_instance_score(instance, current_server, system_state)
        best_server = current_server
        best_score = current_score
        best_paths = current_paths
        memory = float(getattr(instance, "allocated_memory", 1.2))
        for candidate in traditional_servers:
            if candidate.server_id == current_server.server_id:
                continue
            if candidate.available_cpu < instance.allocated_cores or candidate.available_memory < memory:
                continue
            considered += 1
            traditional_considered += 1
            score, consumed_paths = _routing_weighted_instance_score(instance, candidate, system_state)
            if consumed_paths <= 0:
                continue
            if score + 1e-6 < best_score:
                best_server = candidate
                best_score = score
                best_paths = consumed_paths
        improve_ratio = (current_score - best_score) / max(abs(current_score), 1e-6)
        if best_server.server_id != current_server.server_id and improve_ratio >= 0.015:
            _move_traditional_instance(instance_id, best_server, system_state)
            accepted += 1
            traditional_accepted += 1

    ai_accepted = 0
    ai_considered = 0
    ai_servers = [
        server for server in system_state.edge_servers.values()
        if server.server_type.value == "ai_capable"
    ]
    for instance_id in list(system_state.microservice_instances.keys()):
        if accepted >= max_moves:
            break
        instance = system_state.microservice_instances.get(instance_id)
        if not instance or instance.microservice.service_type != "ai":
            continue
        current_server = system_state.edge_servers.get(instance.server_id)
        if not current_server:
            continue
        current_score, current_paths = _routing_weighted_instance_score(instance, current_server, system_state)
        best_server = current_server
        best_score = current_score
        best_paths = current_paths
        for candidate in ai_servers:
            if candidate.server_id == current_server.server_id:
                continue
            if not _ai_instance_move_feasible(instance, candidate):
                continue
            considered += 1
            ai_considered += 1
            score, consumed_paths = _routing_weighted_instance_score(instance, candidate, system_state)
            if consumed_paths <= 0:
                continue
            if score + 1e-6 < best_score:
                best_server = candidate
                best_score = score
                best_paths = consumed_paths
        improve_ratio = (current_score - best_score) / max(abs(current_score), 1e-6)
        if best_server.server_id != current_server.server_id and best_paths > 0 and improve_ratio >= 0.015:
            _move_ai_instance(instance_id, best_server, system_state, context)
            accepted += 1
            ai_accepted += 1

    context["post_routing_refine"] = {
        "routing_aware": True,
        "routing_paths": int(routing_paths),
        "accepted_moves": int(accepted),
        "candidate_checks": int(considered),
        "traditional_accepted_moves": int(traditional_accepted),
        "traditional_candidate_checks": int(traditional_considered),
        "ai_accepted_moves": int(ai_accepted),
        "ai_candidate_checks": int(ai_considered),
        "objective": "routing-weighted delay + communication cost + fragmentation proxy",
        "move_threshold": 0.015,
    }
    return accepted > 0
def _get_successor_dest_placements(system_state: SystemState, flow_id: str,
                                    dest_ms: Microservice) -> Dict[str, float]:
    """获取后继服务可路由目的实例"""
    current_flow_placements = dict(
        system_state.stream_allocated_resources.get(flow_id, {}).get(dest_ms.ms_id, {})
    )
    if current_flow_placements:
        return current_flow_placements
    if dest_ms.service_type == "ai":
        return {}
    # 只有当前flow没有实例时才回退到共享传统实例，避免默认跨flow路由抬高通信成本。
    dest_placements = {}
    for _, resources in system_state.stream_allocated_resources.items():
        for server_id, allocated in resources.get(dest_ms.ms_id, {}).items():
            dest_placements[server_id] = dest_placements.get(server_id, 0) + allocated
    return dest_placements


def build_deterministic_routing_context(system_state: SystemState, context: Dict,
                                        policy_tag: str = "deterministic") -> bool:
    """为FFD/PDRS/LoadAware构建最近可行确定性路由"""
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
            dest_placements = _get_successor_dest_placements(system_state, flow_id, dest_ms)
            if not origin_placements or not dest_placements:
                continue
            for origin_server_id in origin_placements:
                scored = []
                for dest_server_id in dest_placements:
                    delay = 0.0
                    if system_state.network_topology:
                        delay = system_state.network_topology.get_communication_delay(origin_server_id, dest_server_id)
                    spare = _spare_capacity_score(system_state.edge_servers[dest_server_id], dest_ms.service_type)
                    scored.append((delay - 0.35 * spare, dest_server_id))
                _, selected_dest = min(scored, key=lambda row: (row[0], row[1]))
                row_key = f"{flow_id}:{origin_ms.ms_id}@{origin_server_id}->{dest_ms.ms_id}"
                routing_rows[row_key] = {selected_dest: 1.0}
                transfer_key = (origin_server_id, origin_ms.ms_id, selected_dest, dest_ms.ms_id)
                flow_probs[transfer_key] = Fraction(1, 1)
                request_flow.add_routing_probability(origin_server_id, selected_dest, 1.0)
        system_state.stream_transfer_probabilities[flow_id] = flow_probs
    context["deterministic_routing_matrix"] = routing_rows
    context["routing_policy"] = policy_tag
    context["routing_entropy"] = 0.0
    return True


def build_random_routing_context(system_state: SystemState, context: Dict,
                                 seed: int = 0) -> bool:
    """为Random baseline构建seeded随机可行路由"""
    rng = random.Random(seed)
    system_state.stream_transfer_probabilities.clear()
    for request_flow in system_state.request_flows.values():
        request_flow.routing_probabilities = {}
    routing_rows = {}
    entropy_terms = []
    for flow_id, request_flow in system_state.request_flows.items():
        chain = request_flow.service_chain.microservices
        flow_probs = {}
        for idx in range(1, len(chain)):
            origin_ms = chain[idx - 1]
            dest_ms = chain[idx]
            origin_placements = system_state.stream_allocated_resources.get(flow_id, {}).get(origin_ms.ms_id, {})
            dest_placements = _get_successor_dest_placements(system_state, flow_id, dest_ms)
            if not origin_placements or not dest_placements:
                continue
            dest_server_ids = sorted(dest_placements.keys())
            for origin_server_id in origin_placements:
                shuffled = list(dest_server_ids)
                rng.shuffle(shuffled)
                if len(shuffled) <= 1:
                    probs = [1.0]
                    selected = shuffled
                else:
                    selected = shuffled[:min(3, len(shuffled))]
                    probs = [1.0 / len(selected)] * len(selected)
                row_key = f"{flow_id}:{origin_ms.ms_id}@{origin_server_id}->{dest_ms.ms_id}"
                routing_rows[row_key] = dict(zip(selected, probs))
                for dest_server_id, prob in zip(selected, probs):
                    transfer_key = (origin_server_id, origin_ms.ms_id, dest_server_id, dest_ms.ms_id)
                    flow_probs[transfer_key] = Fraction(prob).limit_denominator(1000000)
                    request_flow.add_routing_probability(origin_server_id, dest_server_id, prob)
                    entropy_terms.append(-prob * math.log(prob + 1e-12))
        system_state.stream_transfer_probabilities[flow_id] = flow_probs
    context["random_routing_matrix"] = routing_rows
    context["routing_policy"] = "Random-uniform-feasible"
    context["routing_entropy"] = float(max(0.0, sum(entropy_terms)))
    return True


def run_deterministic_ai_placement(system_state: SystemState, policy_tag: str = "FFD") -> bool:
    """用于FFD等基线的确定性AI物理放置"""
    ai_servers = [
        server for server in system_state.edge_servers.values()
        if server.server_type.value == "ai_capable"
    ]
    if not ai_servers:
        return False
    for flow_id, request_flow in system_state.request_flows.items():
        ai_microservice = request_flow.service_chain.ai_microservice
        if not ai_microservice:
            continue
        predecessor = request_flow.service_chain.get_traditional_microservices()[-1].ms_id
        pred_placements = system_state.stream_allocated_resources.get(flow_id, {}).get(predecessor, {})
        anchor = max(pred_placements.items(), key=lambda row: row[1])[0] if pred_placements else None
        candidates = []
        for server in ai_servers:
            if server.available_cpu < 1:
                continue
            delay = 0.0
            if anchor and system_state.network_topology:
                delay = system_state.network_topology.get_communication_delay(anchor, server.server_id)
            spare = _spare_capacity_score(server, "ai")
            candidates.append((delay - spare, server))
        if not candidates:
            return False
        _, selected_server = min(candidates, key=lambda row: row[0])
        reservation = _estimate_ai_reservation(request_flow, ai_microservice, selected_server)
        _create_ai_instance(flow_id, request_flow, ai_microservice,
                            selected_server, system_state, reservation, policy_tag)
    return True


def run_Random_slow_context(system_state: SystemState, seed: int = 0,
                            max_retries: int = 5) -> bool:
    """
    Random慢层可行部署入口
    只从可行服务器集合采样，失败会恢复并重试。
    """
    from Constant import DecisionVariables

    rng = random.Random(seed)
    reset_slow_deployments(system_state)
    empty_backup = backup_slow_deployment_state(system_state)
    traditional_servers = [
        server for server in system_state.edge_servers.values()
        if server.server_type.value == "traditional"
    ]
    ai_servers = [
        server for server in system_state.edge_servers.values()
        if server.server_type.value == "ai_capable"
    ]
    for retry_count in range(max(1, max_retries)):
        restore_slow_deployment_state(system_state, empty_backup)
        decision_vars = DecisionVariables(system_state.time_frame)
        flow_items = list(system_state.request_flows.items())
        rng.shuffle(flow_items)
        ok = True
        for flow_id, request_flow in flow_items:
            calculate_ms_resource(request_flow)
            for microservice in request_flow.service_chain.get_traditional_microservices():
                cores = request_flow.service_chain.core_allocations.get(microservice.ms_id, 1)
                memory = request_flow.service_chain.memory_allocations.get(microservice.ms_id, 1.2)
                feasible = [
                    server for server in traditional_servers
                    if server.available_cpu >= cores and server.available_memory >= memory
                ]
                if not feasible:
                    ok = False
                    break
                selected_server = rng.choice(feasible)
                if not deploy_single_microservice_to_server(
                    request_flow, microservice, {"cores": cores, "memory": memory},
                    selected_server, system_state, decision_vars
                ):
                    ok = False
                    break
            if not ok:
                break
            ai_microservice = request_flow.service_chain.ai_microservice
            feasible_ai = [server for server in ai_servers if server.available_cpu >= 1]
            if not ai_microservice or not feasible_ai:
                ok = False
                break
            selected_ai = rng.choice(feasible_ai)
            reservation = _estimate_ai_reservation(request_flow, ai_microservice, selected_ai)
            _create_ai_instance(flow_id, request_flow, ai_microservice,
                                selected_ai, system_state, reservation, "Random")
        system_state.random_context = {
            "seed": seed,
            "retry_count": retry_count,
            "flow_order": [flow_id for flow_id, _ in flow_items],
        }
        if ok:
            build_random_routing_context(system_state, system_state.random_context, seed=seed + retry_count)
            return True
        system_state.random_context["failure_reason"] = "随机可行部署失败"
    return False



def _gmda_rmpr_softmax(scores: List[float]) -> List[float]:
    """GMDA-RMPR路由内部使用的稳定softmax。"""
    if not scores:
        return []
    values = np.asarray(scores, dtype=float)
    values = values - np.max(values)
    exp_values = np.exp(values)
    total = float(np.sum(exp_values)) or 1.0
    return [float(v / total) for v in exp_values]


def _gmda_traditional_score(server: EdgeServer, previous_server_id: str,
                            req: Dict[str, float], split_pressure: float,
                            system_state: SystemState) -> float:
    """GMDA传统服务放置评分，越低越好。"""
    cpu_after = float(server.available_cpu) - float(req.get("cores", 1))
    mem_after = float(server.available_memory) - float(req.get("memory", 1.0))
    if cpu_after < 0 or mem_after < 0:
        return float("inf")
    cpu_frag = cpu_after / max(float(server.cpu_cores), 1.0)
    mem_frag = mem_after / max(float(server.memory_capacity), 1.0)
    frag = abs(cpu_frag - mem_frag) + 0.25 * (1.0 - min(cpu_frag, mem_frag))
    delay = 0.0
    if previous_server_id and system_state.network_topology:
        delay = float(system_state.network_topology.get_communication_delay(previous_server_id, server.server_id))
    # resource-splitting压力越高，越偏向残量高且碎片低的节点。
    return 0.52 * delay + 35.0 * frag + 6.0 * float(split_pressure) * (1.0 - min(cpu_frag, mem_frag))


def _build_rmpr_probabilistic_routing_context(system_state: SystemState, context: Dict) -> bool:
    """按RMPR思想构造概率路由，不复用GSLA Softmax上下文。"""
    system_state.stream_transfer_probabilities.clear()
    for request_flow in system_state.request_flows.values():
        request_flow.routing_probabilities = {}
    routing_rows = {}
    entropy_terms = []
    for flow_id, request_flow in system_state.request_flows.items():
        chain = request_flow.service_chain.microservices
        flow_probs = {}
        for idx in range(1, len(chain)):
            origin_ms = chain[idx - 1]
            dest_ms = chain[idx]
            origin_placements = system_state.stream_allocated_resources.get(flow_id, {}).get(origin_ms.ms_id, {})
            dest_placements = _get_successor_dest_placements(system_state, flow_id, dest_ms)
            if not origin_placements or not dest_placements:
                continue
            split_pressure = float(context.get("flow_split_pressure", {}).get(flow_id, 1.0))
            for origin_server_id in sorted(origin_placements.keys()):
                dest_ids = sorted(dest_placements.keys())
                scores = []
                for dest_server_id in dest_ids:
                    server = system_state.edge_servers[dest_server_id]
                    delay = 0.0
                    if system_state.network_topology:
                        delay = float(system_state.network_topology.get_communication_delay(origin_server_id, dest_server_id))
                    spare = _spare_capacity_score(server, dest_ms.service_type)
                    residual_share = (
                        float(getattr(server, "available_cpu", 0.0)) / max(float(getattr(server, "cpu_cores", 1)), 1.0) +
                        float(getattr(server, "available_memory", 0.0)) / max(float(getattr(server, "memory_capacity", 1.0)), 1.0)
                    ) / 2.0
                    # RMPR工程适配：概率偏向低延迟、高残量、低切分压力路径。
                    score = 1.10 * spare + 0.75 * residual_share - 0.08 * delay - 0.18 * split_pressure
                    scores.append(float(score))
                probs = _gmda_rmpr_softmax(scores)
                row_key = f"{flow_id}:{origin_ms.ms_id}@{origin_server_id}->{dest_ms.ms_id}"
                routing_rows[row_key] = dict(zip(dest_ids, probs))
                for dest_server_id, prob in zip(dest_ids, probs):
                    transfer_key = (origin_server_id, origin_ms.ms_id, dest_server_id, dest_ms.ms_id)
                    flow_probs[transfer_key] = Fraction(prob).limit_denominator(1000000)
                    request_flow.add_routing_probability(origin_server_id, dest_server_id, prob)
                    entropy_terms.append(-prob * math.log(prob + 1e-12))
        system_state.stream_transfer_probabilities[flow_id] = flow_probs
    context["rmpr_routing_matrix"] = routing_rows
    context["routing_policy"] = "GMDA-RMPR-probabilistic"
    context["routing_entropy"] = float(max(0.0, sum(entropy_terms)))
    return True


def run_GMDA_RMPR_slow_context(system_state: SystemState) -> bool:
    """
    GMDA-RMPR慢层工程基线
    按Hu等GMDA资源切分部署与RMPR概率路由思想适配到hybrid service-chain。
    """
    from Constant import DecisionVariables

    print("\n=== 开始GMDA-RMPR慢层部署 ===")
    reset_slow_deployments(system_state)
    backup = backup_slow_deployment_state(system_state)
    context = {
        "implementation_boundary": "GMDA/RMPR engineering adaptation",
        "resource_splitting_groups": {},
        "reservation_envelope": {},
        "flow_split_pressure": {},
    }
    traditional_servers = [
        server for server in system_state.edge_servers.values()
        if server.server_type.value == "traditional"
    ]
    ai_servers = [
        server for server in system_state.edge_servers.values()
        if server.server_type.value == "ai_capable"
    ]
    if not traditional_servers or not ai_servers:
        context["failure_reason"] = "缺少传统或AI节点"
        system_state.gmda_rmpr_context = context
        return False

    decision_vars = DecisionVariables(system_state.time_frame)
    demand_ewma = copy.deepcopy(getattr(system_state, "gmda_demand_ewma", {}))
    try:
        ordered_flows = sorted(
            system_state.request_flows.values(),
            key=lambda flow: float(demand_ewma.get(flow.flow_id, flow.arrival_rate)),
            reverse=True,
        )
        for request_flow in ordered_flows:
            calculate_ms_resource(request_flow)
            flow_id = request_flow.flow_id
            prev = float(demand_ewma.get(flow_id, request_flow.arrival_rate))
            demand_hat = 0.60 * prev + 0.40 * float(request_flow.arrival_rate)
            demand_ewma[flow_id] = demand_hat
            chain = request_flow.service_chain.get_traditional_microservices()
            total_cpu = sum(float(request_flow.service_chain.core_allocations.get(ms.ms_id, 1)) for ms in chain)
            total_mem = sum(float(request_flow.service_chain.memory_allocations.get(ms.ms_id, 1.0)) for ms in chain)
            split_pressure = max(1.0, demand_hat / max(float(np.mean([f.arrival_rate for f in system_state.request_flows.values()])), 1e-6))
            context["flow_split_pressure"][flow_id] = float(split_pressure)
            context["resource_splitting_groups"][flow_id] = {
                "traditional_count": len(chain),
                "total_cpu": float(total_cpu),
                "total_memory": float(total_mem),
                "demand_hat": float(demand_hat),
                "split_pressure": float(split_pressure),
            }
            previous_server_id = ""
            for microservice in chain:
                cores = request_flow.service_chain.core_allocations.get(microservice.ms_id, 1)
                memory = request_flow.service_chain.memory_allocations.get(microservice.ms_id, 1.2)
                req = {"cores": cores, "memory": memory}
                feasible = [
                    server for server in traditional_servers
                    if server.available_cpu >= cores and server.available_memory >= memory
                ]
                if not feasible:
                    raise RuntimeError(f"{flow_id}:{microservice.ms_id}无可行传统节点")
                selected = min(
                    feasible,
                    key=lambda server: (
                        _gmda_traditional_score(server, previous_server_id, req, split_pressure, system_state),
                        server.server_id,
                    ),
                )
                if not deploy_single_microservice_to_server(
                    request_flow, microservice, req, selected, system_state, decision_vars
                ):
                    raise RuntimeError(f"{flow_id}:{microservice.ms_id}部署失败")
                previous_server_id = selected.server_id

        residual = {
            server.server_id: {
                "gpu_units": float(server.gpu_units),
                "gpu_memory": float(server.gpu_memory),
                "model_storage": float(server.model_storage),
                "context_units": float(getattr(server, "max_concurrent_contexts", 8)),
            }
            for server in ai_servers
        }
        for request_flow in ordered_flows:
            ai_microservice = request_flow.service_chain.ai_microservice
            if not ai_microservice:
                continue
            predecessor = request_flow.service_chain.get_traditional_microservices()[-1].ms_id
            pred_placements = system_state.stream_allocated_resources.get(request_flow.flow_id, {}).get(predecessor, {})
            anchor = max(pred_placements.items(), key=lambda row: row[1])[0] if pred_placements else ""
            candidates = []
            for server in ai_servers:
                reservation = _estimate_ai_reservation(request_flow, ai_microservice, server, system_state)
                if not _ai_reservation_feasible(server.server_id, residual, reservation):
                    continue
                delay = 0.0
                if anchor and system_state.network_topology:
                    delay = float(system_state.network_topology.get_communication_delay(anchor, server.server_id))
                spare = _spare_capacity_score(server, "ai")
                pressure = (
                    reservation["gpu_units"] / max(float(server.gpu_units), 1.0) +
                    reservation["gpu_memory"] / max(float(server.gpu_memory), 1.0) +
                    reservation["model_storage"] / max(float(server.model_storage), 1.0)
                ) / 3.0
                candidates.append((delay - 3.0 * spare + 8.0 * pressure, server, reservation))
            if not candidates:
                raise RuntimeError(f"{request_flow.flow_id}:AI reservation envelope不可行")
            _, selected_ai, reservation = min(candidates, key=lambda row: (row[0], row[1].server_id))
            _consume_ai_reservation(selected_ai.server_id, residual, reservation)
            _create_ai_instance(
                request_flow.flow_id, request_flow, ai_microservice,
                selected_ai, system_state, reservation, "GMDA-RMPR"
            )
            context["reservation_envelope"][request_flow.flow_id] = {
                "server_id": selected_ai.server_id,
                "gpu_units_reserved": float(reservation["gpu_units"]),
                "gpu_memory_reserved": float(reservation["gpu_memory"]),
                "model_storage_reserved": float(reservation["model_storage"]),
            }
        system_state.gmda_demand_ewma = demand_ewma
        _build_rmpr_probabilistic_routing_context(system_state, context)
        context["status"] = "ok"
        system_state.gmda_rmpr_context = context
        return True
    except Exception as exc:
        restore_slow_deployment_state(system_state, backup)
        context["status"] = "failed"
        context["failure_reason"] = str(exc)
        system_state.gmda_rmpr_context = context
        return False
def run_GSLA(system_state: SystemState) -> bool:
    """
    GSLA慢层部署入口
    使用RAPPA/HAPA/Softmax routing的工程实现。
    """
    print("\n=== 开始GSLA慢层部署 ===")
    context = {
        "implementation_boundary": "engineering-implementation",
    }
    reset_slow_deployments(system_state)
    base_backup = backup_slow_deployment_state(system_state)
    traditional_ok = run_RAPPA_traditional_placement(system_state, context)
    if not traditional_ok:
        restore_slow_deployment_state(system_state, base_backup)
        print("GSLA传统微服务部署失败")
        system_state.gsla_context = context
        return False

    ai_ok = run_HAPA_ai_placement(system_state, context)
    if not ai_ok:
        print("GSLA AI微服务物理部署失败")
        system_state.gsla_context = context
        return False

    build_softmax_routing_context(system_state, context)
    if post_refine_gsla_after_routing(system_state, context):
        # 二次调整移动了实例，必须重建最终routing，避免概率路径指向旧服务器。
        build_softmax_routing_context(system_state, context)
    system_state.previous_ai_routing_state = {
        "stream_transfer_probabilities": copy.deepcopy(system_state.stream_transfer_probabilities),
        "routing_probabilities": {
            flow_id: copy.deepcopy(getattr(request_flow, "routing_probabilities", {}))
            for flow_id, request_flow in system_state.request_flows.items()
        },
    }
    system_state.gsla_context = context
    return True


def _loadaware_server_score(server: EdgeServer, request_flow: RequestFlow,
                            microservice: Microservice, req: Dict,
                            system_state: SystemState,
                            expected_cost_weight: float = 0.0,
                            traffic_cost_weight: float = 0.0) -> float:
    """计算容量感知贪心放置分数，越低越好"""
    cpu_after = server.available_cpu - req.get("cores", 1)
    mem_after = server.available_memory - req.get("memory", 1.2)
    if cpu_after < 0 or mem_after < 0:
        return float("inf")
    cpu_frag = cpu_after / max(server.cpu_cores, 1)
    mem_frag = mem_after / max(server.memory_capacity, 1.0)
    frag = abs(cpu_frag - mem_frag) + 0.25 * (1.0 - min(cpu_frag, mem_frag))
    chain = request_flow.service_chain.get_traditional_microservices()
    ms_ids = [ms.ms_id for ms in chain]
    locality = 0.0
    traffic_cost_term = 0.0
    if microservice.ms_id in ms_ids and system_state.network_topology:
        idx = ms_ids.index(microservice.ms_id)
        pred_ids = [ms_ids[idx - 1]] if idx > 0 else []
        delays = []
        for pred_id in pred_ids:
            for pred_server in system_state.stream_allocated_resources.get(request_flow.flow_id, {}).get(pred_id, {}):
                delays.append(system_state.network_topology.get_communication_delay(pred_server, server.server_id))
        locality = float(np.mean(delays)) if delays else 0.0
        if delays and traffic_cost_weight > 0.0:
            traffic_cost_term = traffic_cost_weight * locality * _flow_traffic_weight(request_flow)
    cost_term = expected_cost_weight * req.get("cores", 1) * 0.5
    return locality + traffic_cost_term + 20.0 * frag + cost_term


def _place_traditional_loadaware(system_state: SystemState, policy_tag: str,
                                 demand_scaled: bool = False) -> Tuple[bool, Dict]:
    """容量感知传统微服务放置，不调用next-fit/GSLA"""
    from Constant import DecisionVariables

    decision_vars = DecisionVariables(system_state.time_frame)
    traditional_servers = [
        server for server in system_state.edge_servers.values()
        if server.server_type.value == "traditional"
    ]
    demand_ewma = copy.deepcopy(getattr(system_state, "pdrs_demand_ewma", {}))
    arrivals = [float(flow.arrival_rate) for flow in system_state.request_flows.values()]
    mean_arrival = float(np.mean(arrivals)) if arrivals else 1.0
    traffic_cost_weight = 0.35 if demand_scaled else 0.0
    placed = []
    scale_out_count = 0
    ordered_flows = sorted(
        system_state.request_flows.values(),
        key=lambda flow: float(demand_ewma.get(flow.flow_id, flow.arrival_rate)),
        reverse=True
    )
    for request_flow in ordered_flows:
        calculate_ms_resource(request_flow)
        previous = float(demand_ewma.get(request_flow.flow_id, request_flow.arrival_rate))
        demand_hat = 0.65 * previous + 0.35 * float(request_flow.arrival_rate)
        demand_ewma[request_flow.flow_id] = demand_hat
        demand_factor = max(1.0, demand_hat / max(mean_arrival, 1e-6)) if demand_scaled else 1.0
        for microservice in request_flow.service_chain.get_traditional_microservices():
            base_cores = request_flow.service_chain.core_allocations.get(microservice.ms_id, 1)
            base_memory = request_flow.service_chain.memory_allocations.get(microservice.ms_id, 1.2)
            replica_count = 2 if demand_scaled and demand_factor > 1.35 and len(traditional_servers) > 1 else 1
            per_replica_cores = max(1, int(math.ceil(base_cores * min(demand_factor, 2.2) / replica_count)))
            per_replica_memory = max(0.5, base_memory * min(demand_factor, 2.2) / replica_count)
            used_servers = set()
            for _ in range(replica_count):
                req = {"cores": per_replica_cores, "memory": per_replica_memory}
                candidates = [
                    server for server in traditional_servers
                    if server.server_id not in used_servers and
                    server.available_cpu >= per_replica_cores and
                    server.available_memory >= per_replica_memory
                ]
                if not candidates:
                    return False, {
                        "failure_reason": f"{request_flow.flow_id}:{microservice.ms_id}无可行传统节点",
                        "pdrs_demand_ewma": demand_ewma,
                    }
                selected = min(
                    candidates,
                    key=lambda server: (
                        _loadaware_server_score(
                            server, request_flow, microservice, req, system_state,
                            traffic_cost_weight=traffic_cost_weight,
                        ),
                        server.server_id
                    )
                )
                ok = deploy_single_microservice_to_server(
                    request_flow, microservice, req, selected, system_state, decision_vars
                )
                if not ok:
                    return False, {
                        "failure_reason": f"{request_flow.flow_id}:{microservice.ms_id}部署失败",
                        "pdrs_demand_ewma": demand_ewma,
                    }
                used_servers.add(selected.server_id)
                placed.append((request_flow.flow_id, microservice.ms_id, selected.server_id))
            if replica_count > 1:
                scale_out_count += replica_count - 1
    return True, {
        "placed": placed,
        "scale_out_count": scale_out_count,
        "pdrs_demand_ewma": demand_ewma,
        "policy": policy_tag,
        "placement_policy": "loadaware-traffic-cost" if demand_scaled else "loadaware",
        "placement_score_weights": {
            "locality": 1.0,
            "traffic_cost": traffic_cost_weight,
            "fragmentation": 20.0,
            "expected_cost": 0.0,
        },
    }


def run_LoadAware_slow_context(system_state: SystemState) -> bool:
    """LoadAware强启发式慢层基线"""
    print("\n=== 开始LoadAware慢层部署 ===")
    reset_slow_deployments(system_state)
    backup = backup_slow_deployment_state(system_state)
    ok, context = _place_traditional_loadaware(system_state, "LoadAware", demand_scaled=False)
    if not ok:
        restore_slow_deployment_state(system_state, backup)
        system_state.loadaware_context = context
        return False
    if not run_deterministic_ai_placement(system_state, policy_tag="LoadAware"):
        restore_slow_deployment_state(system_state, backup)
        context["failure_reason"] = "LoadAware AI确定性放置失败"
        system_state.loadaware_context = context
        return False
    build_deterministic_routing_context(system_state, context, policy_tag="LoadAware-deterministic")
    system_state.loadaware_context = context
    return True


def run_PDRS_slow_context(system_state: SystemState) -> bool:
    """
    PDRS慢层部署入口
    独立动态资源调度工程基线，不调用GSLA/RAPPA/next-fit。
    """
    print("\n=== 开始PDRS动态资源调度慢层部署 ===")
    reset_slow_deployments(system_state)
    backup = backup_slow_deployment_state(system_state)
    ok, context = _place_traditional_loadaware(system_state, "PDRS", demand_scaled=True)
    if not ok:
        restore_slow_deployment_state(system_state, backup)
        context["status"] = "failed"
        system_state.pdrs_context = context
        return False
    system_state.pdrs_demand_ewma = copy.deepcopy(context.get("pdrs_demand_ewma", {}))
    if not run_deterministic_ai_placement(system_state, policy_tag="PDRS"):
        restore_slow_deployment_state(system_state, backup)
        context["status"] = "failed"
        context["failure_reason"] = "PDRS AI确定性放置失败"
        system_state.pdrs_context = context
        return False
    build_deterministic_routing_context(system_state, context, policy_tag="PDRS-loadaware")
    context["status"] = "ok"
    context["boundary"] = "independent DRS engineering baseline, not optimal DRS"
    system_state.pdrs_context = context
    return True


def deploy_microservices_distributed(flow, microservices, ms_requirements, preferred_server,
                                     all_servers, system_state, decision_vars) -> bool:
    """
    分布式部署微服务
    Args:
        flow: 请求流对象
        microservices: 微服务列表
        ms_requirements: 微服务资源需求字典
        preferred_server: 首选服务器
        all_servers: 所有传统服务器列表
        system_state: 系统状态
        decision_vars: 决策变量对象
    Returns:
        bool: 是否部署成功
    """
    deployed_count = 0

    for ms in microservices:
        req = ms_requirements[ms.ms_id]
        deployed = False

        # 首先尝试在首选服务器上部署
        if (preferred_server.available_cpu >= req['cores'] and
                preferred_server.available_memory >= req['memory']):

            deployed = deploy_single_microservice_to_server(
                flow, ms, req, preferred_server, system_state, decision_vars)

            if deployed:
                #print(f"微服务 {ms.ms_id} 部署到当前服务器 {preferred_server.server_id}")
                deployed_count += 1
                continue

        #print(f"首选服务器 {preferred_server.server_id} 资源不足，寻找替代服务器...")


        # 利用NetworkTopology按通信延迟排序服务器
        alternative_servers = get_servers_sorted_by_communication_delay(
            preferred_server.server_id, all_servers, system_state)

        for alt_server in alternative_servers:
            if (alt_server.available_cpu >= req['cores'] and
                    alt_server.available_memory >= req['memory']):

                deployed = deploy_single_microservice_to_server(
                    flow, ms, req, alt_server, system_state, decision_vars)

                if deployed:
                    delay = system_state.network_topology.get_communication_delay(
                        preferred_server.server_id, alt_server.server_id)
                    #print(f"微服务 {ms.ms_id} 部署到 {alt_server.server_id} (通信延迟: {delay:.2f}ms)")
                    preferred_server = alt_server
                    deployed_count += 1
                    break

        if not deployed:
            print(f"⚠️  微服务 {ms.ms_id} 部署失败：所有服务器资源不足")

    return deployed_count == len(microservices)


def deploy_single_microservice_to_server(flow, ms, requirements, server, system_state, decision_vars) -> bool:
    """
    部署单个微服务到指定服务器，使用微服务真实的service_rate属性
    Args:
        flow: 请求流对象
        ms: 微服务对象
        requirements: 资源需求字典
        server: 目标服务器
        system_state: 系统状态
        decision_vars: 决策变量对象
    Returns:
        bool: 是否部署成功
    """
    from Constant import MicroserviceInstance

    req_cores = requirements['cores']
    req_memory = requirements['memory']

    if (server.available_cpu < req_cores or server.available_memory < req_memory):
        return False

    # 创建微服务实例
    instance_id = f"{flow.flow_id}_{ms.ms_id}_{server.server_id}"
    instance = MicroserviceInstance(
        instance_id=instance_id,
        microservice=ms,
        server_id=server.server_id,
        allocated_cores=req_cores,
        arrival_rate=normalize_arrival_rate_for_queue(flow.arrival_rate)
    )
    instance.allocated_cpu_cores = req_cores
    instance.allocated_memory = req_memory

    # 使用微服务的实际service_rate计算服务指标
    if ms.service_rate > 0:
        total_service_rate = req_cores * ms.service_rate
        instance.service_intensity = normalize_arrival_rate_for_queue(flow.arrival_rate) / total_service_rate
        instance.processing_delay = 1.0 / ms.service_rate

        # 这里不需要重新计算队列延迟，因为核心数已经按延迟要求计算好了
        if instance.service_intensity < 1:
            # 这只是理论值，实际延迟已在资源分配阶段保证
            instance.queue_delay = instance.service_intensity / (1 - instance.service_intensity) / ms.service_rate
        else:
            instance.queue_delay = 0.0  # 已分配足够核心，理论上队列延迟很小

    # 更新服务器资源
    server.available_cpu -= req_cores
    server.available_memory -= req_memory

    # 添加到系统状态
    system_state.microservice_instances[instance_id] = instance

    # 更新决策变量
    decision_vars.set_microservice_deployment(ms.ms_id, server.server_id, 1)
    decision_vars.set_core_allocation((ms.ms_id, server.server_id), req_cores)

    # *** 设置路由概率矩阵所需的资源分配信息 ***
    system_state.set_stream_allocated_resource(
        flow_id=flow.flow_id,
        ms_id=ms.ms_id,
        server_id=server.server_id,
        allocated_resource=req_cores
    )

    return True


def get_servers_sorted_by_communication_delay(source_server_id: str, all_servers, system_state):
    """
    利用NetworkTopology按通信延迟排序服务器
    Args:
        source_server_id: 源服务器ID
        all_servers: 所有服务器列表
        system_state: 系统状态对象

    Returns:
        List: 按通信延迟升序排序的服务器列表
    """
    from Constant import ServerType

    if not system_state.network_topology:
        return [s for s in all_servers if s.server_id != source_server_id]

    server_delays = []
    for server in all_servers:
        if server.server_id != source_server_id and server.server_type.value == "traditional":
            delay = system_state.network_topology.get_communication_delay(
                source_server_id, server.server_id)
            server_delays.append((server, delay))

    # 按延迟升序排序
    server_delays.sort(key=lambda x: x[1])
    return [server for server, _ in server_delays]

def deploy_ai_microservices(system_state: SystemState) -> bool:
    """
    部署所有请求流的AI微服务
    每个AI微服务只部署在一个AI服务器上，支持本地推理和云端卸载
    1. 只确定AI微服务的物理部署位置（在哪个AI服务器上）
    2. 不决定处理模式（本地/云端）
    3. 预留资源但不立即消耗
    Args:
        system_state: 系统状态对象
    Returns:
        bool: 是否所有AI微服务都成功部署到AI服务器上
    """
    print("\n=== 开始AI微服务部署 ===")

    # 获取所有AI服务器
    ai_servers = [server for server in system_state.edge_servers.values()
                  if server.server_type.value == "ai_capable"]

    successful_deployments = 0
    total_ai_microservices = 0

    # 收集所有AI微服务并按资源需求排序
    ai_microservice_flows = []
    for flow_id, request_flow in system_state.request_flows.items():
        ai_microservice = request_flow.service_chain.ai_microservice
        if ai_microservice:
            ai_resource_requirement = calculate_ai_microservice_resource_requirement(request_flow, ai_microservice, system_state)
            ai_microservice_flows.append((flow_id, request_flow, ai_microservice, ai_resource_requirement))
            total_ai_microservices += 1

    # 按资源需求降序排序（优先满足高需求的AI微服务）
    ai_microservice_flows.sort(key=lambda x: x[3], reverse=True)

    #print(f"\n需要部署的AI微服务: {total_ai_microservices} 个")

    # 为每个AI微服务选择最佳的物理部署位置
    for flow_id, request_flow, ai_microservice, resource_requirement in ai_microservice_flows:
        #print(f"\n处理AI微服务: {ai_microservice.ms_id} (来自 {flow_id})")
        #print(f"  资源需求评分: {resource_requirement:.2f}")

        # 选择最佳AI服务器进行物理部署
        deployment_result = deploy_ai_microservice_physical_only(
            flow_id, request_flow, ai_microservice, ai_servers, system_state)

        if deployment_result['success']:
            #print(f"  ✓ 物理部署成功到 {deployment_result['server_id']}")
            #print(f"    状态: {deployment_result['status']}")
            successful_deployments += 1
        else:
            print(f"  ✗ 物理部署失败: {deployment_result['reason']}")
    '''
    print(f"\n=== AI微服务物理部署完成 ===")
    print(f"成功部署: {successful_deployments}/{total_ai_microservices}")
    print(f"成功率: {successful_deployments / total_ai_microservices * 100:.1f}%")
    '''
    # 输出部署摘要
    #print_ai_physical_deployment_summary(system_state)

    return successful_deployments == total_ai_microservices


def deploy_ai_microservice_physical_only(flow_id: str, request_flow: RequestFlow,
                                         ai_microservice: Microservice, ai_servers: List[EdgeServer],
                                         system_state: SystemState) -> Dict:
    """
    AI微服务物理部署（只确定位置，不确定处理模式）

    部署策略：
    1. 选择最佳AI服务器（基于资源可用性和位置优势）
    2. 创建AI微服务实例，但处理模式设为"pending"
    3. 预留少量CPU资源，GPU资源不消耗（等待RL决策）

    Args:
        flow_id: 请求流ID
        request_flow: 请求流对象
        ai_microservice: AI微服务对象
        ai_servers: AI服务器列表
        system_state: 系统状态对象

    Returns:
        Dict: 部署结果
    """
    #print(f"\n开始AI微服务物理部署: {ai_microservice.ms_id}")

    # 第一步：选择最佳AI服务器（不考虑具体资源是否充足）
    selected_ai_server = select_best_ai_server_for_physical_deployment(ai_servers, request_flow, ai_microservice, system_state)

    if not selected_ai_server:
        return {
            'success': False,
            'reason': '没有可用的AI服务器进行物理部署',
            'server_id': None,
            'status': 'failed'
        }

    # 第二步：在选定的AI服务器上创建AI微服务实例（物理部署）
    try:
        # 创建AI微服务实例ID
        instance_id = f"{flow_id}_{ai_microservice.ms_id}_{selected_ai_server.server_id}"

        # 创建微服务实例，处理模式设为待定
        ai_instance = MicroserviceInstance(
            instance_id=instance_id,
            microservice=ai_microservice,
            server_id=selected_ai_server.server_id,  # 物理部署位置
            allocated_cores=1,  # 预留1个CPU用于调度（无论后续是本地还是云端）
            arrival_rate=request_flow.arrival_rate
        )

        # 设置为物理部署完成但处理模式待定状态
        ai_instance.processing_mode = "pending_decision"  # 待RL决策
        ai_instance.gpu_memory_allocated = 0.0  # 暂不分配GPU资源
        ai_instance.model_storage_allocated = 0.0  # 暂不分配存储资源
        ai_instance.inference_latency = 0.0  # 待RL决策后计算
        ai_instance.cloud_latency = 0.0  # 待RL决策后计算

        # 只消耗少量CPU资源用于调度，GPU资源等待RL决策
        selected_ai_server.available_cpu -= 1  # 预留1个CPU核心
        # GPU相关资源暂不消耗

        # 添加到系统状态
        system_state.microservice_instances[instance_id] = ai_instance

        # 设置请求流的AI处理选择为待定
        request_flow.ai_processing_choice = "pending"

        # 更新路由概率矩阵（AI微服务已确定物理位置）
        system_state.set_stream_allocated_resource(
            flow_id=flow_id,
            ms_id=ai_microservice.ms_id,
            server_id=selected_ai_server.server_id,
            allocated_resource=1
        )
        '''
        print(f"        AI微服务物理部署成功:")
        print(f"          部署服务器: {selected_ai_server.server_id}")
        print(f"          处理模式: 待RL决策")
        print(f"          CPU预留: 1 核心")
        print(f"          GPU资源: 待分配")
        '''
        return {
            'success': True,
            'server_id': selected_ai_server.server_id,
            'status': 'physical_deployed_pending_decision',
            'instance_id': instance_id
        }

    except Exception as e:
        return {
            'success': False,
            'reason': f'物理部署异常: {str(e)}',
            'server_id': selected_ai_server.server_id,
            'status': 'failed'
        }


def select_best_ai_server_for_physical_deployment(ai_servers: List[EdgeServer],
                                                  request_flow: RequestFlow,
                                                  ai_microservice: Microservice,
                                                  system_state: SystemState) -> Optional[EdgeServer]:
    """
    为物理部署选择最佳AI服务器
    选择策略：
    1. 优先选择未被占用的AI服务器
    2. 评估综合适应性（资源容量、网络位置等）
    3. 不强制要求资源充足（资源分配在RL决策阶段）

    Args:
        ai_servers: AI服务器列表
        request_flow: 请求流对象
        ai_microservice: AI微服务对象
        system_state: 系统状态对象

    Returns:
        Optional[EdgeServer]: 选中的AI服务器，无可用服务器时返回None
    """
    if not ai_servers:
        return None

    # 过滤可用的AI服务器（允许单个AI节点承载多个AI微服务）
    available_servers = []
    max_ai_capacity = 3
    for server in ai_servers:
        current_ai_workload = sum(1 for inst in system_state.microservice_instances.values()
                                  if inst.server_id == server.server_id and
                                  inst.microservice.service_type == "ai")
        if current_ai_workload < max_ai_capacity:
            available_servers.append(server)

    #print(f"  可选AI服务器: {[s.server_id for s in available_servers]}")

    # 评估每个AI服务器的适应性分数
    server_scores = []
    for server in available_servers:
        score = calculate_ai_server_suitability_score(server, request_flow, ai_microservice, system_state)
        server_scores.append((server, score))
        #print(f"    {server.server_id}: 适应性分数 {score:.3f}")

    if not server_scores:
        print("  没有可用AI服务器，启用资源评分兜底选择")
        for server in ai_servers:
            score = calculate_ai_server_suitability_score(server, request_flow, ai_microservice, system_state)
            server_scores.append((server, score))

    # 按适应性分数排序，选择最佳服务器
    server_scores.sort(key=lambda x: x[1], reverse=True)
    selected_server = server_scores[0][0]

    #print(f"  选择AI服务器: {selected_server.server_id} (分数: {server_scores[0][1]:.3f})")
    return selected_server


def calculate_ai_server_suitability_score(server: EdgeServer, request_flow: RequestFlow,
                                          ai_microservice: Microservice, system_state: SystemState) -> float:
    """
    计算AI服务器对特定AI微服务的适应性分数

    评估因素：
    1. 资源容量（GPU内存、存储等）
    2. 当前负载情况
    3. 网络位置优势（与传统微服务的通信）
    4. 能耗效率

    Args:
        server: AI服务器对象
        request_flow: 请求流对象
        ai_microservice: AI微服务对象
        system_state: 系统状态对象

    Returns:
        float: 适应性分数（0-1之间，越高越适合）
    """
    total_score = 0.0

    # 1. 资源容量评分（40%权重）
    gpu_memory_ratio = server.available_gpu_memory / max(server.gpu_memory, 1.0)
    cpu_ratio = server.available_cpu / max(server.cpu_cores, 1)
    storage_ratio = server.available_model_storage / max(server.model_storage, 1.0)

    resource_score = (gpu_memory_ratio * 0.5 + cpu_ratio * 0.3 + storage_ratio * 0.2)
    total_score += resource_score * 0.4

    # 2. 负载均衡评分（30%权重）
    # 当前服务器上AI微服务的数量
    current_ai_workload = sum(1 for inst in system_state.microservice_instances.values()
                              if inst.server_id == server.server_id and
                              inst.microservice.service_type == "ai")

    # 负载越轻，分数越高
    max_ai_capacity = 3  # 假设每个AI服务器最多承载3个AI微服务
    load_balance_score = max(0, (max_ai_capacity - current_ai_workload) / max_ai_capacity)
    total_score += load_balance_score * 0.3

    # 3. 网络位置优势评分（20%权重）
    network_score = calculate_network_advantage_score(server, request_flow, system_state)
    total_score += network_score * 0.2

    # 4. 能耗效率评分（10%权重）
    energy_efficiency = 1.0 - (server.base_power / 150.0)  # 标准化能耗
    total_score += max(0, energy_efficiency) * 0.1

    return min(total_score, 1.0)


def calculate_network_advantage_score(server: EdgeServer, request_flow: RequestFlow,
                                      system_state: SystemState) -> float:
    """
    计算AI服务器的网络位置优势分数

    优势因素：
    - 与该请求流的传统微服务部署位置的平均通信延迟
    - 延迟越低，网络优势越大

    Args:
        server: AI服务器对象
        request_flow: 请求流对象
        system_state: 系统状态对象

    Returns:
        float: 网络优势分数（0-1之间）
    """
    if not system_state.network_topology:
        return 0.5  # 无网络拓扑信息时返回中性分数

    # 找到该请求流的传统微服务部署在哪些服务器上
    traditional_servers = []
    traditional_microservices = request_flow.service_chain.get_traditional_microservices()

    import Waitingtime
    for ms in traditional_microservices:
        ms_server = Waitingtime.find_microservice_server(ms.ms_id, system_state)
        if ms_server and ms_server not in traditional_servers:
            traditional_servers.append(ms_server)

    if not traditional_servers:
        return 0.5  # 无传统微服务时返回中性分数

    # 计算平均通信延迟
    total_delay = 0.0
    for trad_server in traditional_servers:
        delay = system_state.network_topology.get_communication_delay(trad_server, server.server_id)
        total_delay += delay

    avg_delay = total_delay / len(traditional_servers)

    # 将延迟转换为分数（延迟越低分数越高）
    # 假设最大延迟为50ms，最小延迟为1ms
    max_delay = 50.0
    min_delay = 1.0
    normalized_delay = min(max(avg_delay, min_delay), max_delay)
    network_score = 1.0 - ((normalized_delay - min_delay) / (max_delay - min_delay))

    return network_score


def print_ai_physical_deployment_summary(system_state: SystemState):
    """
    打印AI微服务物理部署摘要
    """
    print(f"\n=== AI微服务物理部署摘要 ===")

    pending_instances = []
    total_ai_instances = 0

    for instance_id, instance in system_state.microservice_instances.items():
        if instance.microservice.service_type == "ai":
            total_ai_instances += 1
            processing_mode = getattr(instance, 'processing_mode', 'unknown')

            if processing_mode == "pending_decision":
                pending_instances.append(instance)

    print(f"总AI微服务: {total_ai_instances} 个")
    print(f"待RL决策: {len(pending_instances)} 个")

    if pending_instances:
        print(f"\n待决策的AI微服务:")
        for instance in pending_instances:
            print(f"  {instance.instance_id}: 部署在 {instance.server_id}")
            print(f"    状态: 物理部署完成，等待RL决策处理模式")

    # AI服务器资源预留情况
    ai_servers = [server for server in system_state.edge_servers.values()
                  if server.server_type.value == "ai_capable"]

    if ai_servers:
        print(f"\nAI服务器资源预留情况:")
        for server in ai_servers:
            # 统计该服务器上部署的AI微服务数量
            ai_count = sum(1 for inst in system_state.microservice_instances.values()
                           if inst.server_id == server.server_id and
                           inst.microservice.service_type == "ai")

            print(f"  {server.server_id}:")
            print(f"    部署的AI微服务: {ai_count} 个")
            print(f"    CPU预留: {ai_count} 核心")
            print(f"    可用CPU: {server.available_cpu}/{server.cpu_cores}")
            print(f"    GPU资源: 待RL决策分配")

def calculate_ai_microservice_resource_requirement(request_flow: RequestFlow,
                                                   ai_microservice: Microservice,
                                                   system_state: SystemState) -> float:
    """
    计算AI微服务的资源需求评分
    考虑GPU显存、计算能力、批处理效率等因素
    Args:
        request_flow: 请求流对象
        ai_microservice: AI微服务对象
        system_state: 系统状态对象
    Returns:
        float: 资源需求评分（越高表示需求越大）
    """
    # 基础参数
    arrival_rate = request_flow.arrival_rate  # requests/ms
    input_tokens = request_flow.r_input_data_size  # 输入token数量
    output_tokens = request_flow.r_output_data_size  # 输出token数量
    total_tokens = input_tokens + output_tokens

    # GPU显存需求 (基于token数量和模型大小)
    # 参考vidur中模型的显存需求，Llama-3-8B约需要16GB显存
    base_memory_gb = 16.0  # AI微服务基础显存需求
    token_memory_gb = total_tokens * 0.001  # 每个token额外的显存需求
    gpu_memory_requirement = base_memory_gb + token_memory_gb

    # 计算资源需求 (基于到达率和处理复杂度)
    processing_complexity = math.log(total_tokens + 1)  # 处理复杂度随token数对数增长
    compute_requirement = arrival_rate * processing_complexity

    # 批处理效率考虑
    optimal_batch_size = min(32, max(1, int(arrival_rate * 100)))  # 基于到达率确定最优批处理大小
    batch_efficiency = optimal_batch_size / 32.0  # 标准化到0-1

    # 综合资源需求评分
    gpu_score = gpu_memory_requirement / 16.0  # 标准化显存分数
    compute_score = compute_requirement / 100.0  # 标准化计算分数

    total_score = gpu_score * 0.6 + compute_score * 0.3 + batch_efficiency * 0.1
    '''
    print(f"    AI资源需求分析:")
    print(f"      GPU显存需求: {gpu_memory_requirement:.2f} GB")
    print(f"      计算需求: {compute_requirement:.2f}")
    print(f"      最优批处理大小: {optimal_batch_size}")
    print(f"      综合评分: {total_score:.3f}")
    '''
    return total_score

def is_ai_server_occupied(server_id: str, system_state: SystemState) -> bool:
    """
    检查AI服务器是否已经被AI微服务占用
    Args:
        server_id: AI服务器ID
        system_state: 系统状态对象
    Returns:
        bool: True表示已被占用，False表示可用
    """
    for instance in system_state.microservice_instances.values():
        if (instance.server_id == server_id and
            instance.microservice.service_type == "ai"):
            return True
    return False


def calculate_required_gpu_memory(request_flow: RequestFlow, ai_microservice: Microservice, required_gpu_units: int, verbose: bool = False) -> float:
    """
    基于GPU单元数计算AI微服务所需的GPU显存
    Args:
        request_flow: 请求流对象
        ai_microservice: AI微服务对象
        required_gpu_units: 所需GPU单元数
    Returns:
        float: 总GPU显存需求 (GB)
    """
    # 基础模型显存（每个GPU单元）
    base_model_memory_per_gpu = 16.0  # GB per GPU unit

    # 动态显存需求（基于当前时隙的token数量）
    input_tokens = request_flow.r_input_data_size
    output_tokens = request_flow.r_output_data_size
    context_length = input_tokens + output_tokens

    # KV Cache显存需求（基于到达率和GPU单元数）
    arrival_rate_factor = min(request_flow.arrival_rate / 10.0, 1.5)
    kv_cache_memory_per_gpu = context_length * 0.0002 * arrival_rate_factor

    # 批处理缓冲区（基于到达率和GPU单元数分配）
    batch_buffer_memory_per_gpu = 2.0 + request_flow.arrival_rate * 0.2 / required_gpu_units

    # 每个GPU单元的内存需求
    memory_per_gpu = base_model_memory_per_gpu + kv_cache_memory_per_gpu + batch_buffer_memory_per_gpu
    memory_per_gpu = max(16.0, min(memory_per_gpu, 80.0))

    # 总内存需求
    total_memory = memory_per_gpu * required_gpu_units

    if verbose:
        print(f"        GPU内存需求分析:")
        print(f"          每GPU单元内存: {memory_per_gpu:.1f} GB")
        print(f"          所需GPU单元数: {required_gpu_units}")
        print(f"          总内存需求: {total_memory:.1f} GB")

    return total_memory

def calculate_required_model_storage(ai_microservice: Microservice) -> float:
    """计算AI微服务所需的模型存储空间"""
    # 基础模型大小（如Llama-3-8B约16GB）
    base_model_size = 16.0  # GB
    # 优化模型缓存
    optimization_cache = 4.0  # GB
    return base_model_size + optimization_cache


def calculate_local_ai_inference_latency(request_flow: RequestFlow, ai_microservice: Microservice,
                                         server: EdgeServer, system_state: SystemState,verbose: bool = False) -> tuple[float, float, float, int]:
    """
    使用GPU单元数参与计算的AI微服务本地推理延迟
    核心改进：
    1. 基于服务强度计算所需GPU单元数（类似CPU核心数）
    2. 检查服务器GPU单元充足性
    3. GPU单元数影响并行处理能力
    Args:
        request_flow: 请求流对象
        ai_microservice: AI微服务对象
        server: AI服务器对象
        system_state: 系统状态对象
    Returns:
        tuple: (总延迟, 队列延迟, 处理延迟, 所需GPU单元数) in ms
    """
    # 到达率 (requests/ms)
    arrival_rate = request_flow.arrival_rate

    # ========== 第一步：计算基础处理时间（单GPU单元） ==========
    typical_input_tokens = 1000.0
    typical_output_tokens = 200.0
    actual_input_tokens = request_flow.r_input_data_size
    actual_output_tokens = request_flow.r_output_data_size

    stability_weight = 0.7
    effective_input_tokens = (stability_weight * typical_input_tokens +
                              (1 - stability_weight) * actual_input_tokens)
    effective_output_tokens = (stability_weight * typical_output_tokens +
                               (1 - stability_weight) * actual_output_tokens)

    # 推理速度参数
    prefill_speed = server.prefill_speed_tokens_per_sec
    decode_speed = server.decode_speed_tokens_per_sec

    # 计算单GPU单元的基础处理时间
    effective_prefill_time = effective_input_tokens / prefill_speed
    effective_decode_time = effective_output_tokens / decode_speed
    base_processing_time = (effective_prefill_time + effective_decode_time) * 1000  # ms

    # ========== 第二步：计算所需GPU单元数（类似CPU核心计算） ==========
    # 单GPU单元的基础服务率
    single_gpu_service_rate = 1.0 / base_processing_time  # requests/ms

    # 批处理效率（基于到达率动态调整）
    max_batch_size = server.max_batch_size
    arrival_rate_per_sec = arrival_rate * 1000
    optimal_batch_size = min(max_batch_size, max(1, int(arrival_rate_per_sec * 0.05)))
    batch_efficiency_factor = 0.8
    batch_throughput_multiplier = 1.0 + (optimal_batch_size - 1) * batch_efficiency_factor

    # 考虑批处理后的单GPU单元有效服务率
    single_gpu_effective_service_rate = single_gpu_service_rate * batch_throughput_multiplier

    # ========== 第三步：计算所需GPU单元数以满足服务强度 < 1 ==========
    required_gpu_units = 1  # 最少需要1个GPU单元

    # 如果单个GPU单元无法满足服务强度要求，计算需要的GPU单元数
    if arrival_rate >= single_gpu_effective_service_rate:
        # 计算理论上需要的GPU单元数（向上取整）
        required_gpu_units = int(np.ceil(arrival_rate / single_gpu_effective_service_rate))

        # 限制在服务器最大GPU单元数内
        required_gpu_units = min(required_gpu_units, server.gpu_units)

    # ========== 第四步：计算多GPU并行处理的最终服务率 ==========
    # 多GPU并行效率衰减
    if required_gpu_units > 1:
        parallel_efficiency = 0.95 ** (required_gpu_units - 1)  # 每增加一个GPU，效率衰减5%
    else:
        parallel_efficiency = 1.0

    # 最终总服务率
    total_service_rate = single_gpu_effective_service_rate * required_gpu_units * parallel_efficiency

    # ========== 第五步：排队论模型计算 ==========
    if arrival_rate >= total_service_rate:
        print(
            f"        WARNING: 即使使用{required_gpu_units}个GPU单元仍不稳定 (ρ = {arrival_rate / total_service_rate:.3f} >= 1)")
        return float('inf'), float('inf'), base_processing_time, required_gpu_units

    # 计算服务强度
    rho = arrival_rate / total_service_rate

    # M/M/c排队延迟公式（c = required_gpu_units）
    if required_gpu_units == 1:
        # M/M/1队列
        queue_delay = rho / (total_service_rate * (1 - rho))
    else:
        # M/M/c队列
        queue_delay = rho / (total_service_rate * (1 - rho)) * (1 / required_gpu_units)

    # 总延迟 = 队列延迟 + 处理延迟
    total_latency = queue_delay + base_processing_time

    # 只在verbose=True时输出详细信息
    if verbose:
        print(f"      GPU单元并行计算 - AI推理延迟分析:")
        print(f"        有效tokens: input={effective_input_tokens:.0f}, output={effective_output_tokens:.0f}")
        print(f"        基础处理时间: {base_processing_time:.2f} ms")
        print(f"        最优批处理大小: {optimal_batch_size}/{max_batch_size}")
        print(f"        单GPU有效服务率: {single_gpu_effective_service_rate:.4f} req/ms")
        print(f"        到达率: {arrival_rate:.3f} req/ms")
        print(f"        所需GPU单元数: {required_gpu_units}/{server.gpu_units}")
        print(f"        并行效率: {parallel_efficiency:.3f}")
        print(f"        总服务率: {total_service_rate:.4f} req/ms")
        print(f"        服务强度 ρ: {rho:.3f}")
        print(f"        队列延迟: {queue_delay:.2f} ms")
        print(f"        总延迟: {total_latency:.2f} ms")

    return total_latency, queue_delay, base_processing_time, required_gpu_units


def evaluate_cloud_deployment(request_flow: RequestFlow, ai_microservice: Microservice,
                              system_state: SystemState) -> Dict:
    """
    评估云端卸载部署的性能和成本

    注意：这里计算的是AI服务器到云端的延迟，
    不包括传统服务器到AI服务器的通信延迟（那部分在路径计算中单独处理）
    """
    # 云端基础延迟
    base_latency = request_flow.cloud_latency_base  # ms

    # 网络传输延迟（AI服务器 → 云端 → AI服务器）
    input_data_size = request_flow.r_input_data_size / 1000.0  # token转换为MB估算
    output_data_size = request_flow.r_output_data_size / 1000.0
    total_data_size = input_data_size + output_data_size

    # 双向传输延迟 = (上传 + 下载) / 带宽
    bandwidth_mbps = request_flow.cloud_bandwidth
    transmission_latency = (total_data_size * 8 * 2) / bandwidth_mbps  # *2 for round trip

    # 云端处理延迟
    input_tokens = request_flow.r_input_data_size
    output_tokens = request_flow.r_output_data_size

    # 云端使用更强的GPU，处理速度更快
    cloud_prefill_speed = 3000000.0     # tokens/second
    cloud_decode_speed = 120000.0  # tokens/second

    cloud_processing_time = ((input_tokens / cloud_prefill_speed) +
                             (output_tokens / cloud_decode_speed)) * 1000  # ms

    # 云端资源调度延迟（冷启动等）
    scheduling_latency = random.uniform(4.0, 5.0)  # 4-6ms

    # 总云端延迟（AI服务器发起的卸载延迟）
    total_cloud_latency = base_latency + transmission_latency + cloud_processing_time + scheduling_latency
    '''
    print(f"      云端卸载延迟分析（AI服务器视角）:")
    print(f"        基础延迟: {base_latency:.2f} ms")
    print(f"        网络传输延迟（双向）: {transmission_latency:.2f} ms")
    print(f"        云端处理延迟: {cloud_processing_time:.2f} ms")
    print(f"        AI服务器卸载总延迟: {total_cloud_latency:.2f} ms")
    print(f"        注意：不包括传统服务器→AI服务器的通信延迟")
    '''
    return {
        'total_latency': total_cloud_latency,
        'base_latency': base_latency,
        'transmission_latency': transmission_latency,
        'processing_latency': cloud_processing_time,
        'feasible': True
    }





