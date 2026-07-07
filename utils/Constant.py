import sys

import numpy as np
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import random
from collections import defaultdict, deque
from fractions import Fraction

import Waitingtime


# ================================== 系统模型 ==================================

class ServerType(Enum):
    """服务器类型枚举"""
    TRADITIONAL = "traditional"  # 传统计算节点
    AI_CAPABLE = "ai_capable"  # AI计算节点


class VirtualEnergyQueue:
    """
    虚拟能耗队列模型（仅用于AI服务器）
    对应论文公式36: Y_s(t+1) = max[Y_s(t) + ν(E^t_s - E^max_s), 0]
    """

    def __init__(self, server_id: str, energy_threshold: float, scaling_factor: float):
        self.server_id = server_id
        self.queue_state = 0.0  # Y_s(t): 当前队列状态 (mJ)
        self.energy_threshold = energy_threshold  # E^max_s: 能耗阈值 (J per time slot)
        self.scaling_factor = scaling_factor  # ν: 缩放因子
        self.history = deque(maxlen=10000)  # 历史记录

        # 添加统计信息
        self.violation_count = 0  # 能耗超阈值次数
        self.max_queue_state = 0.0  # 历史最大队列状态

    def update_queue(self, current_energy: float) -> float:
        """
        更新虚拟队列状态
        Y[i_idx,:] = np.maximum(Y[i_idx-1,:] + (energy[i_idx-1,:]- energy_thresh)*nu,0)
        Args:
            current_energy: 当前时刻的能耗 (J per time slot)
        Returns:
            float: 更新后的队列状态
        """
        # 计算队列增量：ν(E^t_s - E^max_s)
        queue_increment = self.scaling_factor * (current_energy - self.energy_threshold)

        # 更新队列状态：max[Y_s(t) + increment, 0]
        # 使用numpy.maximum确保非负性
        self.queue_state = np.maximum(self.queue_state + queue_increment, 0.0)

        # 更新统计信息
        if current_energy > self.energy_threshold:
            self.violation_count += 1

        self.max_queue_state = max(self.max_queue_state, self.queue_state)

        # 记录历史
        self.history.append({
            'timestamp': len(self.history),
            'queue_state': self.queue_state,
            'energy': current_energy,
            'threshold': self.energy_threshold,
            'increment': queue_increment
        })

        return self.queue_state

    def get_queue_state_normalized(self) -> float:
        """
        获取归一化的队列状态
        参考LyDROO中的归一化技巧：Y[i_idx,:]
        """
        return self.queue_state

    def is_queue_stable(self, window_size: int = 100) -> bool:
        """
        检查队列是否稳定（近期平均增长率接近0）
        """
        if len(self.history) < window_size:
            return True

        recent_states = [record['queue_state'] for record in list(self.history)[-window_size:]]
        return (recent_states[-1] - recent_states[0]) / window_size < 0.01

    def get_violation_rate(self) -> float:
        """获取能耗超阈值的比率"""
        if len(self.history) == 0:
            return 0.0
        return self.violation_count / len(self.history)

    def reset_queue(self):
        """重置队列状态"""
        self.queue_state = 0.0
        self.violation_count = 0
        self.max_queue_state = 0.0
        self.history.clear()

class VirtualDelayQueue:
    """
    虚拟延迟队列模型（仅用于AI服务器）
    对应公式: Z_s(t+1) = max[Z_s(t) + μ(D^t_s - D^max_s), 0]
    """

    def __init__(self, server_id: str, delay_threshold: float, scaling_factor: float):
        self.server_id = server_id
        self.queue_state = 0.0  # Z_s(t): 当前延迟队列状态 (ms)
        self.delay_threshold = delay_threshold  # D^max_s: 延迟阈值 (ms)
        self.scaling_factor = scaling_factor  # μ: 缩放因子
        self.history = deque(maxlen=10000)  # 历史记录

        # 添加统计信息
        self.violation_count = 0  # 延迟超阈值次数
        self.max_queue_state = 0.0  # 历史最大队列状态

    def update_queue(self, current_delay: float) -> float:
        """
        更新虚拟延迟队列状态
        Z[i_idx,:] = np.maximum(Z[i_idx-1,:] + (delay[i_idx-1,:]- delay_thresh)*mu,0)
        Args:
            current_delay: 当前时刻的延迟 (ms)
        Returns:
            float: 更新后的队列状态
        """
        # 计算队列增量：μ(D^t_s - D^max_s)
        queue_increment = self.scaling_factor * (current_delay - self.delay_threshold)

        # 更新队列状态：max[Z_s(t) + increment, 0]
        self.queue_state = np.maximum(self.queue_state + queue_increment, 0.0)

        # 更新统计信息
        if current_delay > self.delay_threshold:
            self.violation_count += 1

        self.max_queue_state = max(self.max_queue_state, self.queue_state)

        # 记录历史
        self.history.append({
            'timestamp': len(self.history),
            'queue_state': self.queue_state,
            'delay': current_delay,
            'threshold': self.delay_threshold,
            'increment': queue_increment
        })

        return self.queue_state

    def get_queue_state_normalized(self) -> float:
        """
        获取归一化的延迟队列状态
        """
        return self.queue_state

    def is_queue_stable(self, window_size: int = 100) -> bool:
        """
        检查延迟队列是否稳定（近期平均增长率接近0）
        """
        if len(self.history) < window_size:
            return True

        recent_states = [record['queue_state'] for record in list(self.history)[-window_size:]]
        return (recent_states[-1] - recent_states[0]) / window_size < 0.01

    def get_violation_rate(self) -> float:
        """获取延迟超阈值的比率"""
        if len(self.history) == 0:
            return 0.0
        return self.violation_count / len(self.history)

    def reset_queue(self):
        """重置延迟队列状态"""
        self.queue_state = 0.0
        self.violation_count = 0
        self.max_queue_state = 0.0
        self.history.clear()

@dataclass
class EdgeServer:
    """
    边缘服务器模型
    参考论文中的V, v表示边缘服务器集合
    Cv表示服务器v的CPU核心数量
    """
    server_id: str  # v ∈ V: 服务器标识符
    server_type: ServerType

    # 计算资源 (对应论文中的Cv)
    cpu_cores: int  # Cv: CPU核心数
    memory_capacity: float  # 内存容量(GB)
    available_cpu: int = field(init=False)  # 可用CPU核心数
    available_memory: float = field(init=False)  # 可用内存容量

    # AI计算资源
    gpu_units: int = 0  # GPU计算单元数
    gpu_memory: float = 0.0  # GPU显存容量(GB)
    model_storage: float = 0.0  # 模型存储容量(GB)
    available_gpu_units: int = field(init=False)  # 可用GPU单元数

    # 能耗模型参数
    base_power: float = 0.0  # Wb: 基础功耗 (论文提到的85 MJ)
    cpu_power_coeff: float = 0.0  # CPU功耗系数
    gpu_power_coeff: float = 0.0  # GPU功耗系数

    # 能耗阈值设定
    energy_threshold: float = 0.08  # 能耗阈值 (J per time slot)
    energy_queue_factor: float = 1000.0  # ν: 能耗队列缩放因子
    # 延迟阈值设定
    delay_threshold: float = 50.0  # 延迟阈值 (ms)
    delay_queue_factor: float = 100.0  # μ: 延迟队列缩放因子

    # GPU状态管理
    _gpu_active: bool = field(default=False, init=False)

    # AI推理相关属性
    model_name: str = ""  # 模型名称，如"llama-3-8b"
    max_batch_size: int = 64  # 最大批处理大小
    prefill_speed_tokens_per_sec: float = 0.0  # 预填充速度
    decode_speed_tokens_per_sec: float = 0.0  # 解码速度

    # 当前批处理状态
    current_batch_requests: List = field(default_factory=list)
    batch_processing_start_time: float = 0.0

    def __post_init__(self):
        """初始化后处理"""
        self.available_cpu = self.cpu_cores
        self.available_memory = self.memory_capacity

        # GPU相关属性初始化
        self.available_gpu_units = self.gpu_units
        self.available_gpu_memory = self.gpu_memory
        self.available_model_storage = self.model_storage

        # GPU状态初始化
        self._gpu_active = False

        # AI推理相关属性初始化
        if not hasattr(self, 'current_batch_requests'):
            self.current_batch_requests = []
        if not hasattr(self, 'batch_processing_start_time'):
            self.batch_processing_start_time = 0.0


    def set_gpu_active(self, active: bool):
        """设置GPU激活状态"""
        if self.server_type == ServerType.AI_CAPABLE:
            self._gpu_active = active
        else:
            self._gpu_active = False

    def activate_gpu_for_inference(self):
        """当有AI推理任务时激活GPU """
        if self.server_type == ServerType.AI_CAPABLE and self.gpu_units > 0:
            self._gpu_active = True

    def deactivate_gpu(self):
        """停用GPU """
        self._gpu_active = False

    # 在EdgeServer类中添加以下方法
    def get_cpu_utilization(self) -> float:
        return (self.cpu_cores - self.available_cpu) / max(self.cpu_cores, 1)

    def get_memory_utilization(self) -> float:
        return (self.memory_capacity - self.available_memory) / max(self.memory_capacity, 1.0)

    def get_gpu_unit_utilization(self) -> float:
        if self.server_type != ServerType.AI_CAPABLE:
            return 0.0
        return (self.gpu_units - self.available_gpu_units) / max(self.gpu_units, 1)

    def get_gpu_memory_utilization(self) -> float:
        if self.server_type != ServerType.AI_CAPABLE:
            return 0.0
        return (self.gpu_memory - self.available_gpu_memory) / max(self.gpu_memory, 1.0)

    def get_model_storage_utilization(self) -> float:
        """计算模型存储利用率"""
        if self.server_type != ServerType.AI_CAPABLE:
            return 0.0
        return (self.model_storage - self.available_model_storage) / max(self.model_storage, 1.0)

    def get_comprehensive_gpu_utilization(self) -> float:
        """获取综合GPU利用率（用于能耗计算）"""
        if self.server_type != ServerType.AI_CAPABLE:
            return 0.0
        gpu_unit_util = self.get_gpu_unit_utilization()
        gpu_memory_util = self.get_gpu_memory_utilization()
        return max(gpu_unit_util, gpu_memory_util)

    def get_comprehensive_resource_utilization(self) -> np.ndarray:
        """
        获取全面的资源利用率向量
        包含：CPU、内存、GPU单元、GPU内存、模型存储利用率
        """
        utilization_vector = [
            min(self.get_cpu_utilization(), 1.0),
            min(self.get_memory_utilization(), 1.0)
        ]

        if self.server_type == ServerType.AI_CAPABLE:
            utilization_vector.extend([
                min(self.get_gpu_unit_utilization(), 1.0),
                min(self.get_gpu_memory_utilization(), 1.0),
                min(self.get_model_storage_utilization(), 1.0)
            ])
        else:
            # 传统服务器：GPU相关资源利用率为0
            utilization_vector.extend([0.0, 0.0, 0.0])

        return np.array(utilization_vector, dtype=np.float32)

@dataclass
class Link:
    """
    网络链路模型
    表示两个边缘服务器之间的网络连接
    """
    link_id: str  # 链路标识符
    source_server_id: str  # 源服务器ID
    target_server_id: str  # 目标服务器ID
    bandwidth: float  # 总带宽容量 (Mbps)
    transmission_delay: float  # 传输延时 (ms)
    available_bandwidth: float = field(init=False)  # 可用带宽 (Mbps)


# ================================== 标准化传输延迟矩阵 ==================================
class NetworkTopology:
    """
    网络拓扑模型 (SE: normalized transmission delay matrix)
    """
    def __init__(self, server_ids: List[str]):
        self.server_ids = server_ids
        self.server_count = len(server_ids)
        self.server_id_to_index = {sid: idx for idx, sid in enumerate(server_ids)}

        # e(vi, vj): 原始通信延迟矩阵
        self.raw_delay_matrix = np.zeros((self.server_count, self.server_count))

        # SE: 标准化传输延迟矩阵
        self.normalized_delay_matrix = np.zeros((self.server_count, self.server_count))

    def set_communication_delay(self, source_id: str, target_id: str, delay: float):
        """设置服务器间的通信延迟 e(vi, vj)"""
        if source_id not in self.server_id_to_index or target_id not in self.server_id_to_index:
            raise ValueError(f"Server ID not found: {source_id} or {target_id}")

        source_idx = self.server_id_to_index[source_id]
        target_idx = self.server_id_to_index[target_id]
        self.raw_delay_matrix[source_idx, target_idx] = delay

    def update_normalized_delay_matrix(self):
        """
        更新标准化传输延迟矩阵
        使用最大值归一化方法
        """
        max_delay = np.max(self.raw_delay_matrix)
        if max_delay > 0:
            self.normalized_delay_matrix = self.raw_delay_matrix / max_delay
        else:
            self.normalized_delay_matrix = self.raw_delay_matrix.copy()

    def get_communication_delay(self, source_id: str, target_id: str) -> float:
        """获取服务器间的通信延迟"""
        if source_id not in self.server_id_to_index or target_id not in self.server_id_to_index:
            return 0.0  # 如果服务器不存在，返回0延迟

        source_idx = self.server_id_to_index[source_id]
        target_idx = self.server_id_to_index[target_id]
        return self.raw_delay_matrix[source_idx, target_idx]

    def get_normalized_delay_matrix(self) -> np.ndarray:
        """获取标准化传输延迟矩阵 (论文中的SE)"""
        return self.normalized_delay_matrix.copy()

@dataclass
class Microservice:
    """
    微服务模型 (对应论文中的M, m表示微服务集合)
    """
    ms_id: str  # m ∈ M: 微服务标识符
    service_type: str  # "traditional" 或 "ai"

    # 传统微服务资源需求
    cpu_requirement: int = 0.0  # CPU需求 (cores)
    memory_requirement: float = 0.0  # 内存需求 (GB)
    service_rate: float = 0.0  # μ: 单核服务率 (requests/ms/core)

    # AI微服务资源需求
    model_params: float = 0.0  # 模型参数量(B)，影响推理延迟
    context_window: int = 4096  # 上下文窗口，影响显存需求
    min_batch_size: int = 1  # 最小批处理大小
    max_batch_size: int = 32  # 最大批处理大小

    # 数据传输量
    input_data_size: float = 0.0  # 输入数据大小
    output_data_size: float = 0.0  # 输出数据大小

    def get_min_cores_required(self) -> int:
        """获取最小所需CPU核心数"""
        return max(1, int(np.ceil(self.cpu_requirement)))


@dataclass
class MicroserviceInstance:
    """
    微服务实例模型
    表示部署在特定边缘服务器上的微服务实例
    """
    instance_id: str
    microservice: Microservice
    server_id: str  # AI微服务始终部署在AI服务器上
    allocated_cores: int

    # 队列论模型参数
    arrival_rate: float = 0.0
    service_intensity: float = 0.0
    queue_length: float = 0.0
    queue_delay: float = 0.0
    processing_delay: float = 0.0

    # AI微服务处理模式
    processing_mode: str = "unknown"  # "local_processing" 或 "cloud_offloaded" 或 "pending_decision"

    # AI微服务资源分配
    gpu_memory_allocated: float = 0.0
    model_storage_allocated: float = 0.0

    # AI微服务延迟
    inference_latency: float = 0.0  # 本地推理延迟
    cloud_latency: float = 0.0  # 云端处理延迟

    def update_service_metrics(self, processing_power_per_core_rps: float):
        """
        更新服务强度和延迟指标
        """
        Waitingtime.update_service_metrics(self, processing_power_per_core_rps)


class ServiceChain:
    """
    微服务链模型
    参考论文中的Msc = {m^1_sc, ..., m^L(sc)_sc}
    """

    def __init__(self, chain_id: str, microservices: List[Microservice]):
        self.chain_id = chain_id  # sc ∈ SC: 服务链标识符
        self.microservices = microservices  # Msc: 服务链包含的微服务列表
        self.chain_length = len(microservices)  # L(sc): 服务链长度
        # 直接设置AI微服务为链末尾的微服务（因为一定是AI微服务）
        self.ai_microservice = microservices[-1] if microservices else None
        # 存储每个微服务需要的核心分配数
        self.core_allocations: Dict[str, int] = {}  # {microservice_id: required_cores}
        # 存储每个微服务需要的内存分配数
        self.memory_allocations: Dict[str, float] = {}  # {microservice_id: required_memory_gb}

    def get_traditional_microservices(self) -> List[Microservice]:
        """获取传统微服务列表（除最后一个AI微服务外的所有微服务）"""
        return self.microservices[:-1] if self.microservices else []

    def get_traditional_microservice_chain(self) -> 'ServiceChain':
        """
        获取传统微服务链
        返回一个新的ServiceChain对象，只包含传统微服务（不包含AI微服务）
        Returns:
            ServiceChain: 只包含传统微服务的新服务链对象
        """
        traditional_microservices = self.get_traditional_microservices()
        traditional_chain_id = f"{self.chain_id}_traditional"
        new_chain = ServiceChain(traditional_chain_id, traditional_microservices)

        # 复制资源分配数据
        for ms in traditional_microservices:
            if ms.ms_id in self.core_allocations:
                new_chain.core_allocations[ms.ms_id] = self.core_allocations[ms.ms_id]
            if ms.ms_id in self.memory_allocations:
                new_chain.memory_allocations[ms.ms_id] = self.memory_allocations[ms.ms_id]

        return new_chain

    def get_microservice_sequence(self) -> List[str]:
        """获取微服务ID序列 (支持通信延迟计算)"""
        return [ms.ms_id for ms in self.microservices]

    def set_core_allocation(self, microservice_id: str, required_cores: int):
        """
        设置微服务的核心分配数
        Args:
            microservice_id: 微服务ID
            required_cores: 所需核心数
        """
        self.core_allocations[microservice_id] = required_cores

    def set_memory_allocation(self, microservice_id: str, required_memory: float):
        """
        设置微服务的内存分配数
        Args:
            microservice_id: 微服务ID
            required_memory: 所需内存数量 (GB)
        """
        self.memory_allocations[microservice_id] = required_memory


@dataclass
class RequestFlow:
    """
    请求流模型
    """
    flow_id: str
    service_chain: ServiceChain
    arrival_rate: float  # λ^i_v: 当前时隙请求到达率 (Poisson分布)
    max_latency: float  # 最大容忍延迟阈值
    ca_latency:float #计算得出时延(除去ai处理时延)
    priority: int  # 优先级

    # 网络资源需求
    bandwidth_requirement: float = 0.0  # 带宽需求 (Mbps)

    # 传统计算资源需求
    data_memory_requirement: float = 0.0  # 数据内存需求 (GB)

    # AI计算资源需求
    r_input_data_size: float = 0.0  # 输入token数量
    r_output_data_size: float = 0.0  # 输出token数量

    # AI微服务决策
    ai_processing_choice: str = "local"  # "local" 或 "cloud"

    # 路由概率
    routing_probabilities: Dict[Tuple[str, str], float] = field(default_factory=dict)

    # 云端参数
    cloud_latency_base: float = 50.0  # 云端基础延迟(ms)
    cloud_bandwidth: float = 100.0  # 云端带宽(Mbps)

    def add_routing_probability(self, source_server: str, target_server: str, probability: float):
        """添加路由概率"""
        self.routing_probabilities[(source_server, target_server)] = probability

    def validate_routing_probabilities(self) -> bool:
        """验证路由概率总和是否为1 """
        # 按源服务器分组检查
        source_groups = defaultdict(float)
        for (source, target), prob in self.routing_probabilities.items():
            source_groups[source] += prob

        return all(abs(total - 1.0) < 1e-6 for total in source_groups.values())


# ================================== 系统状态表示 ==================================
class SystemState:
    """
    系统状态模型
    State = {SH, SQ}  # 包含AI服务器性能环境因子和虚拟能耗队列状态
    SH: AI服务器性能环境因子 (类比LyDROO中的信道状态h)
    SQ: 虚拟能耗队列状态
    """
    def __init__(self, time_frame: int):
        self.time_frame = time_frame  # 当前时间帧 t

        # 物理基础设施
        self.edge_servers: Dict[str, EdgeServer] = {}
        self.network_topology: Optional[NetworkTopology] = None
        self.links: Dict[str, Link] = {}  # 存储网络链路

        # 微服务部署状态
        self.microservices: Dict[str, Microservice] = {}
        self.microservice_instances: Dict[str, MicroserviceInstance] = {}
        self.service_chains: Dict[str, ServiceChain] = {}

        # 请求流状态
        self.request_flows: Dict[str, RequestFlow] = {}
        self.current_arrivals: Dict[str, float] = {}  # 当前时刻的到达率

        # 虚拟队列状态
        self.virtual_energy_queues: Dict[str, VirtualEnergyQueue] = {}
        self.virtual_delay_queues: Dict[str, VirtualDelayQueue] = {}

        # 系统性能指标
        self.total_energy_consumption = 0.0
        self.total_latency = 0.0
        self.queue_stability_violations = 0

        # 差异度矩阵
        self.stream_distance_matrix: Optional[np.ndarray] = None

        # ==========路由概率矩阵 ==========
        # 请求流路由转移概率矩阵
        # 格式: {flow_id: {(origin_host, ms_origin, dest_host, ms_dest): probability}}
        # 对应Java中的 ArrayList<Map<ArrayList<Integer>, Fraction>> A_Stream_Transfer_PR
        self.stream_transfer_probabilities: Dict[str, Dict[Tuple[str, str, str, str], Fraction]] = {}

        # 请求流在各服务器上的资源分配矩阵
        # 格式: {flow_id: {ms_id: {server_id: allocated_resource}}}
        # 对应Java中的 int[][][] A_Stream_Allocated_Resource
        self.stream_allocated_resources: Dict[str, Dict[str, Dict[str, int]]] = {}

        # 环境状态管理器（用于计算AI服务器性能因子）
        self.environment_manager = None  # 在create_sample_system中初始化

    def add_link(self, link: Link):
        """添加网络链路"""
        self.links[link.link_id] = link

    def add_edge_server(self, server: EdgeServer):
        """添加边缘服务器，只为AI服务器初始化虚拟队列"""
        self.edge_servers[server.server_id] = server

        # 只为AI服务器初始化虚拟队列
        if server.server_type.value == "ai_capable":
            self.virtual_energy_queues[server.server_id] = VirtualEnergyQueue(
                server.server_id,
                server.energy_threshold,
                server.energy_queue_factor
            )
            self.virtual_delay_queues[server.server_id] = VirtualDelayQueue(
                server.server_id,
                server.delay_threshold,
                server.delay_queue_factor
            )
            print(f"为AI服务器 {server.server_id} 创建虚拟能耗队列和虚拟延迟队列")



        # ========== 路由概率矩阵相关方法 ==========
    def set_stream_allocated_resource(self, flow_id: str, ms_id: str, server_id: str, allocated_resource: int):
        """
        设置请求流在特定服务器上的微服务资源分配
        Args:
            flow_id: 请求流ID
            ms_id: 微服务ID
            server_id: 服务器ID
            allocated_resource: 分配的资源数量(核心数)
        """
        if flow_id not in self.stream_allocated_resources:
            self.stream_allocated_resources[flow_id] = {}
        if ms_id not in self.stream_allocated_resources[flow_id]:
            self.stream_allocated_resources[flow_id][ms_id] = {}

        self.stream_allocated_resources[flow_id][ms_id][server_id] = allocated_resource
        #print(f"设置资源分配: {flow_id}.{ms_id}@{server_id} = {allocated_resource} 核心")

    def get_stream_allocated_resource(self, flow_id: str, ms_id: str, server_id: str) -> int:
        """
        获取请求流在特定服务器上的微服务资源分配
        Returns:
            int: 分配的资源数量，如果未分配则返回0
        """
        return (self.stream_allocated_resources
                    .get(flow_id, {})
                    .get(ms_id, {})
                    .get(server_id, 0))

    def get_microservice_total_allocated_resource(self, flow_id: str, ms_id: str) -> int:
        """
        获取某个微服务在所有服务器上的总分配资源
        Args:
            flow_id: 请求流ID
            ms_id: 微服务ID
        Returns:
            int: 总分配资源数量
        """
        if flow_id not in self.stream_allocated_resources:
            return 0
        if ms_id not in self.stream_allocated_resources[flow_id]:
            return 0

        return sum(self.stream_allocated_resources[flow_id][ms_id].values())

    def calculate_stream_transfer_probabilities(self):
        """考虑微服务的全局资源分配"""
        #print("\n=== 开始计算请求流路由转移概率（全局资源版本）===")
        self.stream_transfer_probabilities.clear()

        for flow_id, request_flow in self.request_flows.items():
            #print(f"\n处理请求流: {flow_id}")

            service_chain = request_flow.service_chain
            traditional_microservices = service_chain.get_traditional_microservices()

            if len(traditional_microservices) <= 1:
                continue

            single_stream_transfer_pr: Dict[Tuple[str, str, str, str], Fraction] = {}

            for j in range(1, len(traditional_microservices)):
                ms_origin_id = traditional_microservices[j - 1].ms_id
                ms_dest_id = traditional_microservices[j].ms_id

                #print(f"  计算转移: {ms_origin_id} → {ms_dest_id}")

                # 获取起点微服务在当前流中的总资源（用于计算起点概率分布）
                origin_flow_resource = sum(
                    self.stream_allocated_resources.get(flow_id, {}).get(ms_origin_id, {}).values())

                if origin_flow_resource == 0:
                    continue

                # 遍历起点微服务在当前流中的部署
                origin_servers = self.stream_allocated_resources.get(flow_id, {}).get(ms_origin_id, {})
                for origin_host, origin_resource in origin_servers.items():
                    if origin_resource <= 0:
                        continue

                    # 计算从该服务器出发的到达率比例
                    origin_arrival_rate = Fraction(origin_resource, origin_flow_resource)

                    # *** 获取目标微服务在整个系统中的总资源 ***
                    dest_global_resource = self.get_microservice_global_allocated_resource(ms_dest_id)

                    if dest_global_resource == 0:
                        continue

                    # 遍历目标微服务在整个系统中的所有部署
                    for dest_flow_id, dest_flow_resources in self.stream_allocated_resources.items():
                        if ms_dest_id not in dest_flow_resources:
                            continue

                        for dest_host, dest_resource in dest_flow_resources[ms_dest_id].items():
                            if dest_resource <= 0:
                                continue

                            # 计算转移概率：起点到达率 × (目标服务器资源 / 目标微服务全局总资源)
                            transfer_probability = origin_arrival_rate * Fraction(dest_resource, dest_global_resource)

                            # 在转移概率设置时改为累加
                            transfer_key = (origin_host, ms_origin_id, dest_host, ms_dest_id)
                            if transfer_key in single_stream_transfer_pr:
                                single_stream_transfer_pr[transfer_key] += transfer_probability
                            else:
                                single_stream_transfer_pr[transfer_key] = transfer_probability
                            '''
                            print(f"    {origin_host}:{ms_origin_id} → {dest_host}:{ms_dest_id}")
                            print(f"      = {float(origin_arrival_rate):.3f} × {dest_resource}/{dest_global_resource}")
                            print(f"      = {float(transfer_probability):.6f}")
                            '''
            self.stream_transfer_probabilities[flow_id] = single_stream_transfer_pr
            #print(f"  请求流{flow_id}转移概率计算完成，共{len(single_stream_transfer_pr)}个相邻转发路径")

    def get_microservice_global_allocated_resource(self, ms_id: str) -> int:
        """获取微服务在整个系统中的总资源分配"""
        total_resource = 0
        for flow_id, flow_resources in self.stream_allocated_resources.items():
            if ms_id in flow_resources:
                total_resource += sum(flow_resources[ms_id].values())
        return total_resource

    def get_transfer_probability(self, flow_id: str, origin_host: str, ms_origin: str,
                                     dest_host: str, ms_dest: str) -> float:
            """
            获取特定路径的转移概率
        Args:
            flow_id: 请求流ID
            origin_host: 起点服务器ID
            ms_origin: 起点微服务ID
            dest_host: 目标服务器ID
            ms_dest: 目标微服务ID
        Returns:
            float: 转移概率值，如果路径不存在返回0.0
        """
            transfer_key = (origin_host, ms_origin, dest_host, ms_dest)
            if flow_id in self.stream_transfer_probabilities:
                fraction_prob = self.stream_transfer_probabilities[flow_id].get(transfer_key, Fraction(0))
                return float(fraction_prob)
            return 0.0

    def get_flow_transfer_probabilities(self, flow_id: str) -> Dict[Tuple[str, str, str, str], float]:
            """
            获取指定请求流的所有转移概率
            Args:
                flow_id: 请求流ID
            Returns:
                Dict: 转移路径到概率的映射
            """
            if flow_id not in self.stream_transfer_probabilities:
                return {}

            result = {}
            for transfer_key, fraction_prob in self.stream_transfer_probabilities[flow_id].items():
                result[transfer_key] = float(fraction_prob)

            return result

    def print_transfer_probabilities(self, flow_id: Optional[str] = None, detailed: bool = True):
            """
            打印转移概率矩阵
            Args:
                flow_id: 可选，指定请求流ID。如果为None则打印所有请求流
                detailed: 是否显示详细信息
            """
            if flow_id:
                flows_to_print = [flow_id] if flow_id in self.stream_transfer_probabilities else []
            else:
                flows_to_print = list(self.stream_transfer_probabilities.keys())

            print(f"\n{'=' * 50}")
            print(f"请求流路由转移概率矩阵")
            print(f"{'=' * 50}")

            for fid in flows_to_print:
                print(f"\n--- 请求流 {fid} ---")
                transfer_probs = self.stream_transfer_probabilities[fid]

                if not transfer_probs:
                    print("  无转移概率数据")
                    continue

                # 按起点分组显示
                if detailed:
                    grouped_by_origin = defaultdict(list)
                    for (origin_host, ms_origin, dest_host, ms_dest), prob in transfer_probs.items():
                        origin_key = f"{origin_host}:{ms_origin}"
                        grouped_by_origin[origin_key].append((dest_host, ms_dest, float(prob)))

                    for origin_key, destinations in grouped_by_origin.items():
                        print(f"  从 {origin_key}:")
                        total_prob = sum(prob for _, _, prob in destinations)
                        for dest_host, ms_dest, prob in destinations:
                            print(f"    → {dest_host}:{ms_dest} = {prob:.6f}")
                        print(f"    总概率: {total_prob:.6f}")
                else:
                    for (origin_host, ms_origin, dest_host, ms_dest), prob in transfer_probs.items():
                        print(f"  {origin_host}:{ms_origin} → {dest_host}:{ms_dest} = {float(prob):.6f}")

    def update_routing_from_deployment(self):
            """
            从当前的微服务部署状态更新路由概率矩阵
            这个方法会根据microservice_instances中的部署信息更新stream_allocated_resources
            然后重新计算转移概率
            """
            print("\n=== 从部署状态更新路由概率矩阵 ===")

            # 清空现有的资源分配记录
            self.stream_allocated_resources.clear()

            # 从微服务实例中重建资源分配信息
            for instance_id, instance in self.microservice_instances.items():
                # 从实例ID中解析出请求流ID（格式为 "flow_id_ms_id_server_id"）
                parts = instance_id.split('_')
                if len(parts) >= 3:
                    flow_id = parts[0] + '_' + parts[1]  # 重建flow_id
                    ms_id = instance.microservice.ms_id
                    server_id = instance.server_id
                    allocated_cores = instance.allocated_cores

                    # 设置资源分配
                    self.set_stream_allocated_resource(flow_id, ms_id, server_id, allocated_cores)

            # 重新计算转移概率
            self.calculate_stream_transfer_probabilities()

            print(f"路由概率矩阵已根据当前部署状态更新")

    def validate_transfer_probabilities(self) -> Dict[str, List[str]]:
        """
        验证转移概率的合法性（修正版本）
        检查每个流中每个微服务转移的所有起点服务器出度概率之和是否为1
        """
        validation_errors = {}

        for flow_id, transfer_probs in self.stream_transfer_probabilities.items():
            errors = []

            # 按微服务转移分组：(ms_origin, ms_dest) -> [(origin_host, prob), ...]
            microservice_transitions = defaultdict(list)
            for (origin_host, ms_origin, dest_host, ms_dest), prob in transfer_probs.items():
                transition_key = (ms_origin, ms_dest)
                microservice_transitions[transition_key].append((origin_host, prob))

            # 检查每个微服务转移的概率和
            for (ms_origin, ms_dest), origin_probs in microservice_transitions.items():
                # 按起点服务器分组，计算每个服务器的出度概率
                server_outgoing_probs = defaultdict(Fraction)
                for origin_host, prob in origin_probs:
                    server_outgoing_probs[origin_host] += prob

                # 计算该微服务转移的总概率（所有起点服务器的概率之和）
                total_transition_prob = sum(server_outgoing_probs.values())

                # 验证总概率是否为1.0
                if abs(float(total_transition_prob) - 1.0) > 1e-6:
                    errors.append(
                        f"微服务转移 {ms_origin}→{ms_dest} 的总出度概率为{float(total_transition_prob):.6f}，应该为1.0"
                    )

                    # 详细显示每个服务器的贡献
                    detail_info = []
                    for server, prob in server_outgoing_probs.items():
                        detail_info.append(f"{server}:{ms_origin}={float(prob):.3f}")
                    errors.append(f"  详情: {' + '.join(detail_info)} = {float(total_transition_prob):.6f}")

            if errors:
                validation_errors[flow_id] = errors

        return validation_errors

    def get_routing_summary(self) -> Dict[str, Dict[str, any]]:
            """
            获取路由概率矩阵的统计摘要
            Returns:
                Dict: 包含各种统计信息的字典
            """
            summary = {}

            for flow_id, transfer_probs in self.stream_transfer_probabilities.items():
                # 统计路径数量
                total_paths = len(transfer_probs)

                # 统计涉及的服务器数量
                servers_involved = set()
                microservices_involved = set()

                for (origin_host, ms_origin, dest_host, ms_dest), prob in transfer_probs.items():
                    servers_involved.add(origin_host)
                    servers_involved.add(dest_host)
                    microservices_involved.add(ms_origin)
                    microservices_involved.add(ms_dest)

                # 概率分布统计
                prob_values = [float(prob) for prob in transfer_probs.values()]

                summary[flow_id] = {
                    'total_paths': total_paths,
                    'servers_count': len(servers_involved),
                    'microservices_count': len(microservices_involved),
                    'avg_probability': np.mean(prob_values) if prob_values else 0,
                    'min_probability': min(prob_values) if prob_values else 0,
                    'max_probability': max(prob_values) if prob_values else 0,
                    'servers_involved': list(servers_involved),
                    'microservices_involved': list(microservices_involved)
                }

            return summary

    def get_state_representation(self) -> np.ndarray:
        """
        获取系统状态表示
        将状态组件展开为一维向量用于神经网络输入
        包含SH和SQ两个组件，类比LyDROO中的状态表示
        """
        # 确保有服务器
        if not self.edge_servers:
            raise ValueError("No edge servers defined")

        # 确保有环境管理器
        if not hasattr(self, 'environment_manager') or not self.environment_manager:
            raise ValueError("Environment manager not initialized")

        return self.environment_manager.get_concatenated_state_vector()

    def get_state_dimension(self) -> int:
        """
        获取状态向量的维度
        """
        if not hasattr(self, 'environment_manager') or not self.environment_manager:
            raise ValueError("Environment manager not initialized")

        dummy_state = self.get_state_representation()
        return len(dummy_state)

    def update_all_energy_queues(self):
        """
        只更新AI服务器的虚拟能耗队列
        参考LyDROO的批量更新方式
        """
        import EnergyConsumption

        ai_energy_consumptions = []
        ai_queue_states = []

        # 只更新AI服务器的能耗队列
        for server_id, server in self.edge_servers.items():
            if server.server_type.value == "ai_capable":
                # 计算当前能耗
                current_energy = EnergyConsumption.calculate_current_energy_consumption(server)
                # 更新对应的虚拟队列
                queue_state = self.virtual_energy_queues[server_id].update_queue(current_energy)

                ai_energy_consumptions.append(current_energy)
                ai_queue_states.append(queue_state)

    def update_all_delay_queues(self):
        """
        只更新AI服务器的虚拟延迟队列
        """
        import Waitingtime

        for server_id, server in self.edge_servers.items():
            if server.server_type.value == "ai_capable":
                # 计算当前AI服务器的延迟
                current_delay = self.calculate_ai_server_current_delay(server_id)
                # 更新对应的虚拟延迟队列
                queue_state = self.virtual_delay_queues[server_id].update_queue(current_delay)

    def calculate_ai_server_current_delay(self, server_id: str) -> float:
        """
        计算AI服务器当前的延迟
        查找部署在该AI服务器上的AI微服务实例的延迟
        """
        for instance_id, instance in self.microservice_instances.items():
            if (instance.server_id == server_id and
                    instance.microservice.service_type == "ai"):

                processing_mode = getattr(instance, 'processing_mode', 'unknown')

                if processing_mode == "local_processing":
                    return getattr(instance, 'inference_latency', 0.0)
                elif processing_mode == "cloud_offloaded":
                    return getattr(instance, 'cloud_latency', 0.0)
                elif processing_mode == "pending_decision":
                    # 待决策状态，返回0
                    return 0

        return 0.0  # 如果没有找到AI微服务，返回0

    def get_lyapunov_function_value(self) -> float:
        """
        计算Lyapunov函数值，包含能耗队列和延迟队列
        L(Y(t), Z(t)) = (1/2) * [Σ_s Y_s(t)^2 + Σ_s Z_s(t)^2]
        """
        total = 0.0
        # 一次遍历处理能耗队列和延迟队列
        for server_id, server in self.edge_servers.items():
            if server.server_type.value == "ai_capable":
                # 能耗队列
                if server_id in self.virtual_energy_queues:
                    energy_queue_state = self.virtual_energy_queues[server_id].queue_state
                    total += energy_queue_state ** 2

                # 延迟队列
                if server_id in self.virtual_delay_queues:
                    delay_queue_state = self.virtual_delay_queues[server_id].queue_state
                    total += delay_queue_state ** 2

        return 0.5 * total

    def get_lyapunov_state_input(self) -> np.ndarray:
        """
        获取Lyapunov算法状态输入
        """
        state_vector = self.get_state_representation()
        return state_vector

    def get_action_space_size(self) -> int:
        """
        获取动作空间大小 (用于强化学习网络设计)
        动作空间是 {0,1}^N（N维二进制向量），类比LyDROO
        每个AI服务器的动作：
        - 0：本地计算（local computing）
        - 1：卸载到云端（offloading）
        """
        # N = AI服务器数量
        num_ai_servers = len([s for s in self.edge_servers.values()
                              if s.server_type == ServerType.AI_CAPABLE])

        # 返回动作向量的维度（N维二进制向量）
        return num_ai_servers

    def update_request_arrivals(self, new_arrivals: Dict[str, float]):
        """
        更新请求到达率 (参考LyDROO的动态到达率更新)
        Args:
            new_arrivals: {flow_id: arrival_rate} 格式的新到达率
        """
        for flow_id, arrival_rate in new_arrivals.items():
            if flow_id in self.request_flows:
                self.request_flows[flow_id].arrival_rate = arrival_rate

        # 更新当前到达率记录
        self.current_arrivals = new_arrivals.copy()

    def calculate_system_reward(self, weights: Dict[str, float] = None) -> float:
        """
        计算系统奖励值
        结合微服务性能指标和约束违反惩罚
        Args:
            weights: 各指标的权重
        Returns:
            float: 系统总奖励值
        """
        if weights is None:
            weights = {
                'delay': -1.0,
                'energy': -0.1,
                'violation': -10.0,
                'queue_stability': -5.0,
                'resource_efficiency': 1.0
            }

        total_reward = 0.0

        # 1. 延迟奖励 (负值，延迟越低奖励越高)
        total_delay = system.total_latency  # 调用Waitingtime模块
        total_reward += weights['delay'] * total_delay

        # 2. 能耗奖励 (负值，能耗越低奖励越高)
        total_reward += weights['energy'] * self.total_energy_consumption

        # 3. 约束违反惩罚
        violations = self.get_energy_constraint_violations()
        violation_count = sum(1 for v in violations.values() if v > 0.1)  # 违反率>10%
        total_reward += weights['violation'] * violation_count

        # 4. 队列稳定性奖励
        total_reward += weights['queue_stability'] * self.queue_stability_violations

        # 5. 资源效率奖励
        resource_efficiency = self.calculate_resource_efficiency()
        total_reward += weights['resource_efficiency'] * resource_efficiency

        return total_reward

    def calculate_resource_efficiency(self) -> float:
        """
        计算资源效率分数
        基于各种资源的利用率平衡性
        """
        if not self.edge_servers:
            return 0.0

        total_efficiency = 0.0
        for server in self.edge_servers.values():
            utilization_vector = server.get_comprehensive_resource_utilization()
            # 计算资源利用率的方差，方差越小表示资源利用越均衡
            utilization_variance = np.var(utilization_vector)
            # 计算平均利用率
            mean_utilization = np.mean(utilization_vector)
            # 效率分数：高利用率且低方差
            efficiency = mean_utilization * (1.0 - utilization_variance)
            total_efficiency += efficiency

        return total_efficiency / len(self.edge_servers)

    def get_energy_constraint_violations(self) -> Dict[str, float]:
        """获取各服务器的能耗约束违反率"""
        violations = {}
        for server_id, queue in self.virtual_energy_queues.items():
            violations[server_id] = queue.get_violation_rate()
        return violations

    def get_delay_constraint_violations(self) -> Dict[str, float]:
        """获取各服务器的延迟约束违反率"""
        violations = {}
        for server_id, queue in self.virtual_delay_queues.items():
            violations[server_id] = queue.get_violation_rate()
        return violations

# ================================== 决策变量 ==================================
class DecisionVariables:
    """
    决策变量集合
    包含系统在时间帧t需要做出的所有决策
    """

    def __init__(self, time_frame: int):
        self.time_frame = time_frame

        # N^m_v: 微服务部署决策 (对应论文公式1约束条件)
        # N^m_v ∈ {0,1,2,...,Cv}: 微服务m在服务器v的实例数
        self.microservice_deployment: Dict[Tuple[str, str], int] = {}
        # 格式: {(microservice_id, server_id): instance_count}

        # P^{vi,ma}_{vj,mb}: 路由概率决策 (对应论文公式2-3)
        self.routing_probabilities: Dict[Tuple[str, str, str], float] = {}
        # 格式: {(flow_id, source_server, target_server): probability}

        # 核心分配决策
        self.core_allocations: Dict[Tuple[str, str], int] = {}
        # 格式: {(microservice_id, server_id): allocated_cores}

    def set_microservice_deployment(self, microservice_id: str, server_id: str, instance_count: int):
        """设置微服务部署决策 (对应论文中的N^m_v)"""
        self.microservice_deployment[(microservice_id, server_id)] = instance_count

    def get_microservice_deployment(self, microservice_id: str, server_id: str) -> int:
        """获取微服务部署实例数"""
        return self.microservice_deployment.get((microservice_id, server_id), 0)

    def get_deployment_matrix(self, server_ids: List[str], microservice_ids: List[str]) -> np.ndarray:
        """
        获取部署决策矩阵 (N^m_v的矩阵表示)
        返回: 形状为(|V|, |M|)的矩阵，元素为N^m_v
        """
        deployment_matrix = np.zeros((len(server_ids), len(microservice_ids)))

        for i, server_id in enumerate(server_ids):
            for j, ms_id in enumerate(microservice_ids):
                deployment_matrix[i, j] = self.get_microservice_deployment(ms_id, server_id)

        return deployment_matrix

    def set_routing_probability(self, flow_id: str, source_server: str,
                                target_server: str, probability: float):
        """设置路由概率 (对应论文中的P^{vi,ma}_{vj,mb})"""
        self.routing_probabilities[(flow_id, source_server, target_server)] = probability

    def get_routing_probability(self, flow_id: str, source_server: str,
                                target_server: str) -> float:
        """获取路由概率"""
        return self.routing_probabilities.get((flow_id, source_server, target_server), 0.0)

    def get_routing_probability_matrix(self, server_ids: List[str]) -> np.ndarray:
        """
        获取路由概率矩阵 (P^{vi,ma}_{vj,mb}的矩阵表示)
        返回: 形状为(|V|, |V|)的矩阵
        """
        routing_matrix = np.zeros((len(server_ids), len(server_ids)))

        # 简化版本：汇总所有流的路由概率
        for i, source_id in enumerate(server_ids):
            for j, target_id in enumerate(server_ids):
                total_prob = 0.0
                for (flow_id, src, tgt), prob in self.routing_probabilities.items():
                    if src == source_id and tgt == target_id:
                        total_prob += prob
                routing_matrix[i, j] = total_prob

        return routing_matrix

    def set_core_allocation(self, microservice_server_tuple: Tuple[str, str], cores: int):
        """
        设置微服务在特定服务器上的核心分配数
        Args:
            microservice_server_tuple: (microservice_id, server_id) 元组
            cores: 分配的核心数
        """
        self.core_allocations[microservice_server_tuple] = cores

    def get_core_allocation(self, microservice_id: str, server_id: str) -> int:
        """
        获取微服务在特定服务器上的核心分配数
        Args:
            microservice_id: 微服务ID
            server_id: 服务器ID
        Returns:
            int: 分配的核心数，如果未分配则返回0
        """
        return self.core_allocations.get((microservice_id, server_id), 0)

    def validate_decisions(self, system_state: SystemState) -> List[str]:
        """
        验证决策变量的合法性 (对应论文约束条件)
        检查公式1, 2, 3, 5等约束条件
        """
        violations = []

        # 验证资源约束 (论文公式1)
        for server_id, server in system_state.edge_servers.items():
            total_cpu_used = 0
            total_memory_used = 0.0

            # 检查微服务资源约束 Σ_{m∈M} N^m_v ≤ Cv
            for (ms_id, srv_id), instance_count in self.microservice_deployment.items():
                if srv_id == server_id and instance_count > 0:
                    microservice = system_state.microservices.get(ms_id)
                    if microservice:
                        # 使用微服务的实际CPU需求作为默认核心分配
                        default_cores = microservice.get_min_cores_required()
                        cores_per_instance = self.core_allocations.get((ms_id, srv_id), default_cores)

                        total_cpu_used += instance_count * cores_per_instance
                        total_memory_used += instance_count * microservice.memory_requirement

            # 验证CPU约束违反情况
            if total_cpu_used > server.cpu_cores:
                violations.append(
                    f"Server {server_id}: CPU constraint violation "
                    f"({total_cpu_used} > {server.cpu_cores}) (论文公式1)"
                )

            if total_memory_used > server.memory_capacity:
                violations.append(
                    f"Server {server_id}: Memory constraint violation "
                    f"({total_memory_used:.2f} > {server.memory_capacity})"
                )

        # 验证路由概率约束 (论文公式2)
        for flow_id in system_state.request_flows.keys():
            source_prob_sums = defaultdict(float)
            for (f_id, source, target), prob in self.routing_probabilities.items():
                if f_id == flow_id:
                    source_prob_sums[source] += prob

            for source, total_prob in source_prob_sums.items():
                if abs(total_prob - 1.0) > 1e-6:
                    violations.append(
                        f"Flow {flow_id}, Source {source}: Routing probability sum = {total_prob:.4f} "
                        f"(should be 1.0) (论文公式2)"
                    )

        # 验证AI微服务只能部署在AI服务器上
        for (ms_id, srv_id), instance_count in self.microservice_deployment.items():
            if instance_count > 0:
                microservice = system_state.microservices.get(ms_id)
                server = system_state.edge_servers.get(srv_id)
                if microservice and server:
                    if (microservice.service_type == "ai" and
                            server.server_type != ServerType.AI_CAPABLE):
                        violations.append(
                            f"AI microservice {ms_id} cannot be deployed on "
                            f"traditional server {srv_id}"
                        )

        return violations

# ================================== 系统初始化 ==================================

def create_sample_system(seed: Optional[int] = None) -> SystemState:
    """
    创建完整的示例系统状态（保持向后兼容）

    Args:
        seed: 随机种子

    Returns:
        SystemState: 初始化完成的系统状态
    """
    # 创建基础系统
    system = create_base_system(seed)

    # 应用Next Fit部署算法
    apply_next_fit_deployment_algorithm(system, use_trained_model=False)

    return system


def create_base_system(seed: int = 42, chain_length_range: tuple = (6, 8),
                       fixed_arrival_rate: float = None, num_edge_nodes: int = 25,
                       ai_node_count: int = 10, request_flow_count: int = 10,
                       arrival_range_req_s: Tuple[float, float] = (4.0, 8.0),
                       input_tokens_range: Tuple[int, int] = (100, 2000),
                       output_tokens_range: Tuple[int, int] = (50, 500),
                       gpu_units_range: Optional[Tuple[int, int]] = None,
                       max_batch_size: int = 64) -> SystemState:
    """
    创建基础系统（不包含部署算法）
    只包含：服务器、网络拓扑、请求流、环境管理器的初始化
    Args:
        seed: 随机种子
        chain_length_range: 服务链长度范围
        fixed_arrival_rate: 固定到达率（如果指定，则所有请求流使用相同的到达率）
        num_edge_nodes: 传统边缘节点数量（默认25，AI节点固定为10）
        ai_node_count: AI计算节点数量
        request_flow_count: 请求流数量
        arrival_range_req_s: 到达率范围(req/s)
        input_tokens_range: 输入token范围
        output_tokens_range: 输出token范围
    Returns:
        SystemState: 基础系统状态对象
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # 在函数内部导入以避免循环导入
    import InitSystem
    system = SystemState(time_frame=0)
    # 创建边缘服务器
    # 传统计算节点（使用可变参数num_edge_nodes）
    InitSystem.initialize_traditional_servers(system, num_edge_nodes, seed=seed)
    # AI计算节点（消融入口可指定数量）
    InitSystem.initialize_ai_servers(
        system,
        ai_node_count,
        seed=seed + 1000 if seed is not None else None,
        gpu_units_range=gpu_units_range,
        max_batch_size=max_batch_size
    )

    # 初始化网络拓扑
    InitSystem.initialize_network_topology(system, seed)

    # 创建请求流
    created_flows = InitSystem.initialize_request_flows(
        system,
        flow_count=request_flow_count,
        seed=seed,
        chain_length_range=chain_length_range,
        fixed_arrival_rate=fixed_arrival_rate,
        arrival_range_req_s=arrival_range_req_s,
        input_tokens_range=input_tokens_range,
        output_tokens_range=output_tokens_range
    )

    # 初始化环境状态管理器
    from Enviroment_state import EnvironmentStateManager
    system.environment_manager = EnvironmentStateManager(system)
    '''
    print(f"基础系统创建完成，包含 {len(system.edge_servers)} 个边缘服务器")
    print(f"创建了 {len(created_flows)} 个请求流")
    '''
    return system

def apply_trained_ai_decisions(system_state: SystemState, offloading_decisions: np.ndarray) -> bool:
    """
    应用训练好的AI模型的卸载决策
    Args:
        system_state: 系统状态对象
        offloading_decisions: AI模型的卸载决策向量 (0=本地, 1=云端)
    Returns:
        bool: 是否应用成功
    """
    print(f"应用训练好的AI模型决策...")

    # 获取AI服务器列表
    ai_servers = [server for server in system_state.edge_servers.values()
                  if server.server_type.value == "ai_capable"]
    ai_server_ids = sorted([server.server_id for server in ai_servers])

    if len(offloading_decisions) != len(ai_server_ids):
        print(f"⚠️ 决策维度({len(offloading_decisions)}) != AI服务器数量({len(ai_server_ids)})")
        return False

    # 获取当前系统状态（用于资源分配函数）
    env_manager = system_state.environment_manager
    if env_manager is None:
        print("⚠️ 环境状态管理器未初始化")
        return False

    # 更新环境状态获取最新的SH, SQ, SZ
    env_manager.update_all_ai_server_states()
    SH, SQ, SZ = env_manager.get_state_components()

    # 预先缓存所有AI服务器的AI实例和请求流信息
    ai_instance_cache = {}
    for server_id in ai_server_ids:
        from ResourceAllocation import find_ai_instance_and_flow
        ai_instance, request_flow = find_ai_instance_and_flow(server_id, system_state)
        ai_instance_cache[server_id] = (ai_instance, request_flow)

    # 复用main.py中的资源分配函数
    from ResourceAllocation import AI_Offloading_Resource_Allocation

    try:
        # 使用完整的资源分配算法
        objective_value, ai_delays, ai_energies = AI_Offloading_Resource_Allocation(
            offloading_mode=offloading_decisions,
            SH=SH,
            SQ=SQ,
            SZ=SZ,
            system_state=system_state,
            V=20.0,
            weights=None,  # 使用默认权重
            ai_instance_cache=ai_instance_cache
        )

        print(f"✓ 资源分配完成:")
        print(f"  目标函数值: {objective_value:.3f}")
        print(f"  AI延迟向量: {ai_delays}")
        print(f"  AI能耗向量: {ai_energies}")

        # 【关键修复】：将计算结果应用到AI微服务实例状态
        success_count = 0
        for i, server_id in enumerate(ai_server_ids):
            decision = offloading_decisions[i]
            ai_instance, request_flow = ai_instance_cache[server_id]

            if ai_instance is None or request_flow is None:
                print(f"  {server_id}: 未找到AI微服务实例 ✗")
                continue

            server = system_state.edge_servers[server_id]

            # 根据决策更新AI微服务实例状态
            if decision == 0:  # 本地处理
                success = apply_local_processing_to_instance(
                    ai_instance, request_flow, server, ai_delays[i], ai_energies[i])
                decision_type = "本地处理"
            else:  # 云端卸载
                success = apply_cloud_offloading_to_instance(
                    ai_instance, request_flow, server, ai_delays[i], ai_energies[i])
                decision_type = "云端卸载"

            if success:
                success_count += 1
                print(f"  {server_id}: {decision_type} ✓")
            else:
                print(f"  {server_id}: 状态更新失败 ✗")

        print(f"决策应用完成: {success_count}/{len(ai_server_ids)} 个成功")
        return success_count == len(ai_server_ids)

    except Exception as e:
        print(f"⚠️ 资源分配过程出错: {e}")
        import traceback
        traceback.print_exc()
        return False


def apply_local_processing_to_instance(ai_instance, request_flow, server, delay, energy):
    """
    将本地处理决策应用到AI微服务实例
    """
    try:
        from Deployment import calculate_local_ai_inference_latency, calculate_required_gpu_memory, \
            calculate_required_model_storage

        # 计算所需资源
        total_latency, queue_delay, processing_delay, required_gpu_units = calculate_local_ai_inference_latency(
            request_flow, ai_instance.microservice, server, None, False)
        required_gpu_memory = calculate_required_gpu_memory(
            request_flow, ai_instance.microservice, required_gpu_units, False)
        required_model_storage = calculate_required_model_storage(ai_instance.microservice)

        # 检查并分配GPU资源
        if (server.available_gpu_units >= required_gpu_units and
                server.available_gpu_memory >= required_gpu_memory and
                server.available_model_storage >= required_model_storage):

            # 分配GPU资源
            server.available_gpu_units -= required_gpu_units
            server.available_gpu_memory -= required_gpu_memory
            server.available_model_storage -= required_model_storage
            server.activate_gpu_for_inference()

            # 更新AI微服务实例状态
            ai_instance.processing_mode = "local_processing"
            ai_instance.gpu_memory_allocated = required_gpu_memory
            ai_instance.model_storage_allocated = required_model_storage
            ai_instance.inference_latency = delay
            ai_instance.cloud_latency = 0.0
            ai_instance.queue_delay = queue_delay
            ai_instance.processing_delay = processing_delay

            # 设置GPU单元分配（如果实例对象支持）
            if hasattr(ai_instance, 'gpu_units_allocated'):
                ai_instance.gpu_units_allocated = required_gpu_units

            return True
        else:
            print(f"      资源不足，无法应用本地处理")
            return False

    except Exception as e:
        print(f"      本地处理状态更新失败: {e}")
        return False


def apply_cloud_offloading_to_instance(ai_instance, request_flow, server, delay, energy):
    """
    将云端卸载决策应用到AI微服务实例
    """
    try:
        # 更新AI微服务实例状态
        ai_instance.processing_mode = "cloud_offloaded"
        ai_instance.gpu_memory_allocated = 0.0  # 云端卸载不占用GPU资源
        ai_instance.model_storage_allocated = 0.0
        ai_instance.inference_latency = 0.0
        ai_instance.cloud_latency = delay
        ai_instance.queue_delay = 0.0  # 云端处理无本地队列延迟
        ai_instance.processing_delay = delay

        # 云端卸载不需要占用本地GPU资源，但需要少量CPU用于协调
        if server.available_cpu > 0:
            # 注意：这里不减少CPU，因为在物理部署时已经预留了1个CPU
            pass

        return True

    except Exception as e:
        print(f"      云端卸载状态更新失败: {e}")
        return False

def apply_next_fit_deployment_algorithm(system: SystemState, use_trained_model: bool = False) -> bool:
    """
    应用Next Fit部署算法到系统
    包含完整的部署流程：差异度矩阵计算、部署执行、性能计算

    Args:
        system: 已经初始化好基础设施的系统状态对象
        use_trained_model: 是否使用训练好的AI模型进行卸载决策
                          True: 用于对比测试，使用训练好的模型
                          False: 用于训练过程，不使用训练好的模型（默认）
    Returns:
        bool: 是否所有微服务都部署成功
    """
    import Deployment, Waitingtime, EnergyConsumption

    print("=== 应用Next Fit部署算法 ===")

    # 获取请求流列表
    created_flows = list(system.request_flows.values())

    # 计算差异度矩阵
    distance_matrix = Deployment.calculate_stream_distance_matrix(created_flows, system)
    system.stream_distance_matrix = distance_matrix

    print(f"差异度矩阵计算完成，矩阵形状: {distance_matrix.shape}")

    # 执行Next Fit部署算法
    Deployment.next_fit_deployment(system)
    ai_deployment_success = Deployment.deploy_ai_microservices(system)

    if ai_deployment_success:
        print(f"✓ 所有AI微服务部署成功")
    else:
        print(f"⚠️  部分AI微服务部署失败，请检查资源配置")
        return False

    # =================== 条件性AI卸载决策 ===================
    if use_trained_model:
        # 对比测试模式：使用训练好的AI模型进行卸载决策
        print(f"\n=== 使用训练好的AI模型进行卸载决策 ===")

        try:
            # 导入并创建训练好的AI推理器
            from ai_inference import create_trained_ai_inference
            ai_inference = create_trained_ai_inference()

            if ai_inference.is_loaded:
                print(f"✓ 训练好的AI模型加载成功")

                # 使用训练好的AI模型进行决策
                offloading_decisions = ai_inference.make_decision(system, decoder_mode='OPN')
                print(f"AI模型决策结果: {offloading_decisions}")
                print(f"决策解释: {['本地处理' if d == 0 else '云端卸载' for d in offloading_decisions]}")

                # 应用AI卸载决策
                apply_success = apply_trained_ai_decisions(system, offloading_decisions)

                if not apply_success:
                    print(f"⚠️ AI模型决策应用失败")
                    return False

                print(f"✓ 训练好的AI模型决策应用成功")
            else:
                print(f"⚠️ 训练好的AI模型加载失败")
                return False

        except Exception as e:
            print(f"⚠️ AI模型决策过程出错: {e}")
            return False
    else:
        # 训练模式：AI微服务保持pending_decision状态，等待RL算法决策
        print(f"\n=== 训练模式：AI微服务保持待决策状态 ===")
        print(f"AI微服务已完成物理部署，等待RL算法进行卸载决策")

        # 验证AI微服务都处于pending_decision状态
        pending_count = 0
        for instance in system.microservice_instances.values():
            if instance.microservice.service_type == "ai":
                processing_mode = getattr(instance, 'processing_mode', 'unknown')
                if processing_mode == "pending_decision":
                    pending_count += 1

        print(f"共有 {pending_count} 个AI微服务处于待决策状态")

    # =================== 原有代码继续 ===================

    # 计算所有请求流的端到端延迟
    system.total_latency = 0.0
    for flow_id in system.request_flows.keys():
        flow_delay = Waitingtime.calculate_flow_end_to_end_delay_global(flow_id, system)
        system.request_flows[flow_id].ca_latency = flow_delay
        system.total_latency += flow_delay

    # 计算平均延迟
    num_flows = len(system.request_flows)
    avg_latency = system.total_latency / max(num_flows, 1)

    # 更新所有服务器的虚拟能耗队列
    system.update_all_energy_queues()

    # 计算系统总能耗
    system.total_energy_consumption = EnergyConsumption.calculate_system_total_energy_consumption(system)

    # 计算Lyapunov函数值
    lyapunov_value = system.get_lyapunov_function_value()

    print(f"部署算法执行完成:")
    print(f"  系统总延迟: {system.total_latency:.2f} ms")
    print(f"  系统平均延迟: {avg_latency:.2f} ms")
    print(f"  系统总能耗: {system.total_energy_consumption:.3f} J")
    print(f"  Lyapunov函数值: {lyapunov_value:.3f}")

    return ai_deployment_success

def print_progress_bar(iteration, total, prefix='Progress', suffix='Complete', length=50):
    """
    显示进度条
    """
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='')
    sys.stdout.flush()

if __name__ == "__main__":
    # 创建示例系统
    system = create_sample_system(42)
    print(f"系统初始化完成，包含 {len(system.edge_servers)} 个边缘服务器")
    # 测试系统状态表示
    SH, SQ = system.environment_manager.get_state_components()

    # 获取完整的状态表示向量（用于神经网络）
    state_vector = system.get_state_representation()
    print(f"\n状态向量维度: {state_vector.shape}")
    print(f"状态向量所有元素: {state_vector[:]}")

