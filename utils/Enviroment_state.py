import numpy as np
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING
from dataclasses import dataclass
from collections import deque
import math
import hashlib

# 使用TYPE_CHECKING避免循环导入
if TYPE_CHECKING:
    from Constant import SystemState, ServerType, EdgeServer, RequestFlow


@dataclass
class AIServerPerformanceState:
    """
    AI服务器性能环境状态
    反映AI服务器当前面临的工作负载环境
    """
    server_id: str
    # 性能环境标量值
    performance_factor: float

    # 3维原始状态向量 [到达率强度, 传统链延迟压力, 传统链能耗压力]
    raw_performance_vector: np.ndarray

    # 历史状态记录（用于状态演化分析）
    history: deque

    # 状态统计信息
    avg_arrival_intensity: float = 0.0
    avg_delay_pressure: float = 0.0
    avg_energy_pressure: float = 0.0
    avg_performance_factor: float = 0.0

    def __post_init__(self):
        if self.raw_performance_vector is None:
            self.raw_performance_vector = np.zeros(3)
        if not hasattr(self, 'history'):
            self.history = deque(maxlen=100)


class EnvironmentStateManager:
    """
    环境状态管理器
    负责计算和更新AI服务器的性能环境状态
    """
    def __init__(self, system_state: 'SystemState'):
        self.system_state = system_state

        # AI服务器性能状态字典
        self.ai_server_states: Dict[str, AIServerPerformanceState] = {}

        # 初始化AI服务器状态
        self._initialize_ai_server_states()

        # 保存每个请求流的初始到达率
        self.initial_arrival_rates: Dict[str, float] = {}
        self._save_initial_arrival_rates()

        # 归一化参数
        self.normalization_factors = {
            'arrival_rate': 12.0,  # 最大到达率归一化因子
            'delay': 50.0,  # 最大延迟归一化因子 (ms)
            'energy': 0.5  # 最大能耗归一化因子 (J)
        }

        # 性能因子合并权重
        self.performance_weights = {
            'arrival_weight': 0.2,   # 到达率权重（降低）
            'delay_weight': -0.5,    # 时延权重（负值）
            'energy_weight': 0.8     # 能耗权重（正值）
        }

        # 时间演化参数
        self.time_frame = 0
        self.update_interval = 1  # 每个时隙更新一次

    def _initialize_ai_server_states(self):
        """初始化所有AI服务器的性能状态"""
        for server_id, server in self.system_state.edge_servers.items():
            if server.server_type.value == "ai_capable":
                self.ai_server_states[server_id] = AIServerPerformanceState(
                    server_id=server_id,
                    performance_factor=0.0,
                    raw_performance_vector=np.zeros(3),
                    history=deque(maxlen=100)
                )
                #print(f"初始化AI服务器 {server_id} 的性能环境状态")

    def update_all_ai_server_states(self, new_arrivals: Dict[str, float] = None,
                                    allow_redeployment: bool = None):
        """
        更新所有AI服务器的性能环境状态
        类比LyDROO中每个时隙的信道状态更新
        Args:
            new_arrivals: 新的到达率字典，如果为None则使用当前到达率
            allow_redeployment: 是否允许到达率更新触发旧Next Fit重部署。
                正式ablation中必须为False，慢层部署只由run_slow_context_for_algorithm显式执行。
        """
        print(f"\n=== 时隙 {self.time_frame}: 更新AI服务器性能环境状态 ===")

        # 如果有新的到达率，先更新系统到达率
        if new_arrivals:
            self._update_request_flow_arrivals(new_arrivals)

        if allow_redeployment is None:
            allow_redeployment = bool(getattr(self.system_state, "allow_environment_redeployment", True))

        # legacy入口仍可选择旧行为；正式复现实验禁止slot更新隐式改慢层部署
        if new_arrivals and allow_redeployment:
            self._trigger_redeployment_if_needed()
        elif new_arrivals:
            self._update_resource_requirements_only()

        # 计算每个AI服务器的性能环境状态
        for server_id in self.ai_server_states.keys():
            self._update_single_ai_server_state(server_id)

        self.time_frame += 1

    def _save_initial_arrival_rates(self):
        """保存每个请求流的初始到达率"""
        for flow_id, request_flow in self.system_state.request_flows.items():
            self.initial_arrival_rates[flow_id] = request_flow.arrival_rate
        print(f"保存了 {len(self.initial_arrival_rates)} 个请求流的初始到达率")

    def _update_request_flow_arrivals(self, new_arrivals: Dict[str, float]):
        """更新系统所有请求流的到达率"""
        self.system_state.update_request_arrivals(new_arrivals)
        print(f"更新系统所有请求流到达率: {new_arrivals}")

    def _trigger_redeployment_if_needed(self):
        """
        每个时隙到达率变化时，触发传统微服务的重新部署和AI资源刷新
        """

        # 重新计算传统微服务资源需求
        for flow_id, request_flow in self.system_state.request_flows.items():
            from Deployment import calculate_ms_resource
            calculate_ms_resource(request_flow)

        # 重新执行Next Fit部署算法
        self._redeploy_traditional_microservices()

        # *** 重新计算AI微服务的资源需求（基于新到达率）***
        self._recalculate_ai_resource_requirements()

    def _update_resource_requirements_only(self):
        """
        只更新负载相关需求，不改变部署和路由。
        formal ablation的slot update使用该路径，避免Next Fit污染GSLA/FFD/PDRS/GMDA慢层状态。
        """
        for flow_id, request_flow in self.system_state.request_flows.items():
            from Deployment import calculate_ms_resource
            calculate_ms_resource(request_flow)
        self._recalculate_ai_resource_requirements()

    def _recalculate_ai_resource_requirements(self):
        """
        基于新的到达率重新计算AI微服务的资源需求
        更新请求流的AI相关参数，但不改变物理部署位置
        """
        #print("\n--- 重新计算AI微服务资源需求 ---")

        for flow_id, request_flow in self.system_state.request_flows.items():
            ai_microservice = request_flow.service_chain.ai_microservice
            if ai_microservice:
                # 基于新到达率重新计算资源需求
                from Deployment import calculate_ai_microservice_resource_requirement
                new_resource_score = calculate_ai_microservice_resource_requirement(
                    request_flow, ai_microservice, self.system_state)
                '''
                print(f"请求流 {flow_id} AI资源需求更新:")
                print(f"  新到达率: {request_flow.arrival_rate:.3f} req/ms")
                print(f"  资源需求评分: {new_resource_score:.3f}")
                '''
                # 更新请求流的AI处理选择为待定（等待RL决策）
                request_flow.ai_processing_choice = "pending"

    def _redeploy_traditional_microservices(self):
        """重新部署传统微服务并重置AI服务器GPU资源"""
        # 清空现有的传统微服务实例
        traditional_instances = []
        for instance_id, instance in list(self.system_state.microservice_instances.items()):
            if instance.microservice.service_type == "traditional":
                traditional_instances.append(instance_id)

        for instance_id in traditional_instances:
            del self.system_state.microservice_instances[instance_id]

        # 重置传统服务器资源
        for server in self.system_state.edge_servers.values():
            if server.server_type.value == "traditional":
                server.available_cpu = server.cpu_cores
                server.available_memory = server.memory_capacity

        #重置AI服务器的GPU资源（保持物理部署位置不变）
        self.reset_ai_server_gpu_resources()

        #到达率更新后重新计算差异度矩阵
        #rint("重新计算差异度矩阵（因为到达率已更新）...")
        from Deployment import calculate_stream_distance_matrix
        request_flows = list(self.system_state.request_flows.values())
        self.system_state.stream_distance_matrix = calculate_stream_distance_matrix(
            request_flows, self.system_state)
        #print(f"差异度矩阵更新完成，形状: {self.system_state.stream_distance_matrix.shape}")

        # 重新执行Next Fit部署算法
        from Deployment import next_fit_deployment
        next_fit_deployment(self.system_state)
        #print("✓ 传统微服务重新部署完成")

    def reset_ai_server_gpu_resources(self):
        """
        重置AI服务器的GPU资源（时隙开始时的完整重置）
        """
        #print(f"\n--- 时隙开始重置AI服务器GPU资源 ---")

        for server_id, server in self.system_state.edge_servers.items():
            if server.server_type.value == "ai_capable":
                # 完全重置GPU资源到初始状态
                server.available_gpu_units = server.gpu_units
                server.available_gpu_memory = server.gpu_memory
                server.available_model_storage = server.model_storage
                server.deactivate_gpu()
                '''
                print(f"AI服务器 {server_id} GPU资源重置:")
                print(f"  GPU单元: 重置为 {server.available_gpu_units} 个")
                print(f"  GPU内存: 重置为 {server.available_gpu_memory:.1f} GB")
                print(f"  模型存储: 重置为 {server.available_model_storage:.1f} GB")
                '''
        # 重置AI微服务实例的资源分配状态为待决策
        for instance_id, instance in self.system_state.microservice_instances.items():
            if instance.microservice.service_type == "ai":
                # 强制重置所有GPU相关属性
                instance.gpu_units_allocated = 0  # 修改：确保重置
                instance.gpu_memory_allocated = 0.0
                instance.model_storage_allocated = 0.0
                instance.processing_mode = "pending_decision"
                instance.inference_latency = 0.0  # 新增：重置延迟
                instance.cloud_latency = 0.0  # 新增：重置云端延迟

                # 新增：确保没有其他GPU相关属性
                if hasattr(instance, 'queue_delay'):
                    instance.queue_delay = 0.0
                if hasattr(instance, 'processing_delay'):
                    instance.processing_delay = 0.0

        #print("时隙GPU资源重置完成 - 所有计算状态已清除")

    def _update_single_ai_server_state(self, server_id: str):
        """
        更新单个AI服务器的性能环境状态
        计算3维向量后合并为1维标量: h = w1*到达率强度 + w2*延迟压力 + w3*能耗压力
        """
        ai_state = self.ai_server_states[server_id]

        # 1. 计算该AI服务器的到达率强度
        arrival_intensity = 0.0
        for instance_id, instance in self.system_state.microservice_instances.items():
             if (instance.server_id == server_id and instance.microservice.service_type == "ai"):
                 # 从instance_id中提取flow_id
                 parts = instance_id.split('_')
                 if len(parts) >= 2:
                     flow_id = f"{parts[0]}_{parts[1]}"
                     request_flow = self.system_state.request_flows.get(flow_id)
                     if request_flow:
                         arrival_intensity = request_flow.arrival_rate
                         break

        # 2. 计算该AI服务器相关链路的传统部分延迟压力
        delay_pressure = self.calculate_delay_pressure(server_id)

        # 3. 计算该AI服务器相关链路的传统部分能耗压力
        energy_pressure = self._calculate_energy_pressure(server_id)

        # 4. 归一化处理（类比LyDROO中的h = h_tmp*CHFACT）
        normalized_arrival = arrival_intensity / self.normalization_factors['arrival_rate']
        normalized_delay = delay_pressure / self.normalization_factors['delay']
        normalized_energy = energy_pressure / self.normalization_factors['energy']
        '''
        print(f"  📊 归一化结果:")
        print(
            f"    到达率: {arrival_intensity:.3f} / {self.normalization_factors['arrival_rate']:.1f} = {normalized_arrival:.3f}")
        print(f"    延迟: {delay_pressure:.3f} / {self.normalization_factors['delay']:.1f} = {normalized_delay:.3f}")
        print(f"    能耗: {energy_pressure:.6f} / {self.normalization_factors['energy']:.1f} = {normalized_energy:.6f}")
        '''
        # 计算加权评分
        weighted_score = (
                self.performance_weights['arrival_weight'] * normalized_arrival +
                self.performance_weights['delay_weight'] * normalized_delay +  # 负权重
                self.performance_weights['energy_weight'] * normalized_energy  # 正权重
        )


        # 6. 添加时变性
        time_varying_factor = self._generate_deterministic_time_factor()
        final_performance_factor = weighted_score * time_varying_factor

        # 7. 确保性能因子在合理范围内
        final_performance_factor = max(0.0, min(final_performance_factor, 2.0))

        # 8. 更新状态
        ai_state.performance_factor = final_performance_factor
        ai_state.raw_performance_vector = np.array([
            normalized_arrival, normalized_delay, normalized_energy
        ])

        # 9. 记录历史
        ai_state.history.append({
            'time_frame': self.time_frame,
            'arrival_intensity': arrival_intensity,
            'delay_pressure': delay_pressure,
            'energy_pressure': energy_pressure,
            'normalized_arrival': normalized_arrival,
            'normalized_delay': normalized_delay,
            'normalized_energy': normalized_energy,
            'performance_factor': final_performance_factor,
            'time_varying_factor': time_varying_factor
        })

        # 10. 更新统计信息
        self._update_statistics(ai_state)
        '''
        print(f"AI服务器 {server_id} 性能环境状态:")
        print(f"  到达率强度: {arrival_intensity:.3f} req/ms (归一化: {normalized_arrival:.3f})")
        print(f"  延迟压力: {delay_pressure:.2f} ms (归一化: {normalized_delay:.3f})")
        print(f"  能耗压力: {energy_pressure:.3f} J (归一化: {normalized_energy:.3f})")
        print(f"  性能因子 h: {final_performance_factor:.3f}")
        '''

    def calculate_delay_pressure(self, server_id: str) -> float:
        """
        计算AI服务器相关链路的传统部分延迟压力
        只考虑传统微服务链的延迟以及到AI微服务的通信延迟，不包括AI微服务处理延迟
        """
        from ResourceAllocation import find_ai_instance_and_flow

        ai_instance, request_flow = find_ai_instance_and_flow(server_id, self.system_state)

        if request_flow is not None:
            return request_flow.ca_latency
        else:
            return 0.0

    def _calculate_energy_pressure(self, server_id: str) -> float:
        """
        计算AI服务器相关链路的传统部分能耗压力
        只考虑传统微服务链的能耗，不包括AI微服务能耗
        """
        total_energy_pressure = 0.0
        related_flows_count = 0

        # 查找与该AI服务器相关的请求流
        for flow_id, request_flow in self.system_state.request_flows.items():
            ai_microservice = request_flow.service_chain.ai_microservice
            if ai_microservice:
                # 检查AI微服务是否部署在该服务器
                ai_deployed_here = False
                for instance_id, instance in self.system_state.microservice_instances.items():
                    if (instance.microservice.ms_id == ai_microservice.ms_id and
                            instance.server_id == server_id and
                            flow_id in instance_id):
                        ai_deployed_here = True
                        break

                if ai_deployed_here:
                    # 计算该请求流传统微服务链的能耗
                    traditional_chain_energy = self._calculate_traditional_chain_energy(request_flow)
                    total_energy_pressure += traditional_chain_energy
                    related_flows_count += 1

        # 返回平均能耗压力
        return total_energy_pressure / max(related_flows_count, 1)

    def _calculate_traditional_chain_energy(self, request_flow: 'RequestFlow') -> float:
        """
        计算请求流传统微服务链的总能耗
        调用EnergyConsumption模块的相关函数
        """
        from EnergyConsumption import calculate_current_energy_consumption

        total_energy = 0.0
        traditional_microservices = request_flow.service_chain.get_traditional_microservices()

        # 统计传统微服务部署在哪些服务器上
        server_workloads = {}  # {server_id: workload_count}

        for ms in traditional_microservices:
            for instance in self.system_state.microservice_instances.values():
                if instance.microservice.ms_id == ms.ms_id:
                    server_id = instance.server_id
                    server_workloads[server_id] = server_workloads.get(server_id, 0) + 1
                    break

        # 根据工作负载估算能耗
        for server_id, workload in server_workloads.items():
            server = self.system_state.edge_servers.get(server_id)
            if server and server.server_type.value == "traditional":
                # 使用EnergyConsumption模块计算准确的服务器能耗
                server_energy = calculate_current_energy_consumption(server)

                # 更精确的能耗分配：基于该请求流在服务器上的资源占用比例
                total_server_cpu_usage = max(server.cpu_cores - server.available_cpu, 1)
                flow_cpu_usage = sum(instance.allocated_cores
                                     for instance in self.system_state.microservice_instances.values()
                                     if (instance.server_id == server_id and
                                         instance.microservice.ms_id in [ms.ms_id for ms in traditional_microservices]))

                # 按CPU占用比例分配能耗
                if total_server_cpu_usage > 0:
                    flow_energy = server_energy * (flow_cpu_usage / total_server_cpu_usage)
                    total_energy += flow_energy

        return total_energy

    def _generate_deterministic_time_factor(self) -> float:
        """
        生成确定性时变因子，移除随机扰动
        只保留确定性的周期性变化
        """
        # 使用确定性的周期性变化
        time_factor = 1.0 + 0.1 * np.sin(self.time_frame * 0.1)  # 周期性变化
        return max(0.8, min(time_factor, 1.2))  # 确保在合理范围内

    def _update_statistics(self, ai_state: AIServerPerformanceState):
        """更新AI服务器状态的统计信息"""
        if len(ai_state.history) > 0:
            recent_history = list(ai_state.history)[-10:]  # 最近10个时隙

            ai_state.avg_arrival_intensity = np.mean([h['arrival_intensity'] for h in recent_history])
            ai_state.avg_delay_pressure = np.mean([h['delay_pressure'] for h in recent_history])
            ai_state.avg_energy_pressure = np.mean([h['energy_pressure'] for h in recent_history])
            ai_state.avg_performance_factor = np.mean([h['performance_factor'] for h in recent_history])

    def get_ai_server_performance_factor(self, server_id: str) -> float:
        """
        获取AI服务器的性能因子（1维标量）
        类比LyDROO中的h值，用于神经网络输入
        """
        if server_id in self.ai_server_states:
            return self.ai_server_states[server_id].performance_factor
        else:
            return 0.0

    def get_all_ai_server_performance_factors(self) -> Dict[str, float]:
        """获取所有AI服务器的性能因子"""
        factors = {}
        for server_id, ai_state in self.ai_server_states.items():
            factors[server_id] = ai_state.performance_factor
        return factors

    def get_state_components(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        获取系统状态的三个组件 (SH, SQ, SZ)
        Returns:
            SH: AI服务器的性能环境因子向量
            SQ: AI服务器的虚拟能耗队列状态向量
            SZ: AI服务器的虚拟延迟队列状态向量（新增）
        """
        ai_server_ids = sorted(self.ai_server_states.keys())  # 确保顺序一致
        num_ai_servers = len(ai_server_ids)

        # SH: AI服务器的性能环境因子向量
        SH = np.zeros(num_ai_servers)
        for i, server_id in enumerate(ai_server_ids):
            SH[i] = self.get_ai_server_performance_factor(server_id)

        # SQ: AI服务器的虚拟能耗队列状态向量
        SQ = np.zeros(num_ai_servers)
        for i, server_id in enumerate(ai_server_ids):
            if server_id in self.system_state.virtual_energy_queues:
                SQ[i] = self.system_state.virtual_energy_queues[server_id].get_queue_state_normalized()

        # SZ: AI服务器的虚拟延迟队列状态向量
        SZ = np.zeros(num_ai_servers)
        for i, server_id in enumerate(ai_server_ids):
            if server_id in self.system_state.virtual_delay_queues:
                SZ[i] = self.system_state.virtual_delay_queues[server_id].get_queue_state_normalized()

        return SH, SQ, SZ

    def get_concatenated_state_vector(self) -> np.ndarray:
        """
        获取连接的状态向量，用于神经网络输入
        类比LyDROO中的 nn_input = np.concatenate((h, Q[i_idx,:]/10000, Y[i_idx,:]/10000, Z[i_idx,:]/10000))
        """
        all_states = []

        # 获取所有AI服务器的性能因子（1维标量）
        ai_server_ids = sorted(self.ai_server_states.keys())  # 确保顺序一致
        for server_id in ai_server_ids:
            performance_factor = self.get_ai_server_performance_factor(server_id)
            all_states.append(performance_factor)

        # 获取虚拟能耗队列状态
        for server_id in ai_server_ids:
            if server_id in self.system_state.virtual_energy_queues:
                queue_state = self.system_state.virtual_energy_queues[server_id].get_queue_state_normalized()
                all_states.append(queue_state)
            else:
                all_states.append(0.0)

        # 获取虚拟延迟队列状态
        for server_id in ai_server_ids:
            if server_id in self.system_state.virtual_delay_queues:
                delay_queue_state = self.system_state.virtual_delay_queues[server_id].get_queue_state_normalized()
                all_states.append(delay_queue_state)
            else:
                all_states.append(0.0)

        return np.array(all_states, dtype=np.float32)


    def generate_time_varying_arrivals(self, time_slot: int) -> Dict[str, float]:
        """
        生成时变到达率
        每个请求流使用独立的时变特性

        Args:
            time_slot: 当前时隙编号
        Returns:
            Dict[str, float]: {flow_id: new_arrival_rate} 格式的新到达率字典
        """
        new_arrivals = {}

        for flow_id in self.system_state.request_flows.keys():
            base_rate = self.initial_arrival_rates.get(flow_id, 5.0)

            # 增加变化幅度 - 每个流有稳定的相位偏移
            stable_hash = hashlib.md5(flow_id.encode("utf-8")).hexdigest()
            flow_hash = int(stable_hash[:8], 16) % 1000
            phase_offset = (flow_hash / 1000.0) * 2 * np.pi

            # 更大的周期性变化 (增加幅度)
            major_cycle = 1.0 * np.sin(time_slot * 0.02 + phase_offset)  # 从1.0增加到1.5
            minor_cycle = 0.5 * np.sin(time_slot * 0.1 + phase_offset)  # 从0.5增加到0.8

            # 突发性变化 (增加频率和强度)

            if getattr(self.system_state, "workload_burst_enabled", False):
                # normal-main profile使用稳定突发项；只依赖flow_id和slot，不使用全局随机数。
                burst_phase = (time_slot + flow_hash) % 17
                burst_factor = 1.0 + (0.35 if burst_phase in (0, 1, 2) else 0.0)
            elif time_slot % 50 == 0:
                burst_factor = 1.0 #np.random.uniform(1.2, 1.5)
            else:
                burst_factor = 1.0

            # 计算原始到达率（不做硬限制）
            raw_arrival_rate = base_rate * (1 + major_cycle + minor_cycle) * burst_factor

            # 使用软边界函数替代硬限制
            # 使用sigmoid函数进行平滑限制，在边界附近仍有变化但变化变慢
            def soft_boundary_limit(x, min_val=2.0, max_val=12.0):
                # 将x映射到(-1, 1)范围
                center = (max_val + min_val) / 2
                range_half = (max_val - min_val) / 2
                normalized_x = (x - center) / (range_half * 1.2)  # 1.2是软化因子

                # 使用tanh进行软限制
                limited_normalized = np.tanh(normalized_x)

                # 映射回目标范围，允许轻微超出边界
                result = center + limited_normalized * range_half

                # 最终硬边界保护（防止极端情况）
                return max(min_val, min(result, max_val))

            if getattr(self.system_state, "scenario_profile", "") == "heterogeneous_burst_main":
                new_arrival_rate = soft_boundary_limit(raw_arrival_rate, min_val=8.0, max_val=18.0)
            else:
                new_arrival_rate = soft_boundary_limit(raw_arrival_rate)
            new_arrivals[flow_id] = new_arrival_rate

        return new_arrivals

