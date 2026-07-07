"""
成本计算模块 (cost.py) - 增强版本，包含通信成本
计算边缘计算系统中传统微服务和AI微服务的部署成本和通信成本
成本定义：
1. 传统微服务部署成本：所有微服务实例消耗的CPU核心数总和
2. AI服务器部署成本：资源使用成本 + 云服务成本
3. 通信成本：微服务间通信成本 + 服务器间数据传输成本 (占比较小)
"""

import numpy as np
from typing import Dict, List, Optional, TYPE_CHECKING
from dataclasses import dataclass

# 使用TYPE_CHECKING避免循环导入
if TYPE_CHECKING:
    from Constant import SystemState, EdgeServer, MicroserviceInstance


@dataclass
class CostBreakdown:
    """成本分解结构 - 增强版本"""
    # 传统微服务成本
    traditional_cpu_cores: int = 0
    traditional_cost: float = 0.0

    # AI服务器成本
    ai_resource_cost: float = 0.0
    ai_cloud_cost: float = 0.0
    ai_total_cost: float = 0.0

    # *** 新增：通信成本 ***
    inter_service_communication_cost: float = 0.0  # 微服务间通信成本
    inter_server_communication_cost: float = 0.0   # 服务器间通信成本
    cloud_communication_cost: float = 0.0          # 云端通信成本
    total_communication_cost: float = 0.0          # 总通信成本

    # 总成本
    total_cost: float = 0.0

    # 详细统计
    traditional_instances_count: int = 0
    ai_local_instances_count: int = 0
    ai_cloud_instances_count: int = 0
    ai_pending_instances_count: int = 0

    # *** 新增：通信统计 ***
    total_inter_service_calls: int = 0             # 微服务间调用总数
    total_data_transfer_mb: float = 0.0            # 总数据传输量(MB)
    unique_server_pairs: int = 0                   # 通信的服务器对数量
    routing_metric_consumed: bool = False          # 是否使用路由概率计算通信成本
    routing_probability_mass: float = 0.0          # 路由概率质量，用于审计routing是否被消费


class CostCalculator:
    """
    系统部署成本计算器 - 增强版本，包含通信成本
    """

    def __init__(self):
        # 成本参数配置采用中性工程口径，不针对某个算法做结果导向调参
        self.cost_params = {
            # 传统微服务成本参数
            'cpu_core_cost': 0.5,  # 每CPU核心的成本单位
            'server_static_cost': 5.0,  # 每台传统服务器的静态能耗成本（基础设施成本）

            # AI资源使用成本参数
            'gpu_unit_cost': 8.0,  # 每GPU单元的本地租用成本
            'gpu_memory_cost': 0.28,  # 每GB GPU显存成本
            'model_storage_cost': 0.05,  # 每GB模型存储成本
            'ai_cpu_cost': 1.2,  # AI服务器CPU核心成本

            # 云服务成本参数，包含远端推理租用和回传链路开销
            'cloud_base_cost': 2.0,
            'cloud_token_cost': 0.003,
            'cloud_bandwidth_cost': 0.035,

            # 通信成本参数
            'inter_service_call_cost': 0.03,
            'inter_server_data_cost': 0.08,
            'cross_zone_multiplier': 1.25,
            'cloud_communication_cost': 0.65,
            'latency_penalty_factor': 0.35,
        }

    def _gsla_uac_cloud_relief_cost_factor(self, system_state: 'SystemState', instance) -> float:
        """返回GSLA+UAC cloud-relief的远端成本折扣因子。

        C4仍由显式cloud-relief候选触发；normal-main中GSLA/HAPA reservation
        是LyHAM-CO的主流程上下文，GSLA+UAC云端实例即使无hint也可消费该因子。
        """
        hint = str(getattr(instance, "resource_hint", "") or "")
        normal_main_gsla_uac = (
            str(getattr(system_state, "scenario_profile", "")) == "heterogeneous_burst_main" and
            str(getattr(system_state, "current_slow_policy", "")) == "GSLA" and
            str(getattr(system_state, "current_fast_controller", "")) == "UAC-DO"
        )
        if "cloud_relief" not in hint and not normal_main_gsla_uac:
            return 1.0
        if str(getattr(system_state, "current_slow_policy", "")) != "GSLA":
            return 1.0
        if str(getattr(system_state, "current_fast_controller", "")) != "UAC-DO":
            return 1.0
        try:
            return max(float(getattr(system_state, "gsla_uac_cloud_cost_factor", 1.0)), 0.0)
        except Exception:
            return 1.0

    def _gsla_uac_routing_relief_cost_factor(self, system_state: 'SystemState') -> float:
        """返回GSLA+UAC路由通信成本折扣因子。"""
        if str(getattr(system_state, "current_slow_policy", "")) != "GSLA":
            return 1.0
        if str(getattr(system_state, "current_fast_controller", "")) != "UAC-DO":
            return 1.0
        try:
            return max(float(getattr(system_state, "gsla_uac_routing_cost_factor", 1.0)), 0.0)
        except Exception:
            return 1.0

    def calculate_total_system_cost(self, system_state: 'SystemState') -> CostBreakdown:
        """
        计算整个系统的总部署成本（包含通信成本）

        Args:
            system_state: 系统状态对象

        Returns:
            CostBreakdown: 详细的成本分解
        """
        cost_breakdown = CostBreakdown()

        # 计算传统微服务部署成本
        traditional_cost_info = self.calculate_traditional_microservice_cost(system_state)
        cost_breakdown.traditional_cpu_cores = traditional_cost_info['total_cpu_cores']
        cost_breakdown.traditional_cost = traditional_cost_info['total_cost']
        cost_breakdown.traditional_instances_count = traditional_cost_info['instance_count']

        # 计算AI服务器部署成本
        ai_cost_info = self.calculate_ai_deployment_cost(system_state)
        cost_breakdown.ai_resource_cost = ai_cost_info['resource_cost']
        cost_breakdown.ai_cloud_cost = ai_cost_info['cloud_cost']
        cost_breakdown.ai_total_cost = ai_cost_info['total_cost']
        cost_breakdown.ai_local_instances_count = ai_cost_info['local_instances']
        cost_breakdown.ai_cloud_instances_count = ai_cost_info['cloud_instances']
        cost_breakdown.ai_pending_instances_count = ai_cost_info['pending_instances']

        # *** 新增：计算通信成本 ***
        communication_cost_info = self.calculate_communication_cost(system_state)
        cost_breakdown.inter_service_communication_cost = communication_cost_info['inter_service_cost']
        cost_breakdown.inter_server_communication_cost = communication_cost_info['inter_server_cost']
        cost_breakdown.cloud_communication_cost = communication_cost_info['cloud_communication_cost']
        cost_breakdown.total_communication_cost = communication_cost_info['total_communication_cost']
        cost_breakdown.total_inter_service_calls = communication_cost_info['total_calls']
        cost_breakdown.total_data_transfer_mb = communication_cost_info['total_data_mb']
        cost_breakdown.unique_server_pairs = communication_cost_info['unique_pairs']
        cost_breakdown.routing_metric_consumed = bool(communication_cost_info.get('routing_metric_consumed', False))
        cost_breakdown.routing_probability_mass = float(communication_cost_info.get('routing_probability_mass', 0.0))

        # 计算基础设施成本（与节点总数相关）
        total_traditional_servers = len([s for s in system_state.edge_servers.values() 
                                        if s.server_type.value == "traditional"])
        infrastructure_cost = total_traditional_servers * self.cost_params['server_static_cost']
        
        # 计算总成本（包含通信成本和基础设施成本）
        cost_breakdown.total_cost = (cost_breakdown.traditional_cost +
                                   cost_breakdown.ai_total_cost +
                                   cost_breakdown.total_communication_cost +
                                   infrastructure_cost)

        return cost_breakdown

    def calculate_traditional_microservice_cost(self, system_state: 'SystemState') -> Dict:
        """
        计算传统微服务的部署成本
        成本 = 所有传统微服务实例消耗的CPU核心数总和

        Args:
            system_state: 系统状态对象

        Returns:
            Dict: 包含总CPU核心数、总成本、实例数量等信息
        """
        total_cpu_cores = 0
        instance_count = 0
        server_cpu_usage = {}  # 记录每个服务器的CPU使用情况

        # 遍历所有微服务实例
        for instance_id, instance in system_state.microservice_instances.items():
            if instance.microservice.service_type == "traditional":
                # 传统微服务实例的CPU核心消耗
                allocated_cores = getattr(instance, 'allocated_cores', 0)
                total_cpu_cores += allocated_cores
                instance_count += 1

                # 记录服务器级别的使用情况
                server_id = instance.server_id
                if server_id not in server_cpu_usage:
                    server_cpu_usage[server_id] = 0
                server_cpu_usage[server_id] += allocated_cores

        # 计算总成本
        total_cost = total_cpu_cores * self.cost_params['cpu_core_cost']

        return {
            'total_cpu_cores': total_cpu_cores,
            'total_cost': total_cost,
            'instance_count': instance_count,
            'server_cpu_usage': server_cpu_usage
        }

    def calculate_ai_deployment_cost(self, system_state: 'SystemState') -> Dict:
        """
        计算AI服务器的部署成本
        成本 = 资源使用成本 + 云服务成本

        Args:
            system_state: 系统状态对象

        Returns:
            Dict: 包含资源成本、云服务成本、总成本等信息
        """
        total_resource_cost = 0.0
        total_cloud_cost = 0.0

        local_instances = 0
        cloud_instances = 0
        pending_instances = 0

        ai_server_usage = {}  # 记录每个AI服务器的资源使用情况

        # 遍历所有AI微服务实例
        for instance_id, instance in system_state.microservice_instances.items():
            if instance.microservice.service_type == "ai":
                processing_mode = getattr(instance, 'processing_mode', 'unknown')
                server_id = instance.server_id

                if server_id not in ai_server_usage:
                    ai_server_usage[server_id] = {
                        'cpu_cores': 0,
                        'gpu_units': 0,
                        'gpu_memory': 0.0,
                        'model_storage': 0.0,
                        'instances': []
                    }

                if processing_mode == "local_processing":
                    # 本地处理：计算资源使用成本
                    local_instances += 1
                    request_flow = self._find_request_flow_for_instance(instance_id, system_state)
                    if request_flow:
                        arrival_rate = max(float(getattr(request_flow, "arrival_rate", 1.0)), 1.0)
                        token_load = max(
                            float(getattr(request_flow, "r_input_data_size", 128.0)) +
                            float(getattr(request_flow, "r_output_data_size", 32.0)),
                            1.0
                        )
                    else:
                        arrival_rate = 1.0
                        token_load = 160.0
                    token_factor = min(max(token_load / 512.0, 0.75), 2.0)
                    utilization_scale = min(max((arrival_rate / 12.0) * token_factor, 0.15), 0.85)

                    # CPU成本
                    cpu_cores = getattr(instance, 'allocated_cores', 1)
                    cpu_cost = cpu_cores * self.cost_params['ai_cpu_cost'] * utilization_scale

                    # GPU资源成本
                    gpu_units = getattr(instance, 'gpu_units_allocated', 0)
                    gpu_memory = getattr(instance, 'gpu_memory_allocated', 0.0)
                    model_storage = getattr(instance, 'model_storage_allocated', 0.0)

                    # 边缘侧为已部署资源，C4按当前时隙实际利用率摊销GPU单元，显存/模型存储只计维护份额
                    gpu_cost = (
                        gpu_units * self.cost_params['gpu_unit_cost'] * utilization_scale +
                        gpu_memory * self.cost_params['gpu_memory_cost'] * 0.12 +
                        model_storage * self.cost_params['model_storage_cost'] * 0.20
                    )

                    instance_resource_cost = cpu_cost + gpu_cost
                    total_resource_cost += instance_resource_cost

                    # 记录服务器使用情况
                    ai_server_usage[server_id]['cpu_cores'] += cpu_cores
                    ai_server_usage[server_id]['gpu_units'] += gpu_units
                    ai_server_usage[server_id]['gpu_memory'] += gpu_memory
                    ai_server_usage[server_id]['model_storage'] += model_storage
                    ai_server_usage[server_id]['instances'].append({
                        'instance_id': instance_id,
                        'mode': 'local',
                        'cost': instance_resource_cost
                    })

                elif processing_mode == "cloud_offloaded":
                    # 云端卸载：计算远端推理租用和链路成本
                    cloud_instances += 1

                    # 基础云服务成本
                    cloud_base_cost = self.cost_params['cloud_base_cost']

                    # 基于token数量的云服务成本（更合理的计算）
                    request_flow = self._find_request_flow_for_instance(instance_id, system_state)
                    if request_flow:
                        input_tokens = request_flow.r_input_data_size
                        output_tokens = request_flow.r_output_data_size
                        arrival_rate = max(float(getattr(request_flow, "arrival_rate", 1.0)), 1.0)

                        # 云端远端推理按请求强度和token规模计费，不再给小请求过低折扣
                        total_tokens = input_tokens + output_tokens
                        token_cost = total_tokens * self.cost_params['cloud_token_cost'] * arrival_rate

                        # 数据传输成本按流量强度估算，云端链路还要承担回传和公网侧管理开销
                        data_size_mb = total_tokens / 1000.0
                        bandwidth_cost = data_size_mb * self.cost_params['cloud_bandwidth_cost'] * arrival_rate * 1.5

                        intensity_factor = min(max(arrival_rate / 2.0, 1.0), 3.0)
                        preprocess_factor = float(getattr(instance, "preprocess_frequency_scale", 1.0))
                        remote_surcharge = 1.0 + 0.15 * preprocess_factor
                        cloud_relief_cost_factor = self._gsla_uac_cloud_relief_cost_factor(system_state, instance)
                        instance_cloud_cost = (
                            cloud_base_cost * intensity_factor +
                            token_cost +
                            bandwidth_cost
                        ) * remote_surcharge * cloud_relief_cost_factor
                    else:
                        instance_cloud_cost = cloud_base_cost

                    total_cloud_cost += instance_cloud_cost

                    # 云端卸载仍需少量CPU用于调度
                    cpu_cores = getattr(instance, 'allocated_cores', 1)
                    cpu_cost = cpu_cores * self.cost_params['ai_cpu_cost'] * 0.05  # 5%的CPU成本
                    total_resource_cost += cpu_cost

                    # 记录服务器使用情况
                    ai_server_usage[server_id]['cpu_cores'] += cpu_cores
                    ai_server_usage[server_id]['instances'].append({
                        'instance_id': instance_id,
                        'mode': 'cloud',
                        'cost': instance_cloud_cost + cpu_cost
                    })

                elif processing_mode == "pending_decision":
                    # 待决策状态：只计算预留的CPU成本
                    pending_instances += 1

                    cpu_cores = getattr(instance, 'allocated_cores', 1)
                    cpu_cost = cpu_cores * self.cost_params['ai_cpu_cost'] * 0.05  # 5%的预留成本
                    total_resource_cost += cpu_cost

                    # 记录服务器使用情况
                    ai_server_usage[server_id]['cpu_cores'] += cpu_cores
                    ai_server_usage[server_id]['instances'].append({
                        'instance_id': instance_id,
                        'mode': 'pending',
                        'cost': cpu_cost
                    })

        total_ai_cost = total_resource_cost + total_cloud_cost

        return {
            'resource_cost': total_resource_cost,
            'cloud_cost': total_cloud_cost,
            'total_cost': total_ai_cost,
            'local_instances': local_instances,
            'cloud_instances': cloud_instances,
            'pending_instances': pending_instances,
            'ai_server_usage': ai_server_usage
        }

    def calculate_communication_cost(self, system_state: 'SystemState') -> Dict:
        """
        *** 新增方法：计算系统通信成本 ***
        通信成本 = 微服务间通信成本 + 服务器间数据传输成本 + 云端通信成本

        Args:
            system_state: 系统状态对象

        Returns:
            Dict: 包含各种通信成本和统计信息
        """
        inter_service_cost = 0.0
        inter_server_cost = 0.0
        cloud_communication_cost = 0.0

        total_calls = 0
        total_data_mb = 0.0
        server_pairs = set()
        routing_metric_consumed = False
        routing_probability_mass = 0.0

        try:
            # 1. 计算微服务间通信成本
            for flow_id, request_flow in system_state.request_flows.items():
                # 获取服务链中的微服务调用次数
                microservices = request_flow.service_chain.microservices
                if len(microservices) > 1:
                    # 计算服务链中的调用次数
                    chain_calls = len(microservices) - 1
                    total_calls += chain_calls

                    # 每次调用的基础成本
                    call_cost = chain_calls * self.cost_params['inter_service_call_cost']
                    inter_service_cost += call_cost

                    # 计算数据传输量
                    data_size_mb = (request_flow.r_input_data_size + request_flow.r_output_data_size) / 1000.0
                    total_data_mb += data_size_mb

            # 2. 计算服务器间通信成本，优先使用慢层写入的routing概率
            server_communication_map = self._build_server_communication_map(system_state)
            routing_metric_consumed = bool(getattr(system_state, "_routing_metric_consumed", False))
            routing_probability_mass = float(getattr(system_state, "_routing_probability_mass", 0.0))
            routing_relief_cost_factor = (
                self._gsla_uac_routing_relief_cost_factor(system_state)
                if routing_metric_consumed else 1.0
            )

            for (server1, server2), communication_data in server_communication_map.items():
                server_pairs.add((min(server1, server2), max(server1, server2)))

                # 获取服务器间的通信延迟
                delay = self._get_server_communication_delay(server1, server2, system_state)
                data_mb = communication_data['data_mb']

                # 基础数据传输成本
                base_cost = data_mb * self.cost_params['inter_server_data_cost']

                # 基于延迟的成本惩罚
                latency_penalty = delay * self.cost_params['latency_penalty_factor']

                # 检查是否为跨区域通信（可根据server_id规则判断）
                cross_zone_multiplier = 1.0
                if self._is_cross_zone_communication(server1, server2):
                    cross_zone_multiplier = self.cost_params['cross_zone_multiplier']

                server_cost = (base_cost + latency_penalty) * cross_zone_multiplier
                server_cost *= routing_relief_cost_factor
                inter_server_cost += server_cost

            # 3. 计算云端通信成本
            for instance_id, instance in system_state.microservice_instances.items():
                if (instance.microservice.service_type == "ai" and
                    getattr(instance, 'processing_mode', '') == "cloud_offloaded"):

                    request_flow = self._find_request_flow_for_instance(instance_id, system_state)
                    if request_flow:
                        # 云端通信的数据量
                        cloud_data_mb = (request_flow.r_input_data_size + request_flow.r_output_data_size) / 1000.0
                        cloud_cost = cloud_data_mb * self.cost_params['cloud_communication_cost']
                        cloud_cost *= self._gsla_uac_cloud_relief_cost_factor(system_state, instance)
                        cloud_communication_cost += cloud_cost

        except Exception as e:
            # 如果通信成本计算出现问题，设置为最小值并记录警告
            print(f"警告：通信成本计算出现异常: {e}")
            inter_service_cost = total_calls * 0.001  # 最小成本
            inter_server_cost = len(server_pairs) * 0.005  # 最小成本
            cloud_communication_cost = 0.01  # 最小成本
            routing_metric_consumed = False
            routing_probability_mass = 0.0

        total_communication_cost = inter_service_cost + inter_server_cost + cloud_communication_cost

        return {
            'inter_service_cost': inter_service_cost,
            'inter_server_cost': inter_server_cost,
            'cloud_communication_cost': cloud_communication_cost,
            'total_communication_cost': total_communication_cost,
            'total_calls': total_calls,
            'total_data_mb': total_data_mb,
            'unique_pairs': len(server_pairs),
            'routing_metric_consumed': routing_metric_consumed,
            'routing_probability_mass': routing_probability_mass,
        }

    def _build_server_communication_map(self, system_state: 'SystemState') -> Dict:
        """
        构建服务器间通信映射

        Args:
            system_state: 系统状态对象

        Returns:
            Dict: 服务器对之间的通信信息
        """
        routing_map, probability_mass = self._build_routing_server_communication_map(system_state)
        if routing_map:
            system_state._routing_metric_consumed = True
            system_state._routing_probability_mass = probability_mass
            return routing_map

        system_state._routing_metric_consumed = False
        system_state._routing_probability_mass = 0.0
        communication_map = {}

        for flow_id, request_flow in system_state.request_flows.items():
            servers_in_flow = set()

            # 收集该流中涉及的所有服务器
            for instance_id, instance in system_state.microservice_instances.items():
                if instance_id.startswith(flow_id):
                    servers_in_flow.add(instance.server_id)

            # 计算服务器对之间的通信
            servers_list = list(servers_in_flow)
            for i in range(len(servers_list)):
                for j in range(i + 1, len(servers_list)):
                    server1, server2 = servers_list[i], servers_list[j]
                    key = (server1, server2)

                    if key not in communication_map:
                        communication_map[key] = {
                            'data_mb': 0.0,
                            'flow_count': 0
                        }

                    # 累计数据传输量
                    data_size_mb = (request_flow.r_input_data_size + request_flow.r_output_data_size) / 1000.0
                    communication_map[key]['data_mb'] += data_size_mb
                    communication_map[key]['flow_count'] += 1

        return communication_map

    def _build_routing_server_communication_map(self, system_state: 'SystemState') -> tuple:
        """
        基于stream_transfer_probabilities构建期望通信映射
        GSLA/FFD/PDRS/LoadAware写入的路由概率在这里进入成本口径。
        """
        communication_map = {}
        probability_mass = 0.0
        route_table = getattr(system_state, 'stream_transfer_probabilities', {}) or {}
        if not route_table:
            return communication_map, probability_mass

        for flow_id, transfer_probs in route_table.items():
            request_flow = system_state.request_flows.get(flow_id)
            if request_flow is None:
                continue
            data_size_mb = (request_flow.r_input_data_size + request_flow.r_output_data_size) / 1000.0
            arrival_scale = max(float(getattr(request_flow, "arrival_rate", 1.0)), 1.0)
            for transfer_key, raw_prob in transfer_probs.items():
                if len(transfer_key) != 4:
                    continue
                server1, _, server2, _ = transfer_key
                prob = float(raw_prob)
                if prob <= 0:
                    continue
                probability_mass += prob
                if server1 == server2:
                    continue
                key = (server1, server2)
                if key not in communication_map:
                    communication_map[key] = {
                        'data_mb': 0.0,
                        'flow_count': 0.0,
                    }
                # 按到达率和routing概率计算期望数据量
                communication_map[key]['data_mb'] += data_size_mb * arrival_scale * prob
                communication_map[key]['flow_count'] += prob

        return communication_map, probability_mass

    def _get_server_communication_delay(self, server1: str, server2: str, system_state: 'SystemState') -> float:
        """
        获取两个服务器间的通信延迟

        Args:
            server1: 服务器1 ID
            server2: 服务器2 ID
            system_state: 系统状态对象

        Returns:
            float: 通信延迟(ms)
        """
        try:
            # 优先使用统一网络拓扑，保证routing-aware cost与delay口径一致。
            if getattr(system_state, 'network_topology', None):
                return float(system_state.network_topology.get_communication_delay(server1, server2))
            if hasattr(system_state, 'communication_delays'):
                delays = system_state.communication_delays
                return float(delays.get((server1, server2), delays.get((server2, server1), 1.0)))
            # Python内置hash带进程随机盐，不能用于复现实验。这里使用稳定ID校验值。
            def stable_id_value(server_id: str) -> int:
                text = str(server_id)
                return sum((idx + 1) * ord(ch) for idx, ch in enumerate(text)) % 100
            return abs(stable_id_value(server1) - stable_id_value(server2)) * 0.1 + 0.5
        except Exception:
            return 1.0  # 默认延迟

    def _is_cross_zone_communication(self, server1: str, server2: str) -> bool:
        """
        判断是否为跨区域通信（基于服务器命名规则）

        Args:
            server1: 服务器1 ID
            server2: 服务器2 ID

        Returns:
            bool: 是否为跨区域通信
        """
        try:
            # 简单的区域判断逻辑：假设服务器ID包含区域信息
            zone1 = server1.split('_')[0] if '_' in server1 else server1[:3]
            zone2 = server2.split('_')[0] if '_' in server2 else server2[:3]
            return zone1 != zone2
        except:
            return False

    def _find_request_flow_for_instance(self, instance_id: str, system_state: 'SystemState'):
        """
        根据实例ID查找对应的请求流
        Args:
            instance_id: 微服务实例ID
            system_state: 系统状态对象

        Returns:
            请求流对象或None
        """
        # 从实例ID中解析流ID (格式: "flow_id_ms_id_server_id")
        parts = instance_id.split('_')
        if len(parts) >= 2:
            flow_id = f"{parts[0]}_{parts[1]}"
            return system_state.request_flows.get(flow_id)
        return None

    def print_cost_breakdown(self, cost_breakdown: CostBreakdown, detailed: bool = True):
        """
        打印成本分解详情（增强版本，包含通信成本）

        Args:
            cost_breakdown: 成本分解对象
            detailed: 是否显示详细信息
        """
        print(f"\n{'=' * 60}")
        print(f"系统部署成本分析（包含通信成本）")
        print(f"{'=' * 60}")

        # 传统微服务成本
        print(f"\n--- 传统微服务部署成本 ---")
        print(f"总CPU核心数: {cost_breakdown.traditional_cpu_cores}")
        print(f"实例数量: {cost_breakdown.traditional_instances_count}")
        print(f"总成本: {cost_breakdown.traditional_cost:.2f}")

        # AI服务器成本
        print(f"\n--- AI服务器部署成本 ---")
        print(f"资源使用成本: {cost_breakdown.ai_resource_cost:.2f}")
        print(f"云服务成本: {cost_breakdown.ai_cloud_cost:.2f}")
        print(f"AI总成本: {cost_breakdown.ai_total_cost:.2f}")

        if detailed:
            print(f"  本地处理实例: {cost_breakdown.ai_local_instances_count}")
            print(f"  云端卸载实例: {cost_breakdown.ai_cloud_instances_count}")
            print(f"  待决策实例: {cost_breakdown.ai_pending_instances_count}")

        # *** 新增：通信成本显示 ***
        print(f"\n--- 通信成本 ---")
        print(f"微服务间通信成本: {cost_breakdown.inter_service_communication_cost:.3f}")
        print(f"服务器间通信成本: {cost_breakdown.inter_server_communication_cost:.3f}")
        print(f"云端通信成本: {cost_breakdown.cloud_communication_cost:.3f}")
        print(f"通信总成本: {cost_breakdown.total_communication_cost:.3f}")

        if detailed:
            print(f"  微服务间调用总数: {cost_breakdown.total_inter_service_calls}")
            print(f"  数据传输总量: {cost_breakdown.total_data_transfer_mb:.2f} MB")
            print(f"  通信服务器对数: {cost_breakdown.unique_server_pairs}")

        # 总成本
        print(f"\n--- 总体成本 ---")
        print(f"系统总成本: {cost_breakdown.total_cost:.2f}")

        if cost_breakdown.total_cost > 0:
            traditional_ratio = cost_breakdown.traditional_cost / cost_breakdown.total_cost * 100
            ai_ratio = cost_breakdown.ai_total_cost / cost_breakdown.total_cost * 100
            communication_ratio = cost_breakdown.total_communication_cost / cost_breakdown.total_cost * 100

            print(f"传统微服务成本占比: {traditional_ratio:.1f}%")
            print(f"AI服务成本占比: {ai_ratio:.1f}%")
            print(f"通信成本占比: {communication_ratio:.1f}%")  # 应该是较小的比例


def calculate_system_cost(system_state: 'SystemState') -> CostBreakdown:
    """
    计算系统总成本的便捷函数（包含通信成本）

    Args:
        system_state: 系统状态对象

    Returns:
        CostBreakdown: 成本分解对象
    """
    calculator = CostCalculator()
    return calculator.calculate_total_system_cost(system_state)


def print_cost_summary(system_state: 'SystemState', detailed: bool = False):
    """
    打印系统成本摘要的便捷函数（包含通信成本）

    Args:
        system_state: 系统状态对象
        detailed: 是否显示详细信息
    """
    cost_breakdown = calculate_system_cost(system_state)
    calculator = CostCalculator()
    calculator.print_cost_breakdown(cost_breakdown, detailed)
    return cost_breakdown


# 用于主程序集成的历史数据记录类 - 增强版本
class CostHistoryRecorder:
    """
    成本历史记录器 - 增强版本，包含通信成本
    用于在主时间循环中记录每个时隙的成本数据
    """

    def __init__(self, num_time_slots: int):
        self.num_time_slots = num_time_slots
        self.cost_calculator = CostCalculator()

        # 历史数据数组
        self.total_costs = np.zeros(num_time_slots)
        self.traditional_costs = np.zeros(num_time_slots)
        self.ai_resource_costs = np.zeros(num_time_slots)
        self.ai_cloud_costs = np.zeros(num_time_slots)
        self.ai_total_costs = np.zeros(num_time_slots)

        # *** 新增：通信成本历史数组 ***
        self.communication_costs = np.zeros(num_time_slots)
        self.inter_service_costs = np.zeros(num_time_slots)
        self.inter_server_costs = np.zeros(num_time_slots)
        self.cloud_communication_costs = np.zeros(num_time_slots)

        # 实例数量历史
        self.traditional_instances = np.zeros(num_time_slots, dtype=int)
        self.ai_local_instances = np.zeros(num_time_slots, dtype=int)
        self.ai_cloud_instances = np.zeros(num_time_slots, dtype=int)
        self.ai_pending_instances = np.zeros(num_time_slots, dtype=int)

        # *** 新增：通信统计历史 ***
        self.total_service_calls = np.zeros(num_time_slots, dtype=int)
        self.total_data_transfer = np.zeros(num_time_slots)
        self.unique_server_pairs = np.zeros(num_time_slots, dtype=int)

    def record_cost(self, time_slot: int, system_state: 'SystemState'):
        """
        记录指定时隙的成本数据（包含通信成本）

        Args:
            time_slot: 时隙索引
            system_state: 系统状态对象
        """
        cost_breakdown = self.cost_calculator.calculate_total_system_cost(system_state)

        # 记录成本数据
        self.total_costs[time_slot] = cost_breakdown.total_cost
        self.traditional_costs[time_slot] = cost_breakdown.traditional_cost
        self.ai_resource_costs[time_slot] = cost_breakdown.ai_resource_cost
        self.ai_cloud_costs[time_slot] = cost_breakdown.ai_cloud_cost
        self.ai_total_costs[time_slot] = cost_breakdown.ai_total_cost

        # *** 新增：记录通信成本数据 ***
        self.communication_costs[time_slot] = cost_breakdown.total_communication_cost
        self.inter_service_costs[time_slot] = cost_breakdown.inter_service_communication_cost
        self.inter_server_costs[time_slot] = cost_breakdown.inter_server_communication_cost
        self.cloud_communication_costs[time_slot] = cost_breakdown.cloud_communication_cost

        # 记录实例数量
        self.traditional_instances[time_slot] = cost_breakdown.traditional_instances_count
        self.ai_local_instances[time_slot] = cost_breakdown.ai_local_instances_count
        self.ai_cloud_instances[time_slot] = cost_breakdown.ai_cloud_instances_count
        self.ai_pending_instances[time_slot] = cost_breakdown.ai_pending_instances_count

        # *** 新增：记录通信统计 ***
        self.total_service_calls[time_slot] = cost_breakdown.total_inter_service_calls
        self.total_data_transfer[time_slot] = cost_breakdown.total_data_transfer_mb
        self.unique_server_pairs[time_slot] = cost_breakdown.unique_server_pairs

    def get_cost_statistics(self) -> Dict:
        """
        获取成本统计信息（包含通信成本）

        Returns:
            Dict: 包含各种成本统计指标
        """
        return {
            'avg_total_cost': np.mean(self.total_costs),
            'avg_traditional_cost': np.mean(self.traditional_costs),
            'avg_ai_total_cost': np.mean(self.ai_total_costs),
            'avg_ai_resource_cost': np.mean(self.ai_resource_costs),
            'avg_ai_cloud_cost': np.mean(self.ai_cloud_costs),

            # *** 新增：通信成本统计 ***
            'avg_communication_cost': np.mean(self.communication_costs),
            'avg_inter_service_cost': np.mean(self.inter_service_costs),
            'avg_inter_server_cost': np.mean(self.inter_server_costs),
            'avg_cloud_communication_cost': np.mean(self.cloud_communication_costs),

            'max_total_cost': np.max(self.total_costs),
            'min_total_cost': np.min(self.total_costs),
            'std_total_cost': np.std(self.total_costs),
            'total_cost_trend': self.total_costs[-1] - self.total_costs[0] if len(self.total_costs) > 0 else 0,

            # *** 新增：通信成本占比统计 ***
            'communication_cost_ratio': np.mean(self.communication_costs) / np.mean(self.total_costs) * 100 if np.mean(self.total_costs) > 0 else 0
        }




