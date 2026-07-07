"""
AI微服务计算卸载的资源分配算法
模仿LyDROO的Algo1_NUM函数逻辑，针对AI微服务场景进行适配

主要功能：
1. 根据卸载决策（二进制向量）为AI服务器分配处理模式
2. 计算本地处理和云端卸载的能耗、延迟
3. 基于Lyapunov优化框架计算目标函数值
4. 更新系统状态和微服务实例

输入：卸载模式、AI服务器状态、虚拟队列状态、Lyapunov参数
输出：目标函数值、延迟向量、能耗向量
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING
import copy
import hashlib
import random
import time

if TYPE_CHECKING:
    from Constant import SystemState, EdgeServer, MicroserviceInstance


UAC_SELECTED_SOURCE_TOKENS = (
    "actor", "uac", "energy_low_dvfs", "energy_cloud_relief",
    "energy_repair", "cost_aware_hybrid", "delay_energy_frontier",
    "replica_ready_local", "queue_pressure_flip",
    "queue_balanced_frontier", "reference_low_impact_neighbor",
    "reference_low_impact_swap", "pair_flip", "hybrid",
)


def is_uac_selected_source(selected_candidate_source: str) -> bool:
    """判断已选候选是否来自UAC快层候选族。"""
    source = str(selected_candidate_source or "")
    if not source or "myopic_seed" in source or source == "myopic_reference_repaired":
        return False
    return any(token in source for token in UAC_SELECTED_SOURCE_TOKENS)


def AI_Offloading_Resource_Allocation(
        offloading_mode: np.ndarray,  # N维二进制向量: 0=本地处理, 1=云端卸载
        SH: np.ndarray,  # AI服务器性能环境因子向量
        SQ: np.ndarray,  # 虚拟能耗队列状态向量
        SZ: np.ndarray,  # 虚拟延迟队列状态向量
        system_state: 'SystemState',  # 系统状态对象
        V: float = 2.0,  # Lyapunov控制参数
        weights: Optional[np.ndarray] = None,  # 权重向量
        ai_instance_cache: Optional[Dict] = None,  # 新增AI实例缓存参数
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    AI微服务计算卸载的资源分配算法
    类比LyDROO的Algo1_NUM，但适配AI微服务场景

    Args:
        offloading_mode: N维二进制决策向量 (0=本地, 1=云端)
        SH: AI服务器性能环境因子向量 [N]
        SQ: 虚拟能耗队列状态向量 [N]
        SZ: 虚拟延迟队列状态向量 [N]
        system_state: 系统状态对象
        V: Lyapunov控制参数
        weights: 权重向量，如果为None则使用默认权重
        ai_instance_cache: AI实例缓存 {server_id: (ai_instance, request_flow)}  # 【修改4】

    Returns:
        Tuple[float, np.ndarray, np.ndarray]: (目标函数值, 延迟向量, 能耗向量)
    """

    # 获取AI服务器列表
    ai_servers = [server for server in system_state.edge_servers.values()
                  if server.server_type.value == "ai_capable"]
    ai_server_ids = sorted([server.server_id for server in ai_servers])
    N = len(ai_server_ids)

    if len(offloading_mode) != N:
        raise ValueError(f"卸载决策维度({len(offloading_mode)}) != AI服务器数量({N})")

    # 初始化权重向量
    if weights is None:
        weights = np.ones(N)  # 默认所有AI服务器权重为1

    # 初始化结果向量
    ai_delays = np.zeros(N)  # AI服务器延迟向量
    ai_energies = np.zeros(N)  # AI服务器能耗向量
    ai_rewards = np.zeros(N)  # AI服务器奖励向量
    '''
    print(f"\n--- AI资源分配算法执行---")
    print(f"卸载决策: {offloading_mode}")
    
    print(f"性能因子(SH): {SH}")
    print(f"能耗队列(SQ): {SQ}")
    print(f"延迟队列(SZ): {SZ}")
    '''
    # 处理每个AI服务器的卸载决策
    for i, server_id in enumerate(ai_server_ids):
        original_decision = offloading_mode[i]  # 原始决策
        actual_decision = original_decision  # 实际执行的决策

        server = system_state.edge_servers[server_id]

        #print(f"\n处理AI服务器 {server_id} (原始决策: {'云端卸载' if original_decision == 1 else '本地处理'}):")

        # 使用缓存的AI实例信息，避免重复查找
        if ai_instance_cache and server_id in ai_instance_cache:
            cached_value = ai_instance_cache[server_id]
            if isinstance(cached_value, list):
                ai_pairs = cached_value
            else:
                ai_pairs = [cached_value]
            #print(f"  使用缓存的实例信息 ✓")
        else:
            # 回退到实时查找方式
            ai_pairs = find_ai_instances_and_flows(server_id, system_state)
            print(f"  回退到实时查找 (缓存未命中)")

        ai_pairs = [
            (ai_instance, request_flow)
            for ai_instance, request_flow in ai_pairs
            if ai_instance is not None and request_flow is not None
        ]
        if not ai_pairs:
            print(f"  未找到AI服务器 {server_id} 上的AI微服务实例")
            continue

        # 资源充足性检查和自动转换
        if original_decision == 0:  # 如果原始决策是本地处理
            if not check_local_resource_sufficiency(server_id, system_state):
                actual_decision = 1  # 资源不足，强制转为云端卸载
                print(f"  ⚠️  本地资源不足，自动转换为云端卸载")
            #else:
                #print(f"  ✓ 本地资源充足，执行本地处理")

        instance_delays = []
        instance_energies = []
        for ai_instance, request_flow in ai_pairs:
            if actual_decision == 0:  # 本地处理
                delay, energy = process_local_ai_decision(
                    ai_instance, request_flow, server, system_state, SH[i], SQ[i], SZ[i], V)
                ai_instance.processing_mode = "local_processing"
                ai_instance.inference_latency = delay
                ai_instance.cloud_latency = 0.0
            else:  # 云端卸载
                delay, energy = process_cloud_offloading_decision(
                    ai_instance, request_flow, server, system_state, SH[i], SQ[i], SZ[i], V)
                ai_instance.processing_mode = "cloud_offloaded"
                ai_instance.inference_latency = 0.0
                ai_instance.cloud_latency = delay
            instance_delays.append(float(delay))
            instance_energies.append(float(energy))

        ai_delays[i] = float(np.mean(instance_delays)) if instance_delays else 0.0
        ai_energies[i] = float(np.sum(instance_energies)) if instance_energies else 0.0

        # 获取基础参数
        arrival_rate = sum(request_flow.arrival_rate for _, request_flow in ai_pairs) if ai_pairs else 1.0

        # 约束阈值
        energy_threshold = server.energy_threshold
        delay_threshold = server.delay_threshold

        if actual_decision == 0:  # 本地处理
            service_reward = arrival_rate * weights[i] * 0.05
        else:  # 云端卸载
            service_reward = arrival_rate * weights[i] * 0.05

        # 计算相对违反程度（标准化到0-1）
        energy_violation_ratio = max(0.0, (ai_energies[i] - energy_threshold) / energy_threshold)
        delay_violation_ratio = max(0.0, (ai_delays[i] - delay_threshold) / delay_threshold)

        # 3. Lyapunov控制项（保持队列稳定性）
        energy_penalty = V * SQ[i] * energy_violation_ratio
        delay_penalty = V * SZ[i] * delay_violation_ratio  * 1.2
        lyapunov_penalty = energy_penalty + delay_penalty

        # 最终奖励：基础奖励 + 服务奖励 - Lyapunov惩罚
        total_reward = 10 + service_reward - lyapunov_penalty

        ai_rewards[i] = total_reward

        # 简化输出信息
        '''
        print(f"  {server_id}: {'本地' if actual_decision == 0 else '云端'} | "
              f"延迟={delay:.1f}ms | 能耗={energy:.4f}J | 奖励={total_reward:.3f} | 奖励构成: 基础= 10 + 服务={service_reward:.2f} - Lyapunov惩罚={lyapunov_penalty:.2f} "
              f"| (能耗={energy_penalty:.2f}+延迟={delay_penalty:.2f})")
        '''

    total_objective = np.sum(ai_rewards)
    print(f"总目标函数值: {total_objective:.3f}")

    return total_objective, ai_delays, ai_energies


def _extract_flow_id_from_instance_id(instance_id: str, system_state: 'SystemState') -> Optional[str]:
    """从实例ID中恢复请求流ID，兼容多实例后缀"""
    for flow_id in sorted(system_state.request_flows.keys(), key=len, reverse=True):
        if instance_id == flow_id or instance_id.startswith(f"{flow_id}_"):
            return flow_id
    parts = instance_id.split('_')
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return None


def find_ai_instances_and_flows(server_id: str, system_state: 'SystemState') -> List[
    Tuple['MicroserviceInstance', 'RequestFlow']]:
    """
    查找指定AI服务器上的全部AI微服务实例和请求流
    HAPA允许单节点多AI实例，快层评估必须聚合同一节点上的实例。
    """
    pairs = []
    for instance_id, instance in system_state.microservice_instances.items():
        if (instance.server_id == server_id and
                instance.microservice.service_type == "ai"):
            flow_id = _extract_flow_id_from_instance_id(instance_id, system_state)
            request_flow = system_state.request_flows.get(flow_id) if flow_id else None
            if request_flow is not None:
                pairs.append((instance, request_flow))
    return pairs


def find_ai_instance_and_flow(server_id: str, system_state: 'SystemState') -> Tuple[
    Optional['MicroserviceInstance'], Optional['RequestFlow']]:
    """
    查找部署在指定AI服务器上的AI微服务实例和对应的请求流
    """
    pairs = find_ai_instances_and_flows(server_id, system_state)
    return pairs[0] if pairs else (None, None)


def _hash_action_bits(bits) -> str:
    """生成动作位串哈希，便于跨CSV追踪"""
    arr = np.asarray(bits, dtype=int).reshape(-1)
    payload = "".join(str(int(v)) for v in arr)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _pair_bits_to_text(bits) -> str:
    """导出pair动作位串"""
    return "".join(str(int(v)) for v in np.asarray(bits, dtype=int).reshape(-1))


def _pair_repair_diagnostics(original_bits, repaired_bits) -> Dict[str, float]:
    """统计pair动作被资源包络修复的幅度"""
    original = np.asarray(original_bits, dtype=int).reshape(-1)
    repaired = np.asarray(repaired_bits, dtype=int).reshape(-1)
    pair_count = min(len(original), len(repaired))
    if pair_count <= 0:
        return {
            "original_pair_action_hash": _hash_action_bits(original),
            "repair_changed_pair_count": 0,
            "repair_changed_ratio": 0.0,
        }
    changed = int(np.sum(original[:pair_count] != repaired[:pair_count]))
    return {
        "original_pair_action_hash": _hash_action_bits(original),
        "repair_changed_pair_count": int(changed),
        "repair_changed_ratio": float(changed / max(pair_count, 1)),
    }


def mark_resource_hint_collapse(resource_hint: str, requested_signature: Dict,
                                executable_signature: Dict) -> bool:
    """判断资源偏好是否在修复后坍缩。

    UAC候选会携带 energy/cost/delay 等资源偏好。若 dry-run 后可执行资源配置
    与候选要求完全一致，说明该偏好没有形成新的资源选择差异，需要在raw中标记。
    """
    if not resource_hint:
        return False
    requested = requested_signature or {}
    executable = executable_signature or {}
    if not requested or not executable:
        return False
    keys = set(requested.keys()) | set(executable.keys())
    for key in keys:
        left = requested.get(key)
        right = executable.get(key)
        try:
            if abs(float(left) - float(right)) > 1e-12:
                return False
        except Exception:
            if left != right:
                return False
    return True


def _resolve_pair_item(item: Dict, system_state: 'SystemState'):
    """根据pair universe中的记录恢复实例、流和服务器"""
    instance = system_state.microservice_instances.get(item.get("instance_id", ""))
    flow_id = item.get("flow_id") or (
        _extract_flow_id_from_instance_id(instance.instance_id, system_state)
        if instance is not None else None
    )
    request_flow = system_state.request_flows.get(flow_id) if flow_id else None
    server = system_state.edge_servers.get(item.get("server_id", ""))
    if instance is None and flow_id and server is not None:
        ms_id = item.get("microservice_id", "")
        for candidate_id, candidate in system_state.microservice_instances.items():
            if (candidate.server_id == server.server_id and
                    candidate.microservice.service_type == "ai" and
                    candidate.microservice.ms_id == ms_id and
                    _extract_flow_id_from_instance_id(candidate_id, system_state) == flow_id):
                instance = candidate
                break
    return instance, request_flow, server


def _pair_hapa_features(item: Dict, system_state: 'SystemState') -> Dict[str, float]:
    """
    读取HAPA慢层反馈
    默认值为中性，避免没有HAPA profile的强baseline被额外惩罚。
    """
    context = getattr(system_state, "gsla_context", {}) or {}
    profile = context.get("hapa_profile", {}) or {}
    flow_id = item.get("flow_id", "")
    server_id = item.get("server_id", "")
    instance_id = item.get("instance_id", "")
    ms_id = item.get("microservice_id", "")
    profile_item = profile.get(instance_id)
    if profile_item is None:
        for candidate in profile.values():
            if (candidate.get("flow_id") == flow_id and
                    candidate.get("server_id") == server_id and
                    candidate.get("microservice") == ms_id):
                profile_item = candidate
                break
    if profile_item is None:
        return {
            "has_hapa_feedback": 0.0,
            "coverage_ratio": 1.0,
            "replica_readiness": 1.0,
            "hapa_psi": 1.0,
            "hapa_d_loc": 1.0,
            "reservation_pressure": 0.0,
        }

    covering = context.get("hapa_demand_covering", {}) or {}
    coverage_item = covering.get(flow_id, {}) or {}
    readiness = (context.get("hapa_replica_readiness", {}) or {}).get(server_id, 1.0)
    return {
        "has_hapa_feedback": 1.0,
        "coverage_ratio": float(coverage_item.get("coverage_ratio", 1.0)),
        "replica_readiness": float(readiness),
        "hapa_psi": float(profile_item.get("hapa_psi", 1.0)),
        "hapa_d_loc": float(profile_item.get("hapa_d_loc", 1.0)),
        "reservation_pressure": float(profile_item.get("hapa_reservation_pressure", 0.0)),
    }


def _apply_hapa_feedback(delay_ms: float, energy_j: float, features: Dict[str, float]) -> Tuple[float, float]:
    """
    将HAPA demand covering和replica readiness反馈到快层指标
    该项是工程闭合：ready且覆盖充分的副本更稳定，覆盖不足的pair承担额外风险。
    """
    if float(features.get("has_hapa_feedback", 0.0)) <= 0.0:
        return float(delay_ms), float(energy_j)
    coverage_gap = max(0.0, 1.0 - float(features.get("coverage_ratio", 1.0)))
    readiness = max(0.0, min(float(features.get("replica_readiness", 1.0)), 1.0))
    readiness_gap = max(0.0, 1.0 - readiness)
    psi_gap = max(0.0, 0.8 - float(features.get("hapa_psi", 1.0)))
    loc_gap = max(0.0, 1.0 - float(features.get("hapa_d_loc", 1.0)))
    reservation_pressure = max(0.0, float(features.get("reservation_pressure", 0.0)))
    ready_gain = max(0.0, readiness - 0.5)
    loc_gain = max(0.0, float(features.get("hapa_d_loc", 1.0)) - 0.7)
    psi_gain = max(0.0, float(features.get("hapa_psi", 1.0)) - 0.8)
    delay_factor = 1.0 + 0.18 * coverage_gap + 0.10 * readiness_gap + 0.08 * loc_gap + 0.04 * psi_gap
    energy_factor = 1.0 + 0.08 * coverage_gap + 0.06 * readiness_gap + 0.04 * psi_gap + 0.03 * reservation_pressure
    # ready副本具备模型热缓存、数据局部性和GPU/VRAM包络匹配优势。
    # 折扣有下界，避免把HAPA工程反馈写成无约束收益。
    delay_factor = max(0.86, delay_factor - 0.08 * ready_gain - 0.04 * loc_gain)
    energy_factor = max(0.78, energy_factor - 0.32 * ready_gain - 0.20 * loc_gain - 0.14 * psi_gain)
    return float(delay_ms) * delay_factor, float(energy_j) * energy_factor


def _ai_server_index_map(system_state: 'SystemState') -> Tuple[List[str], Dict[str, int]]:
    """返回AI服务器顺序和索引"""
    ai_server_ids = sorted([
        server.server_id for server in system_state.edge_servers.values()
        if server.server_type.value == "ai_capable"
    ])
    return ai_server_ids, {server_id: idx for idx, server_id in enumerate(ai_server_ids)}


def _server_queue_value(server_id: str, server_index: Dict[str, int],
                        SQ: np.ndarray, SZ: np.ndarray) -> Tuple[float, float]:
    """按server_id获取虚拟队列值"""
    idx = server_index.get(server_id, -1)
    sq_value = float(SQ[idx]) if 0 <= idx < len(SQ) else 0.0
    sz_value = float(SZ[idx]) if 0 <= idx < len(SZ) else 0.0
    return sq_value, sz_value



def _instance_reservation_envelope(instance, server=None) -> Dict[str, float]:
    """读取AI实例的慢层预留包络，供快层资源枚举和修复使用"""
    if instance is None:
        return {"gpu_units": 0.0, "gpu_memory": 0.0, "model_storage": 0.0, "context_units": 0.0}
    envelope = {
        "gpu_units": float(getattr(instance, "gpu_units_reserved", 0.0)),
        "gpu_memory": float(getattr(instance, "gpu_memory_reserved", 0.0)),
        "model_storage": float(getattr(instance, "model_storage_reserved", 0.0)),
        "context_units": float(getattr(instance, "context_units_reserved", 0.0)),
    }
    if server is not None:
        envelope["gpu_units"] = min(envelope["gpu_units"], float(getattr(server, "gpu_units", envelope["gpu_units"])))
        envelope["gpu_memory"] = min(envelope["gpu_memory"], float(getattr(server, "gpu_memory", envelope["gpu_memory"])))
        envelope["model_storage"] = min(envelope["model_storage"], float(getattr(server, "model_storage", envelope["model_storage"])))
    return envelope
def _repair_pair_action_for_capacity(pair_action: np.ndarray,
                                     pair_universe: List[Dict],
                                     SH: np.ndarray,
                                     SQ: np.ndarray,
                                     SZ: np.ndarray,
                                     system_state: 'SystemState',
                                     V: float,
                                     omega_energy: float = 1.0,
                                     omega_delay: float = 1.0,
                                     resource_hint: str = "") -> Tuple[np.ndarray, Dict[int, Dict], int]:
    """
    修复pair级本地动作的资源包络
    资源不足时只把低收益本地pair转云端，不直接判失败。
    """
    from ablation_resource_models import select_local_ai_config, solve_cloud_preprocess_config

    action = np.asarray(pair_action, dtype=int).copy()
    ai_server_ids, server_index = _ai_server_index_map(system_state)
    local_configs: Dict[int, Dict] = {}
    local_groups: Dict[str, List[int]] = {}
    forced_cloud = 0

    for pos, bit in enumerate(action):
        if int(bit) != 0 or pos >= len(pair_universe):
            continue
        item = pair_universe[pos]
        instance, request_flow, server = _resolve_pair_item(item, system_state)
        if instance is None or request_flow is None or server is None:
            action[pos] = 1
            forced_cloud += 1
            continue
        sq_value, sz_value = _server_queue_value(server.server_id, server_index, SQ, SZ)
        performance_factor = float(SH[server_index.get(server.server_id, 0)]) if len(SH) else 1.0
        reservation_envelope = _instance_reservation_envelope(instance, server)
        local_config = select_local_ai_config(
            request_flow, instance.microservice, server, system_state,
            SQ_value=sq_value, SZ_value=sz_value, performance_factor=performance_factor, V=V,
            omega_energy=omega_energy, omega_delay=omega_delay,
            reservation_envelope=reservation_envelope,
            resource_hint=resource_hint,
        )
        if not local_config:
            action[pos] = 1
            forced_cloud += 1
            continue
        cloud_config = solve_cloud_preprocess_config(
            request_flow, instance.microservice, server, system_state,
            SQ_value=sq_value, SZ_value=sz_value, performance_factor=performance_factor, V=V,
            omega_energy=omega_energy, omega_delay=omega_delay,
            resource_hint=resource_hint
        )
        cloud_energy = float(cloud_config.get("energy_j", local_config.get("energy_j", 0.0))) if cloud_config else float(local_config.get("energy_j", 0.0))
        cloud_latency = float(cloud_config.get("latency_ms", local_config.get("latency_ms", 0.0))) if cloud_config else float(local_config.get("latency_ms", 0.0))
        local_energy = float(local_config.get("energy_j", 0.0))
        local_latency = float(local_config.get("latency_ms", 0.0))
        hapa_features = _pair_hapa_features(item, system_state)
        local_latency, local_energy = _apply_hapa_feedback(local_latency, local_energy, hapa_features)
        local_config["latency_ms"] = float(local_latency)
        local_config["energy_j"] = float(local_energy)
        local_config["hapa_feedback_consumed"] = bool(hapa_features.get("has_hapa_feedback", 0.0) > 0.0)
        # benefit越大越应该保留本地，兼顾能耗队列和延迟队列
        benefit = (
            sq_value * (cloud_energy - local_energy) +
            sz_value * (cloud_latency - local_latency) / 50.0
        )
        local_config["_pair_keep_benefit"] = float(benefit)
        local_config["_reserved_gpu_units"] = float(reservation_envelope.get("gpu_units", 0.0))
        local_config["_reserved_gpu_memory"] = float(reservation_envelope.get("gpu_memory", 0.0))
        local_config["_reserved_model_storage"] = float(reservation_envelope.get("model_storage", 0.0))
        local_configs[pos] = local_config
        local_groups.setdefault(server.server_id, []).append(pos)

    for server_id, positions in local_groups.items():
        server = system_state.edge_servers.get(server_id)
        if server is None:
            continue
        kept = list(positions)

        def totals(items):
            return {
                "gpu_units": sum(float(local_configs[p].get("gpu_units", 0.0)) for p in items),
                "gpu_memory": sum(float(local_configs[p].get("gpu_memory", 0.0)) for p in items),
                "model_storage": sum(float(local_configs[p].get("model_storage", 0.0)) for p in items),
            }

        def effective_budget(items):
            reserved_gpu = sum(float(local_configs[p].get("_reserved_gpu_units", 0.0)) for p in items)
            reserved_memory = sum(float(local_configs[p].get("_reserved_gpu_memory", 0.0)) for p in items)
            reserved_storage = sum(float(local_configs[p].get("_reserved_model_storage", 0.0)) for p in items)
            return {
                "gpu_units": min(float(getattr(server, "gpu_units", 0.0)), max(float(getattr(server, "available_gpu_units", 0.0)), reserved_gpu)),
                "gpu_memory": min(float(getattr(server, "gpu_memory", 0.0)), max(float(getattr(server, "available_gpu_memory", 0.0)), reserved_memory)),
                "model_storage": min(float(getattr(server, "model_storage", 0.0)), max(float(getattr(server, "available_model_storage", 0.0)), reserved_storage)),
            }

        while kept:
            current = totals(kept)
            budget = effective_budget(kept)
            feasible = (
                current["gpu_units"] <= budget["gpu_units"] and
                current["gpu_memory"] <= budget["gpu_memory"] and
                current["model_storage"] <= budget["model_storage"]
            )
            if feasible:
                break
            remove_pos = min(
                kept,
                key=lambda p: (
                    float(local_configs[p].get("_pair_keep_benefit", 0.0)),
                    -float(local_configs[p].get("energy_j", 0.0)),
                    p,
                )
            )
            kept.remove(remove_pos)
            action[remove_pos] = 1
            forced_cloud += 1
    return action, local_configs, forced_cloud



def _pair_endpoint_key(item: Dict) -> str:
    """同一flow/service的候选endpoint归为一组。"""
    return f"{item.get('flow_id', '')}:{item.get('microservice_id', '')}"


def _select_executable_pair_endpoints(repaired_action: np.ndarray,
                                      pair_universe: List[Dict],
                                      local_configs: Dict[int, Dict]) -> Tuple[np.ndarray, set, Dict[str, int]]:
    """
    将pair动作修复为可执行endpoint动作。
    同一flow/service只执行一个endpoint；其它副本只作为候选，不重复产生AI计算。
    """
    action = np.ones(len(repaired_action), dtype=int)
    groups: Dict[str, List[int]] = {}
    for pos, item in enumerate(pair_universe[:len(repaired_action)]):
        groups.setdefault(_pair_endpoint_key(item), []).append(pos)

    executable_positions = set()
    local_count = 0
    cloud_count = 0
    inactive_count = 0
    for positions in groups.values():
        local_positions = [pos for pos in positions if int(repaired_action[pos]) == 0]
        if local_positions:
            selected = max(
                local_positions,
                key=lambda pos: (
                    float(local_configs.get(pos, {}).get("_pair_keep_benefit", 0.0)),
                    -float(local_configs.get(pos, {}).get("energy_j", 0.0)),
                    -float(local_configs.get(pos, {}).get("latency_ms", 0.0)),
                    -pos,
                )
            )
            action[selected] = 0
            local_count += 1
        else:
            selected = min(
                positions,
                key=lambda pos: (
                    str(pair_universe[pos].get("server_id", "")),
                    pos,
                )
            )
            action[selected] = 1
            cloud_count += 1
        executable_positions.add(selected)
        inactive_count += max(len(positions) - 1, 0)

    diagnostics = {
        "executed_pair_count": int(len(executable_positions)),
        "executed_local_pair_count": int(local_count),
        "executed_cloud_pair_count": int(cloud_count),
        "inactive_endpoint_count": int(inactive_count),
    }
    return action, executable_positions, diagnostics


def _edge_queue_energy_for_pair_decision(ai_instance, total_energy: float, decision_bit: int) -> float:
    """返回虚拟能耗队列应消费的边缘节点能耗。

    本地动作由边缘GPU承担，直接计入节点能耗队列；云端动作的远端推理能耗
    仍进入system-wide energy，但节点队列只承担边缘侧预处理和通信能耗。
    """
    if int(decision_bit) == 0:
        return float(total_energy)
    comm_energy = float(getattr(ai_instance, "energy_comm_j", 0.0))
    preprocess_energy = float(getattr(ai_instance, "energy_preprocess_j", 0.0))
    return float(comm_energy + preprocess_energy)


def _format_executed_pair_signature(indices: List[int], pair_ids: List[str]) -> str:
    return ";".join(f"{int(index)}:{pair_id}" for index, pair_id in zip(indices, pair_ids))


def AI_Offloading_Resource_Allocation_Pair(
        pair_action: np.ndarray,
        pair_universe: List[Dict],
        SH: np.ndarray,
        SQ: np.ndarray,
        SZ: np.ndarray,
        system_state: 'SystemState',
        V: float = 20.0,
        omega_energy: float = 1.0,
        omega_delay: float = 1.0,
        resource_hint: str = "") -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    pair级AI卸载资源分配
    0=本地处理，1=云端卸载；同一AI服务器上的不同实例可独立决策。
    """
    ai_server_ids, server_index = _ai_server_index_map(system_state)
    ai_delays_by_server = {server_id: [] for server_id in ai_server_ids}
    ai_energies_by_server = {server_id: [] for server_id in ai_server_ids}
    ai_delay_burden_by_server = {server_id: [] for server_id in ai_server_ids}
    pair_action = np.asarray(pair_action, dtype=int)
    repaired_action, local_configs, forced_cloud_count = _repair_pair_action_for_capacity(
        pair_action, pair_universe, SH, SQ, SZ, system_state, V,
        omega_energy=omega_energy,
        omega_delay=omega_delay,
        resource_hint=resource_hint,
    )
    repaired_action, executable_positions, endpoint_diag = _select_executable_pair_endpoints(
        repaired_action, pair_universe, local_configs
    )
    legacy_reward = 0.0
    executed_local_pair_indices = []
    executed_local_pair_ids = []
    executed_cloud_pair_indices = []
    executed_cloud_pair_ids = []

    # 重置本时隙active pair计数，避免dry-run候选之间互相污染。
    for instance in system_state.microservice_instances.values():
        if instance.microservice.service_type == "ai":
            instance.active_pair_count = 0
            instance.active_local_pair_count = 0
            instance.active_cloud_pair_count = 0

    for pos, item in enumerate(pair_universe):
        if pos not in executable_positions:
            continue
        instance, request_flow, server = _resolve_pair_item(item, system_state)
        if instance is None or request_flow is None or server is None:
            continue
        idx = server_index.get(server.server_id, -1)
        performance_factor = float(SH[idx]) if 0 <= idx < len(SH) else 1.0
        sq_value, sz_value = _server_queue_value(server.server_id, server_index, SQ, SZ)
        decision_bit = int(repaired_action[pos]) if pos < len(repaired_action) else 1
        pair_id = str(item.get("pair_id", f"pair_{pos}"))
        instance.pair_action_bit = decision_bit
        instance.active_pair_count = int(getattr(instance, "active_pair_count", 0)) + 1
        if decision_bit == 0:
            executed_local_pair_indices.append(int(pos))
            executed_local_pair_ids.append(pair_id)
            instance.active_local_pair_count = int(getattr(instance, "active_local_pair_count", 0)) + 1
            delay, energy = process_local_ai_decision(
                instance, request_flow, server, system_state,
                performance_factor, sq_value, sz_value, V,
                omega_energy=omega_energy, omega_delay=omega_delay,
                resource_hint=resource_hint,
            )
            hapa_features = _pair_hapa_features(item, system_state)
            delay, energy = _apply_hapa_feedback(delay, energy, hapa_features)
            instance.hapa_feedback_consumed = bool(hapa_features.get("has_hapa_feedback", 0.0) > 0.0)
            instance.hapa_replica_readiness = float(hapa_features.get("replica_readiness", 1.0))
            instance.hapa_coverage_ratio = float(hapa_features.get("coverage_ratio", 1.0))
            instance.energy_local_gpu_j = float(energy)
            instance.processing_mode = "local_processing"
            instance.inference_latency = float(delay)
            instance.cloud_latency = 0.0
        else:
            executed_cloud_pair_indices.append(int(pos))
            executed_cloud_pair_ids.append(pair_id)
            instance.active_cloud_pair_count = int(getattr(instance, "active_cloud_pair_count", 0)) + 1
            delay, energy = process_cloud_offloading_decision(
                instance, request_flow, server, system_state,
                performance_factor, sq_value, sz_value, V,
                omega_energy=omega_energy, omega_delay=omega_delay,
                resource_hint=resource_hint,
            )
            instance.processing_mode = "cloud_offloaded"
            instance.inference_latency = 0.0
            instance.cloud_latency = float(delay)
        ai_delays_by_server[server.server_id].append(float(delay))
        queue_energy = _edge_queue_energy_for_pair_decision(instance, float(energy), decision_bit)
        ai_energies_by_server[server.server_id].append(float(queue_energy))
        arrival_rate = max(float(getattr(request_flow, "arrival_rate", 1.0)), 1.0)
        ai_delay_burden_by_server[server.server_id].append(float(delay) * arrival_rate)
        legacy_reward += 10.0 + arrival_rate * 0.05

    ai_delays = np.zeros(len(ai_server_ids), dtype=float)
    ai_energies = np.zeros(len(ai_server_ids), dtype=float)
    delay_burden_vec = np.zeros(len(ai_server_ids), dtype=float)
    pair_count_vec = np.zeros(len(ai_server_ids), dtype=float)
    for idx, server_id in enumerate(ai_server_ids):
        delays = ai_delays_by_server.get(server_id, [])
        energies = ai_energies_by_server.get(server_id, [])
        delay_burdens = ai_delay_burden_by_server.get(server_id, [])
        ai_delays[idx] = float(np.mean(delays)) if delays else 0.0
        ai_energies[idx] = float(np.sum(energies)) if energies else 0.0
        delay_burden_vec[idx] = float(np.sum(delay_burdens)) if delay_burdens else 0.0
        pair_count_vec[idx] = float(len(delays))
    system_state._pair_delay_burden_by_server = delay_burden_vec
    system_state._pair_count_by_server = pair_count_vec
    system_state._executed_pair_count = int(endpoint_diag.get("executed_pair_count", len(executable_positions)))
    system_state._executed_local_pair_count = int(endpoint_diag.get("executed_local_pair_count", 0))
    system_state._executed_cloud_pair_count = int(endpoint_diag.get("executed_cloud_pair_count", 0))
    system_state._inactive_pair_endpoint_count = int(endpoint_diag.get("inactive_endpoint_count", 0))
    system_state._executed_local_pair_indices = executed_local_pair_indices
    system_state._executed_local_pair_ids = executed_local_pair_ids
    system_state._executed_local_pair_signature = _format_executed_pair_signature(
        executed_local_pair_indices, executed_local_pair_ids
    )
    system_state._executed_cloud_pair_indices = executed_cloud_pair_indices
    system_state._executed_cloud_pair_ids = executed_cloud_pair_ids
    system_state._executed_cloud_pair_signature = _format_executed_pair_signature(
        executed_cloud_pair_indices, executed_cloud_pair_ids
    )
    return float(legacy_reward), ai_delays, ai_energies, repaired_action, int(forced_cloud_count)


def check_local_resource_sufficiency(server_id: str, system_state: 'SystemState') -> bool:
    """
    检查AI服务器本地资源是否充足（包括GPU单元数检查）
    """
    from Deployment import calculate_local_ai_inference_latency,calculate_required_gpu_memory,calculate_required_model_storage
    from ablation_resource_models import select_local_ai_config
    ai_pairs = find_ai_instances_and_flows(server_id, system_state)
    if not ai_pairs:
        return True

    server = system_state.edge_servers[server_id]

    try:
        total_gpu_units = 0
        total_gpu_memory = 0.0
        total_model_storage = 0.0
        max_latency = 0.0
        processing_feasible = True

        for ai_instance, request_flow in ai_pairs:
            local_config = select_local_ai_config(
                request_flow, ai_instance.microservice, server, system_state,
                SQ_value=0.0, SZ_value=0.0, performance_factor=1.0, V=20.0
            )
            if local_config:
                # C4入口优先使用(g,b,f_GPU)枚举结果，避免旧GPU硬判定把可行本地候选全部压成云端
                required_gpu_units = int(local_config.get("gpu_units", 1))
                total_latency = float(local_config.get("latency_ms", 0.0))
                total_gpu_memory += float(local_config.get("gpu_memory", 0.0))
                total_model_storage += float(local_config.get("model_storage", 0.0))
            else:
                # 枚举失败时保留旧估计作为兼容兜底
                total_latency, queue_delay, processing_delay, required_gpu_units = calculate_local_ai_inference_latency(
                    request_flow, ai_instance.microservice, server, system_state, False)

                if np.isinf(total_latency):
                    processing_feasible = False
                total_gpu_memory += float(calculate_required_gpu_memory(
                    request_flow, ai_instance.microservice, required_gpu_units, False))
                total_model_storage += float(calculate_required_model_storage(ai_instance.microservice))
            if np.isinf(total_latency):
                processing_feasible = False
            max_latency = max(max_latency, float(total_latency))
            total_gpu_units += int(required_gpu_units)

        # 检查GPU单元、显存和模型存储的聚合约束
        gpu_units_sufficient = server.available_gpu_units >= total_gpu_units
        gpu_memory_sufficient = server.available_gpu_memory >= total_gpu_memory
        storage_sufficient = server.available_model_storage >= total_model_storage

        resource_sufficient = (processing_feasible and gpu_units_sufficient and
                              gpu_memory_sufficient and storage_sufficient)

        if not resource_sufficient:
            print(f"    AI服务器 {server_id} 本地资源不足:")
            print(f"      处理可行性: {processing_feasible} ({'稳定' if processing_feasible else '服务强度过载'})")
            print(f"      GPU单元充足性: {gpu_units_sufficient} ({server.available_gpu_units}/{total_gpu_units})")
            print(f"      GPU内存充足性: {gpu_memory_sufficient} ({server.available_gpu_memory:.1f}/{total_gpu_memory:.1f} GB)")
            print(f"      存储充足性: {storage_sufficient} ({server.available_model_storage:.1f}/{total_model_storage:.1f} GB)")
            if processing_feasible:
                print(f"      最大计算延迟: {max_latency:.2f} ms")

        return resource_sufficient

    except Exception as e:
        print(f"    资源检查异常 {server_id}: {e}")
        return False


def process_local_ai_decision(ai_instance: 'MicroserviceInstance',
                              request_flow: 'RequestFlow',
                              server: 'EdgeServer',
                              system_state: 'SystemState',
                              performance_factor: float,
                              SQ_value: float = 0.0,
                              SZ_value: float = 0.0,
                              V: float = 2.0,  # 【修改】新增V参数
                              omega_energy: float = 1.0,
                              omega_delay: float = 1.0,
                              resource_hint: str = ""
                              ) -> Tuple[float, float]:
    """
    处理本地AI处理决策，计算延迟和能耗
    优先使用C4的(g,b,f_GPU)枚举，失败时回退旧GPU单元搜索。
    """
    optimal_config = None
    try:
        from ablation_resource_models import select_local_ai_config
        optimal_config = select_local_ai_config(
            request_flow, ai_instance.microservice, server, system_state,
            SQ_value, SZ_value, performance_factor, V=V,
            omega_energy=omega_energy, omega_delay=omega_delay,
            reservation_envelope=_instance_reservation_envelope(ai_instance, server),
            resource_hint=resource_hint,
        )
    except Exception as exc:
        print(f"      C4本地资源枚举失败，回退旧GPU搜索: {exc}")

    if optimal_config is None:
        # 调用旧GPU资源优化作为兼容回退
        optimal_config = optimize_gpu_allocation_for_local_processing(
            request_flow, ai_instance.microservice, server, system_state,
            SQ_value, SZ_value, performance_factor, V=V
        )

    if optimal_config:
        # 使用优化后的配置
        total_latency = optimal_config.get('latency_ms', optimal_config.get('latency'))
        current_energy = optimal_config.get('energy_j', optimal_config.get('energy'))
        gpu_units = optimal_config['gpu_units']
        gpu_memory = optimal_config['gpu_memory']
        model_storage = optimal_config['model_storage']
        ai_instance.gpu_units_allocated = gpu_units
        ai_instance.gpu_memory_allocated = gpu_memory
        ai_instance.model_storage_allocated = model_storage
        ai_instance.batch_size_allocated = int(optimal_config.get("batch_size", 1))
        ai_instance.gpu_frequency_scale = float(optimal_config.get("gpu_frequency_scale", 1.0))
        ai_instance.resource_config_source = optimal_config.get("source", "legacy_gpu_search")
        ai_instance.inference_latency = float(total_latency)
        ai_instance.energy_local_gpu_j = float(current_energy)
        ai_instance.energy_cloud_compute_j = 0.0
        ai_instance.energy_comm_j = 0.0
        ai_instance.energy_preprocess_j = 0.0

    else:
        # 回退到原始方法
        print(f"      优化失败，使用原始方法...")
        from Deployment import calculate_local_ai_inference_latency, calculate_required_gpu_memory, \
            calculate_required_model_storage
        from EnergyConsumption import calculate_local_ai_processing_energy

        # 原始计算逻辑
        total_latency, queue_delay, processing_delay, gpu_units = calculate_local_ai_inference_latency(
            request_flow, ai_instance.microservice, server, system_state, False)
        gpu_memory = calculate_required_gpu_memory(request_flow, ai_instance.microservice, gpu_units, False)
        model_storage = calculate_required_model_storage(ai_instance.microservice)

        # 使用专用的本地AI处理能耗函数
        current_energy = calculate_local_ai_processing_energy(
            server, gpu_units, gpu_memory, model_storage)
        ai_instance.gpu_units_allocated = gpu_units
        ai_instance.gpu_memory_allocated = gpu_memory
        ai_instance.model_storage_allocated = model_storage
        ai_instance.batch_size_allocated = 1
        ai_instance.gpu_frequency_scale = 1.0
        ai_instance.resource_config_source = "fallback_latency_model"
        ai_instance.energy_local_gpu_j = float(current_energy)
        ai_instance.energy_cloud_compute_j = 0.0
        ai_instance.energy_comm_j = 0.0
        ai_instance.energy_preprocess_j = 0.0

    return total_latency, current_energy


def process_cloud_offloading_decision(ai_instance: 'MicroserviceInstance',
                                      request_flow: 'RequestFlow',
                                      server: 'EdgeServer',
                                      system_state: 'SystemState',
                                      performance_factor: float,
                                      SQ_value: float = 0.0,
                                      SZ_value: float = 0.0,
                                      V: float = 2.0,
                                      omega_energy: float = 1.0,
                                      omega_delay: float = 1.0,
                              resource_hint: str = ""
                                      ) -> Tuple[float, float]:
    """
    处理云端卸载决策，计算延迟和能耗
    """

    optimal_config = None
    try:
        from ablation_resource_models import solve_cloud_preprocess_config
        optimal_config = solve_cloud_preprocess_config(
            request_flow, ai_instance.microservice, server, system_state,
            SQ_value, SZ_value, performance_factor, V,
            omega_energy=omega_energy, omega_delay=omega_delay,
            resource_hint=resource_hint
        )
    except Exception as exc:
        print(f"      C4云端f_pre搜索失败，回退旧传输策略: {exc}")

    if optimal_config is None:
        # 旧LyDROO风格压缩比搜索只作为兼容回退
        optimal_config = optimize_cloud_transmission_strategy(
            request_flow, ai_instance.microservice, server, system_state,
            SQ_value, SZ_value, performance_factor, V)

    if optimal_config:
        # 使用搜索后的云端配置
        adjusted_cloud_latency = optimal_config.get('latency_ms', optimal_config.get('latency'))
        total_cloud_energy = optimal_config.get('energy_j', optimal_config.get('total_energy'))
        compression_ratio = optimal_config.get('compression_ratio', 1.0)
        ai_instance.preprocess_frequency_scale = float(optimal_config.get("f_pre", 1.0))
        ai_instance.compression_ratio = float(compression_ratio)
        ai_instance.resource_config_source = optimal_config.get("source", "legacy_compression_search")
        ai_instance.resource_hint = str(resource_hint or "")
        ai_instance.cloud_latency = float(adjusted_cloud_latency)
        ai_instance.energy_local_gpu_j = 0.0
        ai_instance.energy_cloud_compute_j = float(optimal_config.get("cloud_compute_energy_j", total_cloud_energy))
        ai_instance.energy_comm_j = float(optimal_config.get("communication_energy_j", 0.0))
        ai_instance.energy_preprocess_j = float(optimal_config.get("preprocess_energy_j", 0.0))
        '''
        # 显示约束违反情况（软约束信息）
        if optimal_config['energy_violation'] > 0 or optimal_config['delay_violation'] > 0:
            print(f"        软约束违反: 能耗+{optimal_config['energy_violation']:.4f}J, "
                  f"延迟+{optimal_config['delay_violation']:.1f}ms")
        '''
    else:
        # 回退到基础配置
        print(f"      ⚠️  Lyapunov优化失败，云端...")
        from Deployment import evaluate_cloud_deployment

        # 基础云端评估
        cloud_evaluation = evaluate_cloud_deployment(request_flow, ai_instance.microservice, system_state)
        base_cloud_latency = cloud_evaluation['total_latency']

        # 性能因子调整
        performance_adjustment = 0.9 + 0.2 * performance_factor
        adjusted_cloud_latency = base_cloud_latency * performance_adjustment

        # 使用专用的云端能耗计算函数
        from EnergyConsumption import calculate_cloud_processing_energy, calculate_optimized_communication_energy

        # 云端处理协调能耗
        cloud_processing_energy = calculate_cloud_processing_energy(server, request_flow=request_flow)

        # 通信能耗（无压缩）
        communication_energy = calculate_optimized_communication_energy(
            request_flow, compression_ratio=1.0, server=server)

        total_cloud_energy = cloud_processing_energy + communication_energy
        ai_instance.energy_local_gpu_j = 0.0
        ai_instance.energy_cloud_compute_j = float(cloud_processing_energy)
        ai_instance.energy_comm_j = float(communication_energy)
        ai_instance.energy_preprocess_j = 0.0

        print(f"      基础配置 - 处理能耗: {cloud_processing_energy:.4f}J")
        print(f"      基础配置 - 通信能耗: {communication_energy:.4f}J")
    '''
    print(f"      云端卸载总延迟: {adjusted_cloud_latency:.2f}ms")
    print(f"      云端卸载总能耗: {total_cloud_energy:.4f}J")
    '''
    return adjusted_cloud_latency, total_cloud_energy


def backup_system_state(system_state: 'SystemState', include_full_deployment: bool = True) -> Dict:
    """
    备份关键系统状态用于恢复
    """
    backup = {}

    # 备份服务器资源状态
    backup['server_states'] = {}
    for server_id, server in system_state.edge_servers.items():
        backup['server_states'][server_id] = {
            'available_cpu': getattr(server, 'available_cpu', None),
            'available_memory': getattr(server, 'available_memory', None),
            'available_gpu_units': getattr(server, 'available_gpu_units', None),
            'available_gpu_memory': getattr(server, 'available_gpu_memory', None),
            'available_model_storage': getattr(server, 'available_model_storage', None),
            'gpu_active': getattr(server, '_gpu_active', False),
            'current_batch_requests': list(getattr(server, 'current_batch_requests', [])),
            'batch_processing_start_time': getattr(server, 'batch_processing_start_time', 0.0)
        }

    # 真实执行异常恢复保留全量部署；dry-run只保存实例引用和可变字段，避免候选评分重复深拷贝。
    if include_full_deployment:
        backup['microservice_instances_full'] = copy.deepcopy(system_state.microservice_instances)
        backup['stream_allocated_resources'] = copy.deepcopy(getattr(system_state, 'stream_allocated_resources', {}))
        backup['stream_transfer_probabilities'] = copy.deepcopy(getattr(system_state, 'stream_transfer_probabilities', {}))
        backup['routing_probabilities'] = {
            flow_id: copy.deepcopy(getattr(request_flow, 'routing_probabilities', {}))
            for flow_id, request_flow in system_state.request_flows.items()
        }
        backup['gsla_context'] = copy.deepcopy(getattr(system_state, 'gsla_context', {}))
        backup['ffd_context'] = copy.deepcopy(getattr(system_state, 'ffd_context', {}))
        backup['random_context'] = copy.deepcopy(getattr(system_state, 'random_context', {}))
        backup['pdrs_context'] = copy.deepcopy(getattr(system_state, 'pdrs_context', {}))
        backup['loadaware_context'] = copy.deepcopy(getattr(system_state, 'loadaware_context', {}))
        backup['hapa_demand_ewma'] = copy.deepcopy(getattr(system_state, 'hapa_demand_ewma', {}))
        backup['previous_ai_profile'] = copy.deepcopy(getattr(system_state, 'previous_ai_profile', None))
    else:
        backup['microservice_instance_refs'] = dict(getattr(system_state, 'microservice_instances', {}))

    # 备份微服务实例状态
    backup['instance_states'] = {}
    for instance_id, instance in system_state.microservice_instances.items():
        backup['instance_states'][instance_id] = {
            'server_id': getattr(instance, 'server_id', None),
            'allocated_cpu_cores': getattr(instance, 'allocated_cpu_cores', 0),
            'allocated_memory': getattr(instance, 'allocated_memory', 0.0),
            'processing_mode': getattr(instance, 'processing_mode', 'unknown'),
            'gpu_units_allocated': getattr(instance, 'gpu_units_allocated', 0),
            'gpu_memory_allocated': getattr(instance, 'gpu_memory_allocated', 0.0),
            'model_storage_allocated': getattr(instance, 'model_storage_allocated', 0.0),
            'batch_size_allocated': getattr(instance, 'batch_size_allocated', 1),
            'gpu_frequency_scale': getattr(instance, 'gpu_frequency_scale', 1.0),
            'preprocess_frequency_scale': getattr(instance, 'preprocess_frequency_scale', 1.0),
            'compression_ratio': getattr(instance, 'compression_ratio', 1.0),
            'resource_config_source': getattr(instance, 'resource_config_source', ''),
            'inference_latency': getattr(instance, 'inference_latency', 0.0),
            'cloud_latency': getattr(instance, 'cloud_latency', 0.0),
            'energy_local_gpu_j': getattr(instance, 'energy_local_gpu_j', 0.0),
            'energy_cloud_compute_j': getattr(instance, 'energy_cloud_compute_j', 0.0),
            'energy_comm_j': getattr(instance, 'energy_comm_j', 0.0),
            'energy_preprocess_j': getattr(instance, 'energy_preprocess_j', 0.0),
            'resource_hint': getattr(instance, 'resource_hint', ''),
            'hapa_feedback_consumed': getattr(instance, 'hapa_feedback_consumed', False),
            'hapa_replica_readiness': getattr(instance, 'hapa_replica_readiness', 1.0),
            'hapa_coverage_ratio': getattr(instance, 'hapa_coverage_ratio', 1.0),
            'pair_action_bit': getattr(instance, 'pair_action_bit', None),
            'active_pair_count': getattr(instance, 'active_pair_count', 0),
            'active_local_pair_count': getattr(instance, 'active_local_pair_count', 0),
            'active_cloud_pair_count': getattr(instance, 'active_cloud_pair_count', 0),
        }

    # 备份请求和队列状态
    backup['request_states'] = {}
    for flow_id, request_flow in system_state.request_flows.items():
        backup['request_states'][flow_id] = {
            'arrival_rate': getattr(request_flow, 'arrival_rate', 0.0),
            'ca_latency': getattr(request_flow, 'ca_latency', 0.0)
        }
    backup['energy_queues'] = {
        server_id: queue.queue_state
        for server_id, queue in system_state.virtual_energy_queues.items()
    }
    backup['delay_queues'] = {
        server_id: queue.queue_state
        for server_id, queue in system_state.virtual_delay_queues.items()
    }
    backup['system_metrics'] = {
        'total_energy_consumption': getattr(system_state, 'total_energy_consumption', 0.0),
        'total_latency': getattr(system_state, 'total_latency', 0.0),
        'current_arrivals': dict(getattr(system_state, 'current_arrivals', {})),
        '_routing_energy_consumed': getattr(system_state, '_routing_energy_consumed', False),
        '_routing_energy_probability_mass': getattr(system_state, '_routing_energy_probability_mass', 0.0),
        '_routing_metric_consumed': getattr(system_state, '_routing_metric_consumed', False),
        '_routing_probability_mass': getattr(system_state, '_routing_probability_mass', 0.0),
        '_routing_delay_consumed': getattr(system_state, '_routing_delay_consumed', False),
        '_routing_probability_mass_for_delay': getattr(system_state, '_routing_probability_mass_for_delay', 0.0),
        '_pair_delay_burden_by_server': copy.deepcopy(getattr(system_state, '_pair_delay_burden_by_server', None)),
        '_pair_count_by_server': copy.deepcopy(getattr(system_state, '_pair_count_by_server', None)),
        '_executed_pair_count': getattr(system_state, '_executed_pair_count', 0),
        '_executed_local_pair_count': getattr(system_state, '_executed_local_pair_count', 0),
        '_executed_cloud_pair_count': getattr(system_state, '_executed_cloud_pair_count', 0),
        '_inactive_pair_endpoint_count': getattr(system_state, '_inactive_pair_endpoint_count', 0),
        '_executed_local_pair_indices': copy.deepcopy(getattr(system_state, '_executed_local_pair_indices', [])),
        '_executed_local_pair_ids': copy.deepcopy(getattr(system_state, '_executed_local_pair_ids', [])),
        '_executed_local_pair_signature': getattr(system_state, '_executed_local_pair_signature', ''),
        '_executed_cloud_pair_indices': copy.deepcopy(getattr(system_state, '_executed_cloud_pair_indices', [])),
        '_executed_cloud_pair_ids': copy.deepcopy(getattr(system_state, '_executed_cloud_pair_ids', [])),
        '_executed_cloud_pair_signature': getattr(system_state, '_executed_cloud_pair_signature', ''),
    }

    return backup


def restore_system_state(system_state: 'SystemState', backup: Dict):
    """
    从备份恢复系统状态
    """
    # 恢复AI服务器状态
    for server_id, server_backup in backup['server_states'].items():
        if server_id in system_state.edge_servers:
            server = system_state.edge_servers[server_id]
            for attr in [
                'available_cpu', 'available_memory', 'available_gpu_units',
                'available_gpu_memory', 'available_model_storage',
                'batch_processing_start_time'
            ]:
                if server_backup.get(attr) is not None:
                    setattr(server, attr, server_backup[attr])
            server._gpu_active = server_backup.get('gpu_active', False)
            server.current_batch_requests = list(server_backup.get('current_batch_requests', []))

    # 恢复全量部署和路由状态
    if 'microservice_instances_full' in backup:
        system_state.microservice_instances = copy.deepcopy(backup['microservice_instances_full'])
    elif 'microservice_instance_refs' in backup:
        refs = backup.get('microservice_instance_refs', {})
        system_state.microservice_instances = {
            instance_id: refs[instance_id]
            for instance_id in refs
        }
    if 'stream_allocated_resources' in backup:
        system_state.stream_allocated_resources = copy.deepcopy(backup['stream_allocated_resources'])
    if 'stream_transfer_probabilities' in backup:
        system_state.stream_transfer_probabilities = copy.deepcopy(backup['stream_transfer_probabilities'])
    for flow_id, probabilities in backup.get('routing_probabilities', {}).items():
        if flow_id in system_state.request_flows:
            system_state.request_flows[flow_id].routing_probabilities = copy.deepcopy(probabilities)
    for attr in [
        'gsla_context', 'ffd_context', 'random_context', 'pdrs_context',
        'loadaware_context', 'hapa_demand_ewma', 'previous_ai_profile',
    ]:
        if attr in backup:
            setattr(system_state, attr, copy.deepcopy(backup[attr]))

    # 恢复AI微服务实例状态
    for instance_id, instance_backup in backup['instance_states'].items():
        if instance_id in system_state.microservice_instances:
            instance = system_state.microservice_instances[instance_id]
            for attr, value in instance_backup.items():
                setattr(instance, attr, value)

    # 恢复请求和队列状态
    for flow_id, request_backup in backup.get('request_states', {}).items():
        if flow_id in system_state.request_flows:
            system_state.request_flows[flow_id].arrival_rate = request_backup['arrival_rate']
            system_state.request_flows[flow_id].ca_latency = request_backup['ca_latency']
    for server_id, queue_state in backup.get('energy_queues', {}).items():
        if server_id in system_state.virtual_energy_queues:
            system_state.virtual_energy_queues[server_id].queue_state = queue_state
    for server_id, queue_state in backup.get('delay_queues', {}).items():
        if server_id in system_state.virtual_delay_queues:
            system_state.virtual_delay_queues[server_id].queue_state = queue_state
    metrics = backup.get('system_metrics', {})
    system_state.total_energy_consumption = metrics.get('total_energy_consumption', system_state.total_energy_consumption)
    system_state.total_latency = metrics.get('total_latency', system_state.total_latency)
    system_state.current_arrivals = dict(metrics.get('current_arrivals', getattr(system_state, 'current_arrivals', {})))
    for attr in [
        '_routing_energy_consumed', '_routing_energy_probability_mass',
        '_routing_metric_consumed', '_routing_probability_mass',
        '_routing_delay_consumed', '_routing_probability_mass_for_delay',
        '_pair_delay_burden_by_server', '_pair_count_by_server',
        '_executed_pair_count', '_executed_local_pair_count',
        '_executed_cloud_pair_count', '_inactive_pair_endpoint_count',
        '_executed_local_pair_indices', '_executed_local_pair_ids',
        '_executed_local_pair_signature', '_executed_cloud_pair_indices',
        '_executed_cloud_pair_ids', '_executed_cloud_pair_signature',
    ]:
        if attr in metrics:
            setattr(system_state, attr, metrics[attr])


def calculate_request_level_average_delay(system_state: 'SystemState', fallback_delay_ms: float) -> float:
    """
    计算请求流端到端平均延迟
    若旧Waitingtime路径不可用，则回退到AI分支均值，避免失败时写0。
    """
    try:
        import Waitingtime
        total_delay = 0.0
        valid_count = 0
        for flow_id, request_flow in system_state.request_flows.items():
            flow_delay = Waitingtime.calculate_flow_end_to_end_delay_global(flow_id, system_state)
            if np.isfinite(flow_delay):
                request_flow.ca_latency = float(flow_delay)
                total_delay += float(flow_delay)
                valid_count += 1
        if valid_count > 0:
            return total_delay / valid_count
    except Exception as exc:
        print(f"请求级端到端延迟计算失败，使用AI分支延迟均值: {exc}")
    return float(fallback_delay_ms)


def calculate_request_level_delay_burden_by_server(system_state: 'SystemState',
                                                   ai_server_ids: List[str],
                                                   fallback_delay_ms: float) -> np.ndarray:
    """
    将请求级端到端延迟分摊到相关AI服务器
    这里使用到达率加权平均延迟，避免把多pair吞吐量直接求和后放大DPP队列项。
    """
    delay_sum = np.zeros(len(ai_server_ids), dtype=float)
    weight_sum = np.zeros(len(ai_server_ids), dtype=float)
    server_index = {server_id: idx for idx, server_id in enumerate(ai_server_ids)}
    ai_instances_by_flow = {flow_id: [] for flow_id in system_state.request_flows}
    for instance_id, instance in system_state.microservice_instances.items():
        if instance.microservice.service_type != "ai":
            continue
        flow_id = _extract_flow_id_from_instance_id(str(instance_id), system_state)
        if flow_id in ai_instances_by_flow:
            ai_instances_by_flow[flow_id].append((instance_id, instance))
    try:
        import Waitingtime
    except Exception:
        Waitingtime = None

    for flow_id, request_flow in system_state.request_flows.items():
        flow_delay = float(getattr(request_flow, "ca_latency", float("nan")))
        if not np.isfinite(flow_delay) and Waitingtime is not None:
            try:
                flow_delay = float(Waitingtime.calculate_flow_end_to_end_delay_global(flow_id, system_state))
            except Exception:
                flow_delay = float("nan")
        if not np.isfinite(flow_delay):
            flow_delay = float(fallback_delay_ms)

        related_server_factors = {}
        for instance_id, instance in ai_instances_by_flow.get(flow_id, []):
            # repaired pair action是快层最终可执行动作。
            # 本地pair承担完整延迟负担；云端pair仍经本地网关预处理/出网，保留有界负担。
            pair_bit = getattr(instance, "pair_action_bit", None)
            processing_mode = str(getattr(instance, "processing_mode", ""))
            burden_factor = 1.0
            if pair_bit is not None and int(pair_bit) == 1:
                burden_factor = 0.35
            if processing_mode == "cloud_offloaded":
                burden_factor = min(burden_factor, 0.35)
            previous = float(related_server_factors.get(instance.server_id, 0.0))
            related_server_factors[instance.server_id] = max(previous, burden_factor)
        if not related_server_factors:
            continue

        arrival = max(float(getattr(request_flow, "arrival_rate", 1.0)), 1.0)
        split_weight = arrival / max(len(related_server_factors), 1)
        for server_id, burden_factor in sorted(related_server_factors.items()):
            idx = server_index.get(server_id)
            if idx is None:
                continue
            delay_sum[idx] += flow_delay * float(burden_factor) * split_weight
            weight_sum[idx] += split_weight

    burden_vec = np.zeros(len(ai_server_ids), dtype=float)
    for idx in range(len(ai_server_ids)):
        if weight_sum[idx] > 0:
            burden_vec[idx] = delay_sum[idx] / weight_sum[idx]
    system_state._request_level_delay_burden_by_server = burden_vec
    return burden_vec


def extract_cost_components(cost_breakdown) -> Dict[str, float]:
    """
    将当前CostCalculator结果映射到论文C_topo/C_comp/C_comm字段
    C_topo包含基础设施和管理开销的工程剩余项。
    """
    cost_comm = float(getattr(cost_breakdown, "total_communication_cost", 0.0))
    cost_comp = float(getattr(cost_breakdown, "traditional_cost", 0.0)) + float(getattr(cost_breakdown, "ai_total_cost", 0.0))
    total_cost = float(getattr(cost_breakdown, "total_cost", 0.0))
    cost_topo = max(total_cost - cost_comp - cost_comm, 0.0)
    consistent = abs((cost_topo + cost_comp + cost_comm) - total_cost) <= max(1e-6, abs(total_cost) * 1e-6)
    return {
        "cost_topo": cost_topo,
        "cost_comp": cost_comp,
        "cost_comm": cost_comm,
        "cost_component_consistent": bool(consistent),
        "routing_metric_consumed": bool(getattr(cost_breakdown, "routing_metric_consumed", False)),
        "routing_probability_mass": float(getattr(cost_breakdown, "routing_probability_mass", 0.0)),
    }


def calculate_routing_communication_energy(system_state: 'SystemState') -> float:
    """按routing概率估算服务链通信能耗"""
    route_table = getattr(system_state, "stream_transfer_probabilities", {}) or {}
    total_energy = 0.0
    probability_mass = 0.0
    for flow_id, transfer_probs in route_table.items():
        request_flow = system_state.request_flows.get(flow_id)
        if request_flow is None:
            continue
        data_mb = (
            float(getattr(request_flow, "r_input_data_size", 0.0)) +
            float(getattr(request_flow, "r_output_data_size", 0.0))
        ) / 1000.0
        arrival = max(float(getattr(request_flow, "arrival_rate", 1.0)), 1.0)
        for transfer_key, raw_prob in transfer_probs.items():
            if len(transfer_key) != 4:
                continue
            source_server, _, target_server, _ = transfer_key
            prob = float(raw_prob)
            if prob <= 0:
                continue
            probability_mass += prob
            delay = 1.0
            if getattr(system_state, "network_topology", None):
                delay = system_state.network_topology.get_communication_delay(source_server, target_server)
            distance_factor = 1.0 + float(delay) / 100.0
            total_energy += data_mb * arrival * prob * 0.00045 * distance_factor
    system_state._routing_energy_consumed = probability_mass > 0.0
    system_state._routing_energy_probability_mass = probability_mass
    return float(total_energy)


def calculate_ai_replica_idle_energy(system_state: 'SystemState') -> float:
    """估算AI副本的active/standby缓存摊销能耗。
    active service-chain energy只对当前pair动作消费的副本计高额摊销；未使用副本按低功耗standby计入。
    """
    total_idle = 0.0
    for instance in system_state.microservice_instances.values():
        if instance.microservice.service_type != "ai":
            continue
        server = system_state.edge_servers.get(instance.server_id)
        if server is None:
            continue
        reserved_gpu = float(getattr(instance, "gpu_units_reserved", 0.0))
        reserved_mem = float(getattr(instance, "gpu_memory_reserved", 0.0))
        reserved_storage = float(getattr(instance, "model_storage_reserved", 0.0))
        base_part = 0.0015 * reserved_gpu + 0.00008 * reserved_mem + 0.00003 * reserved_storage
        active_local = int(getattr(instance, "active_local_pair_count", 0))
        active_cloud = int(getattr(instance, "active_cloud_pair_count", 0))
        active_total = int(getattr(instance, "active_pair_count", active_local + active_cloud))
        if active_local > 0:
            # 本地推理副本需要保持模型和GPU上下文热状态。
            factor = 1.0
            cache_part = 0.0008
        elif active_cloud > 0:
            # 云端卸载仍保留轻量预处理/缓存状态，但不按本地GPU热副本收费。
            factor = 0.25
            cache_part = 0.0005
        elif active_total > 0:
            factor = 0.35
            cache_part = 0.0005
        else:
            # 未被当前服务链消费的副本只计低功耗standby，避免慢层副本数掩盖UAC快层动作贡献。
            factor = 0.08
            cache_part = 0.00012
        total_idle += base_part * factor + cache_part
    return float(total_idle)


def calculate_system_active_ai_chain_energy(system_state: 'SystemState',
                                            active_ai_energy_j: float) -> Dict[str, float]:
    """
    计算C4论文表使用的system-wide active AI chain energy
    active_ai_energy_j保留为兼容字段，system项包含服务链通信和副本摊销。
    """
    local_gpu = 0.0
    cloud_compute = 0.0
    cloud_comm = 0.0
    cloud_preprocess = 0.0
    for instance in system_state.microservice_instances.values():
        if instance.microservice.service_type != "ai":
            continue
        local_gpu += float(getattr(instance, "energy_local_gpu_j", 0.0))
        cloud_compute += float(getattr(instance, "energy_cloud_compute_j", 0.0))
        cloud_preprocess += float(getattr(instance, "energy_preprocess_j", 0.0))
        cloud_comm += float(getattr(instance, "energy_comm_j", 0.0))
    routing_comm = calculate_routing_communication_energy(system_state)
    idle_energy = calculate_ai_replica_idle_energy(system_state)
    # f_pre预处理属于云端AI服务链能耗，汇总到cloud_compute口径。
    cloud_compute_total = cloud_compute + cloud_preprocess
    component_sum = local_gpu + cloud_compute_total + cloud_comm + routing_comm + idle_energy
    if component_sum <= 0.0:
        component_sum = float(active_ai_energy_j)
    return {
        "energy_local_gpu_j": float(local_gpu),
        "energy_cloud_compute_j": float(cloud_compute_total),
        "energy_comm_j": float(cloud_comm + routing_comm),
        "energy_idle_replica_j": float(idle_energy),
        "active_ai_energy_j": float(active_ai_energy_j),
        "system_active_ai_chain_energy_j": float(component_sum),
        "routing_energy_consumed": bool(getattr(system_state, "_routing_energy_consumed", False)),
    }


def evaluate_ai_action_shared(
        offloading_mode: np.ndarray,
        system_state: 'SystemState',
        V: float,
        slot: int,
        seed: int,
        algorithm: str,
        slow_policy: str,
        fast_controller: str,
        model_path: str = "",
        slow_context_reused: bool = False,
        paper_dpp_score: float = None,
        scaled_energy_sum: float = None,
        scaled_delay_burden_sum: float = None,
        candidate_count: int = 1,
        selected_candidate_rank: int = 0,
        replay_written: bool = False,
        online_update_step: int = 0,
        solver_gap_vs_lycd: float = 0.0,
        p95_decision_time_ms: float = 0.0,
        routing_entropy: float = 0.0,
        slow_profile_reused: bool = False,
        retry_count: int = 0,
        energy_ref_j: float = 2.0,
        delay_ref_ms: float = 50.0,
        omega_energy: float = 1.0,
        omega_delay: float = 1.0,
        resource_queue_aware: bool = True,
        config=None):
    """
    消融实验共享动作评估函数
    UAC-DO、Myopic和LyCD统一经过这里，保证指标口径一致。
    """
    from ablation_metrics import SlotResult
    from ablation_state_hash import compute_placement_hash, compute_routing_hash, compute_slow_context_hash
    from cost import CostCalculator

    start_time = time.perf_counter()
    decision_time_prior_ms = 0.0
    backup = backup_system_state(system_state)
    try:
        if isinstance(offloading_mode, dict):
            decision_meta = dict(offloading_mode)
            offloading_mode = decision_meta.get("action")
            paper_dpp_score = decision_meta.get("paper_dpp_score", paper_dpp_score)
            scaled_energy_sum = decision_meta.get("scaled_energy_sum", scaled_energy_sum)
            scaled_delay_burden_sum = decision_meta.get("scaled_delay_burden_sum", scaled_delay_burden_sum)
            candidate_count = decision_meta.get("candidate_count", candidate_count)
            selected_candidate_rank = decision_meta.get("selected_candidate_rank", selected_candidate_rank)
            replay_written = decision_meta.get("replay_written", replay_written)
            online_update_step = decision_meta.get("online_update_step", online_update_step)
            solver_gap_vs_lycd = decision_meta.get("solver_gap_vs_lycd", solver_gap_vs_lycd)
            p95_decision_time_ms = decision_meta.get("p95_decision_time_ms", p95_decision_time_ms)
            routing_entropy = decision_meta.get("routing_entropy", routing_entropy)
            slow_profile_reused = decision_meta.get("slow_profile_reused", slow_profile_reused)
            retry_count = decision_meta.get("retry_count", retry_count)
            decision_time_prior_ms = float(decision_meta.get("decision_time_ms", 0.0))
            action_dim = int(decision_meta.get("action_dim", 0))
            pair_action_dim = int(decision_meta.get("pair_action_dim", 0))
            routing_policy = str(decision_meta.get("routing_policy", ""))
            model_mutated = bool(decision_meta.get("model_mutated", False))
            best_local_score = float(decision_meta.get("best_local_score", float("inf")))
            best_cloud_score = float(decision_meta.get("best_cloud_score", float("inf")))
            score_gap_local_vs_cloud = float(decision_meta.get("score_gap_local_vs_cloud", float("inf")))
            local_candidate_feasible_count = int(decision_meta.get("local_candidate_feasible_count", 0))
            candidate_diversity = float(decision_meta.get("candidate_diversity", 0.0))
            repaired_candidate_diversity = float(decision_meta.get("repaired_candidate_diversity", candidate_diversity))
            repaired_source_distribution = str(decision_meta.get("repaired_source_distribution", ""))
            best_repaired_uac_vs_myopic_dpp_gap = float(decision_meta.get("best_repaired_uac_vs_myopic_dpp_gap", 0.0))
            best_energy_candidate_source = str(decision_meta.get("best_energy_candidate_source", ""))
            best_energy_candidate_hash = str(decision_meta.get("best_energy_candidate_hash", ""))
            best_energy_candidate_energy_j = float(decision_meta.get("best_energy_candidate_energy_j", 0.0))
            best_energy_candidate_dpp_score = float(decision_meta.get("best_energy_candidate_dpp_score", 0.0))
            best_energy_candidate_energy_gap = float(decision_meta.get("best_energy_candidate_energy_gap", 0.0))
            best_energy_candidate_dpp_gap = float(decision_meta.get("best_energy_candidate_dpp_gap", 0.0))
            all_cloud_candidate_count = int(decision_meta.get("all_cloud_candidate_count", 0))
            all_local_candidate_count = int(decision_meta.get("all_local_candidate_count", 0))
            pair_action = decision_meta.get("pair_action")
            pair_universe = decision_meta.get("pair_universe", [])
            selected_candidate_source = str(decision_meta.get("candidate_source", decision_meta.get("selected_candidate_source", "")))
            reference_pair_action_bits = str(decision_meta.get("reference_pair_action_bits", ""))
            reference_pair_action_hash = str(decision_meta.get("reference_pair_action_hash", ""))
            reference_pair_action_source = str(decision_meta.get("reference_pair_action_source", ""))
            repaired_hamming_vs_reference = float(decision_meta.get("repaired_hamming_vs_reference", 0.0))
            action_scope = str(decision_meta.get("action_scope", "pair" if pair_action is not None else "server"))
            claim_score = float(decision_meta.get("claim_score", 0.0))
            is_pareto_candidate = bool(decision_meta.get("is_pareto_candidate", False))
            dpp_band_passed = bool(decision_meta.get("dpp_band_passed", False))
            selected_by_dpp_or_claim_band = str(decision_meta.get("selected_by_dpp_or_claim_band", "dpp"))
            per_pair_delta_delay = str(decision_meta.get("per_pair_delta_delay", ""))
            per_pair_delta_energy = str(decision_meta.get("per_pair_delta_energy", ""))
            per_pair_delta_cost = str(decision_meta.get("per_pair_delta_cost", ""))
            resource_hint = str(decision_meta.get("resource_hint", ""))
            decision_resource_queue_aware = bool(decision_meta.get("resource_queue_aware", resource_queue_aware))
            decision_resource_queue_scale = float(decision_meta.get("resource_queue_scale", 1.0))
            if not resource_queue_aware:
                decision_resource_queue_aware = False
                decision_resource_queue_scale = 0.0
            resource_mode = str(decision_meta.get("resource_mode", ""))
            resource_hint_collapsed = bool(decision_meta.get("resource_hint_collapsed", False))
            candidate_source_score_summary = str(decision_meta.get("candidate_source_score_summary", ""))
            candidate_source_family_count = float(decision_meta.get("candidate_source_family_count", 0.0))
            selected_predicted_avg_y = float(decision_meta.get("predicted_avg_y", 0.0))
            selected_predicted_avg_z = float(decision_meta.get("predicted_avg_z", 0.0))
            selected_post_update_queue_drift_term = float(decision_meta.get("post_update_queue_drift_term", 0.0))
            selected_post_update_queue_pressure_term = float(decision_meta.get("post_update_queue_pressure_term", 0.0))
            selected_post_update_queue_delta_term = float(decision_meta.get("post_update_queue_delta_term", 0.0))
            selected_post_update_energy_queue_delta_term = float(decision_meta.get("post_update_energy_queue_delta_term", 0.0))
            selected_post_update_delay_queue_delta_term = float(decision_meta.get("post_update_delay_queue_delta_term", 0.0))
            tail_risk_candidate_source = str(decision_meta.get("tail_risk_candidate_source", ""))
            tail_risk_candidate_hash = str(decision_meta.get("tail_risk_candidate_hash", ""))
            tail_risk_candidate_delay_ms = float(decision_meta.get("tail_risk_candidate_delay_ms", 0.0))
            tail_risk_candidate_energy_j = float(decision_meta.get("tail_risk_candidate_energy_j", 0.0))
            tail_risk_candidate_cost = float(decision_meta.get("tail_risk_candidate_cost", 0.0))
            tail_risk_candidate_claim_score = float(decision_meta.get("tail_risk_candidate_claim_score", 0.0))
            tail_risk_candidate_dpp_score = float(decision_meta.get("tail_risk_candidate_dpp_score", 0.0))
            tail_risk_candidate_predicted_avg_y = float(decision_meta.get("tail_risk_candidate_predicted_avg_y", 0.0))
            tail_risk_candidate_predicted_avg_z = float(decision_meta.get("tail_risk_candidate_predicted_avg_z", 0.0))
            tail_risk_candidate_post_update_queue_drift_term = float(decision_meta.get("tail_risk_candidate_post_update_queue_drift_term", 0.0))
            tail_risk_candidate_upper_excess_score = float(decision_meta.get("tail_risk_candidate_upper_excess_score", 0.0))
            tail_risk_selected_upper_excess_score = float(decision_meta.get("tail_risk_selected_upper_excess_score", 0.0))
            tail_risk_candidate_upper_excess_improvement = float(decision_meta.get("tail_risk_candidate_upper_excess_improvement", 0.0))
            tail_risk_best_relief_source = str(decision_meta.get("tail_risk_best_relief_source", ""))
            tail_risk_best_relief_hash = str(decision_meta.get("tail_risk_best_relief_hash", ""))
            tail_risk_best_relief_delay_ms = float(decision_meta.get("tail_risk_best_relief_delay_ms", 0.0))
            tail_risk_best_relief_energy_j = float(decision_meta.get("tail_risk_best_relief_energy_j", 0.0))
            tail_risk_best_relief_cost = float(decision_meta.get("tail_risk_best_relief_cost", 0.0))
            tail_risk_best_relief_claim_score = float(decision_meta.get("tail_risk_best_relief_claim_score", 0.0))
            tail_risk_best_relief_dpp_score = float(decision_meta.get("tail_risk_best_relief_dpp_score", 0.0))
            tail_risk_best_relief_predicted_avg_y = float(decision_meta.get("tail_risk_best_relief_predicted_avg_y", 0.0))
            tail_risk_best_relief_predicted_avg_z = float(decision_meta.get("tail_risk_best_relief_predicted_avg_z", 0.0))
            tail_risk_best_relief_post_update_queue_drift_term = float(decision_meta.get("tail_risk_best_relief_post_update_queue_drift_term", 0.0))
            tail_risk_best_relief_upper_excess_score = float(decision_meta.get("tail_risk_best_relief_upper_excess_score", 0.0))
            tail_risk_best_relief_improvement = float(decision_meta.get("tail_risk_best_relief_improvement", 0.0))
            tail_risk_best_relief_reject_reason = str(decision_meta.get("tail_risk_best_relief_reject_reason", ""))
            energy_relief_candidate_source = str(decision_meta.get("energy_relief_candidate_source", ""))
            energy_relief_candidate_hash = str(decision_meta.get("energy_relief_candidate_hash", ""))
            energy_relief_candidate_delay_ms = float(decision_meta.get("energy_relief_candidate_delay_ms", 0.0))
            energy_relief_candidate_energy_j = float(decision_meta.get("energy_relief_candidate_energy_j", 0.0))
            energy_relief_candidate_cost = float(decision_meta.get("energy_relief_candidate_cost", 0.0))
            energy_relief_candidate_claim_score = float(decision_meta.get("energy_relief_candidate_claim_score", 0.0))
            energy_relief_candidate_dpp_score = float(decision_meta.get("energy_relief_candidate_dpp_score", 0.0))
            energy_relief_candidate_predicted_avg_y = float(decision_meta.get("energy_relief_candidate_predicted_avg_y", 0.0))
            energy_relief_candidate_predicted_avg_z = float(decision_meta.get("energy_relief_candidate_predicted_avg_z", 0.0))
            energy_relief_candidate_post_update_queue_drift_term = float(decision_meta.get("energy_relief_candidate_post_update_queue_drift_term", 0.0))
            energy_relief_candidate_energy_gain_j = float(decision_meta.get("energy_relief_candidate_energy_gain_j", 0.0))
            energy_relief_candidate_delay_regret_ms = float(decision_meta.get("energy_relief_candidate_delay_regret_ms", 0.0))
            energy_relief_candidate_cost_regret = float(decision_meta.get("energy_relief_candidate_cost_regret", 0.0))
            energy_relief_candidate_dpp_regret = float(decision_meta.get("energy_relief_candidate_dpp_regret", 0.0))
            energy_relief_best_lower_source = str(decision_meta.get("energy_relief_best_lower_source", ""))
            energy_relief_best_lower_hash = str(decision_meta.get("energy_relief_best_lower_hash", ""))
            energy_relief_best_lower_delay_ms = float(decision_meta.get("energy_relief_best_lower_delay_ms", 0.0))
            energy_relief_best_lower_energy_j = float(decision_meta.get("energy_relief_best_lower_energy_j", 0.0))
            energy_relief_best_lower_cost = float(decision_meta.get("energy_relief_best_lower_cost", 0.0))
            energy_relief_best_lower_claim_score = float(decision_meta.get("energy_relief_best_lower_claim_score", 0.0))
            energy_relief_best_lower_dpp_score = float(decision_meta.get("energy_relief_best_lower_dpp_score", 0.0))
            energy_relief_best_lower_predicted_avg_y = float(decision_meta.get("energy_relief_best_lower_predicted_avg_y", 0.0))
            energy_relief_best_lower_predicted_avg_z = float(decision_meta.get("energy_relief_best_lower_predicted_avg_z", 0.0))
            energy_relief_best_lower_post_update_queue_drift_term = float(decision_meta.get("energy_relief_best_lower_post_update_queue_drift_term", 0.0))
            energy_relief_best_lower_energy_gain_j = float(decision_meta.get("energy_relief_best_lower_energy_gain_j", 0.0))
            energy_relief_best_lower_delay_regret_ms = float(decision_meta.get("energy_relief_best_lower_delay_regret_ms", 0.0))
            energy_relief_best_lower_cost_regret = float(decision_meta.get("energy_relief_best_lower_cost_regret", 0.0))
            energy_relief_best_lower_dpp_regret = float(decision_meta.get("energy_relief_best_lower_dpp_regret", 0.0))
            energy_relief_best_lower_reject_reason = str(decision_meta.get("energy_relief_best_lower_reject_reason", ""))
        else:
            action_dim = 0
            pair_action_dim = 0
            routing_policy = ""
            model_mutated = False
            best_local_score = float("inf")
            best_cloud_score = float("inf")
            score_gap_local_vs_cloud = float("inf")
            local_candidate_feasible_count = 0
            candidate_diversity = 0.0
            repaired_candidate_diversity = 0.0
            repaired_source_distribution = ""
            best_repaired_uac_vs_myopic_dpp_gap = 0.0
            best_energy_candidate_source = ""
            best_energy_candidate_hash = ""
            best_energy_candidate_energy_j = 0.0
            best_energy_candidate_dpp_score = 0.0
            best_energy_candidate_energy_gap = 0.0
            best_energy_candidate_dpp_gap = 0.0
            all_cloud_candidate_count = 0
            all_local_candidate_count = 0
            pair_action = None
            pair_universe = []
            selected_candidate_source = ""
            reference_pair_action_bits = ""
            reference_pair_action_hash = ""
            reference_pair_action_source = ""
            repaired_hamming_vs_reference = 0.0
            action_scope = "server"
            claim_score = 0.0
            is_pareto_candidate = False
            dpp_band_passed = False
            selected_by_dpp_or_claim_band = "dpp"
            per_pair_delta_delay = ""
            per_pair_delta_energy = ""
            per_pair_delta_cost = ""
            resource_hint = ""
            decision_resource_queue_aware = bool(resource_queue_aware)
            decision_resource_queue_scale = 1.0 if resource_queue_aware else 0.0
            resource_mode = ""
            resource_hint_collapsed = False
            candidate_source_score_summary = ""
            candidate_source_family_count = 0.0
            selected_predicted_avg_y = 0.0
            selected_predicted_avg_z = 0.0
            selected_post_update_queue_drift_term = 0.0
            selected_post_update_queue_pressure_term = 0.0
            selected_post_update_queue_delta_term = 0.0
            selected_post_update_energy_queue_delta_term = 0.0
            selected_post_update_delay_queue_delta_term = 0.0
            tail_risk_candidate_source = ""
            tail_risk_candidate_hash = ""
            tail_risk_candidate_delay_ms = 0.0
            tail_risk_candidate_energy_j = 0.0
            tail_risk_candidate_cost = 0.0
            tail_risk_candidate_claim_score = 0.0
            tail_risk_candidate_dpp_score = 0.0
            tail_risk_candidate_predicted_avg_y = 0.0
            tail_risk_candidate_predicted_avg_z = 0.0
            tail_risk_candidate_post_update_queue_drift_term = 0.0
            tail_risk_candidate_upper_excess_score = 0.0
            tail_risk_selected_upper_excess_score = 0.0
            tail_risk_candidate_upper_excess_improvement = 0.0
            tail_risk_best_relief_source = ""
            tail_risk_best_relief_hash = ""
            tail_risk_best_relief_delay_ms = 0.0
            tail_risk_best_relief_energy_j = 0.0
            tail_risk_best_relief_cost = 0.0
            tail_risk_best_relief_claim_score = 0.0
            tail_risk_best_relief_dpp_score = 0.0
            tail_risk_best_relief_predicted_avg_y = 0.0
            tail_risk_best_relief_predicted_avg_z = 0.0
            tail_risk_best_relief_post_update_queue_drift_term = 0.0
            tail_risk_best_relief_upper_excess_score = 0.0
            tail_risk_best_relief_improvement = 0.0
            tail_risk_best_relief_reject_reason = ""
            energy_relief_candidate_source = ""
            energy_relief_candidate_hash = ""
            energy_relief_candidate_delay_ms = 0.0
            energy_relief_candidate_energy_j = 0.0
            energy_relief_candidate_cost = 0.0
            energy_relief_candidate_claim_score = 0.0
            energy_relief_candidate_dpp_score = 0.0
            energy_relief_candidate_predicted_avg_y = 0.0
            energy_relief_candidate_predicted_avg_z = 0.0
            energy_relief_candidate_post_update_queue_drift_term = 0.0
            energy_relief_candidate_energy_gain_j = 0.0
            energy_relief_candidate_delay_regret_ms = 0.0
            energy_relief_candidate_cost_regret = 0.0
            energy_relief_candidate_dpp_regret = 0.0
            energy_relief_best_lower_source = ""
            energy_relief_best_lower_hash = ""
            energy_relief_best_lower_delay_ms = 0.0
            energy_relief_best_lower_energy_j = 0.0
            energy_relief_best_lower_cost = 0.0
            energy_relief_best_lower_claim_score = 0.0
            energy_relief_best_lower_dpp_score = 0.0
            energy_relief_best_lower_predicted_avg_y = 0.0
            energy_relief_best_lower_predicted_avg_z = 0.0
            energy_relief_best_lower_post_update_queue_drift_term = 0.0
            energy_relief_best_lower_energy_gain_j = 0.0
            energy_relief_best_lower_delay_regret_ms = 0.0
            energy_relief_best_lower_cost_regret = 0.0
            energy_relief_best_lower_dpp_regret = 0.0
            energy_relief_best_lower_reject_reason = ""

        env_manager = system_state.environment_manager
        SH, SQ, SZ = env_manager.get_state_components()
        allocation_SQ = SQ
        allocation_SZ = SZ
        decision_resource_queue_scale = min(max(float(decision_resource_queue_scale), 0.0), 1.0)
        if not resource_queue_aware:
            # Myopic控制器不读取虚拟队列；队列仍用于事后指标和状态演化。
            allocation_SQ = np.zeros_like(SQ, dtype=float)
            allocation_SZ = np.zeros_like(SZ, dtype=float)
            decision_resource_queue_scale = 0.0
        elif not decision_resource_queue_aware:
            # energy-relaxed候选只放松资源搜索，评分和队列演化仍使用真实Y/Z。
            allocation_SQ = np.zeros_like(SQ, dtype=float)
            allocation_SZ = np.zeros_like(SZ, dtype=float)
            decision_resource_queue_scale = 0.0
        elif decision_resource_queue_scale < 1.0:
            allocation_SQ = np.asarray(SQ, dtype=float) * decision_resource_queue_scale
            allocation_SZ = np.asarray(SZ, dtype=float) * decision_resource_queue_scale

        original_mode = np.array(offloading_mode, dtype=int)
        active_local_pair_indices = []
        active_local_pair_ids = []
        active_local_pair_signature = ""
        active_cloud_pair_indices = []
        active_cloud_pair_ids = []
        active_cloud_pair_signature = ""
        if action_scope == "pair" and pair_action is not None and pair_universe:
            original_pair_action = np.asarray(pair_action, dtype=int)
            legacy_reward, ai_delays, ai_energies, repaired_pair_action, forced_cloud_count = (
                AI_Offloading_Resource_Allocation_Pair(
                    original_pair_action, pair_universe, SH, allocation_SQ, allocation_SZ, system_state, V=V,
                    omega_energy=omega_energy,
                    omega_delay=omega_delay,
                    resource_hint=resource_hint,
                )
            )
            candidate = np.asarray(original_mode, dtype=int)
            local_pair_count = int(getattr(system_state, "_executed_local_pair_count", int(np.sum(repaired_pair_action == 0))))
            cloud_pair_count = int(getattr(system_state, "_executed_cloud_pair_count", int(np.sum(repaired_pair_action == 1))))
            active_local_pair_indices = list(getattr(system_state, "_executed_local_pair_indices", []))
            active_local_pair_ids = list(getattr(system_state, "_executed_local_pair_ids", []))
            active_local_pair_signature = str(getattr(system_state, "_executed_local_pair_signature", ""))
            active_cloud_pair_indices = list(getattr(system_state, "_executed_cloud_pair_indices", []))
            active_cloud_pair_ids = list(getattr(system_state, "_executed_cloud_pair_ids", []))
            active_cloud_pair_signature = str(getattr(system_state, "_executed_cloud_pair_signature", ""))
            pair_action_bits = _pair_bits_to_text(repaired_pair_action)
            pair_action_hash = _hash_action_bits(repaired_pair_action)
            repair_diag = _pair_repair_diagnostics(original_pair_action, repaired_pair_action)
        else:
            candidate = preprocess_candidate_modes([original_mode], system_state)[0]
            forced_cloud_count = int(np.sum((original_mode == 0) & (candidate == 1)))
            legacy_reward, ai_delays, ai_energies = AI_Offloading_Resource_Allocation(
                candidate, SH, allocation_SQ, allocation_SZ, system_state, V=V
            )
            local_pair_count = int(np.sum(candidate == 0))
            cloud_pair_count = int(np.sum(candidate == 1))
            pair_action_bits = _pair_bits_to_text(candidate)
            pair_action_hash = _hash_action_bits(candidate)
            repair_diag = _pair_repair_diagnostics(candidate, candidate)
        action_hash = _hash_action_bits(candidate)
        cost_breakdown = CostCalculator().calculate_total_system_cost(system_state)
        cost_value = cost_breakdown.total_cost
        cost_components = extract_cost_components(cost_breakdown)

        ai_delay_mean = float(np.mean(ai_delays)) if len(ai_delays) else 0.0
        active_ai_energy_j = float(np.sum(ai_energies)) if len(ai_energies) else 0.0
        energy_components = calculate_system_active_ai_chain_energy(system_state, active_ai_energy_j)
        system_active_ai_chain_energy_j = float(energy_components["system_active_ai_chain_energy_j"])
        energy_j = system_active_ai_chain_energy_j
        ai_server_ids = sorted([server.server_id for server in system_state.edge_servers.values()
                                if server.server_type.value == "ai_capable"])
        delay_ms = calculate_request_level_average_delay(system_state, ai_delay_mean)
        request_delay_burden_vec = (
            calculate_request_level_delay_burden_by_server(system_state, ai_server_ids, delay_ms)
            if action_scope == "pair" else None
        )
        post_update_metrics = calculate_post_update_queue_metrics(
            system_state=system_state,
            ai_server_ids=ai_server_ids,
            ai_energies=ai_energies,
            ai_delays=ai_delays,
            active_ai_energy_j=active_ai_energy_j,
            system_energy_j=system_active_ai_chain_energy_j,
            delay_burden_vec=request_delay_burden_vec,
            energy_ref_j=energy_ref_j,
            delay_ref_ms=delay_ref_ms,
            omega_energy=omega_energy,
            omega_delay=omega_delay,
        )
        for i, server_id in enumerate(ai_server_ids):
            if i < len(ai_energies) and server_id in system_state.virtual_energy_queues:
                queue_energy = float(ai_energies[i])
                if active_ai_energy_j > 0:
                    queue_energy *= system_active_ai_chain_energy_j / max(active_ai_energy_j, 1e-9)
                system_state.virtual_energy_queues[server_id].update_queue(queue_energy)
            if i < len(ai_delays) and server_id in system_state.virtual_delay_queues:
                if request_delay_burden_vec is not None and i < len(request_delay_burden_vec):
                    queue_delay = float(request_delay_burden_vec[i])
                else:
                    queue_delay = float(ai_delays[i])
                system_state.virtual_delay_queues[server_id].update_queue(queue_delay)

        updated_y = [queue.queue_state for queue in system_state.virtual_energy_queues.values()]
        updated_z = [queue.queue_state for queue in system_state.virtual_delay_queues.values()]
        avg_y = float(np.mean(updated_y)) if updated_y else 0.0
        avg_z = float(np.mean(updated_z)) if updated_z else 0.0
        system_state.total_energy_consumption = energy_j
        system_state.total_latency = delay_ms * max(len(system_state.request_flows), 1)
        dpp_components = calculate_paper_dpp_components(
            ai_delays, ai_energies, cost_value, SQ, SZ, V,
            energy_ref_j=energy_ref_j,
            delay_ref_ms=delay_ref_ms,
            omega_energy=omega_energy,
            omega_delay=omega_delay,
            system_energy_j=system_active_ai_chain_energy_j,
            delay_burden_vec=request_delay_burden_vec,
        )
        dpp_components = apply_post_update_queue_drift_score(
            dpp_components,
            post_update_metrics,
            config=config,
            queue_aware=bool(resource_queue_aware),
        )
        paper_dpp_score = dpp_components["paper_dpp_score"]
        scaled_energy_sum = dpp_components["scaled_energy_sum"]
        scaled_delay_burden_sum = dpp_components["scaled_delay_burden_sum"]
        local_count = int(local_pair_count)
        cloud_count = int(cloud_pair_count)
        routing_metric_consumed = bool(cost_components.get("routing_metric_consumed", False))
        routing_delay_consumed = bool(getattr(system_state, "_routing_delay_consumed", False))
        routing_energy_consumed = bool(energy_components.get("routing_energy_consumed", False))
        mechanism_gate_passed = (
            routing_metric_consumed and
            routing_delay_consumed and
            routing_energy_consumed and
            not (cloud_count > 0 and local_count == 0)
        )
        execution_time_ms = (time.perf_counter() - start_time) * 1000.0
        decision_time_ms = decision_time_prior_ms + execution_time_ms
        if p95_decision_time_ms == 0.0:
            p95_decision_time_ms = decision_time_ms
        else:
            p95_decision_time_ms = max(float(p95_decision_time_ms), execution_time_ms)
        placement_hash = compute_placement_hash(system_state)
        routing_hash = compute_routing_hash(system_state)
        slow_context_hash = compute_slow_context_hash(system_state, slow_policy)

        return SlotResult(
            slot=slot,
            seed=seed,
            algorithm=algorithm,
            slow_policy=slow_policy,
            fast_controller=fast_controller,
            status="ok",
            failure_reason="",
            delay_ms=delay_ms,
            energy_j=energy_j,
            cost=float(cost_value),
            avg_y=avg_y,
            avg_z=avg_z,
            dpp_score=float(paper_dpp_score),
            legacy_reward=float(legacy_reward),
            feasible=True,
            local_count=local_count,
            cloud_count=cloud_count,
            forced_cloud_count=forced_cloud_count,
            decision_time_ms=decision_time_ms,
            slow_context_reused=slow_context_reused,
            model_path=model_path,
            paper_dpp_score=float(paper_dpp_score),
            scaled_energy_sum=float(scaled_energy_sum),
            scaled_delay_burden_sum=float(scaled_delay_burden_sum),
            candidate_count=int(candidate_count),
            selected_candidate_rank=int(selected_candidate_rank),
            replay_written=bool(replay_written),
            online_update_step=int(online_update_step),
            solver_gap_vs_lycd=float(solver_gap_vs_lycd),
            p95_decision_time_ms=float(p95_decision_time_ms),
            routing_entropy=float(routing_entropy),
            slow_profile_reused=bool(slow_profile_reused),
            retry_count=int(retry_count),
            action_dim=int(action_dim or len(candidate)),
            pair_action_dim=int(pair_action_dim),
            routing_policy=routing_policy,
            cost_topo=float(cost_components["cost_topo"]),
            cost_comp=float(cost_components["cost_comp"]),
            cost_comm=float(cost_components["cost_comm"]),
            cost_component_consistent=bool(cost_components["cost_component_consistent"]),
            active_ai_energy_j=float(energy_components["active_ai_energy_j"]),
            system_active_ai_chain_energy_j=float(system_active_ai_chain_energy_j),
            energy_scope="system_active_ai_chain_energy",
            model_mutated=bool(model_mutated),
            v_cost_term=float(dpp_components["v_cost_term"]),
            energy_queue_term=float(dpp_components["energy_queue_term"]),
            delay_queue_term=float(dpp_components["delay_queue_term"]),
            best_local_score=float(best_local_score),
            best_cloud_score=float(best_cloud_score),
            score_gap_local_vs_cloud=float(score_gap_local_vs_cloud),
            local_candidate_feasible_count=int(local_candidate_feasible_count),
            candidate_diversity=float(candidate_diversity),
            all_cloud_candidate_count=int(all_cloud_candidate_count),
            all_local_candidate_count=int(all_local_candidate_count),
            routing_metric_consumed=routing_metric_consumed,
            routing_delay_consumed=routing_delay_consumed,
            routing_probability_mass=float(cost_components.get("routing_probability_mass", 0.0)),
            mechanism_gate_passed=bool(mechanism_gate_passed),
            energy_local_gpu_j=float(energy_components["energy_local_gpu_j"]),
            energy_cloud_compute_j=float(energy_components["energy_cloud_compute_j"]),
            energy_comm_j=float(energy_components["energy_comm_j"]),
            energy_idle_replica_j=float(energy_components["energy_idle_replica_j"]),
            action_hash=str(action_hash),
            pair_action_hash=str(pair_action_hash),
            repaired_pair_action_hash=str(pair_action_hash),
            original_pair_action_hash=str(repair_diag.get("original_pair_action_hash", "")),
            repair_changed_pair_count=int(repair_diag.get("repair_changed_pair_count", 0)),
            repair_changed_ratio=float(repair_diag.get("repair_changed_ratio", 0.0)),
            pair_action_bits=str(pair_action_bits),
            selected_candidate_source=str(selected_candidate_source),
            active_local_pair_indices=",".join(str(index) for index in active_local_pair_indices),
            active_local_pair_ids=";".join(str(pair_id) for pair_id in active_local_pair_ids),
            active_local_pair_signature=str(active_local_pair_signature),
            active_cloud_pair_indices=",".join(str(index) for index in active_cloud_pair_indices),
            active_cloud_pair_ids=";".join(str(pair_id) for pair_id in active_cloud_pair_ids),
            active_cloud_pair_signature=str(active_cloud_pair_signature),
            reference_pair_action_bits=str(reference_pair_action_bits),
            reference_pair_action_hash=str(reference_pair_action_hash),
            reference_pair_action_source=str(reference_pair_action_source),
            repaired_hamming_vs_reference=float(repaired_hamming_vs_reference),
            local_pair_count=int(local_pair_count),
            cloud_pair_count=int(cloud_pair_count),
            repaired_candidate_diversity=float(repaired_candidate_diversity),
            repaired_source_distribution=str(repaired_source_distribution),
            selected_action_diversity=float(repaired_candidate_diversity),
            uac_source_family_count=float(len({
                token.split('_', 1)[0] if '_' in token else token
                for token in str(repaired_source_distribution).split(';') if token
            })),
            best_repaired_uac_vs_myopic_dpp_gap=float(best_repaired_uac_vs_myopic_dpp_gap),
            best_energy_candidate_source=str(best_energy_candidate_source),
            best_energy_candidate_hash=str(best_energy_candidate_hash),
            best_energy_candidate_energy_j=float(best_energy_candidate_energy_j),
            best_energy_candidate_dpp_score=float(best_energy_candidate_dpp_score),
            best_energy_candidate_energy_gap=float(best_energy_candidate_energy_gap),
            best_energy_candidate_dpp_gap=float(best_energy_candidate_dpp_gap),
            uac_selected_source=is_uac_selected_source(selected_candidate_source),
            placement_hash=placement_hash,
            routing_hash=routing_hash,
            slow_context_hash=slow_context_hash,
            claim_score=float(claim_score),
            is_pareto_candidate=bool(is_pareto_candidate),
            dpp_band_passed=bool(dpp_band_passed),
            selected_by_dpp_or_claim_band=str(selected_by_dpp_or_claim_band),
            per_pair_delta_delay=str(per_pair_delta_delay),
            per_pair_delta_energy=str(per_pair_delta_energy),
            per_pair_delta_cost=str(per_pair_delta_cost),
            resource_hint=str(resource_hint),
            resource_queue_aware=bool(decision_resource_queue_aware),
            resource_queue_scale=float(decision_resource_queue_scale),
            resource_mode=str(resource_mode),
            resource_hint_collapsed=bool(resource_hint_collapsed),
            candidate_source_score_summary=str(candidate_source_score_summary),
            candidate_source_family_count=float(candidate_source_family_count),
            predicted_avg_y=float(post_update_metrics.get("predicted_avg_y", selected_predicted_avg_y)),
            predicted_avg_z=float(post_update_metrics.get("predicted_avg_z", selected_predicted_avg_z)),
            post_update_queue_drift_term=float(dpp_components.get("post_update_queue_drift_term", selected_post_update_queue_drift_term)),
            post_update_queue_pressure_term=float(post_update_metrics.get("post_update_queue_pressure_term", selected_post_update_queue_pressure_term)),
            post_update_queue_delta_term=float(post_update_metrics.get("post_update_queue_delta_term", selected_post_update_queue_delta_term)),
            post_update_energy_queue_delta_term=float(post_update_metrics.get("post_update_energy_queue_delta_term", selected_post_update_energy_queue_delta_term)),
            post_update_delay_queue_delta_term=float(post_update_metrics.get("post_update_delay_queue_delta_term", selected_post_update_delay_queue_delta_term)),
            post_update_queue_drift_enabled=bool(dpp_components.get("post_update_queue_drift_enabled", False)),
            tail_risk_candidate_source=str(tail_risk_candidate_source),
            tail_risk_candidate_hash=str(tail_risk_candidate_hash),
            tail_risk_candidate_delay_ms=float(tail_risk_candidate_delay_ms),
            tail_risk_candidate_energy_j=float(tail_risk_candidate_energy_j),
            tail_risk_candidate_cost=float(tail_risk_candidate_cost),
            tail_risk_candidate_claim_score=float(tail_risk_candidate_claim_score),
            tail_risk_candidate_dpp_score=float(tail_risk_candidate_dpp_score),
            tail_risk_candidate_predicted_avg_y=float(tail_risk_candidate_predicted_avg_y),
            tail_risk_candidate_predicted_avg_z=float(tail_risk_candidate_predicted_avg_z),
            tail_risk_candidate_post_update_queue_drift_term=float(tail_risk_candidate_post_update_queue_drift_term),
            tail_risk_candidate_upper_excess_score=float(tail_risk_candidate_upper_excess_score),
            tail_risk_selected_upper_excess_score=float(tail_risk_selected_upper_excess_score),
            tail_risk_candidate_upper_excess_improvement=float(tail_risk_candidate_upper_excess_improvement),
            tail_risk_best_relief_source=str(tail_risk_best_relief_source),
            tail_risk_best_relief_hash=str(tail_risk_best_relief_hash),
            tail_risk_best_relief_delay_ms=float(tail_risk_best_relief_delay_ms),
            tail_risk_best_relief_energy_j=float(tail_risk_best_relief_energy_j),
            tail_risk_best_relief_cost=float(tail_risk_best_relief_cost),
            tail_risk_best_relief_claim_score=float(tail_risk_best_relief_claim_score),
            tail_risk_best_relief_dpp_score=float(tail_risk_best_relief_dpp_score),
            tail_risk_best_relief_predicted_avg_y=float(tail_risk_best_relief_predicted_avg_y),
            tail_risk_best_relief_predicted_avg_z=float(tail_risk_best_relief_predicted_avg_z),
            tail_risk_best_relief_post_update_queue_drift_term=float(tail_risk_best_relief_post_update_queue_drift_term),
            tail_risk_best_relief_upper_excess_score=float(tail_risk_best_relief_upper_excess_score),
            tail_risk_best_relief_improvement=float(tail_risk_best_relief_improvement),
            tail_risk_best_relief_reject_reason=str(tail_risk_best_relief_reject_reason),
            energy_relief_candidate_source=str(energy_relief_candidate_source),
            energy_relief_candidate_hash=str(energy_relief_candidate_hash),
            energy_relief_candidate_delay_ms=float(energy_relief_candidate_delay_ms),
            energy_relief_candidate_energy_j=float(energy_relief_candidate_energy_j),
            energy_relief_candidate_cost=float(energy_relief_candidate_cost),
            energy_relief_candidate_claim_score=float(energy_relief_candidate_claim_score),
            energy_relief_candidate_dpp_score=float(energy_relief_candidate_dpp_score),
            energy_relief_candidate_predicted_avg_y=float(energy_relief_candidate_predicted_avg_y),
            energy_relief_candidate_predicted_avg_z=float(energy_relief_candidate_predicted_avg_z),
            energy_relief_candidate_post_update_queue_drift_term=float(energy_relief_candidate_post_update_queue_drift_term),
            energy_relief_candidate_energy_gain_j=float(energy_relief_candidate_energy_gain_j),
            energy_relief_candidate_delay_regret_ms=float(energy_relief_candidate_delay_regret_ms),
            energy_relief_candidate_cost_regret=float(energy_relief_candidate_cost_regret),
            energy_relief_candidate_dpp_regret=float(energy_relief_candidate_dpp_regret),
            energy_relief_best_lower_source=str(energy_relief_best_lower_source),
            energy_relief_best_lower_hash=str(energy_relief_best_lower_hash),
            energy_relief_best_lower_delay_ms=float(energy_relief_best_lower_delay_ms),
            energy_relief_best_lower_energy_j=float(energy_relief_best_lower_energy_j),
            energy_relief_best_lower_cost=float(energy_relief_best_lower_cost),
            energy_relief_best_lower_claim_score=float(energy_relief_best_lower_claim_score),
            energy_relief_best_lower_dpp_score=float(energy_relief_best_lower_dpp_score),
            energy_relief_best_lower_predicted_avg_y=float(energy_relief_best_lower_predicted_avg_y),
            energy_relief_best_lower_predicted_avg_z=float(energy_relief_best_lower_predicted_avg_z),
            energy_relief_best_lower_post_update_queue_drift_term=float(energy_relief_best_lower_post_update_queue_drift_term),
            energy_relief_best_lower_energy_gain_j=float(energy_relief_best_lower_energy_gain_j),
            energy_relief_best_lower_delay_regret_ms=float(energy_relief_best_lower_delay_regret_ms),
            energy_relief_best_lower_cost_regret=float(energy_relief_best_lower_cost_regret),
            energy_relief_best_lower_dpp_regret=float(energy_relief_best_lower_dpp_regret),
            energy_relief_best_lower_reject_reason=str(energy_relief_best_lower_reject_reason),
        )
    except Exception as exc:
        restore_system_state(system_state, backup)
        decision_time_ms = (time.perf_counter() - start_time) * 1000.0
        placement_hash = compute_placement_hash(system_state)
        routing_hash = compute_routing_hash(system_state)
        slow_context_hash = compute_slow_context_hash(system_state, slow_policy)
        return SlotResult(
            slot=slot,
            seed=seed,
            algorithm=algorithm,
            slow_policy=slow_policy,
            fast_controller=fast_controller,
            status="failed",
            failure_reason=str(exc),
            delay_ms=float("nan"),
            energy_j=float("nan"),
            cost=float("nan"),
            avg_y=float("nan"),
            avg_z=float("nan"),
            dpp_score=float("nan"),
            legacy_reward=float("nan"),
            feasible=False,
            local_count=0,
            cloud_count=0,
            forced_cloud_count=0,
            decision_time_ms=decision_time_ms,
            slow_context_reused=slow_context_reused,
            model_path=model_path,
            placement_hash=placement_hash,
            routing_hash=routing_hash,
            slow_context_hash=slow_context_hash,
            paper_dpp_score=float("nan"),
            scaled_energy_sum=float("nan"),
            scaled_delay_burden_sum=float("nan"),
            candidate_count=int(candidate_count),
            selected_candidate_rank=int(selected_candidate_rank),
            replay_written=bool(replay_written),
            online_update_step=int(online_update_step),
            solver_gap_vs_lycd=float(solver_gap_vs_lycd),
            p95_decision_time_ms=float(p95_decision_time_ms),
            routing_entropy=float(routing_entropy),
            slow_profile_reused=bool(slow_profile_reused),
            retry_count=int(retry_count),
            action_dim=0,
            pair_action_dim=0,
            routing_policy="",
            cost_topo=float("nan"),
            cost_comp=float("nan"),
            cost_comm=float("nan"),
            cost_component_consistent=False,
            active_ai_energy_j=float("nan"),
            system_active_ai_chain_energy_j=float("nan"),
            energy_scope="active_ai_chain_engineering",
            model_mutated=False,
            v_cost_term=float("nan"),
            energy_queue_term=float("nan"),
            delay_queue_term=float("nan"),
            best_local_score=float("nan"),
            best_cloud_score=float("nan"),
            score_gap_local_vs_cloud=float("nan"),
            local_candidate_feasible_count=0,
            candidate_diversity=0.0,
            all_cloud_candidate_count=0,
            all_local_candidate_count=0,
            routing_metric_consumed=False,
            routing_delay_consumed=False,
            routing_probability_mass=0.0,
        )


def calculate_shared_dpp_score(ai_delays: np.ndarray, ai_energies: np.ndarray,
                               SQ: np.ndarray, SZ: np.ndarray,
                               legacy_reward: float, V: float) -> float:
    """
    计算消融实验共享DPP分数
    分数越小越好，用于LyCD搜索和solver benchmark。
    """
    energy_term = float(np.sum(SQ * ai_energies[:len(SQ)])) if len(SQ) else 0.0
    delay_term = float(np.sum(SZ * ai_delays[:len(SZ)])) if len(SZ) else 0.0
    reward_term = float(legacy_reward)
    return energy_term + delay_term - V * reward_term


def calculate_paper_dpp_components(ai_delays: np.ndarray, ai_energies: np.ndarray,
                                   cost_value: float, SQ: np.ndarray, SZ: np.ndarray,
                                   V: float, energy_ref_j: float = 2.0,
                                   delay_ref_ms: float = 50.0,
                                   omega_energy: float = 1.0,
                                   omega_delay: float = 1.0,
                                   system_energy_j: float = None,
                                   delay_burden_vec: np.ndarray = None) -> Dict[str, float]:
    """
    计算C4论文DPP分量
    Phi = V*C + Y*scaled_energy + Z*scaled_delay，分量单独导出便于审计。
    """
    energy_ref = max(float(energy_ref_j), 1e-9)
    delay_ref = max(float(delay_ref_ms), 1e-9)
    energy_vec = np.asarray(ai_energies, dtype=float)
    delay_vec = np.asarray(ai_delays, dtype=float)
    y_vec = np.asarray(SQ, dtype=float)[:len(energy_vec)]
    z_vec = np.asarray(SZ, dtype=float)[:len(delay_vec)]
    if delay_burden_vec is not None:
        delay_for_queue = np.asarray(delay_burden_vec, dtype=float)[:len(delay_vec)]
    else:
        delay_for_queue = delay_vec
    # 虚拟能耗队列按节点可归属能耗更新；system_energy_j只作为论文表的system-wide能耗导出口径。
    # 云端远端计算能耗不能硬摊回边缘节点队列，否则会掩盖本地/云端卸载的队列释放效果。
    energy_for_queue = energy_vec
    scaled_energy = energy_for_queue[:len(y_vec)] / energy_ref if len(y_vec) else np.array([])
    scaled_delay = delay_for_queue[:len(z_vec)] / delay_ref if len(z_vec) else np.array([])
    energy_burden = float(np.sum(omega_energy * y_vec * scaled_energy)) if len(scaled_energy) else 0.0
    delay_burden = float(np.sum(omega_delay * z_vec * scaled_delay)) if len(scaled_delay) else 0.0
    v_cost_term = float(V) * float(cost_value)
    paper_dpp = v_cost_term + energy_burden + delay_burden
    scaled_energy_sum = (
        float(system_energy_j) / energy_ref if system_energy_j is not None
        else float(np.sum(scaled_energy))
    )
    return {
        "paper_dpp_score": float(paper_dpp),
        "scaled_energy_sum": float(scaled_energy_sum),
        "scaled_delay_burden_sum": float(np.sum(scaled_delay)),
        "v_cost_term": float(v_cost_term),
        "energy_queue_term": float(energy_burden),
        "delay_queue_term": float(delay_burden),
    }


def calculate_post_update_queue_metrics(system_state: 'SystemState',
                                        ai_server_ids: List[str],
                                        ai_energies: np.ndarray,
                                        ai_delays: np.ndarray,
                                        active_ai_energy_j: float = 0.0,
                                        system_energy_j: float = None,
                                        delay_burden_vec: np.ndarray = None,
                                        energy_ref_j: float = 2.0,
                                        delay_ref_ms: float = 50.0,
                                        omega_energy: float = 1.0,
                                        omega_delay: float = 1.0) -> Dict[str, float]:
    """Predict one-step post-update virtual queues without mutating state.

    This mirrors the real queue update in evaluate_ai_action_shared:
    Y'=max(Y+nu(E-Emax),0), Z'=max(Z+mu(D-Dmax),0). The returned pressure terms
    use the same normalized burden convention as paper DPP, but with predicted
    post-update queues for a small drift-risk tie-breaker.
    """
    energy_ref = max(float(energy_ref_j), 1e-9)
    delay_ref = max(float(delay_ref_ms), 1e-9)
    energy_vec = np.asarray(ai_energies, dtype=float).reshape(-1)
    delay_vec = (
        np.asarray(delay_burden_vec, dtype=float).reshape(-1)
        if delay_burden_vec is not None
        else np.asarray(ai_delays, dtype=float).reshape(-1)
    )
    if system_energy_j is not None and float(active_ai_energy_j or 0.0) > 0.0 and len(energy_vec):
        energy_vec = energy_vec * (float(system_energy_j) / max(float(active_ai_energy_j), 1e-9))

    current_y = []
    current_z = []
    predicted_y = []
    predicted_z = []
    current_energy_term = 0.0
    current_delay_term = 0.0
    post_energy_term = 0.0
    post_delay_term = 0.0

    for idx, server_id in enumerate(ai_server_ids):
        energy_queue = getattr(system_state, "virtual_energy_queues", {}).get(server_id)
        delay_queue = getattr(system_state, "virtual_delay_queues", {}).get(server_id)

        if energy_queue is not None and idx < len(energy_vec):
            y_now = float(getattr(energy_queue, "queue_state", 0.0))
            threshold = float(getattr(energy_queue, "energy_threshold", 0.0))
            scale = float(getattr(energy_queue, "scaling_factor", 1.0))
            energy = float(energy_vec[idx])
            y_next = max(y_now + scale * (energy - threshold), 0.0)
            scaled_energy = energy / energy_ref
            current_y.append(y_now)
            predicted_y.append(y_next)
            current_energy_term += float(omega_energy) * y_now * scaled_energy
            post_energy_term += float(omega_energy) * y_next * scaled_energy

        if delay_queue is not None and idx < len(delay_vec):
            z_now = float(getattr(delay_queue, "queue_state", 0.0))
            threshold = float(getattr(delay_queue, "delay_threshold", 0.0))
            scale = float(getattr(delay_queue, "scaling_factor", 1.0))
            delay = float(delay_vec[idx])
            z_next = max(z_now + scale * (delay - threshold), 0.0)
            scaled_delay = delay / delay_ref
            current_z.append(z_now)
            predicted_z.append(z_next)
            current_delay_term += float(omega_delay) * z_now * scaled_delay
            post_delay_term += float(omega_delay) * z_next * scaled_delay

    current_pressure = current_energy_term + current_delay_term
    post_pressure = post_energy_term + post_delay_term
    energy_delta = post_energy_term - current_energy_term
    delay_delta = post_delay_term - current_delay_term
    return {
        "current_avg_y": float(np.mean(current_y)) if current_y else 0.0,
        "current_avg_z": float(np.mean(current_z)) if current_z else 0.0,
        "predicted_avg_y": float(np.mean(predicted_y)) if predicted_y else 0.0,
        "predicted_avg_z": float(np.mean(predicted_z)) if predicted_z else 0.0,
        "current_energy_queue_term": float(current_energy_term),
        "current_delay_queue_term": float(current_delay_term),
        "post_update_energy_queue_term": float(post_energy_term),
        "post_update_delay_queue_term": float(post_delay_term),
        "post_update_queue_pressure_term": float(post_pressure),
        "post_update_queue_delta_term": float(post_pressure - current_pressure),
        "post_update_energy_queue_delta_term": float(energy_delta),
        "post_update_delay_queue_delta_term": float(delay_delta),
    }


def apply_post_update_queue_drift_score(dpp_components: Dict[str, float],
                                        post_update_metrics: Dict[str, float],
                                        config=None,
                                        queue_aware: bool = True) -> Dict[str, float]:
    """Add an optional one-step queue drift-risk term to queue-aware DPP scoring."""
    enabled = bool(queue_aware and getattr(config, "post_update_queue_drift_enabled", False))
    weight = max(float(getattr(config, "post_update_queue_drift_weight", 0.0)), 0.0)
    delta = float(post_update_metrics.get("post_update_queue_delta_term", 0.0))
    energy_delta = post_update_metrics.get("post_update_energy_queue_delta_term", None)
    delay_delta = post_update_metrics.get("post_update_delay_queue_delta_term", None)
    if energy_delta is None or delay_delta is None:
        positive_delta = max(delta, 0.0)
    else:
        positive_delta = max(float(energy_delta), 0.0) + max(float(delay_delta), 0.0)
    drift_term = float(weight * positive_delta) if enabled and np.isfinite(delta) else 0.0
    base_score = float(dpp_components.get("paper_dpp_score", float("inf")))
    if np.isfinite(base_score) and np.isfinite(drift_term):
        dpp_components["paper_dpp_score"] = float(base_score + drift_term)
    dpp_components["post_update_queue_drift_term"] = float(drift_term)
    dpp_components["post_update_queue_drift_enabled"] = bool(enabled)
    return dpp_components


def calculate_paper_dpp_score(ai_delays: np.ndarray, ai_energies: np.ndarray,
                              cost_value: float, SQ: np.ndarray, SZ: np.ndarray,
                              V: float, energy_ref_j: float = 2.0,
                              delay_ref_ms: float = 50.0,
                              omega_energy: float = 1.0,
                              omega_delay: float = 1.0) -> Tuple[float, float, float]:
    """兼容旧调用，返回DPP、scaled energy和scaled delay"""
    components = calculate_paper_dpp_components(
        ai_delays, ai_energies, cost_value, SQ, SZ, V,
        energy_ref_j=energy_ref_j,
        delay_ref_ms=delay_ref_ms,
        omega_energy=omega_energy,
        omega_delay=omega_delay,
    )
    return (
        components["paper_dpp_score"],
        components["scaled_energy_sum"],
        components["scaled_delay_burden_sum"],
    )


def evaluate_action_dry_run(offloading_mode: np.ndarray, system_state: 'SystemState',
                            config=None, queue_aware: bool = True) -> Dict:
    """
    评估候选动作但恢复系统状态
    UAC-DO、Myopic和LyCD都通过这里比较候选，避免搜索污染真实状态。
    """
    from cost import CostCalculator

    V = float(getattr(config, "V", 20.0))
    energy_ref = float(getattr(config, "energy_ref_j", 2.0))
    delay_ref = float(getattr(config, "delay_ref_ms", 50.0))
    omega_energy = float(getattr(config, "omega_energy", 1.0))
    omega_delay = float(getattr(config, "omega_delay", 1.0))
    backup = backup_system_state(system_state, include_full_deployment=False)
    try:
        candidate_meta = dict(offloading_mode) if isinstance(offloading_mode, dict) else {}
        pair_action = candidate_meta.get("pair_action")
        pair_universe = candidate_meta.get("pair_universe", [])
        action_scope = str(candidate_meta.get("action_scope", "pair" if pair_action is not None else "server"))
        selected_candidate_source = str(candidate_meta.get("candidate_source", ""))
        resource_hint = str(candidate_meta.get("resource_hint", ""))
        candidate_resource_queue_aware = bool(candidate_meta.get("resource_queue_aware", queue_aware))
        candidate_resource_queue_scale = float(candidate_meta.get("resource_queue_scale", 1.0))
        if not queue_aware:
            candidate_resource_queue_aware = False
            candidate_resource_queue_scale = 0.0
        resource_mode = str(candidate_meta.get("resource_mode", ""))
        requested_resource_signature = candidate_meta.get("requested_resource_signature", {})
        executable_resource_signature = candidate_meta.get("executable_resource_signature", {})
        resource_hint_collapsed = mark_resource_hint_collapse(
            resource_hint,
            requested_resource_signature,
            executable_resource_signature,
        )
        candidate_action = candidate_meta.get("action", offloading_mode)
        env_manager = system_state.environment_manager
        SH, score_SQ, score_SZ = env_manager.get_state_components()
        allocation_SQ = score_SQ
        allocation_SZ = score_SZ
        candidate_resource_queue_scale = min(max(float(candidate_resource_queue_scale), 0.0), 1.0)
        if not queue_aware:
            score_SQ = np.zeros_like(score_SQ, dtype=float)
            score_SZ = np.zeros_like(score_SZ, dtype=float)
            allocation_SQ = score_SQ
            allocation_SZ = score_SZ
            candidate_resource_queue_scale = 0.0
        elif not candidate_resource_queue_aware:
            allocation_SQ = np.zeros_like(score_SQ, dtype=float)
            allocation_SZ = np.zeros_like(score_SZ, dtype=float)
            candidate_resource_queue_scale = 0.0
        elif candidate_resource_queue_scale < 1.0:
            allocation_SQ = np.asarray(score_SQ, dtype=float) * candidate_resource_queue_scale
            allocation_SZ = np.asarray(score_SZ, dtype=float) * candidate_resource_queue_scale

        original_mode = np.array(candidate_action, dtype=int)
        active_local_pair_indices = []
        active_local_pair_ids = []
        active_local_pair_signature = ""
        active_cloud_pair_indices = []
        active_cloud_pair_ids = []
        active_cloud_pair_signature = ""
        if action_scope == "pair" and pair_action is not None and pair_universe:
            original_pair_action = np.asarray(pair_action, dtype=int)
            legacy_reward, ai_delays, ai_energies, repaired_pair_action, forced_cloud_count = (
                AI_Offloading_Resource_Allocation_Pair(
                    original_pair_action, pair_universe, SH, allocation_SQ, allocation_SZ, system_state, V=V,
                    omega_energy=omega_energy,
                    omega_delay=omega_delay,
                    resource_hint=resource_hint,
                )
            )
            candidate = original_mode
            local_pair_count = int(getattr(system_state, "_executed_local_pair_count", int(np.sum(repaired_pair_action == 0))))
            cloud_pair_count = int(getattr(system_state, "_executed_cloud_pair_count", int(np.sum(repaired_pair_action == 1))))
            active_local_pair_indices = list(getattr(system_state, "_executed_local_pair_indices", []))
            active_local_pair_ids = list(getattr(system_state, "_executed_local_pair_ids", []))
            active_local_pair_signature = str(getattr(system_state, "_executed_local_pair_signature", ""))
            active_cloud_pair_indices = list(getattr(system_state, "_executed_cloud_pair_indices", []))
            active_cloud_pair_ids = list(getattr(system_state, "_executed_cloud_pair_ids", []))
            active_cloud_pair_signature = str(getattr(system_state, "_executed_cloud_pair_signature", ""))
            pair_action_bits = _pair_bits_to_text(repaired_pair_action)
            pair_action_hash = _hash_action_bits(repaired_pair_action)
            repair_diag = _pair_repair_diagnostics(original_pair_action, repaired_pair_action)
        else:
            candidate = preprocess_candidate_modes([original_mode], system_state)[0]
            forced_cloud_count = int(np.sum((original_mode == 0) & (candidate == 1)))
            legacy_reward, ai_delays, ai_energies = AI_Offloading_Resource_Allocation(
                candidate, SH, allocation_SQ, allocation_SZ, system_state, V=V
            )
            local_pair_count = int(np.sum(candidate == 0))
            cloud_pair_count = int(np.sum(candidate == 1))
            pair_action_bits = _pair_bits_to_text(candidate)
            pair_action_hash = _hash_action_bits(candidate)
            repair_diag = _pair_repair_diagnostics(candidate, candidate)
        action_hash = _hash_action_bits(candidate)
        cost_breakdown = CostCalculator().calculate_total_system_cost(system_state)
        cost_value = cost_breakdown.total_cost
        cost_components = extract_cost_components(cost_breakdown)
        active_ai_energy_j = float(np.sum(ai_energies)) if len(ai_energies) else 0.0
        energy_components = calculate_system_active_ai_chain_energy(system_state, active_ai_energy_j)
        system_active_ai_chain_energy_j = float(energy_components["system_active_ai_chain_energy_j"])
        ai_delay_mean = float(np.mean(ai_delays)) if len(ai_delays) else 0.0
        delay_ms = calculate_request_level_average_delay(system_state, ai_delay_mean)
        request_delay_burden_vec = (
            calculate_request_level_delay_burden_by_server(
                system_state,
                sorted([server.server_id for server in system_state.edge_servers.values()
                        if server.server_type.value == "ai_capable"]),
                delay_ms,
            )
            if action_scope == "pair" else None
        )
        ai_server_ids = sorted([server.server_id for server in system_state.edge_servers.values()
                                if server.server_type.value == "ai_capable"])
        post_update_metrics = calculate_post_update_queue_metrics(
            system_state=system_state,
            ai_server_ids=ai_server_ids,
            ai_energies=ai_energies,
            ai_delays=ai_delays,
            active_ai_energy_j=active_ai_energy_j,
            system_energy_j=system_active_ai_chain_energy_j,
            delay_burden_vec=request_delay_burden_vec,
            energy_ref_j=energy_ref,
            delay_ref_ms=delay_ref,
            omega_energy=omega_energy,
            omega_delay=omega_delay,
        )
        dpp_components = calculate_paper_dpp_components(
            ai_delays, ai_energies, cost_value, score_SQ, score_SZ, V,
            energy_ref_j=energy_ref,
            delay_ref_ms=delay_ref,
            omega_energy=omega_energy,
            omega_delay=omega_delay,
            system_energy_j=system_active_ai_chain_energy_j,
            delay_burden_vec=request_delay_burden_vec,
        )
        if not queue_aware:
            # Myopic只看即时成本、能耗和延迟，不读取虚拟队列Y/Z
            dpp_components["energy_queue_term"] = omega_energy * dpp_components["scaled_energy_sum"]
            dpp_components["delay_queue_term"] = omega_delay * dpp_components["scaled_delay_burden_sum"]
            dpp_components["paper_dpp_score"] = (
                dpp_components["v_cost_term"] +
                dpp_components["energy_queue_term"] +
                dpp_components["delay_queue_term"]
            )
        dpp_components = apply_post_update_queue_drift_score(
            dpp_components,
            post_update_metrics,
            config=config,
            queue_aware=bool(queue_aware),
        )
        energy_j = system_active_ai_chain_energy_j
        return {
            "action": candidate,
            "pair_action": np.asarray(repaired_pair_action if action_scope == "pair" and pair_action is not None and pair_universe else candidate, dtype=int),
            "pair_universe": pair_universe,
            "action_scope": action_scope,
            "feasible": True,
            "failure_reason": "",
            "paper_dpp_score": float(dpp_components["paper_dpp_score"]),
            "delay_ms": delay_ms,
            "energy_j": energy_j,
            "cost": float(cost_value),
            "scaled_energy_sum": float(dpp_components["scaled_energy_sum"]),
            "scaled_delay_burden_sum": float(dpp_components["scaled_delay_burden_sum"]),
            "v_cost_term": float(dpp_components["v_cost_term"]),
            "energy_queue_term": float(dpp_components["energy_queue_term"]),
            "delay_queue_term": float(dpp_components["delay_queue_term"]),
            "predicted_avg_y": float(post_update_metrics.get("predicted_avg_y", 0.0)),
            "predicted_avg_z": float(post_update_metrics.get("predicted_avg_z", 0.0)),
            "post_update_energy_queue_term": float(post_update_metrics.get("post_update_energy_queue_term", 0.0)),
            "post_update_delay_queue_term": float(post_update_metrics.get("post_update_delay_queue_term", 0.0)),
            "post_update_queue_pressure_term": float(post_update_metrics.get("post_update_queue_pressure_term", 0.0)),
            "post_update_queue_delta_term": float(post_update_metrics.get("post_update_queue_delta_term", 0.0)),
            "post_update_energy_queue_delta_term": float(post_update_metrics.get("post_update_energy_queue_delta_term", 0.0)),
            "post_update_delay_queue_delta_term": float(post_update_metrics.get("post_update_delay_queue_delta_term", 0.0)),
            "post_update_queue_drift_term": float(dpp_components.get("post_update_queue_drift_term", 0.0)),
            "post_update_queue_drift_enabled": bool(dpp_components.get("post_update_queue_drift_enabled", False)),
            "legacy_reward": float(legacy_reward),
            "local_count": int(local_pair_count),
            "cloud_count": int(cloud_pair_count),
            "forced_cloud_count": forced_cloud_count,
            "cost_topo": float(cost_components["cost_topo"]),
            "cost_comp": float(cost_components["cost_comp"]),
            "cost_comm": float(cost_components["cost_comm"]),
            "cost_component_consistent": bool(cost_components["cost_component_consistent"]),
            "routing_metric_consumed": bool(cost_components.get("routing_metric_consumed", False)),
            "routing_delay_consumed": bool(getattr(system_state, "_routing_delay_consumed", False)),
            "routing_probability_mass": float(cost_components.get("routing_probability_mass", 0.0)),
            "active_ai_energy_j": float(energy_components["active_ai_energy_j"]),
            "system_active_ai_chain_energy_j": float(system_active_ai_chain_energy_j),
            "energy_scope": "system_active_ai_chain_energy",
            "energy_local_gpu_j": float(energy_components["energy_local_gpu_j"]),
            "energy_cloud_compute_j": float(energy_components["energy_cloud_compute_j"]),
            "energy_comm_j": float(energy_components["energy_comm_j"]),
            "energy_idle_replica_j": float(energy_components["energy_idle_replica_j"]),
            "action_hash": str(action_hash),
            "pair_action_hash": str(pair_action_hash),
            "repaired_pair_action_hash": str(pair_action_hash),
            "original_pair_action_hash": str(repair_diag.get("original_pair_action_hash", "")),
            "repair_changed_pair_count": int(repair_diag.get("repair_changed_pair_count", 0)),
            "repair_changed_ratio": float(repair_diag.get("repair_changed_ratio", 0.0)),
            "pair_action_bits": str(pair_action_bits),
            "active_local_pair_indices": [int(index) for index in active_local_pair_indices],
            "active_local_pair_ids": [str(pair_id) for pair_id in active_local_pair_ids],
            "active_local_pair_signature": str(active_local_pair_signature),
            "active_cloud_pair_indices": [int(index) for index in active_cloud_pair_indices],
            "active_cloud_pair_ids": [str(pair_id) for pair_id in active_cloud_pair_ids],
            "active_cloud_pair_signature": str(active_cloud_pair_signature),
            "selected_candidate_source": selected_candidate_source,
            "candidate_source": selected_candidate_source,
            "resource_hint": resource_hint,
            "resource_queue_aware": bool(candidate_resource_queue_aware),
            "resource_queue_scale": float(candidate_resource_queue_scale),
            "resource_mode": resource_mode,
            "resource_hint_collapsed": bool(resource_hint_collapsed),
            "local_pair_count": int(local_pair_count),
            "cloud_pair_count": int(cloud_pair_count),
            "uac_selected_source": is_uac_selected_source(selected_candidate_source),
        }
    except Exception as exc:
        failed_action = offloading_mode.get("action") if isinstance(offloading_mode, dict) else offloading_mode
        return {
            "action": np.array(failed_action, dtype=int),
            "feasible": False,
            "failure_reason": str(exc),
            "paper_dpp_score": float("inf"),
            "delay_ms": float("nan"),
            "energy_j": float("nan"),
            "cost": float("nan"),
            "scaled_energy_sum": float("nan"),
            "scaled_delay_burden_sum": float("nan"),
            "v_cost_term": float("inf"),
            "energy_queue_term": float("inf"),
            "delay_queue_term": float("inf"),
            "legacy_reward": float("nan"),
            "local_count": 0,
            "cloud_count": 0,
            "forced_cloud_count": 0,
            "cost_topo": float("inf"),
            "cost_comp": float("inf"),
            "cost_comm": float("inf"),
            "cost_component_consistent": False,
            "routing_metric_consumed": False,
            "routing_delay_consumed": False,
            "routing_probability_mass": 0.0,
            "active_ai_energy_j": float("inf"),
            "system_active_ai_chain_energy_j": float("inf"),
            "energy_scope": "active_ai_chain_engineering",
            "resource_hint": str(candidate_meta.get("resource_hint", "")) if isinstance(offloading_mode, dict) else "",
            "resource_hint_collapsed": False,
        }
    finally:
        restore_system_state(system_state, backup)


def evaluate_ai_action_for_search(offloading_mode: np.ndarray, SH: np.ndarray, SQ: np.ndarray,
                                  SZ: np.ndarray, system_state: 'SystemState', V: float) -> float:
    """
    LyCD搜索专用评估函数
    每次候选动作评估后恢复系统状态，避免搜索污染真实运行状态。
    """
    backup = backup_system_state(system_state, include_full_deployment=False)
    try:
        legacy_reward, ai_delays, ai_energies = AI_Offloading_Resource_Allocation(
            offloading_mode, SH, SQ, SZ, system_state, V=V
        )
        from cost import CostCalculator
        cost_value = CostCalculator().calculate_total_system_cost(system_state).total_cost
        paper_dpp, _, _ = calculate_paper_dpp_score(ai_delays, ai_energies, cost_value, SQ, SZ, V)
        return paper_dpp
    except Exception:
        return float("inf")
    finally:
        restore_system_state(system_state, backup)


def preprocess_candidate_modes(candidate_modes: List[np.ndarray],
                               system_state: 'SystemState') -> List[np.ndarray]:
    """
    预处理候选动作：检查AI服务器资源充足性，强制转换资源不足的服务器为云端卸载
    Args:
        candidate_modes: 原始候选动作列表
        system_state: 系统状态对象

    Returns:
        List[np.ndarray]: 预处理后的候选动作列表
    """
    #print(f"\n--- 候选动作预处理：资源充足性检查---")

    # 获取AI服务器列表
    ai_servers = [server for server in system_state.edge_servers.values()
                  if server.server_type.value == "ai_capable"]
    ai_server_ids = sorted([server.server_id for server in ai_servers])
    N = len(ai_server_ids)

    if not candidate_modes or len(candidate_modes[0]) != N:
        print("候选动作为空或维度不匹配")
        return candidate_modes

    # 预先检查每个AI服务器的资源充足性（只检查一次）
    resource_sufficient = np.ones(N, dtype=bool)  # 默认资源充足

    #print(f"一次性检查 {N} 个AI服务器的资源充足性:")
    for i, server_id in enumerate(ai_server_ids):
        sufficient = check_local_resource_sufficiency(server_id, system_state)
        resource_sufficient[i] = sufficient

        if not sufficient:
            print(f"  AI服务器 {i} ({server_id}): 资源不足 → 强制云端卸载")


    # 预处理所有候选动作
    processed_modes = []
    conversion_count = 0

    for mode_idx, original_mode in enumerate(candidate_modes):
        processed_mode = original_mode.copy()
        mode_changed = False

        # 检查每个AI服务器
        for server_idx in range(N):
            # 如果原始决策是本地处理(0)但资源不足，强制转为云端卸载(1)
            if original_mode[server_idx] == 0 and not resource_sufficient[server_idx]:
                processed_mode[server_idx] = 1
                mode_changed = True

        processed_modes.append(processed_mode)

        if mode_changed:
            conversion_count += 1
            print(f"  候选{mode_idx + 1}: {original_mode} → {processed_mode} (强制转换)")


    #print(f"预处理完成: {conversion_count}/{len(candidate_modes)} 个候选动作被修改")

    return processed_modes


def optimize_gpu_allocation_for_local_processing(
        request_flow: 'RequestFlow',
        ai_microservice: 'Microservice',
        server: 'EdgeServer',
        system_state: 'SystemState',
        SQ_value: float,  # 虚拟能耗队列状态
        SZ_value: float,  # 虚拟延迟队列状态
        performance_factor: float = 1.0,
        V: float = 2.0,  # 【修改1】新增V参数，与主函数保持一致
        weight: float = 1.0  # 【修改2】新增权重参数，默认为1.0
) -> Dict:
    """
    优化本地AI处理的GPU资源分配
    【修改】目标函数与AI_Offloading_Resource_Allocation保持完全一致
    """
    from Deployment import calculate_required_gpu_memory, calculate_required_model_storage
    import EnergyConsumption

    # 获取约束条件
    energy_threshold = server.energy_threshold
    delay_threshold = server.delay_threshold

    best_config = None
    max_objective = float('-inf')

    # 搜索GPU单元数范围
    max_gpu_units = min(server.available_gpu_units, 16)

    if max_gpu_units <= 0:
        print(f"      GPU优化失败: 无可用GPU单元 (available_gpu_units={server.available_gpu_units})")
        return None

    valid_configs = 0

    for gpu_units in range(1, max_gpu_units + 1):
        try:
            # 计算GPU内存和存储需求
            required_gpu_memory = calculate_required_gpu_memory(request_flow, ai_microservice, gpu_units, False)
            required_model_storage = calculate_required_model_storage(ai_microservice)

            # 检查资源约束
            if required_gpu_memory > server.available_gpu_memory:
                continue
            if required_model_storage > server.available_model_storage:
                continue

            # 基于指定GPU单元数计算延迟
            latency_result = calculate_latency_with_fixed_gpu_units(request_flow, ai_microservice, server, gpu_units,
                                                                    system_state)

            if latency_result is None:
                continue
            elif latency_result[0] >= 20.0:
                continue

            total_latency, queue_delay, processing_delay = latency_result

            # 计算优化后的能耗
            energy = calculate_optimized_local_energy(
                server, gpu_units, required_gpu_memory, required_model_storage)

            # 【修改3】使用与AI_Offloading_Resource_Allocation完全一致的目标函数
            arrival_rate = request_flow.arrival_rate

            # 1. 基础奖励和服务奖励（与主函数一致）
            base_reward = 10
            service_reward = arrival_rate * weight * 0.05  # 【修改】使用0.05系数

            # 2. 计算相对违反程度（标准化到0-1，与主函数一致）
            energy_violation_ratio = max(0.0, (energy - energy_threshold) / energy_threshold)
            delay_violation_ratio = max(0.0, (total_latency - delay_threshold) / delay_threshold)

            # 3. Lyapunov控制项（与主函数完全一致）
            energy_penalty = V * SQ_value * energy_violation_ratio
            delay_penalty = V * SZ_value * delay_violation_ratio * 1.2  # 【修改】添加1.2系数

            # 4. 最终目标函数（与主函数完全一致）
            objective = base_reward + service_reward - energy_penalty - delay_penalty

            valid_configs += 1

            # 最大化目标函数
            if objective > max_objective:
                max_objective = objective
                best_config = {
                    'gpu_units': gpu_units,
                    'gpu_memory': required_gpu_memory,
                    'model_storage': required_model_storage,
                    'latency': total_latency,
                    'energy': energy,
                    'queue_delay': queue_delay,
                    'processing_delay': processing_delay,
                    'objective': objective,
                    # 【修改4】新增详细信息用于调试
                    'energy_violation_ratio': energy_violation_ratio,
                    'delay_violation_ratio': delay_violation_ratio,
                    'energy_penalty': energy_penalty,
                    'delay_penalty': delay_penalty
                }

        except Exception as e:
            print(f"        GPU单元数{gpu_units}优化失败: {e}")
            continue

    return best_config

def calculate_latency_with_fixed_gpu_units(request_flow, ai_microservice, server, gpu_units, system_state, batch_size=None):
    """
    基于固定GPU单元数计算AI推理延迟
    """
    # 到达率
    arrival_rate = request_flow.arrival_rate

    # 计算基础处理时间（单GPU单元）
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

    # 单GPU单元的基础处理时间
    effective_prefill_time = effective_input_tokens / prefill_speed
    effective_decode_time = effective_output_tokens / decode_speed
    base_processing_time = (effective_prefill_time + effective_decode_time) * 1000

    # 单GPU单元服务率使用 req/s。base_processing_time 已是 ms，不能直接与 req/s 到达率比较。
    single_gpu_service_rate = 1000.0 / max(base_processing_time, 1e-9)

    # 批处理效率。默认保持旧调用口径；本地(g,b,f_GPU)枚举会显式传入候选batch_size，
    # 避免先用batch=1稳定性门禁把可由批处理救回的本地候选直接丢弃。
    max_batch_size = server.max_batch_size
    if batch_size is None:
        batch_window_s = 0.05
        optimal_batch_size = min(max_batch_size, max(1, int(np.ceil(arrival_rate * batch_window_s))))
    else:
        optimal_batch_size = min(max_batch_size, max(1, int(batch_size)))
    batch_efficiency_factor = 0.8
    batch_throughput_multiplier = 1.0 + (optimal_batch_size - 1) * batch_efficiency_factor

    # 考虑批处理后的单GPU单元有效服务率
    single_gpu_effective_service_rate = single_gpu_service_rate * batch_throughput_multiplier

    # 关键修复：直接使用指定的GPU单元数，而不是重新计算
    # 多GPU并行效率衰减
    if gpu_units > 1:
        parallel_efficiency = 0.95 ** (gpu_units - 1)
    else:
        parallel_efficiency = 1.0

    # 最终总服务率（基于指定的GPU单元数）
    total_service_rate = single_gpu_effective_service_rate * gpu_units * parallel_efficiency

    # 排队论模型计算
    if arrival_rate >= total_service_rate:
        return None  # 系统不稳定

    # 计算服务强度
    rho = arrival_rate / total_service_rate

    # M/M/c近似排队延迟。服务率是 req/s，因此等待时间先按秒计算，再转为 ms。
    if gpu_units == 1:
        queue_delay_s = rho / (total_service_rate * (1 - rho))
    else:
        queue_delay_s = rho / (total_service_rate * (1 - rho)) * (1 / gpu_units)
    queue_delay = queue_delay_s * 1000.0

    # 总延迟 = 队列延迟 + 处理延迟
    total_latency = queue_delay + base_processing_time

    return total_latency, queue_delay, base_processing_time


def calculate_optimized_local_energy(server, gpu_units, gpu_memory, model_storage):
    """
    计算优化配置下的本地处理能耗
    修正版本：调用专用的本地AI处理能耗函数
    """
    try:
        # 【修正】直接调用专用的本地AI处理能耗函数
        from EnergyConsumption import calculate_local_ai_processing_energy

        energy = calculate_local_ai_processing_energy(server, gpu_units, gpu_memory, model_storage)

        # 检查能耗计算结果
        if np.isinf(energy) or np.isnan(energy) or energy < 0:
            print(f"          能耗计算结果异常: {energy}")
            return 0.1  # 返回默认值

        return energy

    except Exception as e:
        print(f"          能耗计算异常: {e}")
        return 0.1  # 返回默认值


def optimize_cloud_transmission_strategy(
        request_flow: 'RequestFlow',
        ai_microservice: 'Microservice',
        server: 'EdgeServer',
        system_state: 'SystemState',
        SQ_value: float,
        SZ_value: float,
        performance_factor: float = 1.0,
        V: float = 2.0  # Lyapunov控制参数
) -> Dict:
    """
    优化云端卸载的传输策略
    使用LyDROO风格的Lyapunov优化：最大化(服务奖励 - V*队列惩罚)
    """
    from Deployment import evaluate_cloud_deployment

    # 获取基础云端评估
    base_cloud_eval = evaluate_cloud_deployment(request_flow, ai_microservice, system_state)

    # 约束阈值（用于计算违反程度，不是硬约束）
    delay_threshold = server.delay_threshold
    energy_threshold = server.energy_threshold

    def evaluate_lyapunov_objective(compression_ratio):
        """
        评估Lyapunov目标函数：最大化(服务奖励 - V*队列惩罚)
        完全模仿LyDROO的目标函数设计
        """
        try:
            # 计算压缩开销
            if compression_ratio >= 0.9:
                compression_overhead = 0.0
            elif compression_ratio >= 0.75:
                compression_overhead = 0.8
            elif compression_ratio >= 0.55:
                compression_overhead = 2.2
            else:
                compression_overhead = 4.5

            strategy = {'ratio': compression_ratio, 'overhead': compression_overhead}
            metrics = calculate_optimized_cloud_metrics(
                request_flow, base_cloud_eval, strategy, performance_factor, server)

            if metrics is None:
                return -float('inf')  # 计算失败时返回最小值

            latency, comm_energy, proc_energy = metrics
            total_energy = comm_energy + proc_energy

            # 获取基础参数
            arrival_rate = request_flow.arrival_rate

            # 1. 服务奖励：与主代码保持一致
            service_reward = arrival_rate * 0.1

            # 2. 新增：节能奖励（能耗越低奖励越高）
            energy_efficiency_ratio = max(0.0, (energy_threshold - total_energy) / energy_threshold)
            energy_saving_reward = energy_efficiency_ratio  # 节能奖励系数

            # 2. 计算相对违反程度（标准化到0-1）：与主代码保持一致
            energy_violation_ratio = max(0.0, (total_energy - energy_threshold) / energy_threshold)
            delay_violation_ratio = max(0.0, (latency - delay_threshold) / delay_threshold)

            # 3. Lyapunov控制项（保持队列稳定性）：与主代码保持一致
            energy_penalty = V * SQ_value * energy_violation_ratio * 10
            delay_penalty = V * SZ_value * delay_violation_ratio * 10
            lyapunov_penalty = energy_penalty + delay_penalty

            # 4. 最终目标函数：基础奖励 + 服务奖励 - Lyapunov惩罚（与主代码保持一致）
            objective = 10 + service_reward - lyapunov_penalty + energy_saving_reward

            return objective

        except Exception:
            return -float('inf')

    # 三分搜索找最优压缩比（最大化目标函数）
    left, right = 0.3, 1.0
    epsilon = 1e-3

    #print(f"      三分法搜索最优压缩比，区间[{left}, {right}]...")

    iteration = 0
    while right - left > epsilon and iteration < 25:
        mid1 = left + (right - left) / 3
        mid2 = right - (right - left) / 3

        obj1 = evaluate_lyapunov_objective(mid1)
        obj2 = evaluate_lyapunov_objective(mid2)

        # 最大化目标函数
        if obj1 < obj2:
            left = mid1
        else:
            right = mid2

        iteration += 1

    optimal_ratio = (left + right) / 2
    optimal_objective = evaluate_lyapunov_objective(optimal_ratio)

    #print(f"      优化完成: {iteration}次迭代, 最优目标值={optimal_objective:.3f}")

    # 如果优化失败，使用保守策略
    if optimal_objective == -float('inf'):
        print(f"      优化失败，使用无压缩策略")
        optimal_ratio = 1.0

    # 计算最优配置的性能指标
    final_compression_overhead = 0.0 if optimal_ratio >= 0.9 else (
        0.8 if optimal_ratio >= 0.75 else (
            2.2 if optimal_ratio >= 0.55 else 4.5))

    final_strategy = {'ratio': optimal_ratio, 'overhead': final_compression_overhead}
    final_metrics = calculate_optimized_cloud_metrics(request_flow, base_cloud_eval, final_strategy, performance_factor, server)

    if final_metrics is None:
        print(f"      最终计算失败，使用默认配置")
        return None

    total_latency, communication_energy, processing_energy = final_metrics
    total_energy = communication_energy + processing_energy

    # 计算约束违反情况（信息展示，非硬约束）
    energy_violation = max(0.0, total_energy - energy_threshold)
    delay_violation = max(0.0, total_latency - delay_threshold)

    #print(f"      最优策略: 压缩比={optimal_ratio:.3f}")
    #print(f"      性能指标: 延迟={total_latency:.1f}ms, 能耗={total_energy:.4f}J")


    return {
        'compression_ratio': optimal_ratio,
        'compression_overhead': final_compression_overhead,
        'latency': total_latency,
        'communication_energy': communication_energy,
        'processing_energy': processing_energy,
        'total_energy': total_energy,
        'lyapunov_objective': optimal_objective,
        'energy_violation': energy_violation,
        'delay_violation': delay_violation
    }


def calculate_optimized_cloud_metrics(
        request_flow: 'RequestFlow',
        base_cloud_eval: Dict,
        compression_strategy: Dict,
        performance_factor: float,
        server: 'EdgeServer'
) -> Optional[Tuple[float, float, float]]:
    """
    计算优化压缩策略下的云端处理指标
    保持与原函数相同的计算逻辑
    """
    try:
        # 基础延迟组件
        base_latency = base_cloud_eval['base_latency']
        base_transmission_latency = base_cloud_eval['transmission_latency']
        base_processing_latency = base_cloud_eval['processing_latency']

        # 压缩参数
        compression_ratio = compression_strategy['ratio']
        compression_overhead = compression_strategy['overhead']

        # === 延迟计算 ===
        # 1. 压缩后的传输延迟
        optimized_transmission_latency = base_transmission_latency * compression_ratio

        # 2. 压缩处理开销
        compression_processing_overhead = compression_overhead * (1.0 - compression_ratio)

        # 3. 总延迟
        total_latency = (base_latency + optimized_transmission_latency +
                         base_processing_latency + compression_processing_overhead)

        # 4. 性能因子调整

        performance_adjustment = 0.95 + 0.03 / max(performance_factor, 0.1)
        total_latency *= performance_adjustment

        # === 能耗计算 ===
        from EnergyConsumption import calculate_optimized_communication_energy, calculate_cloud_processing_energy

        # 1. 通信能耗（考虑压缩）
        communication_energy = calculate_optimized_communication_energy(
            request_flow, compression_ratio, server)

        # 2. 云端处理协调能耗
        processing_energy = calculate_cloud_processing_energy(server)

        # 3. 压缩计算能耗
        if compression_ratio < 1.0:
            compression_energy = compression_overhead * 0.002
            processing_energy += compression_energy

        return total_latency, communication_energy, processing_energy

    except Exception as e:
        print(f"        云端指标计算失败: {e}")
        return None


















