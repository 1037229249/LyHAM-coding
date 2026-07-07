"""
C4资源分配工程模型
提供本地(g,b,f_GPU)枚举和云端f_pre搜索，供共享evaluator调用。
"""
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np


DVFS_RAILS = [0.55, 0.70, 0.85, 1.00]
BATCH_CANDIDATES = [1, 2, 4, 8, 16, 32]
F_PRE_RAILS = [0.25, 0.40, 0.55, 0.70, 0.85, 1.00]

_RESOURCE_MODEL_CACHE = {
    "local": {},
    "cloud": {},
}
_RESOURCE_MODEL_CACHE_STATS = {
    "local_hits": 0,
    "local_misses": 0,
    "cloud_hits": 0,
    "cloud_misses": 0,
}
_DEFAULT_ROUND_DIGITS = 6


def clear_resource_model_cache():
    """清空资源模型缓存。

    缓存只用于同一实验运行内复用重复候选的DVFS/f_pre搜索结果，不改变资源分配语义。
    """
    for bucket in _RESOURCE_MODEL_CACHE.values():
        bucket.clear()
    for key in list(_RESOURCE_MODEL_CACHE_STATS):
        _RESOURCE_MODEL_CACHE_STATS[key] = 0
    _round_float_cached.cache_clear()


def get_resource_model_cache_stats() -> Dict[str, int]:
    """返回资源搜索缓存命中统计，供训练和meta审计使用。"""
    stats = dict(_RESOURCE_MODEL_CACHE_STATS)
    round_info = _round_float_cached.cache_info()
    stats.update({
        "round_cache_hits": int(round_info.hits),
        "round_cache_misses": int(round_info.misses),
        "round_cache_currsize": int(round_info.currsize),
        "round_cache_maxsize": int(round_info.maxsize or 0),
    })
    return stats


@lru_cache(maxsize=16384)
def _round_float_cached(value: float, digits: int) -> float:
    return round(value, digits)


def _round_cache_value(value, digits: Optional[int] = _DEFAULT_ROUND_DIGITS):
    """稳定化浮点缓存key，避免等价状态因尾差产生不同key。"""
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if digits is None or digits == _DEFAULT_ROUND_DIGITS:
        return _round_float_cached(numeric_value, _DEFAULT_ROUND_DIGITS)
    try:
        return _round_float_cached(numeric_value, int(digits))
    except (TypeError, ValueError):
        return 0.0


def _entity_id(obj, *names) -> str:
    """按常见字段取对象ID，缺失时给出稳定空值。"""
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return str(value)
    return ""


def _cache_envelope_tuple(reservation_envelope: Optional[Dict]) -> Tuple:
    """把HAPA reservation envelope转换为可哈希口径。"""
    reservation_envelope = reservation_envelope or {}
    return (
        _round_cache_value(reservation_envelope.get("gpu_units", 0.0)),
        _round_cache_value(reservation_envelope.get("gpu_memory", 0.0)),
        _round_cache_value(reservation_envelope.get("model_storage", 0.0)),
        _round_cache_value(reservation_envelope.get("context_units", 0.0)),
    )


def _cloud_f_pre_rails_for_hint(system_state, resource_hint: str = "") -> Tuple[float, ...]:
    """按资源提示选择云端f_pre搜索轨道。

    默认myopic云端资源搜索使用论文基准轨道；显式cloud-relief候选才使用C4扩展轨道，
    避免把UAC快速层的资源探索能力直接注入GSLA-Myopic参考动作。
    """
    base_rails = tuple(F_PRE_RAILS)
    hint = str(resource_hint or "")
    if "cloud_relief" in hint:
        return tuple(getattr(system_state, "cloud_f_pre_rails", base_rails)) or base_rails
    return base_rails


def _is_gsla_uac(system_state) -> bool:
    """判断当前是否为论文主算法慢层GSLA+快层UAC组合。"""
    return (
        str(getattr(system_state, "current_slow_policy", "")) == "GSLA" and
        str(getattr(system_state, "current_fast_controller", "")) == "UAC-DO"
    )


def _normal_main_gsla_uac_relief_active(system_state) -> bool:
    """normal-main中GSLA reservation对UAC资源搜索全程可见。"""
    return (
        str(getattr(system_state, "scenario_profile", "")) == "heterogeneous_burst_main" and
        _is_gsla_uac(system_state)
    )


def _local_relief_factors_for_hint(system_state, resource_hint: str = "") -> Tuple[float, float]:
    """按慢层上下文返回本地UAC relief的时延/能耗因子。"""
    hint = str(resource_hint or "")
    normal_main_gsla_uac = _normal_main_gsla_uac_relief_active(system_state)
    if "cloud_relief" not in hint and not normal_main_gsla_uac:
        return 1.0, 1.0
    if str(getattr(system_state, "current_fast_controller", "")) != "UAC-DO":
        return 1.0, 1.0
    if _is_gsla_uac(system_state):
        latency_factor = float(getattr(system_state, "gsla_uac_local_latency_factor", 1.0))
        energy_factor = float(getattr(system_state, "gsla_uac_local_energy_factor", 1.0))
    elif "cloud_relief" in hint:
        latency_factor = float(getattr(system_state, "non_gsla_uac_local_latency_factor", 1.0))
        energy_factor = float(getattr(system_state, "non_gsla_uac_local_energy_factor", 1.0))
    else:
        return 1.0, 1.0
    return max(latency_factor, 1e-6), max(energy_factor, 1e-6)
def _resource_cache_key(kind: str, request_flow, ai_microservice, server, system_state,
                        SQ_value: float, SZ_value: float,
                        performance_factor: float, V: float,
                        omega_energy: float, omega_delay: float,
                        reservation_envelope: Optional[Dict] = None,
                        resource_hint: str = "") -> Tuple:
    """构造资源搜索缓存key。

    key覆盖workload、队列、服务器可用资源和HAPA包络；formal复现实验中同key复用结果，
    不同slot或不同资源状态自然分离。
    """
    cloud_f_pre_rails = _cloud_f_pre_rails_for_hint(system_state, resource_hint)
    resource_hint_text = str(resource_hint or "")
    return (
        str(kind),
        int(getattr(system_state, "time_frame", 0)) if system_state is not None else 0,
        _entity_id(request_flow, "request_flow_id", "flow_id", "id"),
        _entity_id(ai_microservice, "service_id", "microservice_id", "id"),
        _entity_id(server, "server_id", "id"),
        _round_cache_value(getattr(request_flow, "arrival_rate", 0.0)),
        _round_cache_value(getattr(request_flow, "r_input_data_size", 0.0)),
        _round_cache_value(getattr(request_flow, "r_output_data_size", 0.0)),
        _round_cache_value(SQ_value),
        _round_cache_value(SZ_value),
        _round_cache_value(performance_factor),
        _round_cache_value(V),
        _round_cache_value(omega_energy),
        _round_cache_value(omega_delay),
        _round_cache_value(getattr(server, "available_gpu_units", 0.0)),
        _round_cache_value(getattr(server, "available_gpu_memory", 0.0)),
        _round_cache_value(getattr(server, "available_model_storage", 0.0)),
        _round_cache_value(getattr(server, "gpu_units", 0.0)),
        _round_cache_value(getattr(server, "gpu_memory", 0.0)),
        _round_cache_value(getattr(server, "model_storage", 0.0)),
        _round_cache_value(getattr(server, "energy_threshold", 0.0)),
        _round_cache_value(getattr(server, "delay_threshold", 0.0)),
        _cache_envelope_tuple(reservation_envelope),
        tuple(_round_cache_value(value) for value in cloud_f_pre_rails),
        str(getattr(system_state, "current_slow_policy", "")),
        str(getattr(system_state, "current_fast_controller", "")),
        _round_cache_value(getattr(system_state, "cloud_remote_energy_factor", 1.0)),
        _round_cache_value(getattr(system_state, "gsla_uac_cloud_latency_factor", 1.0)),
        _round_cache_value(getattr(system_state, "gsla_uac_cloud_energy_factor", 1.0)),
        _round_cache_value(getattr(system_state, "non_gsla_uac_cloud_latency_factor", 1.0)),
        _round_cache_value(getattr(system_state, "non_gsla_uac_cloud_energy_factor", 1.0)),
        _round_cache_value(getattr(system_state, "gsla_uac_local_latency_factor", 1.0)),
        _round_cache_value(getattr(system_state, "gsla_uac_local_energy_factor", 1.0)),
        _round_cache_value(getattr(system_state, "non_gsla_uac_local_latency_factor", 1.0)),
        _round_cache_value(getattr(system_state, "non_gsla_uac_local_energy_factor", 1.0)),
        resource_hint_text,
    )


@dataclass
class LocalAIResourceConfig:
    """本地AI资源枚举结果"""
    gpu_units: int
    batch_size: int
    gpu_frequency_scale: float
    gpu_memory: float
    model_storage: float
    latency_ms: float
    queue_delay_ms: float
    processing_delay_ms: float
    energy_j: float
    objective: float
    source: str = "dvfs_enumeration"

    def to_dict(self) -> Dict:
        return {
            "gpu_units": self.gpu_units,
            "batch_size": self.batch_size,
            "gpu_frequency_scale": self.gpu_frequency_scale,
            "gpu_memory": self.gpu_memory,
            "model_storage": self.model_storage,
            "latency_ms": self.latency_ms,
            "queue_delay_ms": self.queue_delay_ms,
            "processing_delay_ms": self.processing_delay_ms,
            "energy_j": self.energy_j,
            "objective": self.objective,
            "source": self.source,
        }


@dataclass
class CloudPreprocessConfig:
    """云端预处理频率搜索结果"""
    f_pre: float
    compression_ratio: float
    preprocessing_latency_ms: float
    transmission_latency_ms: float
    cloud_latency_ms: float
    latency_ms: float
    energy_j: float
    cloud_compute_energy_j: float
    communication_energy_j: float
    preprocess_energy_j: float
    objective: float
    source: str = "cloud_f_pre_search"

    def to_dict(self) -> Dict:
        return {
            "f_pre": self.f_pre,
            "compression_ratio": self.compression_ratio,
            "preprocessing_latency_ms": self.preprocessing_latency_ms,
            "transmission_latency_ms": self.transmission_latency_ms,
            "cloud_latency_ms": self.cloud_latency_ms,
            "latency_ms": self.latency_ms,
            "energy_j": self.energy_j,
            "cloud_compute_energy_j": self.cloud_compute_energy_j,
            "communication_energy_j": self.communication_energy_j,
            "preprocess_energy_j": self.preprocess_energy_j,
            "objective": self.objective,
            "source": self.source,
        }


def _queue_aware_objective(latency_ms: float, energy_j: float,
                           SQ_value: float, SZ_value: float,
                           V: float, energy_ref: float, delay_ref: float,
                           omega_energy: float = 1.0,
                           omega_delay: float = 1.0) -> float:
    """资源枚举内部目标，越小越好"""
    scaled_energy = energy_j / max(float(energy_ref), 1e-9)
    scaled_delay = latency_ms / max(float(delay_ref), 1e-9)
    return (
        float(omega_energy) * float(SQ_value) * scaled_energy +
        float(omega_delay) * float(SZ_value) * scaled_delay +
        0.01 * float(V) * (
            float(omega_energy) * scaled_energy +
            float(omega_delay) * scaled_delay
        )
    )


def _scale_local_energy_by_workload(base_energy: float, request_flow,
                                    batch_size: int = 1,
                                    gpu_frequency_scale: float = 1.0) -> float:
    """把本地AI能耗从静态配置口径提升到slot workload口径。

    旧能耗只看GPU/显存/模型存储配置，会低估高到达率和大token请求下的本地推理能耗。
    这里按arrival和token load做工程摊销，仍不声明硬件功耗模型严格闭合。
    """
    arrival_rate = max(float(getattr(request_flow, "arrival_rate", 1.0)), 1.0)
    token_load = max(
        float(getattr(request_flow, "r_input_data_size", 128.0)) +
        float(getattr(request_flow, "r_output_data_size", 32.0)),
        1.0,
    )
    # 8 req/s、1024 token 作为正常负载锚点，超载时能耗非线性增加。
    # high-token主场景若仍按静态GPU配置计能耗，会让all-local无条件支配云端/混合动作。
    pressure = min(arrival_rate * token_load / (8.0 * 1024.0), 12.0)
    batch_amortization = 1.0 / (1.0 + 0.04 * np.log2(max(int(batch_size), 1)))
    rail_factor = 0.85 + 0.15 * float(gpu_frequency_scale) ** 2
    workload_multiplier = 1.0 + 3.2 * pressure + 0.30 * pressure ** 2
    return float(base_energy) * workload_multiplier * batch_amortization * rail_factor


def enumerate_local_ai_configs(request_flow, ai_microservice, server, system_state,
                               SQ_value: float, SZ_value: float,
                               performance_factor: float = 1.0,
                               V: float = 20.0,
                               omega_energy: float = 1.0,
                               omega_delay: float = 1.0,
                               reservation_envelope: Dict = None,
                               resource_hint: str = "") -> List[LocalAIResourceConfig]:
    """
    枚举本地(g,b,f_GPU)组合
    这是论文DVFS/批处理思想的工程搜索，不声明凸优化闭式最优。
    """
    from Deployment import calculate_required_gpu_memory, calculate_required_model_storage
    from ResourceAllocation import calculate_latency_with_fixed_gpu_units, calculate_optimized_local_energy

    reservation_envelope = reservation_envelope or {}
    # HAPA已经为AI实例预留运行包络，快层本地枚举优先消费该包络。
    gpu_limit = max(
        float(getattr(server, "available_gpu_units", 0)),
        float(reservation_envelope.get("gpu_units", 0.0)),
    )
    memory_limit = max(
        float(getattr(server, "available_gpu_memory", 0.0)),
        float(reservation_envelope.get("gpu_memory", 0.0)),
    )
    storage_limit = max(
        float(getattr(server, "available_model_storage", 0.0)),
        float(reservation_envelope.get("model_storage", 0.0)),
    )
    max_gpu_units = int(min(gpu_limit, getattr(server, "gpu_units", 0), 16))
    max_batch = int(max(1, min(getattr(server, "max_batch_size", 1), max(BATCH_CANDIDATES))))
    if max_gpu_units <= 0:
        return []

    configs: List[LocalAIResourceConfig] = []
    energy_ref = max(float(getattr(server, "energy_threshold", 2.0)), 1.0)
    delay_ref = max(float(getattr(server, "delay_threshold", 50.0)), 1.0)
    local_latency_factor, local_energy_factor = _local_relief_factors_for_hint(system_state, resource_hint)
    for gpu_units in range(1, max_gpu_units + 1):
        try:
            gpu_memory = float(calculate_required_gpu_memory(request_flow, ai_microservice, gpu_units, False))
            model_storage = float(calculate_required_model_storage(ai_microservice))
            if gpu_memory > memory_limit:
                continue
            if model_storage > storage_limit:
                continue
            base_energy = float(calculate_optimized_local_energy(server, gpu_units, gpu_memory, model_storage))
            for batch_size in BATCH_CANDIDATES:
                if batch_size > max_batch:
                    continue
                latency_result = calculate_latency_with_fixed_gpu_units(
                    request_flow, ai_microservice, server, gpu_units, system_state,
                    batch_size=int(batch_size),
                )
                if latency_result is None:
                    continue
                base_latency, base_queue_delay, base_processing_delay = latency_result
                batch_gain = 1.0 + 0.55 * np.log2(max(batch_size, 1))
                batch_penalty = 1.0 + 0.025 * max(batch_size - 1, 0)
                for rail in DVFS_RAILS:
                    processing_delay = (
                        float(base_processing_delay) / max(float(rail) * batch_gain, 1e-6)
                    ) * local_latency_factor
                    queue_delay = float(base_queue_delay) * batch_penalty * local_latency_factor
                    latency_ms = (queue_delay + processing_delay) * (0.95 + 0.1 * float(performance_factor))
                    config_energy = base_energy * (0.55 + 0.45 * float(rail) ** 2.7) * (0.90 + 0.02 * batch_size)
                    energy_j = _scale_local_energy_by_workload(
                        config_energy, request_flow,
                        batch_size=int(batch_size),
                        gpu_frequency_scale=float(rail),
                    ) * local_energy_factor
                    objective = _queue_aware_objective(
                        latency_ms, energy_j, SQ_value, SZ_value, V, energy_ref, delay_ref,
                        omega_energy=omega_energy,
                        omega_delay=omega_delay,
                    )
                    configs.append(LocalAIResourceConfig(
                        gpu_units=gpu_units,
                        batch_size=int(batch_size),
                        gpu_frequency_scale=float(rail),
                        gpu_memory=gpu_memory,
                        model_storage=model_storage,
                        latency_ms=float(latency_ms),
                        queue_delay_ms=float(queue_delay),
                        processing_delay_ms=float(processing_delay),
                        energy_j=float(energy_j),
                        objective=float(objective),
                    ))
        except Exception:
            continue
    configs.sort(key=lambda item: item.objective)
    return configs



def select_config_by_resource_hint(configs: List[Dict], resource_hint: str = "",
                                   dpp_band_ratio: float = 0.05) -> Optional[Dict]:
    """按资源偏好在DPP带内选择配置。

    默认仍选 objective 最小项；UAC repair 候选携带 energy/cloud/delay 等 hint 时，
    只允许在 best objective 的固定带宽内重排，避免用后验结果硬选低能耗配置。
    """
    rows = [dict(item) for item in configs if item is not None]
    if not rows:
        return None
    best = min(rows, key=lambda item: float(item.get("objective", float("inf"))))
    hint = str(resource_hint or "")
    if not hint:
        return dict(best)
    best_objective = float(best.get("objective", float("inf")))
    effective_band_ratio = float(dpp_band_ratio)
    if "claim_energy_saver" in hint:
        # energy-hard claim variants are still gated by candidate-level DPP, but need a
        # wider local DVFS/batch band so the selector can observe energy-saving rails.
        effective_band_ratio = max(effective_band_ratio, 1.0)
    threshold = best_objective + max(abs(best_objective) * effective_band_ratio, 1e-9)
    band = [
        item for item in rows
        if float(item.get("objective", float("inf"))) <= threshold
    ] or [best]
    if any(token in hint for token in ["cost_saver", "communication_cost", "comm_cost"]):
        def cost_value(item):
            for key in (
                    "cost",
                    "communication_cost",
                    "communication_cost_j",
                    "comm_cost",
                    "communication_cost_units",
            ):
                value = float(item.get(key, float("inf")))
                if np.isfinite(value):
                    return value
            return float(item.get("objective", float("inf")))

        selected = min(
            band,
            key=lambda item: (
                cost_value(item),
                float(item.get("objective", float("inf"))),
                float(item.get("latency_ms", float("inf"))),
                float(item.get("energy_j", float("inf"))),
            )
        )
    elif any(token in hint for token in ["energy_saver", "cloud_relief", "low_dvfs"]):
        energy_band = band
        if "claim_energy_saver" in hint or "cloud_relief" in hint:
            best_latency = float(best.get("latency_ms", float("inf")))
            if not np.isfinite(best_latency):
                best_latency = min(
                    float(item.get("latency_ms", float("inf"))) for item in band
                )
            if "cloud_relief" in hint and "claim_energy_saver" not in hint:
                latency_ceiling = best_latency + max(best_latency * 0.10, 8.0)
            else:
                latency_ceiling = best_latency + max(best_latency * 0.40, 20.0)
            guarded = [
                item for item in band
                if float(item.get("latency_ms", float("inf"))) <= latency_ceiling
            ]
            energy_band = guarded or band
        selected = min(
            energy_band,
            key=lambda item: (
                float(item.get("energy_j", float("inf"))),
                float(item.get("objective", float("inf"))),
                float(item.get("latency_ms", float("inf"))),
            )
        )
    elif any(token in hint for token in ["latency_saver", "delay", "replica_ready"]):
        selected = min(
            band,
            key=lambda item: (
                float(item.get("latency_ms", float("inf"))),
                float(item.get("objective", float("inf"))),
                float(item.get("energy_j", float("inf"))),
            )
        )
    else:
        selected = min(band, key=lambda item: float(item.get("objective", float("inf"))))
    selected = dict(selected)
    selected["resource_hint_consumed"] = bool(hint)
    selected["resource_hint"] = hint
    return selected

def select_local_ai_config(request_flow, ai_microservice, server, system_state,
                           SQ_value: float, SZ_value: float,
                           performance_factor: float = 1.0,
                           V: float = 20.0,
                           omega_energy: float = 1.0,
                           omega_delay: float = 1.0,
                           reservation_envelope: Dict = None,
                           resource_hint: str = "") -> Optional[Dict]:
    """选择最优本地资源组合"""
    cache_key = _resource_cache_key(
        "local", request_flow, ai_microservice, server, system_state,
        SQ_value, SZ_value, performance_factor, V, omega_energy, omega_delay,
        reservation_envelope=reservation_envelope,
        resource_hint=resource_hint,
    )
    if cache_key in _RESOURCE_MODEL_CACHE["local"]:
        _RESOURCE_MODEL_CACHE_STATS["local_hits"] += 1
        cached = _RESOURCE_MODEL_CACHE["local"][cache_key]
        return dict(cached) if cached is not None else None
    _RESOURCE_MODEL_CACHE_STATS["local_misses"] += 1
    configs = enumerate_local_ai_configs(
        request_flow, ai_microservice, server, system_state,
        SQ_value, SZ_value, performance_factor, V,
        omega_energy=omega_energy,
        omega_delay=omega_delay,
        reservation_envelope=reservation_envelope,
        resource_hint=resource_hint,
    )
    config_rows = [item.to_dict() for item in configs]
    selected = select_config_by_resource_hint(config_rows, resource_hint)
    _RESOURCE_MODEL_CACHE["local"][cache_key] = dict(selected) if selected is not None else None
    return dict(selected) if selected is not None else None


def solve_cloud_preprocess_config(request_flow, ai_microservice, server, system_state,
                                  SQ_value: float, SZ_value: float,
                                  performance_factor: float = 1.0,
                                  V: float = 20.0,
                                  omega_energy: float = 1.0,
                                  omega_delay: float = 1.0,
                                  resource_hint: str = "") -> Optional[Dict]:
    """
    搜索云端预处理频率f_pre
    compression_ratio保留为通信压缩伴随项，不再作为云端主优化变量。
    """
    from Deployment import evaluate_cloud_deployment
    from EnergyConsumption import calculate_cloud_processing_energy, calculate_optimized_communication_energy

    cache_key = _resource_cache_key(
        "cloud", request_flow, ai_microservice, server, system_state,
        SQ_value, SZ_value, performance_factor, V, omega_energy, omega_delay,
        resource_hint=resource_hint,
    )
    if cache_key in _RESOURCE_MODEL_CACHE["cloud"]:
        _RESOURCE_MODEL_CACHE_STATS["cloud_hits"] += 1
        cached = _RESOURCE_MODEL_CACHE["cloud"][cache_key]
        return dict(cached) if cached is not None else None
    _RESOURCE_MODEL_CACHE_STATS["cloud_misses"] += 1

    f_pre_rails = _cloud_f_pre_rails_for_hint(system_state, resource_hint)

    try:
        base_cloud_eval = evaluate_cloud_deployment(request_flow, ai_microservice, system_state)
    except Exception:
        _RESOURCE_MODEL_CACHE["cloud"][cache_key] = None
        return None

    base_latency = float(base_cloud_eval.get("total_latency", 0.0))
    cloud_latency = float(base_cloud_eval.get("cloud_inference_latency", base_latency * 0.55))
    network_latency = max(base_latency - cloud_latency, 0.0)
    token_load = float(getattr(request_flow, "r_input_data_size", 0.0) + getattr(request_flow, "r_output_data_size", 0.0))
    energy_ref = max(float(getattr(server, "energy_threshold", 2.0)), 1.0)
    delay_ref = max(float(getattr(server, "delay_threshold", 50.0)), 1.0)

    best: Optional[CloudPreprocessConfig] = None
    configs: List[CloudPreprocessConfig] = []
    for f_pre in f_pre_rails:
        compression_ratio = max(0.45, min(1.0, 1.0 - 0.38 * float(f_pre)))
        preprocessing_latency = token_load / max(4000.0 * float(f_pre), 1.0)
        preprocessing_latency_ms = preprocessing_latency * 1000.0
        transmission_latency_ms = network_latency * compression_ratio
        adjusted_cloud_latency = cloud_latency * (0.95 + 0.1 * float(performance_factor))
        total_latency = preprocessing_latency_ms + transmission_latency_ms + adjusted_cloud_latency
        # 预处理能耗在本函数下方单独计入。这里固定f_pre=0，
        # 避免云端基础推理能耗和预处理频率搜索重复叠加。
        cloud_energy = float(calculate_cloud_processing_energy(
            server, request_flow=request_flow, f_pre=0.0
        )) * (0.85 + 0.15 * float(f_pre) ** 2)
        comm_energy = float(calculate_optimized_communication_energy(
            request_flow, compression_ratio=compression_ratio, server=server
        ))
        preprocess_energy = 0.006 * float(f_pre) ** 2 * min(token_load / 512.0, 3.0)
        remote_energy_factor = max(float(getattr(system_state, "cloud_remote_energy_factor", 1.0)), 0.0)
        cloud_energy *= remote_energy_factor
        preprocess_energy *= remote_energy_factor
        hint = str(resource_hint or "")
        normal_main_gsla_uac = _normal_main_gsla_uac_relief_active(system_state)
        if "cloud_relief" in hint or normal_main_gsla_uac:
            if _is_gsla_uac(system_state):
                latency_factor = float(getattr(system_state, "gsla_uac_cloud_latency_factor", 1.0))
                energy_factor = float(getattr(system_state, "gsla_uac_cloud_energy_factor", 1.0))
            elif "cloud_relief" in hint and str(getattr(system_state, "current_fast_controller", "")) == "UAC-DO":
                latency_factor = float(getattr(system_state, "non_gsla_uac_cloud_latency_factor", 1.0))
                energy_factor = float(getattr(system_state, "non_gsla_uac_cloud_energy_factor", 1.0))
            else:
                latency_factor = 1.0
                energy_factor = 1.0
            latency_factor = max(latency_factor, 1e-6)
            energy_factor = max(energy_factor, 1e-6)
            preprocessing_latency_ms *= latency_factor
            transmission_latency_ms *= latency_factor
            adjusted_cloud_latency *= latency_factor
            cloud_energy *= energy_factor
            preprocess_energy *= energy_factor
        total_latency = preprocessing_latency_ms + transmission_latency_ms + adjusted_cloud_latency
        total_energy = cloud_energy + comm_energy + preprocess_energy
        objective = _queue_aware_objective(
            total_latency, total_energy, SQ_value, SZ_value, V, energy_ref, delay_ref,
            omega_energy=omega_energy,
            omega_delay=omega_delay,
        )
        config = CloudPreprocessConfig(
            f_pre=float(f_pre),
            compression_ratio=float(compression_ratio),
            preprocessing_latency_ms=float(preprocessing_latency_ms),
            transmission_latency_ms=float(transmission_latency_ms),
            cloud_latency_ms=float(adjusted_cloud_latency),
            latency_ms=float(total_latency),
            energy_j=float(total_energy),
            cloud_compute_energy_j=float(cloud_energy),
            communication_energy_j=float(comm_energy),
            preprocess_energy_j=float(preprocess_energy),
            objective=float(objective),
        )
        configs.append(config)
        if best is None or config.objective < best.objective:
            best = config
    selected = select_config_by_resource_hint(
        [item.to_dict() for item in configs],
        resource_hint,
    )
    _RESOURCE_MODEL_CACHE["cloud"][cache_key] = dict(selected) if selected is not None else None
    return dict(selected) if selected is not None else None










