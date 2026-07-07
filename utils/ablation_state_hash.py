"""
复现实验状态哈希工具
用于确认slot内arrival更新没有污染慢层placement/routing。

工程边界：
- 哈希只覆盖慢层可观测状态，不覆盖到达率、队列和快层临时资源分配。
- Fraction、numpy标量等对象统一转成稳定JSON表示，避免同配置复跑漂移。
"""

import hashlib
import json
from fractions import Fraction
from typing import Any, Dict

import numpy as np


def _stable_value(value: Any) -> Any:
    """把实验对象转换为稳定JSON值。"""
    if isinstance(value, Fraction):
        return {"num": int(value.numerator), "den": int(value.denominator)}
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return [_stable_value(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {
            str(key): _stable_value(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple, set)):
        return [_stable_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _digest(payload: Dict[str, Any]) -> str:
    """计算稳定哈希前缀。"""
    text = json.dumps(_stable_value(payload), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def compute_placement_hash(system_state) -> str:
    """
    计算慢层部署哈希。
    只包含实例位置、服务类型和慢层资源包络，arrival变化不应改变该哈希。
    """
    instances = {}
    for instance_id, instance in sorted(system_state.microservice_instances.items()):
        instances[instance_id] = {
            "server_id": getattr(instance, "server_id", ""),
            "ms_id": getattr(instance.microservice, "ms_id", ""),
            "service_type": getattr(instance.microservice, "service_type", ""),
            "cpu_cores_allocated": getattr(instance, "cpu_cores_allocated", 0),
            "memory_allocated": getattr(instance, "memory_allocated", 0.0),
            "gpu_units_reserved": getattr(instance, "gpu_units_reserved", 0.0),
            "gpu_memory_reserved": getattr(instance, "gpu_memory_reserved", 0.0),
            "model_storage_reserved": getattr(instance, "model_storage_reserved", 0.0),
        }
    return _digest({
        "instances": instances,
        "stream_allocated_resources": getattr(system_state, "stream_allocated_resources", {}),
    })


def compute_routing_hash(system_state) -> str:
    """
    计算慢层路由哈希。
    覆盖系统级stream_transfer_probabilities和每条flow上的routing_probabilities。
    """
    flow_routing = {}
    for flow_id, flow in sorted(system_state.request_flows.items()):
        flow_routing[flow_id] = getattr(flow, "routing_probabilities", {})
    return _digest({
        "stream_transfer_probabilities": getattr(system_state, "stream_transfer_probabilities", {}),
        "flow_routing": flow_routing,
    })


def _context_for_policy(system_state, slow_policy: str) -> Dict[str, Any]:
    """按论文算法名获取当前慢层上下文。"""
    mapping = {
        "GSLA": "gsla_context",
        "FFD": "ffd_context",
        "PDRS": "pdrs_context",
        "Random": "random_context",
        "LoadAware": "loadaware_context",
        "GMDA-RMPR": "gmda_rmpr_context",
    }
    attr = mapping.get(slow_policy, "")
    return dict(getattr(system_state, attr, {}) or {})


def compute_slow_context_hash(system_state, slow_policy: str = "") -> str:
    """
    计算慢层上下文综合哈希。
    该哈希用于验证slow epoch内部slot更新不会改写部署、路由或算法上下文。
    """
    return _digest({
        "slow_policy": slow_policy,
        "context": _context_for_policy(system_state, slow_policy),
        "placement_hash": compute_placement_hash(system_state),
        "routing_hash": compute_routing_hash(system_state),
    })
