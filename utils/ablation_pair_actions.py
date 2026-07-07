"""
C4 pair级动作工具
文件作用：统一active (flow, AI service, server)动作空间、动作投影和诊断哈希。
论文映射：对应UAC-DO中按active pair生成候选集合Omega^t的工程实现。
工程边界：当前旧checkpoint仍是server级输出，本模块只负责把旧输出扩展为pair级候选。
"""
from typing import Dict, Iterable, List
import hashlib
import json

import numpy as np


def normalize_action(action, expected_len: int = None) -> np.ndarray:
    """整理二进制动作向量，1表示云端，0表示本地。"""
    arr = np.array(action, dtype=float).reshape(-1)
    if expected_len is not None:
        if len(arr) < expected_len:
            arr = np.pad(arr, (0, expected_len - len(arr)), constant_values=1)
        elif len(arr) > expected_len:
            arr = arr[:expected_len]
    return (arr >= 0.5).astype(int)


def get_sorted_ai_servers(system_state) -> List:
    """按稳定顺序获取AI服务器，保证动作维度可复现。"""
    return sorted([
        server for server in system_state.edge_servers.values()
        if server.server_type.value == "ai_capable"
    ], key=lambda server: server.server_id)


def get_ai_action_dimension(system_state) -> int:
    """获取旧server级AI卸载动作维度。"""
    return len(get_sorted_ai_servers(system_state))


def build_active_pair_universe(system_state) -> List[Dict]:
    """
    构建active pair动作边界
    每个pair对应一个已部署AI实例和所属请求流，可独立选择本地或云端。
    """
    ai_servers = get_sorted_ai_servers(system_state)
    server_index = {server.server_id: idx for idx, server in enumerate(ai_servers)}
    flow_ids = sorted(system_state.request_flows.keys(), key=len, reverse=True)
    pairs = []
    for instance_id, instance in sorted(system_state.microservice_instances.items()):
        if instance.microservice.service_type != "ai":
            continue
        if instance.server_id not in server_index:
            continue
        flow_id = next(
            (candidate for candidate in flow_ids
             if instance_id == candidate or instance_id.startswith(f"{candidate}_")),
            instance_id.split("_")[0],
        )
        pairs.append({
            "pair_id": f"{flow_id}:{instance.microservice.ms_id}@{instance.server_id}",
            "flow_id": flow_id,
            "microservice_id": instance.microservice.ms_id,
            "server_id": instance.server_id,
            "server_index": int(server_index[instance.server_id]),
            "instance_id": instance_id,
        })
    return pairs


def expand_server_action_to_pair_action(action: np.ndarray, pair_universe: List[Dict]) -> np.ndarray:
    """将旧server级动作扩展到pair级动作。"""
    action = np.asarray(action, dtype=int)
    pair_action = []
    for item in pair_universe:
        idx = int(item["server_index"])
        pair_action.append(int(action[idx]) if idx < len(action) else 1)
    return np.asarray(pair_action, dtype=int)


def project_pair_action_to_server_action(pair_action: np.ndarray,
                                         pair_universe: List[Dict],
                                         action_dim: int) -> np.ndarray:
    """
    将pair动作投影回server级动作
    只用于兼容旧模型和旧导出字段；正式评分仍保留pair_action。
    """
    server_action = np.ones(action_dim, dtype=int)
    for bit, item in zip(np.asarray(pair_action, dtype=int), pair_universe):
        idx = int(item["server_index"])
        if idx < action_dim and int(bit) == 0:
            server_action[idx] = 0
    return server_action


def pair_bits_to_text(bits) -> str:
    """导出pair动作位串。"""
    return "".join(str(int(v)) for v in np.asarray(bits, dtype=int).reshape(-1))


def pair_action_hamming(left, right) -> float:
    """计算两个pair动作的归一化Hamming距离。"""
    left_bits = pair_bits_to_text(left)
    right_bits = pair_bits_to_text(right)
    if not left_bits or not right_bits or len(left_bits) != len(right_bits):
        return 0.0
    diff = sum(1 for a, b in zip(left_bits, right_bits) if a != b)
    return diff / max(len(left_bits), 1)


def deduplicate_pair_candidates(candidates: Iterable[Dict], limit: int = None) -> List[Dict]:
    """按pair_action去重，保持生成顺序。"""
    unique = []
    seen = set()
    for candidate in candidates:
        pair_action = np.asarray(candidate.get("pair_action", []), dtype=int)
        key = tuple(int(v) for v in pair_action)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
        if limit is not None and len(unique) >= limit:
            break
    return unique


def wrap_candidates_with_pair_projection(candidates: Iterable, system_state) -> List[Dict]:
    """
    包装候选动作元数据
    server级候选会被扩展成pair级候选；pair级候选会同步生成兼容server动作。
    """
    action_dim = get_ai_action_dimension(system_state)
    pair_universe = build_active_pair_universe(system_state)
    wrapped = []
    for candidate in candidates:
        if isinstance(candidate, dict):
            action = normalize_action(candidate.get("action"), action_dim)
            meta = dict(candidate)
        else:
            action = normalize_action(candidate, action_dim)
            meta = {}
        meta.setdefault("candidate_source", "legacy_server_candidate")
        pair_action = meta.get("pair_action")
        if pair_action is None:
            pair_action = expand_server_action_to_pair_action(action, pair_universe)
        else:
            pair_action = np.asarray(pair_action, dtype=int)
            action = project_pair_action_to_server_action(pair_action, pair_universe, action_dim)
        meta.update({
            "action": action,
            "pair_action": pair_action,
            "action_scope": "pair" if len(pair_universe) else "server",
            "action_dim": int(action_dim),
            "pair_action_dim": int(len(pair_universe)),
            "pair_universe": [dict(item) for item in pair_universe],
        })
        wrapped.append(meta)
    return wrapped

def pair_universe_hash(pair_universe: List[Dict]) -> str:
    """计算active pair universe稳定哈希，用于复现实验追踪。"""
    payload = []
    for item in pair_universe:
        payload.append({
            "pair_id": str(item.get("pair_id", "")),
            "flow_id": str(item.get("flow_id", "")),
            "microservice_id": str(item.get("microservice_id", "")),
            "server_id": str(item.get("server_id", "")),
            "server_index": int(item.get("server_index", 0)),
            "instance_id": str(item.get("instance_id", "")),
        })
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def pair_action_hash(bits) -> str:
    """计算pair动作位串稳定哈希。"""
    text = pair_bits_to_text(bits)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def pair_action_distance_report(left, right, pair_universe: List[Dict]) -> Dict:
    """
    生成pair动作差异报告
    1表示云端，0表示本地；报告用于UAC相对Myopic的机制门禁。
    """
    left_bits = normalize_action(left)
    right_bits = normalize_action(right, expected_len=len(left_bits))
    if len(right_bits) != len(left_bits):
        right_bits = normalize_action(right_bits, expected_len=len(left_bits))
    pair_count = min(len(left_bits), len(right_bits), len(pair_universe))
    if pair_count <= 0:
        return {
            "pair_count": 0,
            "hamming": 0.0,
            "changed_pair_count": 0,
            "local_to_cloud_count": 0,
            "cloud_to_local_count": 0,
            "changed_service_count": 0,
            "changed_server_count": 0,
            "changed_services": "",
            "changed_servers": "",
        }
    changed_services = set()
    changed_servers = set()
    local_to_cloud = 0
    cloud_to_local = 0
    changed = 0
    for idx in range(pair_count):
        l_bit = int(left_bits[idx])
        r_bit = int(right_bits[idx])
        if l_bit == r_bit:
            continue
        changed += 1
        item = pair_universe[idx]
        changed_services.add(str(item.get("microservice_id", "")))
        changed_servers.add(str(item.get("server_id", "")))
        if l_bit == 0 and r_bit == 1:
            local_to_cloud += 1
        elif l_bit == 1 and r_bit == 0:
            cloud_to_local += 1
    return {
        "pair_count": int(pair_count),
        "hamming": float(changed / max(pair_count, 1)),
        "changed_pair_count": int(changed),
        "local_to_cloud_count": int(local_to_cloud),
        "cloud_to_local_count": int(cloud_to_local),
        "changed_service_count": int(len(changed_services)),
        "changed_server_count": int(len(changed_servers)),
        "changed_services": ";".join(sorted(changed_services)),
        "changed_servers": ";".join(sorted(changed_servers)),
    }

