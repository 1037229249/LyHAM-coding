"""
pair-level UAC actor离线训练入口
文件作用：为UAC-DO提供active (flow, AI service, server)粒度的离线监督训练流程。
论文映射：状态/动作空间对齐论文中的Omega^t候选集合和Lyapunov evaluator评分。
工程边界：本文件只生成非formal seed warm-start checkpoint和训练元数据；
formal主实验由run_ablation中的在线UAC-DO继续因果更新，不回写旧checkpoint。
"""
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Optional
import argparse
import contextlib
import hashlib
import io
import json

import numpy as np

from ablation_config import AblationExperimentConfig, get_utils_dir
from Constant import create_base_system
from Deployment import run_GSLA
from ablation_algorithms import build_active_pair_universe, build_pair_repair_candidates
from ablation_resource_models import clear_resource_model_cache, get_resource_model_cache_stats
from ResourceAllocation import evaluate_action_dry_run, _pair_hapa_features


@dataclass
class PairActorTrainingConfig:
    """pair actor训练配置，formal种子默认禁止进入训练集。"""
    version: str = "v1"
    train_seeds: List[int] = field(default_factory=lambda: [28, 29, 30, 31, 32, 33, 34, 35, 36, 37])
    val_seeds: List[int] = field(default_factory=lambda: [43, 44])
    formal_seed_blacklist: List[int] = field(default_factory=lambda: [38, 39, 40, 41, 42])
    time_slots: int = 20
    train_slots: int = None
    max_candidates_per_seed: int = 64
    max_candidates_per_slot: int = None
    max_evaluated_candidates_per_slot: Optional[int] = None
    hidden_dim: int = 128
    epochs: int = 30
    lr: float = 1e-3
    loss_mode: str = "bce"
    hard_negative_loss_weight: float = 1.5
    verbose: bool = False
    checkpoint_every_epochs: int = 0
    resume: bool = False
    output_dir: str = field(default_factory=lambda: str(get_utils_dir().parent / "Training results" / "PairUAC"))

    def __post_init__(self):
        """应用v2/v3/v4默认训练范围和CLI别名。"""
        if self.version in {"v2", "v3"}:
            self.train_seeds = list(range(20, 38)) + list(range(43, 53))
            self.val_seeds = [53, 54, 55, 56, 57]
        if self.version == "v4":
            # v4用于energy-hard正式前的离线pair actor重训，默认提高样本覆盖。
            self.train_seeds = list(range(20, 38)) + list(range(43, 53))
            self.val_seeds = [53, 54, 55, 56, 57]
            self.time_slots = max(int(self.time_slots), 100)
            self.max_candidates_per_seed = max(int(self.max_candidates_per_seed), 128)
            if self.max_evaluated_candidates_per_slot is None:
                # v4保留128候选池，但只对高多样性子集做shared evaluator精评。
                # 训练阶段不参与formal选择，16个精评候选能覆盖source quota并保持可执行速度。
                self.max_evaluated_candidates_per_slot = 16
            self.loss_mode = "pairwise_hard_negative_weighted_bce"
            self.hard_negative_loss_weight = max(float(self.hard_negative_loss_weight), 1.5)
        if self.train_slots is not None:
            self.time_slots = int(self.train_slots)
        if self.max_candidates_per_slot is not None:
            self.max_candidates_per_seed = int(self.max_candidates_per_slot)
        if self.max_evaluated_candidates_per_slot is None:
            self.max_evaluated_candidates_per_slot = int(self.max_candidates_per_seed)
        self.max_evaluated_candidates_per_slot = max(1, min(int(self.max_evaluated_candidates_per_slot), int(self.max_candidates_per_seed)))
        self.checkpoint_every_epochs = max(0, int(self.checkpoint_every_epochs or 0))
        self.resume = bool(self.resume)

    def to_dict(self) -> Dict:
        """转换为可哈希配置。"""
        return asdict(self)


def stable_hash(payload: Dict) -> str:
    """生成稳定哈希，用于训练数据和配置追踪。"""
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def training_identity_payload(config: PairActorTrainingConfig) -> Dict:
    """返回可跨目标epoch复用的训练身份配置。"""
    payload = dict(config.to_dict())
    for key in ("epochs", "output_dir", "resume", "checkpoint_every_epochs", "verbose"):
        payload.pop(key, None)
    return payload


def _load_torch_checkpoint(path: Path):
    """兼容不同PyTorch版本读取含optimizer/history的训练checkpoint。"""
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _quiet_call(verbose: bool, func, *args, **kwargs):
    """按需静默调用旧仿真函数，避免训练阶段大量打印拖慢执行。"""
    if verbose:
        return func(*args, **kwargs)
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)

def assert_no_formal_seed_leakage(config: PairActorTrainingConfig) -> None:
    """训练集不能包含formal seed，避免数据泄漏。"""
    forbidden = set(config.formal_seed_blacklist)
    leaked = sorted((set(config.train_seeds) | set(config.val_seeds)) & forbidden)
    if leaked:
        raise ValueError(f"pair actor训练seed不能包含formal seed: {leaked}")


def apply_training_virtual_queue_profile(system_state, seed: int, slot: int) -> Dict:
    """为离线训练生成可复现的异构虚拟队列。

    正式实验的队列由slot执行自然演化；训练集需要覆盖不同能耗/延迟压力，否则
    shared evaluator 会长期把标签压到全本地或全云端动作。
    """
    energy_values = {}
    delay_values = {}
    for idx, server_id in enumerate(sorted(system_state.virtual_energy_queues.keys())):
        energy_value = 20.0 + float((seed * 17 + slot * 31 + idx * 13) % 9) * 22.0
        if (idx + seed + slot) % 4 == 0:
            energy_value += 160.0
        queue = system_state.virtual_energy_queues.get(server_id)
        if queue is not None:
            queue.queue_state = float(energy_value)
        energy_values[server_id] = float(energy_value)

    for idx, server_id in enumerate(sorted(system_state.virtual_delay_queues.keys())):
        delay_value = 15.0 + float((seed * 11 + slot * 19 + idx * 7) % 8) * 18.0
        if (2 * idx + seed + slot) % 5 == 0:
            delay_value += 120.0
        queue = system_state.virtual_delay_queues.get(server_id)
        if queue is not None:
            queue.queue_state = float(delay_value)
        delay_values[server_id] = float(delay_value)

    return {
        "energy_queue_unique": int(len(set(round(value, 6) for value in energy_values.values()))),
        "delay_queue_unique": int(len(set(round(value, 6) for value in delay_values.values()))),
        "energy_queue_max": float(max(energy_values.values(), default=0.0)),
        "delay_queue_max": float(max(delay_values.values(), default=0.0)),
    }


def make_training_system(seed: int, ablation_config: AblationExperimentConfig):
    """构造训练用系统。

    v4训练复用run_ablation的系统构造，确保云端profile、burst和禁止隐式重部署一致。
    """
    if getattr(ablation_config, "experiment_type", "") == "normal_main":
        from run_ablation import create_ablation_system

        return create_ablation_system(seed, ablation_config)
    return create_base_system(
        seed=seed,
        chain_length_range=ablation_config.chain_length_range,
        fixed_arrival_rate=ablation_config.fixed_arrival_rate,
        num_edge_nodes=ablation_config.traditional_nodes,
        ai_node_count=ablation_config.ai_nodes,
        request_flow_count=ablation_config.request_flow_count,
        arrival_range_req_s=ablation_config.arrival_range_req_s,
        input_tokens_range=ablation_config.input_tokens_range,
        output_tokens_range=ablation_config.output_tokens_range,
    )


def extract_pair_features(item: Dict, system_state, ablation_config: AblationExperimentConfig) -> List[float]:
    """
    提取pair级状态特征
    单位统一做轻量缩放，训练目标只用于候选排序，不直接进入论文结果。
    """
    server = system_state.edge_servers.get(item.get("server_id"))
    flow = system_state.request_flows.get(item.get("flow_id"))
    hapa = _pair_hapa_features(item, system_state)
    server_index = float(item.get("server_index", 0)) / max(float(ablation_config.ai_nodes), 1.0)
    arrival = float(getattr(flow, "arrival_rate", 0.0)) / 20.0 if flow is not None else 0.0
    input_tokens = float(getattr(flow, "r_input_data_size", 0.0)) / 1024.0 if flow is not None else 0.0
    output_tokens = float(getattr(flow, "r_output_data_size", 0.0)) / 512.0 if flow is not None else 0.0
    y_queue = float(getattr(system_state.virtual_energy_queues.get(item.get("server_id")), "queue_state", 0.0)) / 1000.0
    z_queue = float(getattr(system_state.virtual_delay_queues.get(item.get("server_id")), "queue_state", 0.0)) / 1000.0
    gpu_free = float(getattr(server, "available_gpu_units", 0.0)) / 16.0 if server is not None else 0.0
    vram_free = float(getattr(server, "available_gpu_memory", 0.0)) / 128.0 if server is not None else 0.0
    storage_free = float(getattr(server, "available_model_storage", 0.0)) / 512.0 if server is not None else 0.0
    return [
        server_index,
        arrival,
        input_tokens,
        output_tokens,
        y_queue,
        z_queue,
        gpu_free,
        vram_free,
        storage_free,
        float(hapa.get("coverage_ratio", 1.0)),
        float(hapa.get("replica_readiness", 1.0)),
        float(hapa.get("hapa_psi", 1.0)),
        float(hapa.get("hapa_d_loc", 1.0)),
        float(hapa.get("reservation_pressure", 0.0)),
        1.0 if float(hapa.get("has_hapa_feedback", 0.0)) > 0.0 else 0.0,
    ]



def extract_repaired_label_from_eval_result(candidate: Dict, eval_result: Dict) -> np.ndarray:
    """从shared evaluator结果提取修复后的可执行pair动作标签。"""
    repaired = eval_result.get("pair_action")
    if repaired is None or len(np.asarray(repaired).reshape(-1)) == 0:
        bits = str(eval_result.get("pair_action_bits", ""))
        if bits:
            repaired = [int(ch) for ch in bits if ch in {"0", "1"}]
    if repaired is None or len(np.asarray(repaired).reshape(-1)) == 0:
        repaired = candidate.get("pair_action", [])
    return np.asarray(repaired, dtype=np.float32).reshape(-1)



def _pair_bits_key(bits) -> str:
    """生成训练标签位串，用于统计repaired label多样性。"""
    arr = np.asarray(bits, dtype=int).reshape(-1)
    return "".join(str(int(value)) for value in arr)


def summarize_hard_negative_quality(evaluated_rows: List[Tuple], best_score: float,
                                    best_bits: np.ndarray) -> Dict:
    """统计hard negative类型和repaired label多样性。
    evaluated_rows元素为(score, candidate, repaired_bits, eval_result)。
    """
    type_counts = {
        "myopic_repaired_action": 0,
        "all_local_collapse": 0,
        "all_cloud_collapse": 0,
        "near_hamming_worse_dpp": 0,
        "low_energy_high_delay": 0,
        "low_delay_high_energy": 0,
    }
    if not evaluated_rows:
        return {
            "hard_negative_count": 0,
            "hard_negative_type_counts": type_counts,
            "repaired_label_hash_count": 0,
            "repaired_label_diversity": 0.0,
        }

    best_row = min(evaluated_rows, key=lambda row: float(row[0]))
    best_result = best_row[3] if len(best_row) > 3 and isinstance(best_row[3], dict) else {}
    best_energy = float(best_result.get("energy_j", 0.0))
    best_delay = float(best_result.get("delay_ms", 0.0))
    best_bits = np.asarray(best_bits, dtype=int).reshape(-1)
    label_hashes = set()
    hard_negative_count = 0

    for score, candidate, bits, result in evaluated_rows:
        bits = np.asarray(bits, dtype=int).reshape(-1)
        if len(bits) == 0:
            continue
        label_hashes.add(_pair_bits_key(bits))
        source = str(candidate.get("candidate_source", "")) if isinstance(candidate, dict) else ""
        is_worse = float(score) > float(best_score) + 1e-12
        if not is_worse:
            continue
        hard_negative_count += 1
        if "myopic" in source:
            type_counts["myopic_repaired_action"] += 1
        if int(np.sum(bits)) == 0:
            type_counts["all_local_collapse"] += 1
        if int(np.sum(bits)) == len(bits):
            type_counts["all_cloud_collapse"] += 1
        if len(bits) == len(best_bits):
            hamming = float(np.mean(bits != best_bits))
            if hamming <= 0.25:
                type_counts["near_hamming_worse_dpp"] += 1
        result = result if isinstance(result, dict) else {}
        energy = float(result.get("energy_j", best_energy))
        delay = float(result.get("delay_ms", best_delay))
        if best_energy > 0 and best_delay > 0:
            if energy <= best_energy and delay >= best_delay * 1.25:
                type_counts["low_energy_high_delay"] += 1
            if delay <= best_delay and energy >= best_energy * 1.10:
                type_counts["low_delay_high_energy"] += 1

    return {
        "hard_negative_count": int(hard_negative_count),
        "hard_negative_type_counts": type_counts,
        "repaired_label_hash_count": int(len(label_hashes)),
        "repaired_label_diversity": float(len(label_hashes) / max(len(evaluated_rows), 1)),
    }


def make_hard_negative_bit_weights(evaluated_rows: List[Tuple], best_score: float,
                                   best_bits: np.ndarray, base_weight: float = 1.0,
                                   hard_negative_weight: float = 1.5) -> np.ndarray:
    """把hard negative转成pair位训练权重。

    score更差且与最优repaired action不同的pair位会增加权重，形成轻量pairwise约束。
    """
    best_bits = np.asarray(best_bits, dtype=np.float32).reshape(-1)
    weights = np.ones(len(best_bits), dtype=np.float32) * float(base_weight)
    if len(best_bits) == 0:
        return weights
    for score, _candidate, bits, _result in evaluated_rows:
        if float(score) <= float(best_score) + 1e-12:
            continue
        bits = np.asarray(bits, dtype=np.float32).reshape(-1)
        if len(bits) != len(best_bits):
            continue
        changed = bits != best_bits
        if np.any(changed):
            weights[changed] += float(hard_negative_weight)
    return weights


def _row_metric(row: Tuple, key: str) -> float:
    """读取候选评估指标，缺失时给无穷大避免误选。"""
    result = row[3] if len(row) > 3 and isinstance(row[3], dict) else {}
    try:
        return float(result.get(key, float("inf")))
    except Exception:
        return float("inf")


def _is_metric_dominated(row: Tuple, rows: List[Tuple]) -> bool:
    """判断delay/energy/cost三指标是否被另一个候选支配。"""
    values = [_row_metric(row, key) for key in ("delay_ms", "energy_j", "cost")]
    for other in rows:
        if other is row:
            continue
        other_values = [_row_metric(other, key) for key in ("delay_ms", "energy_j", "cost")]
        if all(left <= right + 1e-12 for left, right in zip(other_values, values)) and any(
            left < right - 1e-12 for left, right in zip(other_values, values)
        ):
            return True
    return False


def select_frontier_training_rows(evaluated_rows: List[Tuple], max_positive: int = 4) -> List[Tuple]:
    """从repaired executable候选中选择训练正样本frontier。

    训练actor时不能只学习单个best DPP动作，否则label会坍缩。
    这里保留best DPP、三指标非支配候选，以及最低delay/energy/cost边界动作。
    """
    if not evaluated_rows:
        return []
    max_positive = max(1, int(max_positive))
    ordered = sorted(evaluated_rows, key=lambda row: float(row[0]))
    selected: List[Tuple] = []
    seen = set()

    def add(row: Tuple) -> None:
        if len(selected) >= max_positive:
            return
        key = _pair_bits_key(row[2])
        if key in seen:
            return
        if row is not ordered[0] and _is_metric_dominated(row, evaluated_rows):
            return
        seen.add(key)
        selected.append(row)

    add(ordered[0])
    for key in ("energy_j", "delay_ms", "cost"):
        add(min(evaluated_rows, key=lambda row: (_row_metric(row, key), float(row[0]))))
    for row in ordered:
        add(row)
        if len(selected) >= max_positive:
            break
    return selected


def _candidate_bits(candidate: Dict) -> np.ndarray:
    """读取候选pair动作位串，训练预筛只依赖动作本身。"""
    return np.asarray(candidate.get("pair_action", []), dtype=int).reshape(-1)


def select_training_candidate_subset(candidates: List[Dict], max_eval: int) -> List[Dict]:
    """确定性预筛训练候选。

    训练阶段仍生成完整候选池，但shared evaluator只精评保留下来的高多样性子集。
    该函数不改变formal实验选择规则，只降低离线标签生成成本。
    """
    if max_eval <= 0 or len(candidates) <= max_eval:
        return list(candidates)
    selected: List[Dict] = []
    seen_keys = set()

    def action_key(candidate: Dict):
        return tuple(int(value) for value in _candidate_bits(candidate))

    def add(candidate: Dict) -> bool:
        if len(selected) >= max_eval:
            return False
        key = action_key(candidate)
        source = str(candidate.get("candidate_source", ""))
        source_key = (source, key)
        if source_key in seen_keys:
            return False
        seen_keys.add(source_key)
        selected.append(candidate)
        return True

    # 每类来源至少保留一个，避免训练标签被单一repair来源主导。
    seen_source = set()
    for candidate in candidates:
        source = str(candidate.get("candidate_source", ""))
        if source not in seen_source and add(candidate):
            seen_source.add(source)
        if len(selected) >= max_eval:
            return selected

    def cloud_count(candidate: Dict) -> int:
        return int(np.sum(_candidate_bits(candidate)))

    # 保留全本地、全云端和均衡动作，形成hard negative和边界样本。
    ordered = sorted(enumerate(candidates), key=lambda item: cloud_count(item[1]))
    for _, candidate in ordered[:max(1, min(6, len(ordered)))]:
        add(candidate)
    for _, candidate in ordered[-max(1, min(6, len(ordered))):]:
        add(candidate)
    if candidates:
        pair_dim = max(len(_candidate_bits(item)) for item in candidates)
        midpoint = pair_dim / 2.0
        balanced = sorted(enumerate(candidates), key=lambda item: abs(cloud_count(item[1]) - midpoint))
        for _, candidate in balanced[:max(1, min(8, len(balanced)))]:
            add(candidate)

    # 剩余名额按原候选顺序均匀抽样，保持确定性。
    remaining = [candidate for candidate in candidates if (str(candidate.get("candidate_source", "")), action_key(candidate)) not in seen_keys]
    if remaining and len(selected) < max_eval:
        step = max(len(remaining) / float(max_eval - len(selected)), 1.0)
        cursor = 0.0
        while len(selected) < max_eval and int(cursor) < len(remaining):
            add(remaining[int(cursor)])
            cursor += step
    for candidate in candidates:
        if len(selected) >= max_eval:
            break
        add(candidate)
    return selected[:max_eval]

def collect_seed_examples(seed: int, train_config: PairActorTrainingConfig,
                          ablation_config: AblationExperimentConfig) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """用shared evaluator为一个seed生成监督样本。"""
    system_state = _quiet_call(train_config.verbose, make_training_system, seed, ablation_config)
    workload_trace = [
        system_state.environment_manager.generate_time_varying_arrivals(slot)
        for slot in range(max(int(train_config.time_slots), 1))
    ]
    meta = {
        "seed": seed,
        "status": "ok",
        "candidate_count": 0,
        "evaluated_candidate_count": 0,
        "candidate_prescreen_policy": "source_quota_action_diversity_ratio_sampling",
        "pair_count": 0,
        "slot_count": 0,
        "hard_negative_count": 0,
        "hard_negative_type_counts": {},
    }
    if workload_trace:
        system_state.time_frame = 0
        _quiet_call(
            train_config.verbose,
            system_state.environment_manager.update_all_ai_server_states,
            workload_trace[0],
            allow_redeployment=False,
        )
    if not _quiet_call(train_config.verbose, run_GSLA, system_state):
        meta["status"] = "gsla_failed"
        meta["failure_reason"] = getattr(system_state, "gsla_context", {}).get("hapa_failure_reason", "")
        return np.zeros((0, 15), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32), meta

    universe = build_active_pair_universe(system_state)
    if not universe:
        meta["status"] = "empty_universe_or_candidates"
        return np.zeros((0, 15), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32), meta

    features = []
    labels = []
    sample_weights = []
    label_hashes = set()
    candidate_total = 0
    evaluated_candidate_total = 0
    feasible_slots = 0
    for slot in range(max(int(train_config.time_slots), 1)):
        system_state.time_frame = slot
        if slot < len(workload_trace):
            _quiet_call(
                train_config.verbose,
                system_state.environment_manager.update_all_ai_server_states,
                workload_trace[slot],
                allow_redeployment=False,
            )
        # 训练阶段只扰动虚拟队列，避免重新构造慢层；正式实验仍使用真实slot演化。
        queue_profile = apply_training_virtual_queue_profile(system_state, seed, slot)
        meta.setdefault("queue_profile_unique", []).append(queue_profile)

        candidate_pool = build_pair_repair_candidates(
            system_state, ablation_config, queue_aware=True,
            seed_candidates=None, source_prefix="pair_train"
        )[:train_config.max_candidates_per_seed]
        candidate_total += len(candidate_pool)
        candidates = select_training_candidate_subset(
            candidate_pool,
            int(train_config.max_evaluated_candidates_per_slot),
        )
        evaluated_candidate_total += len(candidates)
        if not candidates:
            continue
        best = None
        evaluated_rows = []
        for candidate in candidates:
            result = evaluate_action_dry_run(candidate, system_state, ablation_config, queue_aware=True)
            if not result.get("feasible", False):
                continue
            score = float(result.get("paper_dpp_score", float("inf")))
            repaired_bits = extract_repaired_label_from_eval_result(candidate, result)
            evaluated_rows.append((score, candidate, repaired_bits, result))
            if best is None or score < best[0]:
                best = (score, candidate, repaired_bits, result)
        if best is None:
            continue
        feasible_slots += 1
        best_bits = np.asarray(best[2], dtype=np.float32)
        # hard negative进入pair位权重；formal使用该checkpoint warm-start后在各seed内在线更新。
        quality = summarize_hard_negative_quality(evaluated_rows, best_score=best[0], best_bits=best_bits)
        meta["hard_negative_count"] = int(meta.get("hard_negative_count", 0)) + int(quality["hard_negative_count"])
        type_counts = meta.setdefault("hard_negative_type_counts", {})
        for key, value in quality["hard_negative_type_counts"].items():
            type_counts[key] = int(type_counts.get(key, 0)) + int(value)
        positive_rows = select_frontier_training_rows(evaluated_rows, max_positive=4)
        for row_index, positive in enumerate(positive_rows):
            positive_bits = np.asarray(positive[2], dtype=np.float32)
            bit_weights = make_hard_negative_bit_weights(
                evaluated_rows, best_score=best[0], best_bits=positive_bits,
                base_weight=1.0,
                hard_negative_weight=float(getattr(train_config, "hard_negative_loss_weight", 1.5)),
            )
            # best DPP动作权重最高，frontier边界动作提供多样化监督。
            frontier_weight = 1.0 if row_index == 0 else 0.45
            label_hashes.add(_pair_bits_key(positive_bits))
            for pos, item in enumerate(universe):
                if pos >= len(positive_bits):
                    continue
                features.append(extract_pair_features(item, system_state, ablation_config))
                labels.append(float(positive_bits[pos]))
                weight = float(bit_weights[pos]) if pos < len(bit_weights) else 1.0
                sample_weights.append(weight * frontier_weight)
    meta["candidate_count"] = candidate_total
    meta["evaluated_candidate_count"] = evaluated_candidate_total
    meta["pair_count"] = len(universe)
    meta["slot_count"] = int(train_config.time_slots)
    meta["feasible_slot_count"] = feasible_slots
    meta["repaired_label_hash_count"] = int(len(label_hashes))
    meta["repaired_label_diversity"] = float(len(label_hashes) / max(feasible_slots, 1))
    meta["repaired_label_hashes"] = sorted(label_hashes)
    if not features:
        meta["status"] = "no_feasible_candidate"
        return np.zeros((0, 15), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32), meta
    meta["loss_mode"] = train_config.loss_mode
    meta["mean_sample_weight"] = float(np.mean(sample_weights)) if sample_weights else 1.0
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.float32),
        np.asarray(sample_weights, dtype=np.float32),
        meta,
    )


def build_dataset(config: PairActorTrainingConfig,
                  ablation_config: AblationExperimentConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
    """生成训练样本，样本标签来自统一Lyapunov evaluator。"""
    assert_no_formal_seed_leakage(config)
    xs = []
    ys = []
    ws = []
    metas = []
    for seed in config.train_seeds:
        x_seed, y_seed, w_seed, meta = collect_seed_examples(seed, config, ablation_config)
        metas.append(meta)
        if len(x_seed):
            xs.append(x_seed)
            ys.append(y_seed)
            ws.append(w_seed)
    if not xs:
        return np.zeros((0, 15), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32), metas
    return np.vstack(xs), np.concatenate(ys), np.concatenate(ws), metas


def build_training_ablation_config(config: PairActorTrainingConfig) -> AblationExperimentConfig:
    """构造训练侧ablation配置。

    v4用于normal-main正式前训练，必须和主实验异构profile对齐；
    否则actor会在default小场景上学到全云端/全本地模板。
    """
    ablation_config = AblationExperimentConfig(
        include_energy_claim=True,
        experiment_type="normal_main" if config.version == "v4" else "c4_ablation",
    )
    if config.version == "v4":
        from run_ablation import apply_heterogeneous_burst_main_profile

        apply_heterogeneous_burst_main_profile(ablation_config, preserve_runtime_overrides=False)
        ablation_config.strict_pair_actor_required = True
    ablation_config.omega_energy = ablation_config.energy_claim_omega_energy
    ablation_config.uac_pair_repair_limit = max(
        int(getattr(ablation_config, "uac_pair_repair_limit", 24)),
        int(config.max_candidates_per_seed),
    )
    return ablation_config


def _mean_meta_value(items: List[Dict], key: str) -> float:
    """计算训练meta字段均值，只统计status=ok的seed。"""
    values = [
        float(item.get(key, 0.0))
        for item in items
        if isinstance(item, dict) and item.get("status", "ok") == "ok"
    ]
    return float(np.mean(values)) if values else 0.0


def summarize_training_label_quality(payload: Dict, min_diversity: float = 0.05) -> Dict:
    """汇总repaired label质量，阻断坍缩标签checkpoint误入formal。"""
    dataset_meta = list(payload.get("dataset_meta", []) or [])
    validation_meta = list(payload.get("validation_summary", []) or [])
    train_diversity = _mean_meta_value(dataset_meta, "repaired_label_diversity")
    val_diversity = _mean_meta_value(validation_meta, "repaired_label_diversity")
    train_hash_count = _mean_meta_value(dataset_meta, "repaired_label_hash_count")
    val_hash_count = _mean_meta_value(validation_meta, "repaired_label_hash_count")
    gate_passed = (
        train_diversity >= float(min_diversity)
        and val_diversity >= float(min_diversity)
        and train_hash_count >= 2.0
        and val_hash_count >= 2.0
    )
    reason = ""
    if not gate_passed:
        reason = (
            "label diversity不足: "
            f"train={train_diversity:.4f}, validation={val_diversity:.4f}, "
            f"train_hash={train_hash_count:.2f}, validation_hash={val_hash_count:.2f}"
        )
    return {
        "label_quality_gate_passed": bool(gate_passed),
        "label_quality_failure_reason": reason,
        "training_label_diversity_mean": train_diversity,
        "validation_label_diversity_mean": val_diversity,
        "training_repaired_label_hash_count_mean": train_hash_count,
        "validation_repaired_label_hash_count_mean": val_hash_count,
        "label_quality_min_diversity": float(min_diversity),
    }


def fit_pair_actor_model(x_train: np.ndarray, y_train: np.ndarray,
                         sample_weights: np.ndarray,
                         config: PairActorTrainingConfig,
                         x_val: Optional[np.ndarray] = None,
                         y_val: Optional[np.ndarray] = None,
                         validation_weights: Optional[np.ndarray] = None,
                         resume_state: Optional[Dict] = None,
                         epoch_callback: Optional[Callable] = None):
    """训练pair actor并返回逐epoch loss历史。"""
    import torch
    from torch import nn

    torch.manual_seed(20260621)
    model = nn.Sequential(
        nn.Linear(x_train.shape[1], config.hidden_dim),
        nn.ReLU(),
        nn.Linear(config.hidden_dim, config.hidden_dim),
        nn.ReLU(),
        nn.Linear(config.hidden_dim, 1),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    x_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
    w_tensor = torch.tensor(sample_weights.reshape(-1, 1), dtype=torch.float32)
    has_validation = x_val is not None and y_val is not None and len(y_val) > 0
    if has_validation:
        x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(np.asarray(y_val).reshape(-1, 1), dtype=torch.float32)
        if validation_weights is None or len(validation_weights) == 0:
            val_w_tensor = torch.ones_like(y_val_tensor, dtype=torch.float32)
        else:
            val_w_tensor = torch.tensor(np.asarray(validation_weights).reshape(-1, 1), dtype=torch.float32)
    else:
        x_val_tensor = None
        y_val_tensor = None
        val_w_tensor = None
    loss_history = []
    completed_epoch = 0
    if resume_state:
        model.load_state_dict(resume_state["state_dict"])
        optimizer_state = resume_state.get("optimizer_state_dict")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        completed_epoch = int(resume_state.get("completed_epoch", 0))
        loaded_history = resume_state.get("loss_history", [])
        loss_history = [
            dict(item) for item in loaded_history
            if int(item.get("epoch", 0)) <= completed_epoch
        ]
    target_positive_ratio = float(torch.mean(y_tensor).detach().cpu().item()) if y_tensor.numel() else 0.0
    start_epoch = min(int(completed_epoch), int(config.epochs)) + 1
    for epoch in range(start_epoch, int(config.epochs) + 1):
        optimizer.zero_grad()
        logits = model(x_tensor)
        loss = (criterion(logits, y_tensor) * w_tensor).mean()
        loss.backward()

        grad_sq_sum = 0.0
        gradient_nonfinite_count = 0
        for parameter in model.parameters():
            if parameter.grad is None:
                continue
            grad = parameter.grad.detach()
            finite_mask = torch.isfinite(grad)
            gradient_nonfinite_count += int((~finite_mask).sum().cpu().item())
            if bool(finite_mask.any().cpu().item()):
                grad_sq_sum += float(torch.sum(torch.square(grad[finite_mask])).cpu().item())
        gradient_l2_norm = float(grad_sq_sum ** 0.5)

        optimizer.step()

        parameter_nonfinite_count = 0
        for parameter in model.parameters():
            values = parameter.detach()
            parameter_nonfinite_count += int((~torch.isfinite(values)).sum().cpu().item())

        validation_loss = None
        validation_logit_nonfinite_count = 0
        with torch.no_grad():
            updated_logits = model(x_tensor)
            logit_nonfinite_count = int((~torch.isfinite(updated_logits)).sum().cpu().item())
            probabilities = torch.sigmoid(updated_logits[torch.isfinite(updated_logits)])
            probability_mean = float(probabilities.mean().cpu().item()) if probabilities.numel() else float("nan")
            predicted_positive_ratio = float((probabilities >= 0.5).float().mean().cpu().item()) if probabilities.numel() else 0.0
            if has_validation:
                validation_logits = model(x_val_tensor)
                validation_logit_nonfinite_count = int((~torch.isfinite(validation_logits)).sum().cpu().item())
                validation_loss_tensor = (criterion(validation_logits, y_val_tensor) * val_w_tensor).mean()
                validation_loss = float(validation_loss_tensor.detach().cpu().item())

        loss_value = float(loss.detach().cpu().item())
        actor_collapse_detected = bool(
            probability_mean <= 0.01
            or probability_mean >= 0.99
            or predicted_positive_ratio <= 0.01
            or predicted_positive_ratio >= 0.99
        )
        instability_detected = bool(
            (not np.isfinite(loss_value))
            or gradient_nonfinite_count > 0
            or parameter_nonfinite_count > 0
            or logit_nonfinite_count > 0
            or validation_logit_nonfinite_count > 0
            or (validation_loss is not None and not np.isfinite(validation_loss))
        )
        history_item = {
            "epoch": int(epoch),
            "loss": loss_value,
            "lr": float(config.lr),
            "gradient_l2_norm": gradient_l2_norm,
            "gradient_nonfinite_count": int(gradient_nonfinite_count),
            "parameter_nonfinite_count": int(parameter_nonfinite_count),
            "logit_nonfinite_count": int(logit_nonfinite_count),
            "probability_mean": probability_mean,
            "predicted_positive_ratio": predicted_positive_ratio,
            "target_positive_ratio": target_positive_ratio,
            "actor_collapse_detected": actor_collapse_detected,
            "instability_detected": instability_detected,
        }
        if validation_loss is not None:
            history_item["validation_loss"] = validation_loss
            history_item["validation_logit_nonfinite_count"] = int(validation_logit_nonfinite_count)
        loss_history.append(history_item)
        if epoch_callback is not None:
            epoch_callback(model, optimizer, int(epoch), list(loss_history), dict(history_item))
    return model, loss_history


def train_pair_actor(config: PairActorTrainingConfig, dry_run: bool = False) -> Dict:
    """训练pair-level actor；dry-run只生成数据和元信息，不写checkpoint。"""
    clear_resource_model_cache()
    ablation_config = build_training_ablation_config(config)
    x_train, y_train, sample_weights, dataset_meta = build_dataset(config, ablation_config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_meta = []
    validation_xs = []
    validation_ys = []
    validation_ws = []
    for val_seed in config.val_seeds:
        x_val_seed, y_val_seed, w_val_seed, val_meta = collect_seed_examples(val_seed, config, ablation_config)
        validation_meta.append(val_meta)
        if len(x_val_seed):
            validation_xs.append(x_val_seed)
            validation_ys.append(y_val_seed)
            validation_ws.append(w_val_seed)
    if validation_xs:
        x_val = np.vstack(validation_xs)
        y_val = np.concatenate(validation_ys)
        validation_weights = np.concatenate(validation_ws)
    else:
        x_val = None
        y_val = None
        validation_weights = None
    payload = {
        "pair_actor": True,
        "label_source": "shared_evaluator_repaired_pair_action",
        "model_mutated": False,
        "formal_seed_excluded": True,
        "train_config": config.to_dict(),
        "ablation_config": ablation_config.to_dict(),
        "dataset_meta": dataset_meta,
        "validation_summary": validation_meta,
        "sample_count": int(len(y_train)),
        "validation_sample_count": int(len(y_val)) if y_val is not None else 0,
        "feature_dim": int(x_train.shape[1]) if x_train.ndim == 2 else 0,
        "loss_mode": config.loss_mode,
        "hard_negative_loss_weight": float(config.hard_negative_loss_weight),
        "mean_sample_weight": float(np.mean(sample_weights)) if len(sample_weights) else 1.0,
        "resource_model_cache_stats": get_resource_model_cache_stats(),
    }
    payload.update(summarize_training_label_quality(payload))
    payload["train_config_hash"] = stable_hash(config.to_dict())
    payload["training_dataset_hash"] = stable_hash({
        "shape": list(x_train.shape),
        "label_sum": float(np.sum(y_train)) if len(y_train) else 0.0,
        "meta": dataset_meta,
    })
    payload["training_identity_hash"] = stable_hash(training_identity_payload(config))
    resume_checkpoint_path = output_dir / (
        f"pair_uac_actor_resume_{payload['training_identity_hash']}_{payload['training_dataset_hash']}.pth"
    )
    progress_path = output_dir / (
        f"pair_uac_actor_resume_{payload['training_identity_hash']}_{payload['training_dataset_hash']}.progress.json"
    )
    progress_enabled = int(config.checkpoint_every_epochs) > 0 or bool(config.resume)
    payload["checkpoint_every_epochs"] = int(config.checkpoint_every_epochs)
    payload["resume_requested"] = bool(config.resume)
    payload["resume_loaded"] = False
    payload["resumed_from_epoch"] = 0
    payload["resume_checkpoint_path"] = str(resume_checkpoint_path)
    payload["progress_path"] = str(progress_path)
    if dry_run or len(y_train) == 0:
        meta_path = output_dir / "pair_actor_dry_run_meta.json"
        meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        payload["meta_path"] = str(meta_path)
        return payload

    import torch

    resume_state = None
    if bool(config.resume) and resume_checkpoint_path.exists():
        resume_state = _load_torch_checkpoint(resume_checkpoint_path)
        if resume_state.get("training_identity_hash") != payload["training_identity_hash"]:
            raise ValueError("resume checkpoint training identity hash mismatch")
        if resume_state.get("training_dataset_hash") != payload["training_dataset_hash"]:
            raise ValueError("resume checkpoint dataset hash mismatch")
        payload["resume_loaded"] = True
        payload["resumed_from_epoch"] = int(resume_state.get("completed_epoch", 0))

    last_checkpoint_path = str(resume_checkpoint_path) if resume_checkpoint_path.exists() else ""

    def write_training_progress(status: str, current_epoch: int,
                                loss_history: List[Dict], last_epoch: Dict) -> None:
        if not progress_enabled:
            return
        progress_payload = {
            "status": status,
            "current_epoch": int(current_epoch),
            "target_epochs": int(config.epochs),
            "train_config_hash": payload["train_config_hash"],
            "training_identity_hash": payload["training_identity_hash"],
            "training_dataset_hash": payload["training_dataset_hash"],
            "checkpoint_every_epochs": int(config.checkpoint_every_epochs),
            "resume_requested": bool(config.resume),
            "resume_loaded": bool(payload["resume_loaded"]),
            "resumed_from_epoch": int(payload["resumed_from_epoch"]),
            "loss_history": loss_history,
            "last_epoch": last_epoch,
            "last_checkpoint_path": last_checkpoint_path,
            "nonfinite_counters": {
                "gradient_nonfinite_count": int(last_epoch.get("gradient_nonfinite_count", 0)) if last_epoch else 0,
                "parameter_nonfinite_count": int(last_epoch.get("parameter_nonfinite_count", 0)) if last_epoch else 0,
                "logit_nonfinite_count": int(last_epoch.get("logit_nonfinite_count", 0)) if last_epoch else 0,
                "validation_logit_nonfinite_count": int(last_epoch.get("validation_logit_nonfinite_count", 0)) if last_epoch else 0,
            },
        }
        progress_path.write_text(json.dumps(progress_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def on_training_epoch(model_obj, optimizer_obj, epoch: int,
                          loss_history: List[Dict], last_epoch: Dict) -> None:
        nonlocal last_checkpoint_path
        should_checkpoint = (
            progress_enabled
            and int(config.checkpoint_every_epochs) > 0
            and (epoch % int(config.checkpoint_every_epochs) == 0 or epoch == int(config.epochs))
        )
        if should_checkpoint:
            torch.save({
                "pair_actor": True,
                "state_dict": model_obj.state_dict(),
                "optimizer_state_dict": optimizer_obj.state_dict(),
                "feature_dim": int(x_train.shape[1]),
                "hidden_dim": int(config.hidden_dim),
                "train_config_hash": payload["train_config_hash"],
                "training_identity_hash": payload["training_identity_hash"],
                "training_dataset_hash": payload["training_dataset_hash"],
                "completed_epoch": int(epoch),
                "target_epochs": int(config.epochs),
                "loss_history": loss_history,
                "last_epoch": last_epoch,
            }, resume_checkpoint_path)
            last_checkpoint_path = str(resume_checkpoint_path)
        write_training_progress("running", epoch, loss_history, last_epoch)

    model, loss_history = fit_pair_actor_model(
        x_train,
        y_train,
        sample_weights,
        config,
        x_val=x_val,
        y_val=y_val,
        validation_weights=validation_weights,
        resume_state=resume_state,
        epoch_callback=on_training_epoch,
    )
    checkpoint_path = output_dir / f"pair_uac_actor_{payload['train_config_hash']}_{payload['training_dataset_hash']}.pth"
    torch.save({
        "pair_actor": True,
        "state_dict": model.state_dict(),
        "feature_dim": int(x_train.shape[1]),
        "hidden_dim": int(config.hidden_dim),
        "train_config_hash": payload["train_config_hash"],
        "training_dataset_hash": payload["training_dataset_hash"],
    }, checkpoint_path)
    payload["checkpoint_path"] = str(checkpoint_path)
    payload["model_architecture"] = "MLP(15-hidden-hidden-1)"
    payload["loss_history"] = loss_history
    payload["final_train_loss"] = float(loss_history[-1]["loss"]) if loss_history else float("nan")
    current_epoch = int(loss_history[-1]["epoch"]) if loss_history else int(payload["resumed_from_epoch"])
    last_epoch = dict(loss_history[-1]) if loss_history else {}
    write_training_progress("complete", current_epoch, list(loss_history), last_epoch)
    meta_path = checkpoint_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["meta_path"] = str(meta_path)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    """构造训练入口CLI，供LR sweep复用。"""
    parser = argparse.ArgumentParser(description="离线训练pair-level UAC actor")
    parser.add_argument("--dry-run", action="store_true", help="只生成数据集元信息，不训练checkpoint")
    parser.add_argument("--version", default="v1", choices=["v1", "v2", "v3", "v4"], help="训练配置版本")
    parser.add_argument("--train-slots", type=int, default=None, help="每个训练seed生成的时隙数")
    parser.add_argument("--max-candidates-per-slot", type=int, default=None, help="每个slot生成的候选池规模")
    parser.add_argument("--max-evaluated-candidates-per-slot", type=int, default=None, help="每个slot进入shared evaluator精评的候选数")
    parser.add_argument("--lr", type=float, default=PairActorTrainingConfig.lr, help="Adam学习率")
    parser.add_argument("--epochs", type=int, default=PairActorTrainingConfig.epochs, help="训练epoch数")
    parser.add_argument("--output-dir", default=None, help="checkpoint/meta输出目录")
    parser.add_argument("--checkpoint-every-epochs", type=int, default=PairActorTrainingConfig.checkpoint_every_epochs, help="每N个epoch写入可恢复checkpoint和progress JSON；0表示关闭")
    parser.add_argument("--resume", action="store_true", help="从同一训练身份和数据hash的最新可恢复checkpoint继续训练")
    parser.add_argument("--verbose", action="store_true", help="训练数据生成时保留旧仿真打印")
    return parser


def main() -> None:
    """命令行入口。"""
    parser = build_arg_parser()
    args = parser.parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = PairActorTrainingConfig().output_dir
    result = train_pair_actor(PairActorTrainingConfig(
        version=args.version,
        train_slots=args.train_slots,
        max_candidates_per_slot=args.max_candidates_per_slot,
        max_evaluated_candidates_per_slot=args.max_evaluated_candidates_per_slot,
        lr=args.lr,
        epochs=args.epochs,
        output_dir=output_dir,
        checkpoint_every_epochs=args.checkpoint_every_epochs,
        resume=args.resume,
        verbose=args.verbose,
    ), dry_run=args.dry_run)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()























