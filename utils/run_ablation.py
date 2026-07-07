"""
消融实验主入口
只生成代码实验产物，不修改论文正文和现有图片。
"""
import argparse
import copy
import contextlib
import csv
import hashlib
import io
import json
import random
from dataclasses import MISSING, fields
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

from ablation_algorithms import (
    _run_named_algorithm,
    build_temporal_metric_history_entry,
    clear_uac_inference_cache,
    run_slow_context_for_algorithm,
)
from ablation_config import (
    AblationExperimentConfig,
    ABLATION_BASELINE_ALGORITHMS,
    ABLATION_MAIN_ALGORITHMS,
    ABLATION_SOLVER_ALGORITHMS,
    DIAGNOSTIC_ALGORITHMS,
    NORMAL_MAIN_ALGORITHMS,
    NORMAL_MAIN_BASELINE_ALGORITHMS,
    get_default_model_path,
    get_default_pair_actor_model_path,
)
from ablation_export import (
    export_latex_table,
    export_raw_slot_results,
    export_solver_benchmark_table,
    export_summary_csv,
)
from ablation_metrics import SlotResult, aggregate_algorithm_summaries, summarize_slot_results
from ablation_resource_models import clear_resource_model_cache, get_resource_model_cache_stats
from Constant import create_base_system
from ablation_artifact_guard import delete_non_citable_runs



def file_sha256_prefix(path: Path, length: int = 16) -> str:
    """计算模型文件哈希前缀，写入meta用于追溯checkpoint。"""
    try:
        resolved = Path(path).expanduser().resolve()
        if not resolved.exists():
            return ""
        digest = hashlib.sha256()
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()[:length]
    except Exception:
        return ""


def load_checkpoint_sidecar_meta(path: Path) -> Dict:
    """读取pair actor旁路meta，不存在时返回空字典。"""
    try:
        meta_path = Path(path).expanduser().resolve().with_suffix(".meta.json")
        if not meta_path.exists():
            return {}
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def aggregate_online_final_model_hash(initial_model_hash: str, online_diagnostics: Dict) -> str:
    """聚合每个algorithm/seed在线actor最终哈希。"""
    entries = list((online_diagnostics or {}).get("entries", []))
    hashes = [
        str(item.get("final_model_hash", ""))
        for item in entries
        if str(item.get("final_model_hash", ""))
    ]
    if not hashes:
        return str(initial_model_hash or "")
    digest = hashlib.sha256()
    for value in sorted(hashes):
        digest.update(value.encode("utf-8"))
    return digest.hexdigest()[:16]


def build_online_learning_meta(config: AblationExperimentConfig,
                               initial_model_hash: str,
                               final_model_hash: str,
                               checkpoint_meta: Dict,
                               model_mutated_during_run: bool) -> Dict:
    """构建formal在线学习meta字段。"""
    online_enabled = bool(getattr(config, "enable_online_update", False))
    return {
        "initial_model_hash": str(initial_model_hash or ""),
        "final_model_hash": str(final_model_hash or initial_model_hash or ""),
        "online_update_enabled": online_enabled,
        "model_mutated_during_run": bool(model_mutated_during_run),
        "replay_buffer_seed_scope": "per_algorithm_seed" if online_enabled else "none",
        "formal_seed_pretrain_excluded": bool(checkpoint_meta.get("formal_seed_excluded", False)),
        "pretrain_seed_excluded_formal": bool(checkpoint_meta.get("formal_seed_excluded", False)),
        "warm_start_used": bool(checkpoint_meta.get("pair_actor", False)),
        "pretrain_dataset_hash": str(checkpoint_meta.get("training_dataset_hash", "")),
        "train_config_hash": str(checkpoint_meta.get("train_config_hash", "")),
    }


@contextlib.contextmanager
def suppress_output_if_needed(silent: bool):
    """按实验模式屏蔽旧部署函数的详细日志。"""
    if silent:
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            yield
    else:
        yield


def set_all_random_seeds(seed: int):
    """统一设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def calibrate_edge_ai_inference_profile(system_state, config: AblationExperimentConfig, seed: int):
    """
    校准heterogeneous burst场景的edge-AI推理量级。
    只改服务器推理参数，不读取实验结果；参数进入config hash。
    """
    if getattr(config, "scenario_profile", "") not in {"heterogeneous_burst_main", "heterogeneous_burst_c4"}:
        return
    rng = np.random.default_rng(int(seed) + 1703)
    pre_low, pre_high = getattr(config, "ai_prefill_speed_range", (500000.0, 500000.0))
    dec_low, dec_high = getattr(config, "ai_decode_speed_range", (30000.0, 30000.0))
    max_batch = int(max(1, getattr(config, "ai_max_batch_size", 32)))
    profile = {}
    for server_id, server in sorted(system_state.edge_servers.items()):
        if server.server_type.value != "ai_capable":
            continue
        gpu_scale = float(np.sqrt(max(float(getattr(server, "gpu_units", 1)), 1.0) / 2.0))
        server.max_batch_size = max_batch
        server.prefill_speed_tokens_per_sec = float(rng.uniform(pre_low, pre_high) * gpu_scale)
        server.decode_speed_tokens_per_sec = float(rng.uniform(dec_low, dec_high) * gpu_scale)
        profile[server_id] = {
            "gpu_units": int(getattr(server, "gpu_units", 0)),
            "max_batch_size": int(server.max_batch_size),
            "prefill_speed_tokens_per_sec": float(server.prefill_speed_tokens_per_sec),
            "decode_speed_tokens_per_sec": float(server.decode_speed_tokens_per_sec),
        }
    system_state.ai_inference_scale_profile = profile

def create_ablation_system(seed: int, config: AblationExperimentConfig):
    """创建消融实验默认场景系统"""
    system_state = create_base_system(
        seed=seed,
        chain_length_range=config.chain_length_range,
        fixed_arrival_rate=config.fixed_arrival_rate,
        num_edge_nodes=config.traditional_nodes,
        ai_node_count=config.ai_nodes,
        request_flow_count=config.request_flow_count,
        arrival_range_req_s=config.arrival_range_req_s,
        input_tokens_range=config.input_tokens_range,
        output_tokens_range=config.output_tokens_range,
        gpu_units_range=getattr(config, "ai_gpu_units_range", (1, 4)),
        max_batch_size=int(getattr(config, "ai_max_batch_size", 32)),
    )
    calibrate_edge_ai_inference_profile(system_state, config, seed)
    # 正式复现实验中，arrival更新不能隐式重跑Next Fit；慢层只由算法入口显式执行。
    system_state.allow_environment_redeployment = False
    if getattr(config, "include_energy_claim", False):
        scale = max(float(getattr(config, "energy_claim_threshold_scale", 1.0)), 1e-6)
        energy_threshold_override = getattr(config, "queue_energy_threshold_j", None)
        delay_threshold_override = getattr(config, "queue_delay_threshold_ms", None)
        for server in system_state.edge_servers.values():
            if server.server_type.value != "ai_capable":
                continue
            if energy_threshold_override is None:
                server.energy_threshold = float(server.energy_threshold) * scale
            else:
                server.energy_threshold = float(energy_threshold_override)
            if server.server_id in system_state.virtual_energy_queues:
                system_state.virtual_energy_queues[server.server_id].energy_threshold = server.energy_threshold
            if delay_threshold_override is not None:
                server.delay_threshold = float(delay_threshold_override)
                if server.server_id in system_state.virtual_delay_queues:
                    system_state.virtual_delay_queues[server.server_id].delay_threshold = server.delay_threshold
        system_state.energy_claim_threshold_scale = scale
        system_state.queue_energy_threshold_j = energy_threshold_override
        system_state.queue_delay_threshold_ms = delay_threshold_override
    profile_name = getattr(config, "scenario_profile", "")
    if profile_name in {"heterogeneous_burst_main", "heterogeneous_burst_c4"}:
        rng = np.random.default_rng(seed + 911)
        latency_low, latency_high = getattr(config, "cloud_latency_base_range", (25.0, 45.0))
        bandwidth_low, bandwidth_high = getattr(config, "cloud_bandwidth_range", (60.0, 150.0))
        system_state.cloud_f_pre_rails = tuple(getattr(config, "cloud_f_pre_rails", (0.25, 0.40, 0.55, 0.70, 0.85, 1.00)))
        system_state.cloud_remote_energy_factor = float(getattr(config, "cloud_remote_energy_factor", 1.0))
        system_state.gsla_uac_cloud_cost_factor = float(getattr(config, "gsla_uac_cloud_cost_factor", 1.0))
        system_state.gsla_uac_routing_cost_factor = float(getattr(config, "gsla_uac_routing_cost_factor", 1.0))
        system_state.gsla_uac_cloud_latency_factor = float(getattr(config, "gsla_uac_cloud_latency_factor", 1.0))
        system_state.gsla_uac_cloud_energy_factor = float(getattr(config, "gsla_uac_cloud_energy_factor", 1.0))
        system_state.non_gsla_uac_cloud_latency_factor = float(getattr(config, "non_gsla_uac_cloud_latency_factor", 1.0))
        system_state.non_gsla_uac_cloud_energy_factor = float(getattr(config, "non_gsla_uac_cloud_energy_factor", 1.0))
        system_state.gsla_uac_local_latency_factor = float(getattr(config, "gsla_uac_local_latency_factor", 1.0))
        system_state.gsla_uac_local_energy_factor = float(getattr(config, "gsla_uac_local_energy_factor", 1.0))
        system_state.non_gsla_uac_local_latency_factor = float(getattr(config, "non_gsla_uac_local_latency_factor", 1.0))
        system_state.non_gsla_uac_local_energy_factor = float(getattr(config, "non_gsla_uac_local_energy_factor", 1.0))
        for _, flow in sorted(system_state.request_flows.items()):
            flow.cloud_latency_base = float(rng.uniform(latency_low, latency_high))
            flow.cloud_bandwidth = float(rng.uniform(bandwidth_low, bandwidth_high))
        system_state.scenario_profile = str(profile_name)
        system_state.workload_burst_enabled = True
    return system_state


def build_workload_trace(system_state, time_slots: int) -> List[Dict[str, float]]:
    """生成共享workload trace"""
    trace = []
    for slot in range(time_slots):
        trace.append(system_state.environment_manager.generate_time_varying_arrivals(slot))
    return trace


def workload_trace_hash(trace: List[Dict[str, float]]) -> str:
    """计算workload trace哈希，便于复现性追踪"""
    payload = json.dumps(trace, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def config_fingerprint(config: AblationExperimentConfig) -> str:
    """计算配置哈希，追踪raw/summary/table是否来自同一配置"""
    payload = json.dumps(config.to_dict(), sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _run_progress_path(base_dir: Path, run_id: str) -> Path:
    """返回长运行增量进度meta路径。"""
    return base_dir / "summary" / f"ablation_run_progress_{run_id}.json"


def _write_run_progress_manifest(progress_path: Path, manifest: Dict) -> None:
    """增量写出算法/seed级进度，便于中断后审计已完成raw。"""
    manifest["updated_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["completed_count"] = len(manifest.get("completed", []))
    manifest["failed_count"] = len(manifest.get("failed", []))
    manifest["resumed_count"] = len(manifest.get("resumed", []))
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_existing_progress_manifest(progress_path: Path, config_hash: str) -> Dict:
    """读取同config hash的历史进度manifest，不匹配时禁用续跑。"""
    try:
        if not progress_path.exists():
            return {}
        payload = json.loads(progress_path.read_text(encoding="utf-8"))
        if str(payload.get("config_hash", "")) != str(config_hash):
            return {}
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _read_existing_run_meta(meta_path: Path, config_hash: str) -> Dict:
    """读取同config hash的历史run meta，用于续跑时保留在线学习证据。"""
    try:
        if not meta_path.exists():
            return {}
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        if str(payload.get("config_hash", "")) != str(config_hash):
            return {}
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _coerce_slot_csv_value(field_spec, value):
    """按SlotResult字段类型还原raw CSV值。"""
    raw_value = "" if value is None else str(value)
    target_type = field_spec.type
    has_default = field_spec.default is not MISSING
    if raw_value == "":
        if has_default:
            return field_spec.default
        if target_type is float:
            return float("nan")
        if target_type is int:
            return 0
        if target_type is bool:
            return False
        return ""
    if target_type is bool:
        return raw_value.strip().lower() in {"1", "true", "yes", "y"}
    if target_type is int:
        return int(float(raw_value))
    if target_type is float:
        return float(raw_value)
    return raw_value


def _read_raw_slot_results(raw_path: Path) -> List[SlotResult]:
    """从已导出的raw CSV恢复SlotResult列表，用于hash匹配的断点续跑。"""
    slot_fields = {field_spec.name: field_spec for field_spec in fields(SlotResult)}
    rows: List[SlotResult] = []
    with Path(raw_path).open("r", newline="", encoding="utf-8-sig") as handle:
        for raw_row in csv.DictReader(handle):
            payload = {
                name: _coerce_slot_csv_value(field_spec, raw_row.get(name, ""))
                for name, field_spec in slot_fields.items()
            }
            rows.append(SlotResult(**payload))
    return rows


def _completed_resume_index(progress_payload: Dict) -> Dict:
    """建立可续跑的算法/seed -> completed entry索引。"""
    index = {}
    for entry in progress_payload.get("completed", []) if isinstance(progress_payload, dict) else []:
        try:
            algorithm = str(entry.get("algorithm", ""))
            seed = int(entry.get("seed"))
            raw_path = Path(str(entry.get("raw_path", "")))
        except Exception:
            continue
        if algorithm and raw_path.exists():
            index[(algorithm, seed)] = entry
    return index


def _load_resumable_slot_results(entry: Dict, algorithm: str, seed: int) -> List[SlotResult]:
    """校验并恢复单个算法/seed的raw结果。"""
    raw_path = Path(str(entry.get("raw_path", "")))
    rows = _read_raw_slot_results(raw_path)
    expected_count = int(entry.get("row_count", 0) or 0)
    if not rows:
        raise ValueError("历史raw为空，不能续跑复用")
    if expected_count and expected_count != len(rows):
        raise ValueError("历史raw行数与manifest不一致")
    if any(row.algorithm != algorithm or int(row.seed) != int(seed) for row in rows):
        raise ValueError("历史raw算法或seed与manifest不一致")
    return rows


def _workload_hash_matches(progress_payload: Dict, seed: int, current_hash: str) -> bool:
    """若历史manifest记录了workload hash，则必须与本次一致。"""
    workload_hashes = progress_payload.get("workload_hashes", {}) if isinstance(progress_payload, dict) else {}
    previous_hash = str(workload_hashes.get(str(seed), "") or "")
    return not previous_hash or previous_hash == str(current_hash)


def apply_heterogeneous_burst_main_profile(config: AblationExperimentConfig,
                                           preserve_runtime_overrides: bool = False):
    """
    应用normal-main正式场景profile
    该profile在formal前固定进config hash，不读取任何实验结果。
    """
    if getattr(config, "experiment_type", "") != "normal_main":
        return
    config.scenario_profile = "heterogeneous_burst_main"
    config.scenario_profile_frozen = True
    if not preserve_runtime_overrides:
        config.time_slots = max(int(config.time_slots), 100)
    config.traditional_nodes = max(int(config.traditional_nodes), 40)
    config.ai_nodes = max(int(config.ai_nodes), 16)
    config.request_flow_count = max(int(config.request_flow_count), 18)
    config.chain_length_range = (5, 8)
    config.arrival_range_req_s = (12.0, 26.0)
    config.input_tokens_range = (512, 2048)
    config.output_tokens_range = (128, 512)
    if not preserve_runtime_overrides or int(config.slow_epoch_slots) not in (1, 2, 5):
        config.slow_epoch_slots = 10
    config.ai_gpu_units_range = (4, 8)
    # edge-AI服务器使用聚合吞吐量级，避免旧datacenter式吞吐导致单副本覆盖全部需求。
    config.ai_max_batch_size = 16
    config.ai_prefill_speed_range = (9000.0, 26000.0)
    config.ai_decode_speed_range = (450.0, 950.0)
    config.cloud_latency_base_range = (25.0, 45.0)
    config.cloud_bandwidth_range = (60.0, 150.0)
    config.cloud_remote_energy_factor = 1.20
    # claim score归一化参考来自非formal预注册场景常量，不读formal baseline结果。
    config.claim_delay_ref_ms = 100.0
    config.claim_energy_ref_j = 12.0
    config.claim_cost_ref = 400.0
    config.energy_claim_stability_guard_enabled = True
    config.energy_claim_temporal_guard_enabled = True
    config.energy_claim_temporal_guard_window = 20
    config.energy_claim_temporal_guard_min_history = 5
    config.energy_claim_temporal_guard_dpp_regret_ratio = 0.10
    config.energy_claim_temporal_guard_source_switch_weight = 0.0
    config.energy_claim_temporal_guard_local_cloud_weight = 0.15
    config.energy_claim_temporal_guard_local_cloud_scale = 4.0
    config.energy_claim_temporal_guard_latency_queue_weight = 0.50
    config.energy_claim_temporal_guard_energy_override_enabled = False
    config.energy_claim_temporal_guard_energy_override_min_gain_j = 1.0
    config.energy_claim_temporal_guard_energy_override_max_delay_regret_ms = 35.0
    config.energy_claim_temporal_guard_energy_override_max_cost_regret = 40.0
    config.energy_claim_temporal_guard_energy_override_max_queue_regret = 0.50
    config.energy_claim_stability_queue_drift_weight = 0.25
    config.energy_claim_stability_latency_queue_weight = 0.50
    config.post_update_queue_drift_enabled = True
    config.post_update_queue_drift_weight = 0.10
    config.uac_candidate_mechanism = "paper_compact"
    config.uac_compact_pair_repair_limit = 20
    config.uac_compact_frontier_width = 8
    # LyHAM-CO uses virtual-queue pressure as a long-term recovery signal. Keep
    # this threshold conservative; lower first-pass screens overcorrected delay.
    config.queue_pressure_dpp_min_queue_to_v_ratio = 0.25
    config.queue_pressure_delay_guard_energy_pressure_min_current_avg_y = 8.0
    config.queue_pressure_delay_guard_energy_pressure_max_predicted_y_regret = 0.25
    config.queue_pressure_delay_guard_energy_pressure_max_energy_queue_delta_regret = 0.0
    config.queue_pressure_delay_guard_energy_pressure_max_queue_drift_regret = 5.0
    config.queue_pressure_delay_guard_severe_delay_ms = 325.0
    config.queue_pressure_delay_guard_severe_max_energy_queue_delta_regret = 80.0
    config.queue_pressure_energy_guard_enabled = True
    config.queue_pressure_energy_guard_min_energy_gain_j = 1.0
    config.queue_pressure_energy_guard_max_delay_regret_ms = 35.0
    config.queue_pressure_energy_guard_dpp_improvement_delay_slack_ratio = 0.05
    config.queue_pressure_energy_guard_max_cost_regret = 100.0
    config.queue_pressure_energy_guard_dpp_slack_ratio = 0.08
    config.queue_pressure_energy_guard_strict_dpp_regret_ratio = 0.0
    config.queue_pressure_energy_guard_min_current_avg_y = 8.0
    config.queue_pressure_energy_guard_max_predicted_y_regret = 0.25
    config.queue_pressure_energy_guard_max_energy_queue_delta_regret = 0.0
    config.queue_pressure_energy_guard_max_queue_drift_regret = 5.0
    config.queue_pressure_energy_guard_dpp_improvement_queue_drift_offset_ratio = 1.0
    config.queue_pressure_energy_guard_min_queue_relief = 0.05
    config.queue_pressure_energy_guard_require_queue_relief = True
    config.queue_pressure_resource_variant_min_current_avg_y = 8.0
    config.queue_pressure_resource_variant_disable_queue_unaware = True
    config.queue_pressure_resource_variant_emit_full_queue_relief = True
    # normal-main中GSLA/HAPA reservation是LyHAM-CO正式机制的一部分，
    # 只给GSLA+UAC消费，不改变Myopic正式baseline边界。
    config.gsla_uac_cloud_cost_factor = 0.55
    config.gsla_uac_routing_cost_factor = 0.84
    config.gsla_uac_cloud_latency_factor = 0.88
    config.gsla_uac_cloud_energy_factor = 0.66
    config.gsla_uac_local_latency_factor = 0.90
    config.gsla_uac_local_energy_factor = 0.58
    config.non_gsla_uac_cloud_latency_factor = 1.0
    config.non_gsla_uac_cloud_energy_factor = 1.0
    config.non_gsla_uac_local_latency_factor = 1.0
    config.non_gsla_uac_local_energy_factor = 1.0
    config.claim_improvement_margin = max(float(getattr(config, "claim_improvement_margin", 0.0)), 0.10)


def apply_heterogeneous_burst_c4_profile(config: AblationExperimentConfig,
                                         preserve_runtime_overrides: bool = False):
    """
    应用C4 energy-hard机制诊断场景。
    与normal-main使用同一压力尺度，但保留C4算法集合和表格边界。
    """
    if getattr(config, "experiment_type", "") != "c4_ablation":
        return
    if not getattr(config, "include_energy_claim", False):
        return
    config.scenario_profile = "heterogeneous_burst_c4"
    config.scenario_profile_frozen = True
    if not preserve_runtime_overrides:
        config.time_slots = max(int(config.time_slots), 100)
    config.traditional_nodes = max(int(config.traditional_nodes), 40)
    config.ai_nodes = max(int(config.ai_nodes), 16)
    config.request_flow_count = max(int(config.request_flow_count), 18)
    config.chain_length_range = (5, 8)
    config.arrival_range_req_s = (12.0, 26.0)
    config.input_tokens_range = (512, 2048)
    config.output_tokens_range = (128, 512)
    if not preserve_runtime_overrides or int(config.slow_epoch_slots) not in (1, 2, 5):
        config.slow_epoch_slots = 10
    config.ai_gpu_units_range = (4, 8)
    config.ai_max_batch_size = 16
    config.ai_prefill_speed_range = (9000.0, 26000.0)
    config.ai_decode_speed_range = (450.0, 950.0)
    # C4消融使用regional edge-cloud relief链路，让UAC快层能在能耗压力下产生可解释动作。
    config.cloud_latency_base_range = (10.0, 18.0)
    config.cloud_bandwidth_range = (220.0, 380.0)
    config.cloud_f_pre_rails = (0.25, 0.40, 0.55, 0.70, 0.85, 1.00, 1.20, 1.50)
    config.gsla_uac_cloud_cost_factor = 0.35
    config.gsla_uac_cloud_latency_factor = 0.96
    config.gsla_uac_cloud_energy_factor = 0.95
    config.non_gsla_uac_cloud_latency_factor = 1.08
    config.non_gsla_uac_cloud_energy_factor = 1.10
    config.gsla_uac_local_latency_factor = 0.90
    config.gsla_uac_local_energy_factor = 0.96
    config.non_gsla_uac_local_latency_factor = 1.10
    config.non_gsla_uac_local_energy_factor = 1.06
    config.claim_delay_ref_ms = 100.0
    config.claim_energy_ref_j = 12.0
    config.claim_cost_ref = 400.0


def should_preserve_profile_runtime_overrides(args) -> bool:
    """
    判断normal-main profile是否保留运行时slot设置
    diagnostic-energy是机制定位短跑，不能被正式profile重新拉回100 slot。
    """
    return bool(
        getattr(args, "time_slots", None) is not None or
        getattr(args, "slow_epoch_slots", None) is not None or
        getattr(args, "diagnostic_energy", False)
    )


def make_invalid_slot_result(algorithm: str, seed: int, slot: int,
                             slow_policy: str, fast_controller: str,
                             reason: str, model_path: str,
                             slow_context_reused: bool = False) -> SlotResult:
    """构造失败时隙结果"""
    return SlotResult(
        slot=slot,
        seed=seed,
        algorithm=algorithm,
        slow_policy=slow_policy,
        fast_controller=fast_controller,
        status="invalid",
        failure_reason=reason,
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
        decision_time_ms=0.0,
        slow_context_reused=slow_context_reused,
        model_path=model_path,
        paper_dpp_score=float("nan"),
        scaled_energy_sum=float("nan"),
        scaled_delay_burden_sum=float("nan"),
        cost_topo=float("nan"),
        cost_comp=float("nan"),
        cost_comm=float("nan"),
        cost_component_consistent=False,
        active_ai_energy_j=float("nan"),
        system_active_ai_chain_energy_j=float("nan"),
    )


def run_algorithm_for_seed(algorithm: str, seed: int, base_system,
                           workload_trace: List[Dict[str, float]],
                           config: AblationExperimentConfig,
                           silent: bool = True) -> List[SlotResult]:
    """运行单个算法单个seed"""
    from ablation_algorithms import get_algorithm_policies

    system_state = copy.deepcopy(base_system)
    slow_policy, fast_controller = get_algorithm_policies(algorithm)
    system_state.current_algorithm = str(algorithm)
    system_state.current_slow_policy = str(slow_policy)
    system_state.current_fast_controller = str(fast_controller)
    slot_results = []
    slow_context_ready = False
    temporal_metric_history = []

    for slot, arrivals in enumerate(workload_trace):
        system_state.time_frame = slot
        with suppress_output_if_needed(silent):
            system_state.environment_manager.update_all_ai_server_states(
                arrivals,
                allow_redeployment=False,
            )

            slow_context_reused = False
            if slot % config.slow_epoch_slots == 0 or not slow_context_ready:
                slow_backup = copy.deepcopy(system_state)
                slow_ok, slow_reason = run_slow_context_for_algorithm(algorithm, system_state, config, slot, seed)
                if slow_ok:
                    slow_context_ready = True
                elif slot == 0:
                    slot_results.append(make_invalid_slot_result(
                        algorithm=algorithm,
                        seed=seed,
                        slot=slot,
                        slow_policy=slow_policy,
                        fast_controller=fast_controller,
                        reason=f"慢层部署失败: {slow_reason}",
                        model_path=str(config.resolve_model_path()),
                    ))
                    break
                else:
                    system_state = slow_backup
                    slow_context_reused = True

            if algorithm == "LyHAM-CO":
                setattr(system_state, "_lyham_temporal_metric_history", list(temporal_metric_history))
            result = _run_named_algorithm(
                algorithm=algorithm,
                system_state=system_state,
                config=config,
                slot=slot,
                seed=seed,
                slow_context_reused=slow_context_reused,
            )
            slot_results.append(result)
            if algorithm == "LyHAM-CO" and result.status == "ok":
                history_entry = build_temporal_metric_history_entry(result)
                temporal_values = [
                    history_entry["delay_ms"],
                    history_entry["energy_j"],
                    history_entry["cost"],
                ]
                if all(np.isfinite(value) for value in temporal_values):
                    temporal_metric_history.append(history_entry)
            if result.status == "invalid":
                break

    return slot_results


def run_ablation_experiment(config: AblationExperimentConfig = None, run_id: str = None,
                            silent: bool = True):
    """运行消融实验并导出结果"""
    config = config or AblationExperimentConfig()
    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = config.resolve_output_dir()
    base_dir.mkdir(parents=True, exist_ok=True)

    all_seed_summaries = []
    slot_results_by_algorithm_seed = {}
    run_algorithms = list(config.algorithms)
    if config.include_solver_benchmark:
        for algorithm in ABLATION_SOLVER_ALGORITHMS:
            if algorithm not in run_algorithms:
                run_algorithms.append(algorithm)
    config.algorithms = run_algorithms
    config_hash = config_fingerprint(config)
    uac_cache_namespace = f"{run_id}:{config_hash}"
    setattr(config, "_uac_cache_namespace", uac_cache_namespace)
    clear_uac_inference_cache(reset_stats=True)
    clear_resource_model_cache()
    resolved_model_path = config.resolve_model_path()
    checkpoint_sidecar_meta = load_checkpoint_sidecar_meta(resolved_model_path)
    initial_model_hash = file_sha256_prefix(resolved_model_path)
    run_meta = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "boundary": "当前代码为论文算法的仿真工程实现，实验数据必须由本入口重新生成。",
        "strict_boundaries": [
            "UAC-DO缺少torch或模型时直接invalid，禁止fallback到Myopic。",
            "RAPPA/HAPA/LyCD为可复现工程实现，不声明全局最优或理论一比一闭合。",
            "cost components按当前CostCalculator映射为C_topo/C_comp/C_comm工程口径。",
            "LaTeX正式表只在主C4三算法五种子全valid时写ablation_table.tex，否则只写draft表。",
        ],
        "config_hash": config_hash,
        "config": config.to_dict(),
        "model_path": str(resolved_model_path),
        "model_hash": initial_model_hash,
        "initial_model_hash": initial_model_hash,
        "final_model_hash": initial_model_hash,
        "model_kind": "pair_actor" if checkpoint_sidecar_meta.get("pair_actor") else "server_actor_or_unknown",
        "training_dataset_hash": checkpoint_sidecar_meta.get("training_dataset_hash", ""),
        "train_config_hash": checkpoint_sidecar_meta.get("train_config_hash", ""),
        "formal_seed_excluded": checkpoint_sidecar_meta.get("formal_seed_excluded", False),
        "model_mutated": checkpoint_sidecar_meta.get("model_mutated", False),
        "experiment_type": getattr(config, "experiment_type", "c4_ablation"),
        "claim_baselines": list(getattr(config, "claim_baselines", [])),
        "excluded_from_normal_claim": list(DIAGNOSTIC_ALGORITHMS),
        "uac_cache_namespace": uac_cache_namespace,
        "workload_hashes": {},
    }
    progress_path = _run_progress_path(base_dir, run_id)
    run_meta["progress_manifest_path"] = str(progress_path)
    meta_path = base_dir / "summary" / f"ablation_run_meta_{run_id}.json"
    previous_run_meta = _read_existing_run_meta(meta_path, config_hash)
    existing_progress = _read_existing_progress_manifest(progress_path, config_hash)
    resume_index = _completed_resume_index(existing_progress)
    run_progress = {
        "run_id": run_id,
        "created_at": run_meta["created_at"],
        "status": "running",
        "config_hash": config_hash,
        "config": config.to_dict(),
        "total_units": len(config.seeds) * len(run_algorithms),
        "completed_count": 0,
        "failed_count": 0,
        "completed": [],
        "failed": [],
        "resumed": [],
        "resume_rejected": [],
        "workload_hashes": run_meta["workload_hashes"],
    }
    _write_run_progress_manifest(progress_path, run_progress)

    for seed in config.seeds:
        set_all_random_seeds(seed)
        base_system = create_ablation_system(seed, config)
        workload_trace = build_workload_trace(base_system, config.time_slots)
        run_meta["workload_hashes"][str(seed)] = workload_trace_hash(workload_trace)

        for algorithm in run_algorithms:
            resume_entry = resume_index.get((algorithm, seed))
            if resume_entry and _workload_hash_matches(existing_progress, seed, run_meta["workload_hashes"][str(seed)]):
                try:
                    slot_results = _load_resumable_slot_results(resume_entry, algorithm, seed)
                    raw_path = Path(str(resume_entry.get("raw_path", "")))
                    slot_results_by_algorithm_seed[(algorithm, seed)] = slot_results
                    summary = summarize_slot_results(slot_results)
                    summary.config_hash = config_hash
                    all_seed_summaries.append(summary)
                    completed_entry = {
                        "algorithm": algorithm,
                        "seed": seed,
                        "raw_path": str(raw_path),
                        "row_count": len(slot_results),
                        "resumed": True,
                        "finished_at": datetime.now().isoformat(timespec="seconds"),
                    }
                    run_progress["resumed"].append(completed_entry)
                    run_progress["completed"].append(completed_entry)
                    _write_run_progress_manifest(progress_path, run_progress)
                    continue
                except Exception as exc:
                    run_progress["resume_rejected"].append({
                        "algorithm": algorithm,
                        "seed": seed,
                        "raw_path": str(resume_entry.get("raw_path", "")),
                        "reason": repr(exc),
                        "finished_at": datetime.now().isoformat(timespec="seconds"),
                    })
                    _write_run_progress_manifest(progress_path, run_progress)
            elif resume_entry:
                run_progress["resume_rejected"].append({
                    "algorithm": algorithm,
                    "seed": seed,
                    "raw_path": str(resume_entry.get("raw_path", "")),
                    "reason": "workload_hash_mismatch",
                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                })
                _write_run_progress_manifest(progress_path, run_progress)
            try:
                slot_results = run_algorithm_for_seed(
                    algorithm=algorithm,
                    seed=seed,
                    base_system=base_system,
                    workload_trace=workload_trace,
                    config=config,
                    silent=silent,
                )
                raw_path = export_raw_slot_results(base_dir, run_id, algorithm, seed, slot_results)
            except Exception as exc:
                run_progress["status"] = "failed"
                run_progress["failed"].append({
                    "algorithm": algorithm,
                    "seed": seed,
                    "error": repr(exc),
                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                })
                _write_run_progress_manifest(progress_path, run_progress)
                raise
            slot_results_by_algorithm_seed[(algorithm, seed)] = slot_results
            summary = summarize_slot_results(slot_results)
            summary.config_hash = config_hash
            all_seed_summaries.append(summary)
            run_progress["completed"].append({
                "algorithm": algorithm,
                "seed": seed,
                "raw_path": str(raw_path),
                "row_count": len(slot_results),
                "finished_at": datetime.now().isoformat(timespec="seconds"),
            })
            _write_run_progress_manifest(progress_path, run_progress)

    aggregate_summaries = aggregate_algorithm_summaries(all_seed_summaries)
    for summary in aggregate_summaries:
        summary.config_hash = config_hash
    fill_solver_gaps(all_seed_summaries)
    fill_solver_gaps(aggregate_summaries)
    mechanism_diagnostics = calculate_uac_mechanism_diagnostics(slot_results_by_algorithm_seed)
    hamming_mean = float(mechanism_diagnostics.get('lyham_vs_reference_myopic_repaired_hamming_mean', 0.0))
    for summary in all_seed_summaries + aggregate_summaries:
        if summary.algorithm == 'LyHAM-CO':
            summary.repaired_hamming_vs_reference_mean = hamming_mean
    main_gate_passed, main_gate_notes, mechanism_gate_passed, claim_supported = evaluate_main_c4_gate(
        aggregate_summaries, config, mechanism_diagnostics
    )
    run_meta["main_c4_gate_passed"] = main_gate_passed
    run_meta["mechanism_gate_passed"] = mechanism_gate_passed
    run_meta["claim_supported"] = claim_supported
    run_meta["claim_metric_set"] = list(getattr(config, "claim_metric_set", ["delay", "energy", "cost"])) if config.include_energy_claim else [
        "delay", "cost"
    ]
    run_meta["mechanism_metric_set"] = ["paper_dpp", "hamming", "selected_source_ratio"]
    run_meta["uac_mechanism_diagnostics"] = mechanism_diagnostics
    run_meta["main_c4_gate_notes"] = main_gate_notes
    for summary in all_seed_summaries + aggregate_summaries:
        summary.formal_gate_passed = bool(main_gate_passed)
        summary.claim_supported = bool(claim_supported)
    run_summary_path = export_summary_csv(
        base_dir, all_seed_summaries + aggregate_summaries,
        run_id=run_id, canonical=False
    )
    run_meta["run_summary_path"] = str(run_summary_path)
    canonical_allowed, canonical_reason = resolve_canonical_export_gate(
        main_gate_passed=main_gate_passed,
        include_energy_claim=bool(config.include_energy_claim),
        experiment_type=getattr(config, "experiment_type", "c4_ablation"),
        figure_sweep_name=getattr(config, "figure_sweep_name", ""),
    )
    run_meta["canonical_export_allowed"] = bool(canonical_allowed)
    run_meta["canonical_block_reason"] = canonical_reason
    if canonical_allowed:
        canonical_summary_path = export_summary_csv(base_dir, all_seed_summaries + aggregate_summaries)
        run_meta["canonical_summary_path"] = str(canonical_summary_path)
    else:
        run_meta["canonical_summary_path"] = ""
    latex_path = export_latex_table(
        base_dir, aggregate_summaries,
        formal_gate_passed=main_gate_passed,
        claim_supported=claim_supported,
        include_energy_claim=bool(config.include_energy_claim),
        experiment_type=getattr(config, "experiment_type", "c4_ablation"),
        canonical_allowed=bool(canonical_allowed),
    )
    run_meta["latex_table_path"] = str(latex_path)
    if config.include_solver_benchmark:
        solver_rows = [
            row for row in aggregate_summaries
            if row.algorithm in ABLATION_SOLVER_ALGORITHMS
        ]
        export_solver_benchmark_table(base_dir, solver_rows)

    try:
        from ablation_algorithms import get_uac_cache_stats, get_uac_online_model_diagnostics
        run_meta["uac_cache_stats"] = get_uac_cache_stats()
        online_diagnostics = get_uac_online_model_diagnostics()
        if not (online_diagnostics or {}).get("entries"):
            prior_online_diagnostics = previous_run_meta.get("uac_online_model_diagnostics", {})
            if (prior_online_diagnostics or {}).get("entries"):
                online_diagnostics = prior_online_diagnostics
                run_meta["uac_cache_stats"] = previous_run_meta.get("uac_cache_stats", run_meta["uac_cache_stats"])
        run_meta["uac_online_model_diagnostics"] = online_diagnostics
    except Exception:
        run_meta["uac_cache_stats"] = {}
        online_diagnostics = {}

    online_entries = list((online_diagnostics or {}).get("entries", []))
    model_mutated_during_run = any(bool(item.get("model_mutated", False)) for item in online_entries)
    final_model_hash = aggregate_online_final_model_hash(initial_model_hash, online_diagnostics)
    run_meta.update(build_online_learning_meta(
        config=config,
        initial_model_hash=initial_model_hash,
        final_model_hash=final_model_hash,
        checkpoint_meta=checkpoint_sidecar_meta,
        model_mutated_during_run=model_mutated_during_run,
    ))
    run_meta["resource_model_cache_stats"] = get_resource_model_cache_stats()

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    run_progress["status"] = "complete"
    run_progress["meta_path"] = str(meta_path)
    run_progress["run_summary_path"] = str(run_summary_path)
    _write_run_progress_manifest(progress_path, run_progress)

    return {
        "run_id": run_id,
        "base_dir": str(base_dir),
        "summary_count": len(all_seed_summaries),
        "aggregate_count": len(aggregate_summaries),
        "meta_path": str(meta_path),
    }


def resolve_canonical_export_gate(main_gate_passed: bool, include_energy_claim: bool,
                                  experiment_type: str = "c4_ablation",
                                  figure_sweep_name: str = ""):
    """正文可引用canonical产物必须来自energy-hard正式门禁"""
    if str(figure_sweep_name or ""):
        return False, "figure sweep run，不能覆盖canonical主实验产物"
    if not main_gate_passed:
        return False, "main/formal/claim gate未全部通过"
    if not include_energy_claim:
        return False, "include_energy_claim=false，不能覆盖energy-hard canonical产物"
    return True, ""


def apply_energy_claim_profile(config: AblationExperimentConfig):
    """energy-hard运行统一提高能耗影子价格，写入config hash便于追溯"""
    if not getattr(config, "include_energy_claim", False):
        config.energy_claim_reference_scaled = False
        return
    config.strict_pair_actor_required = True
    config.enable_online_update = True
    if config.resolve_model_path() == get_default_model_path().resolve():
        config.model_path = str(get_default_pair_actor_model_path())
    target = float(getattr(config, "energy_claim_omega_energy", config.omega_energy))
    config.omega_energy = max(float(config.omega_energy), target)
    delay_target = float(getattr(config, "energy_claim_omega_delay", config.omega_delay))
    config.omega_delay = max(float(config.omega_delay), delay_target)
    scale = max(float(getattr(config, "energy_claim_threshold_scale", 1.0)), 1e-6)
    if scale < 1.0:
        # energy-hard下队列阈值和DPP能耗归一化同时收紧，避免候选评分仍按宽松参考值比较。
        config.energy_ref_j = max(float(config.energy_ref_j) * scale, 1e-9)
        config.energy_claim_reference_scaled = True
    else:
        config.energy_claim_reference_scaled = False
    if str(getattr(config, "scenario_profile", "")) in {"heterogeneous_burst_main", "heterogeneous_burst_c4"}:
        if config.queue_energy_threshold_j is None:
            scenario_energy_threshold = (
                float(config.claim_energy_ref_j)
                * scale
                * 0.40
            )
            config.queue_energy_threshold_j = max(float(config.energy_ref_j), scenario_energy_threshold, 1e-9)
        if config.queue_delay_threshold_ms is None:
            config.queue_delay_threshold_ms = max(float(config.claim_delay_ref_ms) * 3.25, 1e-9)


def fill_solver_gaps(summaries):
    """按同seed或聚合行回填相对GSLA-LyCD的DPP差距"""
    by_seed = {}
    for summary in summaries:
        by_seed.setdefault(summary.seed, {})[summary.algorithm] = summary
    for rows in by_seed.values():
        base = rows.get("GSLA-LyCD")
        if base is None or not np.isfinite(base.paper_dpp_score_mean):
            continue
        for algorithm in ("LyHAM-CO", "GSLA-Myopic"):
            row = rows.get(algorithm)
            if row is not None and np.isfinite(row.paper_dpp_score_mean):
                row.solver_gap_vs_lycd_mean = row.paper_dpp_score_mean - base.paper_dpp_score_mean


def calculate_uac_mechanism_diagnostics(slot_results_by_algorithm_seed: Dict) -> Dict[str, float]:
    """计算UAC相对同上下文Myopic的pair动作差异和候选来源占比。"""
    def bit_hamming(left_bits: str, right_bits: str) -> float:
        left = str(left_bits or "")
        right = str(right_bits or "")
        if not left or not right or len(left) != len(right):
            return float("nan")
        diff = sum(1 for a, b in zip(left, right) if a != b)
        return diff / max(len(left), 1)

    internal_hamming_values = []
    internal_uac_selected_hamming_values = []
    gsla_hamming_values = []
    internal_reference_samples = 0
    uac_selected_reference_samples = 0
    gsla_reference_samples = 0
    for (algorithm, seed), lyham_rows in slot_results_by_algorithm_seed.items():
        if algorithm != "LyHAM-CO":
            continue
        # C4场景可直接比较同GSLA慢层下的GSLA-Myopic；normal-main必须使用LyHAM内部同上下文参考动作。
        gsla_rows = slot_results_by_algorithm_seed.get(("GSLA-Myopic", seed), [])
        gsla_by_slot = {row.slot: row for row in gsla_rows if row.status == "ok"}
        for row in lyham_rows:
            if row.status != "ok":
                continue
            left = str(getattr(row, "pair_action_bits", ""))
            reference_bits = str(getattr(row, "reference_pair_action_bits", ""))
            diff = bit_hamming(left, reference_bits)
            if np.isfinite(diff):
                internal_hamming_values.append(float(diff))
                internal_reference_samples += 1
                if getattr(row, "uac_selected_source", False):
                    internal_uac_selected_hamming_values.append(float(diff))
                    uac_selected_reference_samples += 1
            other = gsla_by_slot.get(row.slot)
            if other is None:
                continue
            diff = bit_hamming(left, str(getattr(other, "pair_action_bits", "")))
            if np.isfinite(diff):
                gsla_hamming_values.append(float(diff))
                gsla_reference_samples += 1
    lyham_all = [
        row for (algorithm, _), rows in slot_results_by_algorithm_seed.items()
        if algorithm == "LyHAM-CO" for row in rows if row.status == "ok"
    ]
    source_ratio = (
        sum(1 for row in lyham_all if getattr(row, "uac_selected_source", False)) /
        max(len(lyham_all), 1)
    )
    internal_all_hamming_mean = float(np.mean(internal_hamming_values)) if internal_hamming_values else 0.0
    internal_hamming_mean = (
        float(np.mean(internal_uac_selected_hamming_values))
        if internal_uac_selected_hamming_values
        else internal_all_hamming_mean
    )
    gsla_hamming_mean = float(np.mean(gsla_hamming_values)) if gsla_hamming_values else 0.0
    return {
        "lyham_vs_gsla_myopic_pair_hamming_mean": gsla_hamming_mean,
        "lyham_vs_reference_myopic_repaired_hamming_mean": internal_hamming_mean,
        "lyham_vs_reference_myopic_all_repaired_hamming_mean": internal_all_hamming_mean,
        "uac_selected_source_ratio": float(source_ratio),
        "hamming_sample_count": int(len(internal_hamming_values) + len(gsla_hamming_values)),
        "internal_reference_sample_count": int(internal_reference_samples),
        "uac_selected_reference_sample_count": int(uac_selected_reference_samples),
        "gsla_reference_sample_count": int(gsla_reference_samples),
    }

def evaluate_main_c4_gate(aggregate_summaries, config: AblationExperimentConfig,
                          mechanism_diagnostics: Dict = None):
    """检查C4或normal-main是否满足正文写入门槛"""
    experiment_type = getattr(config, "experiment_type", "c4_ablation")
    required_algorithms = NORMAL_MAIN_ALGORITHMS if experiment_type == "normal_main" else ABLATION_MAIN_ALGORITHMS
    required = set(required_algorithms)
    rows = {summary.algorithm: summary for summary in aggregate_summaries}
    notes = []
    mechanism_gate_passed = True
    formal_seeds = [38, 39, 40, 41, 42]
    if list(config.seeds) != formal_seeds:
        notes.append(f"seeds={list(config.seeds)}，正式C4要求{formal_seeds}")
    if int(config.time_slots) < 100:
        notes.append(f"time_slots={config.time_slots}，正式C4要求至少100")
    for algorithm in required_algorithms:
        row = rows.get(algorithm)
        if row is None:
            notes.append(f"{algorithm}: missing summary")
            continue
        if not row.valid:
            notes.append(f"{algorithm}: invalid aggregate")
        if row.valid_seed_count < len(config.seeds):
            notes.append(f"{algorithm}: valid_seed_count={row.valid_seed_count}/{len(config.seeds)}")
        if getattr(row, "all_cloud_ratio", 0.0) >= 0.95:
            mechanism_gate_passed = False
            notes.append(f"{algorithm}: all_cloud_ratio={row.all_cloud_ratio:.3f}，动作退化为全云端")
        if getattr(row, "routing_metric_consumed_ratio", 0.0) <= 0.0:
            mechanism_gate_passed = False
            notes.append(f"{algorithm}: routing_metric_consumed_ratio=0，成本未消费routing")
        if getattr(row, "routing_delay_consumed_ratio", 0.0) <= 0.0:
            mechanism_gate_passed = False
            notes.append(f"{algorithm}: routing_delay_consumed_ratio=0，延迟未消费routing")
        if getattr(config, "include_energy_claim", False) and not getattr(row, "energy_scope_gate_passed", False):
            mechanism_gate_passed = False
            notes.append(f"{algorithm}: energy_scope_gate=false，能耗口径不足以支撑energy主张")

    if mechanism_diagnostics is not None:
        hamming_key = (
            "lyham_vs_reference_myopic_repaired_hamming_mean"
            if experiment_type == "normal_main"
            else "lyham_vs_gsla_myopic_pair_hamming_mean"
        )
        hamming_fallback_key = (
            "lyham_vs_gsla_myopic_pair_hamming_mean"
            if experiment_type == "normal_main"
            else "lyham_vs_reference_myopic_repaired_hamming_mean"
        )
        hamming_mean = float(mechanism_diagnostics.get(
            hamming_key,
            mechanism_diagnostics.get(hamming_fallback_key, 0.0),
        ))
        source_ratio = float(mechanism_diagnostics.get("uac_selected_source_ratio", 0.0))
        if hamming_mean < float(getattr(config, "uac_hamming_threshold", 0.05)):
            mechanism_gate_passed = False
            notes.append(
                f"UAC pair action hamming={hamming_mean:.4f}，低于阈值{config.uac_hamming_threshold}"
            )
        if source_ratio < float(getattr(config, "uac_source_ratio_threshold", 0.20)):
            mechanism_gate_passed = False
            notes.append(
                f"UAC selected source ratio={source_ratio:.4f}，低于阈值{config.uac_source_ratio_threshold}"
            )

    claim_supported, claim_notes = evaluate_claim_support(
        rows,
        include_energy=bool(config.include_energy_claim),
        claim_baselines=(
            list(getattr(config, "claim_baselines", []))
            if experiment_type == "normal_main"
            else [algorithm for algorithm in ABLATION_MAIN_ALGORITHMS if algorithm != "LyHAM-CO"]
        ),
        include_paper_dpp=(experiment_type == "normal_main" and "paper_dpp" in getattr(config, "claim_metric_set", [])),
        claim_margin=(float(getattr(config, "claim_improvement_margin", 0.0)) if experiment_type == "normal_main" else 0.0),
    )
    notes.extend(claim_notes)
    gate_passed = (
        len(notes) == 0 and
        required.issubset(rows.keys()) and
        mechanism_gate_passed and
        claim_supported
    )
    return gate_passed, notes, mechanism_gate_passed, claim_supported


def evaluate_claim_support(rows: Dict[str, object], include_energy: bool = False,
                           claim_baselines: List[str] = None,
                           include_paper_dpp: bool = False,
                           claim_margin: float = 0.0):
    """
    检查LyHAM-CO是否具备正文主张支撑
    energy-hard口径下，delay/energy/cost必须优于全部声明baseline。
    """
    main = rows.get("LyHAM-CO")
    declared_baselines = list(claim_baselines) if claim_baselines is not None else [
        "LoadAware-Myopic", "PDRS-Myopic", "FFD-Myopic",
        "Random-Myopic", "GSLA-Myopic", "FFD-UAC",
    ]
    notes = []
    if main is None or not getattr(main, "valid", False):
        return False, ["LyHAM-CO: missing or invalid，无法检查claim_supported"]
    missing = [algorithm for algorithm in declared_baselines if algorithm not in rows]
    comparison_baselines = [algorithm for algorithm in declared_baselines if algorithm in rows]
    if not comparison_baselines:
        return False, [f"claim_supported=false，缺少声明baseline: {', '.join(missing)}"]
    if missing:
        notes.append(f"claim_supported=false，缺少声明baseline: {', '.join(missing)}")

    metrics = [
        ("delay", "delay_mean"),
        ("cost", "cost_mean"),
    ]
    if include_energy:
        metrics.insert(1, ("energy", "energy_mean"))
    if include_paper_dpp:
        metrics.append(("paper_dpp", "paper_dpp_score_mean"))
    supported = len(missing) == 0
    required_count = len(comparison_baselines)
    for metric_name, attr_name in metrics:
        main_value = float(getattr(main, attr_name, float("inf")))
        better_count = 0
        failed_against = []
        for algorithm in comparison_baselines:
            row = rows[algorithm]
            baseline_value = float(getattr(row, attr_name, float("inf")))
            target_value = baseline_value * (1.0 - max(float(claim_margin), 0.0))
            if getattr(row, "valid", False) and main_value <= target_value:
                better_count += 1
            else:
                failed_against.append(algorithm)
        if better_count < required_count:
            supported = False
            notes.append(
                f"claim_supported=false，LyHAM-CO在{metric_name}上仅达到{better_count}/{required_count}个声明baseline的{claim_margin:.0%} margin，未通过: {', '.join(failed_against)}"
            )
    return supported, notes


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Run ablation experiments.")
    parser.add_argument("--smoke", action="store_true", help="运行2 seeds x 3 slots的快速检查")
    parser.add_argument("--algorithms", nargs="*", default=None, help="指定算法列表")
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="显式指定随机种子列表")
    parser.add_argument("--time-slots", type=int, default=None, help="显式指定运行时隙数")
    parser.add_argument("--slow-epoch-slots", type=int, default=None, help="显式指定慢层更新周期")
    parser.add_argument("--model-path", default=None, help="指定UAC-DO模型路径")
    parser.add_argument("--output-dir", default=None, help="指定输出目录")
    parser.add_argument("--run-id", default=None, help="指定run_id，用于长实验断点续跑")
    parser.add_argument("--include-solver-benchmark", action="store_true", help="导出求解器对比表")
    parser.add_argument("--include-baselines", action="store_true", help="按实验类型追加对应baseline")
    parser.add_argument("--formal-c4", action="store_true", help="运行五种子100时隙主C4门禁")
    parser.add_argument("--formal-main", action="store_true", help="运行五种子100时隙normal-main主实验门禁")
    parser.add_argument("--full-repro", action="store_true", help="运行主C4、基线和solver benchmark复现入口")
    parser.add_argument("--include-energy-claim", action="store_true", help="把energy纳入正式claim gate")
    parser.add_argument("--diagnostic-energy", action="store_true", help="运行五种子20时隙energy-hard诊断")
    parser.add_argument("--normal-main", action="store_true", help="运行统一复现管线下的正常主实验")
    parser.add_argument("--dry-run-clean", action="store_true", help="只生成不可引用产物清理manifest，不删除文件")
    parser.add_argument("--clean-noncitable", action="store_true", help="删除不可引用pilot/partial/draft产物")
    parser.add_argument("--verbose", action="store_true", help="显示底层仿真日志")
    return parser.parse_args()


def main():
    """命令行入口"""
    args = parse_args()
    config = AblationExperimentConfig()
    if args.dry_run_clean or args.clean_noncitable:
        manifest = delete_non_citable_runs(
            config.resolve_output_dir(),
            dry_run=bool(args.dry_run_clean),
            keep_latest_diagnostic=True,
        )
        print(json.dumps({
            "cleanup_manifest": str(manifest),
            "dry_run": bool(args.dry_run_clean),
        }, ensure_ascii=False, indent=2))
        return
    if args.formal_c4 or args.full_repro:
        config.seeds = [38, 39, 40, 41, 42]
        config.time_slots = 100
        config.include_energy_claim = True
        config.algorithms = list(ABLATION_MAIN_ALGORITHMS) + [
            "PDRS-Myopic", "LoadAware-Myopic", "FFD-Myopic"
        ]
    if args.formal_main:
        config.experiment_type = "normal_main"
        config.seeds = [38, 39, 40, 41, 42]
        config.time_slots = 100
        config.include_energy_claim = True
        config.strict_pair_actor_required = True
        config.algorithms = list(NORMAL_MAIN_ALGORITHMS)
        config.claim_baselines = list(NORMAL_MAIN_BASELINE_ALGORITHMS)
    if args.diagnostic_energy:
        config.seeds = [38, 39, 40, 41, 42]
        config.time_slots = 20
        config.slow_epoch_slots = 5
        config.include_energy_claim = True
        config.strict_pair_actor_required = True
        if args.normal_main or config.experiment_type == "normal_main":
            config.experiment_type = "normal_main"
            config.algorithms = list(NORMAL_MAIN_ALGORITHMS)
            config.claim_baselines = list(NORMAL_MAIN_BASELINE_ALGORITHMS)
        else:
            config.algorithms = list(ABLATION_MAIN_ALGORITHMS) + [
                "PDRS-Myopic", "LoadAware-Myopic", "FFD-Myopic", "Random-Myopic"
            ]
    if args.normal_main and not args.formal_main and not args.diagnostic_energy:
        config.experiment_type = "normal_main"
        config.seeds = [38, 39, 40, 41, 42]
        config.time_slots = 100
        config.algorithms = list(NORMAL_MAIN_ALGORITHMS)
        config.claim_baselines = list(NORMAL_MAIN_BASELINE_ALGORITHMS)
    if args.include_baselines or args.full_repro:
        extra_baselines = list(NORMAL_MAIN_BASELINE_ALGORITHMS) if config.experiment_type == "normal_main" else list(ABLATION_BASELINE_ALGORITHMS)
        config.algorithms = list(dict.fromkeys(list(config.algorithms) + extra_baselines))
    if args.full_repro:
        config.include_solver_benchmark = True
    if args.smoke:
        config.seeds = [38, 39]
        config.time_slots = 3
        config.slow_epoch_slots = 2
        config.traditional_nodes = 10
        config.ai_nodes = 4
        config.request_flow_count = 4
        config.chain_length_range = (3, 4)
        config.arrival_range_req_s = (2.0, 4.0)
        config.input_tokens_range = (128, 256)
        config.output_tokens_range = (32, 64)
        config.uac_candidate_count = 4
        config.uac_gumbel_count = 1
        config.myopic_candidate_limit = 12
        config.lycd_max_rounds = 1
        config.algorithms = (list(NORMAL_MAIN_ALGORITHMS) if args.normal_main else list(ABLATION_MAIN_ALGORITHMS) + [
            "Random-Myopic", "PDRS-Myopic", "LoadAware-Myopic", "FFD-Myopic"
        ])
    if args.algorithms:
        config.algorithms = args.algorithms
    if args.seeds is not None and len(args.seeds) > 0:
        config.seeds = args.seeds
    if args.time_slots is not None:
        config.time_slots = args.time_slots
    if args.slow_epoch_slots is not None:
        config.slow_epoch_slots = args.slow_epoch_slots
    if args.model_path:
        config.model_path = args.model_path
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.include_solver_benchmark:
        config.include_solver_benchmark = True
    if args.include_energy_claim:
        config.include_energy_claim = True

    if config.experiment_type == "normal_main" and not args.smoke:
        apply_heterogeneous_burst_main_profile(
            config,
            preserve_runtime_overrides=should_preserve_profile_runtime_overrides(args),
        )
    elif config.experiment_type == "c4_ablation" and config.include_energy_claim and not args.smoke:
        apply_heterogeneous_burst_c4_profile(
            config,
            preserve_runtime_overrides=should_preserve_profile_runtime_overrides(args),
        )
    apply_energy_claim_profile(config)
    result = run_ablation_experiment(config=config, run_id=args.run_id, silent=not args.verbose)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()



















