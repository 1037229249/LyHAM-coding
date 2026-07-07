"""Replay LyHAM-CO dry-run candidates and summarize resource economics."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
UTILS_DIR = REPO_ROOT / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from ablation_algorithms import (  # noqa: E402
    _run_named_algorithm,
    build_temporal_metric_history_entry,
    calculate_claim_score,
    get_algorithm_policies,
    normalize_candidate_source_family,
    run_slow_context_for_algorithm,
)
import ablation_algorithms as ablation_module  # noqa: E402
from ablation_config import AblationExperimentConfig  # noqa: E402
from run_ablation import (  # noqa: E402
    build_workload_trace,
    create_ablation_system,
    set_all_random_seeds,
    suppress_output_if_needed,
)


SELECTED_DIAGNOSTIC_FIELDS = (
    "best_energy_candidate_source",
    "best_energy_candidate_hash",
    "best_energy_candidate_energy_j",
    "best_energy_candidate_dpp_score",
    "best_energy_candidate_energy_gap",
    "best_energy_candidate_dpp_gap",
    "energy_relief_candidate_source",
    "energy_relief_candidate_hash",
    "energy_relief_candidate_delay_ms",
    "energy_relief_candidate_energy_j",
    "energy_relief_candidate_cost",
    "energy_relief_candidate_claim_score",
    "energy_relief_candidate_dpp_score",
    "energy_relief_candidate_predicted_avg_y",
    "energy_relief_candidate_predicted_avg_z",
    "energy_relief_candidate_post_update_queue_drift_term",
    "energy_relief_candidate_energy_gain_j",
    "energy_relief_candidate_delay_regret_ms",
    "energy_relief_candidate_cost_regret",
    "energy_relief_candidate_dpp_regret",
    "energy_relief_best_lower_source",
    "energy_relief_best_lower_hash",
    "energy_relief_best_lower_delay_ms",
    "energy_relief_best_lower_energy_j",
    "energy_relief_best_lower_cost",
    "energy_relief_best_lower_claim_score",
    "energy_relief_best_lower_dpp_score",
    "energy_relief_best_lower_predicted_avg_y",
    "energy_relief_best_lower_predicted_avg_z",
    "energy_relief_best_lower_post_update_queue_drift_term",
    "energy_relief_best_lower_energy_gain_j",
    "energy_relief_best_lower_delay_regret_ms",
    "energy_relief_best_lower_cost_regret",
    "energy_relief_best_lower_dpp_regret",
    "energy_relief_best_lower_reject_reason",
    "predicted_avg_y",
    "predicted_avg_z",
    "active_cloud_pair_signature",
    "post_update_queue_drift_term",
    "post_update_queue_pressure_term",
    "post_update_queue_delta_term",
    "post_update_energy_queue_delta_term",
    "post_update_delay_queue_delta_term",
    "v_cost_term",
    "energy_queue_term",
    "delay_queue_term",
    "dpp_band_passed",
    "is_pareto_candidate",
    "best_local_score",
    "best_cloud_score",
    "score_gap_local_vs_cloud",
    "local_candidate_feasible_count",
    "all_cloud_candidate_count",
    "all_local_candidate_count",
)


def _coerce_config_value(key: str, value):
    if isinstance(value, list) and (
        key.endswith("_range")
        or key in {
            "cloud_f_pre_rails",
            "algorithms",
            "seeds",
            "claim_baselines",
        }
    ):
        return tuple(value)
    return value


def _load_config(meta_payload: dict) -> AblationExperimentConfig:
    config = AblationExperimentConfig()
    for key, value in (meta_payload.get("config") or {}).items():
        if hasattr(config, key):
            setattr(config, key, _coerce_config_value(key, value))
    return config


def _float(value, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _bits(values) -> str:
    try:
        return "".join(str(int(x)) for x in values)
    except Exception:
        return ""


def _pair_action_detail(bits: str, pair_universe: list[dict]) -> dict:
    values = [int(ch) for ch in str(bits) if ch in {"0", "1"}]
    universe = list(pair_universe or [])
    pair_count = min(len(values), len(universe)) if universe else len(values)
    local_pairs = []
    cloud_pairs = []
    for index in range(pair_count):
        bit = int(values[index])
        item = dict(universe[index]) if index < len(universe) else {}
        entry = {
            "pair_index": int(index),
            "bit": bit,
            "mode": "cloud" if bit == 1 else "local",
            "pair_id": str(item.get("pair_id", "")),
            "flow_id": str(item.get("flow_id", "")),
            "microservice_id": str(item.get("microservice_id", "")),
            "server_id": str(item.get("server_id", "")),
            "server_index": int(item.get("server_index", 0)) if str(item.get("server_index", "")).strip() else 0,
            "instance_id": str(item.get("instance_id", "")),
        }
        if bit == 1:
            cloud_pairs.append(entry)
        else:
            local_pairs.append(entry)
    return {
        "pair_count": int(pair_count),
        "local_pair_indices": [int(item["pair_index"]) for item in local_pairs],
        "cloud_pair_indices": [int(item["pair_index"]) for item in cloud_pairs],
        "local_pairs": local_pairs,
        "cloud_pairs": cloud_pairs,
    }


def _selected_diagnostics(result) -> dict:
    diagnostics = {}
    for field in SELECTED_DIAGNOSTIC_FIELDS:
        value = getattr(result, field, None)
        if isinstance(value, (bool, str)):
            diagnostics[field] = value
        elif value is None:
            diagnostics[field] = None
        else:
            try:
                diagnostics[field] = float(value)
            except (TypeError, ValueError):
                diagnostics[field] = str(value)
    return diagnostics


def _instance_rows(system_state) -> list[dict]:
    rows = []
    for instance_id, instance in sorted(system_state.microservice_instances.items()):
        if instance.microservice.service_type != "ai":
            continue
        active_pairs = int(getattr(instance, "active_pair_count", 0))
        mode = str(getattr(instance, "processing_mode", "") or "")
        if active_pairs <= 0 and not mode:
            continue
        pair_bit = getattr(instance, "pair_action_bit", None)
        rows.append({
            "instance_id": str(instance_id),
            "server_id": str(getattr(instance, "server_id", "")),
            "processing_mode": mode,
            "pair_action_bit": None if pair_bit is None else int(pair_bit),
            "active_pair_count": active_pairs,
            "active_local_pair_count": int(getattr(instance, "active_local_pair_count", 0)),
            "active_cloud_pair_count": int(getattr(instance, "active_cloud_pair_count", 0)),
            "gpu_units_allocated": _float(getattr(instance, "gpu_units_allocated", 0.0)),
            "batch_size_allocated": int(getattr(instance, "batch_size_allocated", 1)),
            "gpu_frequency_scale": _float(getattr(instance, "gpu_frequency_scale", 1.0)),
            "preprocess_frequency_scale": _float(getattr(instance, "preprocess_frequency_scale", 0.0)),
            "compression_ratio": _float(getattr(instance, "compression_ratio", 1.0)),
            "resource_config_source": str(getattr(instance, "resource_config_source", "") or ""),
            "resource_hint": str(getattr(instance, "resource_hint", "") or ""),
            "inference_latency_ms": _float(getattr(instance, "inference_latency", 0.0)),
            "cloud_latency_ms": _float(getattr(instance, "cloud_latency", 0.0)),
            "energy_local_gpu_j": _float(getattr(instance, "energy_local_gpu_j", 0.0)),
            "energy_cloud_compute_j": _float(getattr(instance, "energy_cloud_compute_j", 0.0)),
            "energy_comm_j": _float(getattr(instance, "energy_comm_j", 0.0)),
            "energy_preprocess_j": _float(getattr(instance, "energy_preprocess_j", 0.0)),
        })
    return rows


def _resource_summary(rows: list[dict]) -> dict:
    local = [row for row in rows if row["processing_mode"] == "local_processing"]
    cloud = [row for row in rows if row["processing_mode"] == "cloud_offloaded"]
    batch_counter = Counter(int(row["batch_size_allocated"]) for row in local)
    gpu_counter = Counter(int(row["gpu_units_allocated"]) for row in local)
    f_gpu_counter = Counter(round(float(row["gpu_frequency_scale"]), 4) for row in local)
    f_pre_counter = Counter(round(float(row["preprocess_frequency_scale"]), 4) for row in cloud)
    compression_values = [float(row["compression_ratio"]) for row in cloud]
    local_latencies = [float(row["inference_latency_ms"]) for row in local]
    cloud_latencies = [float(row["cloud_latency_ms"]) for row in cloud]
    return {
        "active_local_instances": len(local),
        "active_cloud_instances": len(cloud),
        "local_gpu_energy_j": sum(float(row["energy_local_gpu_j"]) for row in rows),
        "cloud_compute_energy_j": sum(float(row["energy_cloud_compute_j"]) for row in rows),
        "cloud_comm_energy_j": sum(float(row["energy_comm_j"]) for row in rows),
        "cloud_preprocess_energy_j": sum(float(row["energy_preprocess_j"]) for row in rows),
        "local_latency_mean_ms": mean(local_latencies) if local_latencies else 0.0,
        "local_latency_max_ms": max(local_latencies) if local_latencies else 0.0,
        "cloud_latency_mean_ms": mean(cloud_latencies) if cloud_latencies else 0.0,
        "cloud_latency_max_ms": max(cloud_latencies) if cloud_latencies else 0.0,
        "compression_mean": mean(compression_values) if compression_values else 0.0,
        "batch_distribution": {str(k): v for k, v in sorted(batch_counter.items())},
        "gpu_units_distribution": {str(k): v for k, v in sorted(gpu_counter.items())},
        "f_gpu_distribution": {f"{k:.4f}": v for k, v in sorted(f_gpu_counter.items())},
        "f_pre_distribution": {f"{k:.4f}": v for k, v in sorted(f_pre_counter.items())},
    }


def _raw_rows(path: Path | None) -> list[dict]:
    if not path:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _rank_key(row: dict, key: str) -> tuple:
    return (
        not bool(row.get("feasible", False)),
        _float(row.get(key), float("inf")),
        _float(row.get("paper_dpp_score"), float("inf")),
        int(row.get("eval_index", 0)),
    )


def _best(rows: list[dict], key: str) -> dict:
    return min(rows, key=lambda row: _rank_key(row, key)) if rows else {}


def _brief(row: dict) -> dict:
    if not row:
        return {}
    resource = row.get("resource_summary", {})
    return {
        "eval_index": int(row.get("eval_index", 0)),
        "candidate_source": str(row.get("candidate_source", "")),
        "candidate_family": str(row.get("candidate_family", "")),
        "resource_hint": str(row.get("resource_hint", "")),
        "requested_pair_bits": str(row.get("requested_pair_bits", "")),
        "repaired_pair_bits": str(row.get("repaired_pair_bits", "")),
        "active_cloud_pair_signature": str(row.get("active_cloud_pair_signature", "")),
        "local_pair_count": int(row.get("local_pair_count", 0)),
        "cloud_pair_count": int(row.get("cloud_pair_count", 0)),
        "delay_ms": _float(row.get("delay_ms"), float("nan")),
        "energy_j": _float(row.get("energy_j"), float("nan")),
        "cost": _float(row.get("cost"), float("nan")),
        "claim_score": _float(row.get("claim_score"), float("nan")),
        "paper_dpp_score": _float(row.get("paper_dpp_score"), float("nan")),
        "energy_local_gpu_j": _float(row.get("energy_local_gpu_j"), 0.0),
        "energy_cloud_compute_j": _float(row.get("energy_cloud_compute_j"), 0.0),
        "energy_comm_j": _float(row.get("energy_comm_j"), 0.0),
        "energy_idle_replica_j": _float(row.get("energy_idle_replica_j"), 0.0),
        "active_local_instances": int(resource.get("active_local_instances", 0)),
        "active_cloud_instances": int(resource.get("active_cloud_instances", 0)),
        "cloud_compute_energy_j": _float(resource.get("cloud_compute_energy_j"), 0.0),
        "cloud_comm_energy_j": _float(resource.get("cloud_comm_energy_j"), 0.0),
        "cloud_preprocess_energy_j": _float(resource.get("cloud_preprocess_energy_j"), 0.0),
        "cloud_latency_mean_ms": _float(resource.get("cloud_latency_mean_ms"), 0.0),
        "cloud_latency_max_ms": _float(resource.get("cloud_latency_max_ms"), 0.0),
        "f_pre_distribution": dict(resource.get("f_pre_distribution", {})),
        "compression_mean": _float(resource.get("compression_mean"), 0.0),
        "local_latency_mean_ms": _float(resource.get("local_latency_mean_ms"), 0.0),
        "batch_distribution": dict(resource.get("batch_distribution", {})),
        "f_gpu_distribution": dict(resource.get("f_gpu_distribution", {})),
    }


def _guard_row_brief(row: dict) -> dict:
    if not row:
        return {}
    brief = {
        "candidate_source": str(row.get("candidate_source", "")),
        "repaired_pair_action_hash": str(
            row.get("repaired_pair_action_hash") or row.get("pair_action_hash") or row.get("action_hash") or ""
        ),
        "local_pair_count": int(row.get("local_pair_count", row.get("local_count", 0))),
        "cloud_pair_count": int(row.get("cloud_pair_count", row.get("cloud_count", 0))),
        "active_cloud_pair_signature": str(row.get("active_cloud_pair_signature", "")),
        "delay_ms": _float(row.get("delay_ms"), float("nan")),
        "energy_j": _float(row.get("energy_j"), float("nan")),
        "cost": _float(row.get("cost"), float("nan")),
        "claim_score": _float(row.get("claim_score"), float("nan")),
        "paper_dpp_score": _float(row.get("paper_dpp_score"), float("nan")),
        "predicted_avg_y": _float(row.get("predicted_avg_y"), float("nan")),
        "post_update_energy_queue_delta_term": _float(
            row.get("post_update_energy_queue_delta_term"), float("nan")
        ),
        "post_update_queue_drift_term": _float(row.get("post_update_queue_drift_term"), float("nan")),
        "energy_queue_term": _float(row.get("energy_queue_term"), 0.0),
        "delay_queue_term": _float(row.get("delay_queue_term"), 0.0),
        "v_cost_term": _float(row.get("v_cost_term"), 0.0),
        "eval_rank": int(row.get("eval_rank", row.get("eval_index", 0))),
    }
    if "energy_guard_strict_dpp_baseline_score" in row:
        brief["energy_guard_strict_dpp_baseline_score"] = _float(
            row.get("energy_guard_strict_dpp_baseline_score"), 0.0
        )
    if "energy_guard_strict_dpp_baseline_source" in row:
        brief["energy_guard_strict_dpp_baseline_source"] = str(
            row.get("energy_guard_strict_dpp_baseline_source", "")
        )
    if "energy_guard_strict_dpp_baseline_hash" in row:
        brief["energy_guard_strict_dpp_baseline_hash"] = str(
            row.get("energy_guard_strict_dpp_baseline_hash", "")
        )
    return brief


def _energy_guard_invocation_summary(rows: list[dict], config: AblationExperimentConfig,
                                     current_row: dict, selected_row: dict) -> dict:
    current_energy = _float(current_row.get("energy_j"), float("inf"))
    current_delay = _float(current_row.get("delay_ms"), float("inf"))
    current_cost = _float(current_row.get("cost"), float("inf"))
    current_dpp = _float(current_row.get("paper_dpp_score"), float("inf"))
    min_energy_gain = max(float(getattr(config, "queue_pressure_energy_guard_min_energy_gain_j", 1.0)), 0.0)
    max_delay_regret = max(float(getattr(config, "queue_pressure_energy_guard_max_delay_regret_ms", 20.0)), 0.0)
    max_cost_regret = max(float(getattr(config, "queue_pressure_energy_guard_max_cost_regret", 80.0)), 0.0)
    relief_reason = ablation_module._energy_guard_queue_relief_reason(current_row, current_row, config)
    high_queue_pressure = relief_reason in {"queue_relief_absent", ""}
    dpp_slack = 0.0
    if high_queue_pressure:
        dpp_slack = max(
            abs(current_dpp) *
            max(float(getattr(config, "queue_pressure_energy_guard_dpp_slack_ratio", 0.0)), 0.0),
            0.0,
        )

    reason_counts = Counter()
    hybrid_reason_counts = Counter()
    source_counts = Counter()
    hybrid_source_counts = Counter()
    eligible = []
    lower_energy = []
    lower_energy_hybrid = []
    for row in rows:
        source_family = normalize_candidate_source_family(str(row.get("candidate_source", "")))
        source_counts[source_family] += 1
        is_hybrid_or_cloud = int(row.get("cloud_pair_count", row.get("cloud_count", 0))) > 0
        if is_hybrid_or_cloud:
            hybrid_source_counts[source_family] += 1
        if row is current_row:
            continue
        row_energy = _float(row.get("energy_j"), float("inf"))
        row_delay = _float(row.get("delay_ms"), float("inf"))
        row_cost = _float(row.get("cost"), float("inf"))
        row_dpp = _float(row.get("paper_dpp_score"), float("inf"))
        if not bool(row.get("feasible", False)):
            reason = "infeasible"
        elif not all(np.isfinite(value) for value in [row_energy, row_delay, row_cost, row_dpp]):
            reason = "nonfinite"
        elif current_energy - row_energy < min_energy_gain:
            reason = "energy_gain_below_min"
        elif row_dpp > current_dpp + dpp_slack:
            reason = "dpp_regret"
        elif ablation_module._energy_guard_strict_dpp_regret_reason(row, current_row, config):
            reason = "strict_dpp_regret"
        elif ablation_module._energy_guard_delay_regret_reason(
            row_delay,
            current_delay,
            row_dpp,
            current_dpp,
            high_queue_pressure,
            max_delay_regret,
            config,
        ):
            reason = "delay_regret"
        elif row_cost - current_cost > max_cost_regret:
            reason = "cost_regret"
        elif high_queue_pressure:
            reason = ablation_module._energy_guard_queue_relief_reason(row, current_row, config)
        else:
            reason = ""
        if (
            not reason and
            ablation_module._energy_guard_is_relaxed_cloud_resource(row) and
            any(
                ablation_module._energy_guard_is_same_cloud_recovery_probe(row, probe_row, config)
                for probe_row in rows
            )
        ):
            reason = "same_cloud_recovery_regret"
        reason = str(reason or "eligible")
        reason_counts[reason] += 1
        if is_hybrid_or_cloud:
            hybrid_reason_counts[reason] += 1
        if bool(row.get("feasible", False)) and np.isfinite(row_energy) and current_energy - row_energy > 0.0:
            lower_energy.append(row)
            if is_hybrid_or_cloud:
                lower_energy_hybrid.append(row)
        if reason == "eligible":
            eligible.append(row)

    return {
        "row_count": len(rows),
        "feasible_count": sum(1 for row in rows if bool(row.get("feasible", False))),
        "current_relief_reason": str(relief_reason),
        "high_queue_pressure": bool(high_queue_pressure),
        "dpp_slack": float(dpp_slack),
        "min_energy_gain_j": float(min_energy_gain),
        "max_delay_regret_ms": float(max_delay_regret),
        "dpp_improvement_delay_slack_ratio": float(
            getattr(config, "queue_pressure_energy_guard_dpp_improvement_delay_slack_ratio", 0.0)
        ),
        "dpp_improvement_queue_drift_offset_ratio": float(
            getattr(config, "queue_pressure_energy_guard_dpp_improvement_queue_drift_offset_ratio", 0.0)
        ),
        "strict_dpp_regret_ratio": float(
            getattr(config, "queue_pressure_energy_guard_strict_dpp_regret_ratio", 0.0)
        ),
        "max_cost_regret": float(max_cost_regret),
        "current": _guard_row_brief(current_row),
        "selected_by_energy_guard": _guard_row_brief(selected_row),
        "selected_energy_gain_j": float(current_energy - _float(selected_row.get("energy_j"), current_energy))
        if selected_row else 0.0,
        "lower_energy_count": len(lower_energy),
        "lower_energy_hybrid_or_cloud_count": len(lower_energy_hybrid),
        "eligible_count": len(eligible),
        "eligible_hybrid_or_cloud_count": sum(
            1 for row in eligible if int(row.get("cloud_pair_count", row.get("cloud_count", 0))) > 0
        ),
        "reason_counts": {str(k): int(v) for k, v in sorted(reason_counts.items())},
        "hybrid_or_cloud_reason_counts": {str(k): int(v) for k, v in sorted(hybrid_reason_counts.items())},
        "source_family_counts": {str(k): int(v) for k, v in sorted(source_counts.items())},
        "hybrid_or_cloud_source_family_counts": {
            str(k): int(v) for k, v in sorted(hybrid_source_counts.items())
        },
        "best_lower_energy": _guard_row_brief(min(lower_energy, key=lambda row: _float(row.get("energy_j"), float("inf"))))
        if lower_energy else {},
        "best_lower_energy_hybrid_or_cloud": _guard_row_brief(
            min(lower_energy_hybrid, key=lambda row: _float(row.get("energy_j"), float("inf")))
        ) if lower_energy_hybrid else {},
        "top_lower_energy_hybrid_or_cloud": [
            _guard_row_brief(row)
            for row in sorted(
                lower_energy_hybrid,
                key=lambda row: (
                    _float(row.get("energy_j"), float("inf")),
                    _float(row.get("paper_dpp_score"), float("inf")),
                    int(row.get("eval_rank", row.get("eval_index", 0))),
                ),
            )[:5]
        ],
    }


def _delta(candidate: dict, reference: dict) -> dict:
    if not candidate or not reference:
        return {}
    fields = ["delay_ms", "energy_j", "cost", "claim_score", "paper_dpp_score"]
    return {field: _float(candidate.get(field), 0.0) - _float(reference.get(field), 0.0) for field in fields}


def _source_family_summary(rows: list[dict]) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("candidate_family", ""))].append(row)
    summary = {}
    for family, items in sorted(groups.items()):
        feasible = [row for row in items if bool(row.get("feasible", False))]
        source_rows = feasible or items
        best_claim = _best(source_rows, "claim_score")
        best_dpp = _best(source_rows, "paper_dpp_score")
        best_energy = _best(source_rows, "energy_j")
        summary[family] = {
            "count": len(items),
            "feasible_count": len(feasible),
            "best_claim": _brief(best_claim),
            "best_dpp": _brief(best_dpp),
            "best_energy": _brief(best_energy),
        }
    return summary


def _summarize_slots(slots: list[dict]) -> dict:
    all_deltas = []
    source_counter = Counter()
    hybrid_source_counter = Counter()
    for slot in slots:
        for row in slot["candidate_records"]:
            source_counter[str(row.get("candidate_family", ""))] += 1
            if int(row.get("cloud_pair_count", 0)) > 0 and int(row.get("local_pair_count", 0)) > 0:
                hybrid_source_counter[str(row.get("candidate_family", ""))] += 1
        if slot.get("best_hybrid_or_cloud_by_claim") and slot.get("best_all_local_by_claim"):
            all_deltas.append(_delta(slot["best_hybrid_or_cloud_by_claim"], slot["best_all_local_by_claim"]))
    mean_delta = {}
    for field in ["delay_ms", "energy_j", "cost", "claim_score", "paper_dpp_score"]:
        values = [item[field] for item in all_deltas if field in item and math.isfinite(item[field])]
        mean_delta[field] = mean(values) if values else 0.0
    return {
        "slot_count": len(slots),
        "candidate_count": sum(len(slot["candidate_records"]) for slot in slots),
        "hybrid_or_cloud_better_claim_slots": sum(
            1
            for slot in slots
            if slot.get("best_hybrid_or_cloud_by_claim")
            and slot.get("best_all_local_by_claim")
            and _float(slot["best_hybrid_or_cloud_by_claim"].get("claim_score"), float("inf"))
            < _float(slot["best_all_local_by_claim"].get("claim_score"), float("inf"))
        ),
        "hybrid_or_cloud_better_dpp_slots": sum(
            1
            for slot in slots
            if slot.get("best_hybrid_or_cloud_by_dpp")
            and slot.get("best_all_local_by_dpp")
            and _float(slot["best_hybrid_or_cloud_by_dpp"].get("paper_dpp_score"), float("inf"))
            < _float(slot["best_all_local_by_dpp"].get("paper_dpp_score"), float("inf"))
        ),
        "mean_hybrid_or_cloud_minus_all_local_claim_best": mean_delta,
        "candidate_family_distribution": dict(sorted(source_counter.items())),
        "hybrid_family_distribution": dict(sorted(hybrid_source_counter.items())),
    }


def _markdown(payload: dict) -> str:
    summary = payload["summary"]
    lines = [
        "# P5 Candidate Economics Audit",
        "",
        f"- Meta: `{payload['meta_path']}`",
        f"- Raw comparison: `{payload.get('raw_path', '')}`",
        f"- Algorithm: `{payload['algorithm']}`",
        f"- Seed: {payload['seed']}",
        f"- Slots replayed: {summary['slot_count']}",
        f"- Dry-run candidates captured: {summary['candidate_count']}",
        f"- Hybrid/cloud better by claim slots: {summary['hybrid_or_cloud_better_claim_slots']}",
        f"- Hybrid/cloud better by DPP slots: {summary['hybrid_or_cloud_better_dpp_slots']}",
        "",
        "## Mean Hybrid/Cloud Minus All-Local Delta",
        "",
        "| metric | mean delta |",
        "|---|---:|",
    ]
    for key, value in summary["mean_hybrid_or_cloud_minus_all_local_claim_best"].items():
        lines.append(f"| {key} | {value:.6f} |")
    lines.extend([
        "",
        "## Per-Slot Best Claim Rows",
        "",
        "| slot | selected | all-local source | all-local d/e/c/claim | hybrid/cloud source | hybrid/cloud d/e/c/claim | delta d/e/c/claim |",
        "|---:|---|---|---|---|---|---|",
    ])
    for slot in payload["slots"]:
        local = slot.get("best_all_local_by_claim", {})
        hybrid = slot.get("best_hybrid_or_cloud_by_claim", {})
        delta = _delta(hybrid, local)
        lines.append(
            f"| {slot['slot']} | {slot['selected_candidate_source']} | "
            f"{local.get('candidate_source', '')} | "
            f"{_float(local.get('delay_ms')):.3f}/{_float(local.get('energy_j')):.3f}/{_float(local.get('cost')):.3f}/{_float(local.get('claim_score')):.3f} | "
            f"{hybrid.get('candidate_source', '')} | "
            f"{_float(hybrid.get('delay_ms')):.3f}/{_float(hybrid.get('energy_j')):.3f}/{_float(hybrid.get('cost')):.3f}/{_float(hybrid.get('claim_score')):.3f} | "
            f"{_float(delta.get('delay_ms')):.3f}/{_float(delta.get('energy_j')):.3f}/{_float(delta.get('cost')):.3f}/{_float(delta.get('claim_score')):.3f} |"
        )
    lines.extend([
        "",
        "## Hybrid/Cloud Resource Components",
        "",
        "| slot | source | local/cloud pairs | cloud compute | cloud comm | preprocess | cloud latency mean/max | f_pre |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ])
    for slot in payload["slots"]:
        hybrid = slot.get("best_hybrid_or_cloud_by_claim", {})
        lines.append(
            f"| {slot['slot']} | {hybrid.get('candidate_source', '')} | "
            f"{int(hybrid.get('local_pair_count', 0))}/{int(hybrid.get('cloud_pair_count', 0))} | "
            f"{_float(hybrid.get('cloud_compute_energy_j')):.6f} | "
            f"{_float(hybrid.get('cloud_comm_energy_j')):.6f} | "
            f"{_float(hybrid.get('cloud_preprocess_energy_j')):.6f} | "
            f"{_float(hybrid.get('cloud_latency_mean_ms')):.3f}/{_float(hybrid.get('cloud_latency_max_ms')):.3f} | "
            f"{json.dumps(hybrid.get('f_pre_distribution', {}), sort_keys=True)} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", required=True, type=Path)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--algorithm", default="LyHAM-CO")
    parser.add_argument("--slot-limit", type=int, default=12)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--markdown", required=True, type=Path)
    parser.add_argument(
        "--include-resource-detail",
        action="store_true",
        help="Include per-pair repaired action details and per-instance resource rows in candidate records.",
    )
    args = parser.parse_args()

    meta_path = args.meta.resolve()
    raw_path = args.raw.resolve() if args.raw else None
    meta_payload = json.loads(meta_path.read_text(encoding="utf-8-sig"))
    config = _load_config(meta_payload)
    config.algorithms = (args.algorithm,)
    config.seeds = (int(args.seed),)
    if args.slot_limit > 0:
        config.time_slots = min(int(config.time_slots), int(args.slot_limit))

    set_all_random_seeds(args.seed)
    base_system = create_ablation_system(args.seed, config)
    workload_trace = build_workload_trace(base_system, config.time_slots)
    system_state = copy.deepcopy(base_system)
    slow_policy, fast_controller = get_algorithm_policies(args.algorithm)
    system_state.current_algorithm = str(args.algorithm)
    system_state.current_slow_policy = str(slow_policy)
    system_state.current_fast_controller = str(fast_controller)

    import ResourceAllocation as resource_allocation  # noqa: WPS433

    original_eval = resource_allocation.evaluate_action_dry_run
    original_restore = resource_allocation.restore_system_state
    original_energy_guard = ablation_module.queue_pressure_energy_guard_candidate
    current_eval: dict = {}
    captured_records: list[dict] = []
    current_slot = {"slot": -1}
    guard_records: list[dict] = []

    def capture_restore(state, backup):
        if current_eval.get("active"):
            rows = _instance_rows(state)
            current_eval["resource_rows"] = rows
            current_eval["resource_summary"] = _resource_summary(rows)
        return original_restore(state, backup)

    def eval_wrapper(offloading_mode, state, cfg, queue_aware=True):
        meta = dict(offloading_mode) if isinstance(offloading_mode, dict) else {}
        pair_action = meta.get("pair_action", [])
        eval_index = len(captured_records)
        current_eval.clear()
        current_eval.update({
            "active": True,
            "eval_index": eval_index,
            "resource_rows": [],
            "resource_summary": {},
        })
        try:
            result = original_eval(offloading_mode, state, cfg, queue_aware=queue_aware)
            requested_pair_bits = _bits(pair_action)
            repaired_pair_bits = str(result.get("pair_action_bits", ""))
            row = {
                "slot": int(current_slot["slot"]),
                "eval_index": int(eval_index),
                "candidate_source": str(meta.get("candidate_source", result.get("candidate_source", ""))),
                "candidate_family": normalize_candidate_source_family(
                    str(meta.get("candidate_source", result.get("candidate_source", "")))
                ),
                "resource_hint": str(meta.get("resource_hint", result.get("resource_hint", ""))),
                "resource_mode": str(meta.get("resource_mode", result.get("resource_mode", ""))),
                "base_resource_hint": str(meta.get("base_resource_hint", result.get("base_resource_hint", ""))),
                "resource_queue_aware": bool(meta.get("resource_queue_aware", result.get("resource_queue_aware", queue_aware))),
                "resource_queue_scale": _float(meta.get("resource_queue_scale", result.get("resource_queue_scale", 1.0))),
                "requested_pair_bits": requested_pair_bits,
                "repaired_pair_bits": repaired_pair_bits,
                "active_local_pair_indices": list(result.get("active_local_pair_indices", [])),
                "active_local_pair_ids": list(result.get("active_local_pair_ids", [])),
                "active_local_pair_signature": str(result.get("active_local_pair_signature", "")),
                "active_cloud_pair_indices": list(result.get("active_cloud_pair_indices", [])),
                "active_cloud_pair_ids": list(result.get("active_cloud_pair_ids", [])),
                "active_cloud_pair_signature": str(result.get("active_cloud_pair_signature", "")),
                "feasible": bool(result.get("feasible", False)),
                "failure_reason": str(result.get("failure_reason", "")),
                "delay_ms": _float(result.get("delay_ms"), float("nan")),
                "energy_j": _float(result.get("energy_j"), float("nan")),
                "cost": _float(result.get("cost"), float("nan")),
                "paper_dpp_score": _float(result.get("paper_dpp_score"), float("inf")),
                "claim_score": calculate_claim_score(result, cfg) if bool(result.get("feasible", False)) else float("inf"),
                "local_pair_count": int(result.get("local_pair_count", result.get("local_count", 0))),
                "cloud_pair_count": int(result.get("cloud_pair_count", result.get("cloud_count", 0))),
                "energy_local_gpu_j": _float(result.get("energy_local_gpu_j"), 0.0),
                "energy_cloud_compute_j": _float(result.get("energy_cloud_compute_j"), 0.0),
                "energy_comm_j": _float(result.get("energy_comm_j"), 0.0),
                "energy_idle_replica_j": _float(result.get("energy_idle_replica_j"), 0.0),
                "cost_topo": _float(result.get("cost_topo"), 0.0),
                "cost_comp": _float(result.get("cost_comp"), 0.0),
                "cost_comm": _float(result.get("cost_comm"), 0.0),
                "energy_queue_term": _float(result.get("energy_queue_term"), 0.0),
                "delay_queue_term": _float(result.get("delay_queue_term"), 0.0),
                "v_cost_term": _float(result.get("v_cost_term"), 0.0),
                "predicted_avg_y": _float(result.get("predicted_avg_y"), float("nan")),
                "predicted_avg_z": _float(result.get("predicted_avg_z"), float("nan")),
                "post_update_energy_queue_delta_term": _float(
                    result.get("post_update_energy_queue_delta_term"),
                    float("nan"),
                ),
                "post_update_delay_queue_delta_term": _float(
                    result.get("post_update_delay_queue_delta_term"),
                    float("nan"),
                ),
                "post_update_queue_delta_term": _float(
                    result.get("post_update_queue_delta_term"),
                    float("nan"),
                ),
                "post_update_queue_drift_term": _float(
                    result.get("post_update_queue_drift_term"),
                    float("nan"),
                ),
                "post_update_queue_pressure_term": _float(result.get("post_update_queue_pressure_term"), 0.0),
                "resource_summary": dict(current_eval.get("resource_summary", {})),
            }
            if args.include_resource_detail:
                pair_universe = meta.get("pair_universe", result.get("pair_universe", [])) or []
                row["requested_pair_detail"] = _pair_action_detail(requested_pair_bits, pair_universe)
                row["repaired_pair_detail"] = _pair_action_detail(repaired_pair_bits, pair_universe)
                row["resource_rows"] = [dict(item) for item in current_eval.get("resource_rows", [])]
            captured_records.append(row)
            return result
        finally:
            current_eval.clear()

    resource_allocation.evaluate_action_dry_run = eval_wrapper
    resource_allocation.restore_system_state = capture_restore

    def energy_guard_wrapper(rows, cfg, current_row):
        row_list = list(rows)
        selected = original_energy_guard(row_list, cfg, current_row)
        guard_records.append({
            "slot": int(current_slot["slot"]),
            **_energy_guard_invocation_summary(row_list, cfg, current_row, selected),
        })
        return selected

    ablation_module.queue_pressure_energy_guard_candidate = energy_guard_wrapper

    raw_rows = _raw_rows(raw_path) if raw_path else []
    slow_context_ready = False
    temporal_metric_history = []
    slots = []
    try:
        for slot, arrivals in enumerate(workload_trace):
            current_slot["slot"] = int(slot)
            slot_start = len(captured_records)
            system_state.time_frame = slot
            with suppress_output_if_needed(True):
                system_state.environment_manager.update_all_ai_server_states(
                    arrivals,
                    allow_redeployment=False,
                )
                slow_context_reused = False
                if slot % config.slow_epoch_slots == 0 or not slow_context_ready:
                    slow_backup = copy.deepcopy(system_state)
                    slow_ok, slow_reason = run_slow_context_for_algorithm(
                        args.algorithm, system_state, config, slot, args.seed
                    )
                    if slow_ok:
                        slow_context_ready = True
                    elif slot == 0:
                        raise RuntimeError(f"slow context failed at slot 0: {slow_reason}")
                    else:
                        system_state = slow_backup
                        slow_context_reused = True
                if args.algorithm == "LyHAM-CO":
                    setattr(
                        system_state,
                        "_lyham_temporal_metric_history",
                        list(temporal_metric_history),
                    )
                result = _run_named_algorithm(
                    algorithm=args.algorithm,
                    system_state=system_state,
                    config=config,
                    slot=slot,
                    seed=args.seed,
                    slow_context_reused=slow_context_reused,
                )

            slot_records = [dict(row) for row in captured_records[slot_start:]]
            slot_guard_records = [
                dict(row) for row in guard_records
                if int(row.get("slot", -1)) == int(slot)
            ]
            feasible = [row for row in slot_records if bool(row.get("feasible", False))]
            all_local = [
                row for row in feasible
                if int(row.get("local_pair_count", 0)) > 0 and int(row.get("cloud_pair_count", 0)) == 0
            ]
            hybrid_or_cloud = [
                row for row in feasible
                if int(row.get("cloud_pair_count", 0)) > 0
            ]
            slot_payload = {
                "slot": int(slot),
                "selected_candidate_source": str(result.selected_candidate_source),
                "selected_by": str(result.selected_by_dpp_or_claim_band),
                "selected_delay_ms": float(result.delay_ms),
                "selected_energy_j": float(result.energy_j),
                "selected_cost": float(result.cost),
                "selected_claim_score": float(result.claim_score),
                "selected_paper_dpp_score": float(result.paper_dpp_score),
                "selected_local_pair_count": int(result.local_pair_count),
                "selected_cloud_pair_count": int(result.cloud_pair_count),
                "selected_decision_diagnostics": _selected_diagnostics(result),
                "energy_guard_invocations": slot_guard_records,
                "candidate_records": slot_records,
                "source_family_summary": _source_family_summary(slot_records),
                "best_all_local_by_claim": _brief(_best(all_local, "claim_score")),
                "best_all_local_by_dpp": _brief(_best(all_local, "paper_dpp_score")),
                "best_hybrid_or_cloud_by_claim": _brief(_best(hybrid_or_cloud, "claim_score")),
                "best_hybrid_or_cloud_by_dpp": _brief(_best(hybrid_or_cloud, "paper_dpp_score")),
                "best_hybrid_or_cloud_by_energy": _brief(_best(hybrid_or_cloud, "energy_j")),
                "raw_delta": {},
            }
            if slot < len(raw_rows):
                raw_row = raw_rows[slot]
                for field, attr in (
                    ("delay_ms", "delay_ms"),
                    ("energy_j", "energy_j"),
                    ("cost", "cost"),
                ):
                    slot_payload["raw_delta"][field] = _float(getattr(result, attr), 0.0) - _float(raw_row.get(field), 0.0)
            slots.append(slot_payload)

            if args.algorithm == "LyHAM-CO" and result.status == "ok":
                history_entry = build_temporal_metric_history_entry(result)
                temporal_values = [
                    history_entry["delay_ms"],
                    history_entry["energy_j"],
                    history_entry["cost"],
                ]
                if all(math.isfinite(value) for value in temporal_values):
                    temporal_metric_history.append(history_entry)
            if result.status == "invalid":
                break
    finally:
        resource_allocation.evaluate_action_dry_run = original_eval
        resource_allocation.restore_system_state = original_restore
        ablation_module.queue_pressure_energy_guard_candidate = original_energy_guard

    payload = {
        "meta_path": str(meta_path),
        "raw_path": str(raw_path) if raw_path else "",
        "algorithm": str(args.algorithm),
        "seed": int(args.seed),
        "summary": _summarize_slots(slots),
        "slots": slots,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.markdown.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
