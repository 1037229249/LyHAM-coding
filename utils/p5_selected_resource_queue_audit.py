"""Replay a P5 LyHAM-CO smoke and audit selected resource/queue details."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from statistics import mean


REPO_ROOT = Path(__file__).resolve().parents[1]
UTILS_DIR = REPO_ROOT / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from ablation_algorithms import (  # noqa: E402
    _run_named_algorithm,
    build_temporal_metric_history_entry,
    get_algorithm_policies,
    run_slow_context_for_algorithm,
)
from ablation_config import AblationExperimentConfig  # noqa: E402
from run_ablation import (  # noqa: E402
    build_workload_trace,
    create_ablation_system,
    set_all_random_seeds,
    suppress_output_if_needed,
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


def _instance_flow_id(instance_id: str, system_state) -> str:
    for flow_id in sorted(system_state.request_flows.keys(), key=len, reverse=True):
        if instance_id == flow_id or instance_id.startswith(f"{flow_id}_"):
            return str(flow_id)
    return ""


def _queue_snapshot(system_state) -> dict:
    energy = []
    delay = []
    for server_id, queue in sorted(system_state.virtual_energy_queues.items()):
        last = queue.history[-1] if queue.history else {}
        energy.append({
            "server_id": server_id,
            "queue_state": _float(getattr(queue, "queue_state", 0.0)),
            "energy_j": _float(last.get("energy", 0.0)),
            "threshold_j": _float(getattr(queue, "energy_threshold", 0.0)),
            "increment": _float(last.get("increment", 0.0)),
            "positive_increment": _float(last.get("increment", 0.0)) > 0.0,
        })
    for server_id, queue in sorted(system_state.virtual_delay_queues.items()):
        last = queue.history[-1] if queue.history else {}
        delay.append({
            "server_id": server_id,
            "queue_state": _float(getattr(queue, "queue_state", 0.0)),
            "delay_ms": _float(last.get("delay", 0.0)),
            "threshold_ms": _float(getattr(queue, "delay_threshold", 0.0)),
            "increment": _float(last.get("increment", 0.0)),
            "positive_increment": _float(last.get("increment", 0.0)) > 0.0,
        })
    return {"energy": energy, "delay": delay}


def _resource_snapshot(system_state) -> list[dict]:
    rows = []
    for instance_id, instance in sorted(system_state.microservice_instances.items()):
        if instance.microservice.service_type != "ai":
            continue
        active_pairs = int(getattr(instance, "active_pair_count", 0))
        mode = str(getattr(instance, "processing_mode", "") or "")
        if active_pairs <= 0 and not mode:
            continue
        rows.append({
            "instance_id": str(instance_id),
            "flow_id": _instance_flow_id(str(instance_id), system_state),
            "server_id": str(getattr(instance, "server_id", "")),
            "processing_mode": mode,
            "active_pair_count": active_pairs,
            "active_local_pair_count": int(getattr(instance, "active_local_pair_count", 0)),
            "active_cloud_pair_count": int(getattr(instance, "active_cloud_pair_count", 0)),
            "gpu_units_allocated": _float(getattr(instance, "gpu_units_allocated", 0.0)),
            "batch_size_allocated": int(getattr(instance, "batch_size_allocated", 1)),
            "gpu_frequency_scale": _float(getattr(instance, "gpu_frequency_scale", 1.0)),
            "resource_config_source": str(getattr(instance, "resource_config_source", "") or ""),
            "inference_latency_ms": _float(getattr(instance, "inference_latency", 0.0)),
            "cloud_latency_ms": _float(getattr(instance, "cloud_latency", 0.0)),
            "energy_local_gpu_j": _float(getattr(instance, "energy_local_gpu_j", 0.0)),
            "energy_cloud_compute_j": _float(getattr(instance, "energy_cloud_compute_j", 0.0)),
            "energy_comm_j": _float(getattr(instance, "energy_comm_j", 0.0)),
            "energy_preprocess_j": _float(getattr(instance, "energy_preprocess_j", 0.0)),
            "resource_hint": str(getattr(instance, "resource_hint", "") or ""),
        })
    return rows


def _raw_rows(path: Path) -> list[dict]:
    if not path:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _summarize(slots: list[dict]) -> dict:
    local_config_counter = Counter()
    source_counter = Counter()
    batch_counter = Counter()
    gpu_counter = Counter()
    rail_counter = Counter()
    max_queue_energy_inputs = []
    max_queue_delay_inputs = []
    positive_energy_slots = 0
    positive_delay_slots = 0
    active_local_instances = 0
    active_cloud_instances = 0
    for slot in slots:
        energy_rows = slot["queue_snapshot"]["energy"]
        delay_rows = slot["queue_snapshot"]["delay"]
        if any(row["positive_increment"] for row in energy_rows):
            positive_energy_slots += 1
        if any(row["positive_increment"] for row in delay_rows):
            positive_delay_slots += 1
        if energy_rows:
            max_queue_energy_inputs.append(max(row["energy_j"] for row in energy_rows))
        if delay_rows:
            max_queue_delay_inputs.append(max(row["delay_ms"] for row in delay_rows))
        for item in slot["resource_snapshot"]:
            if item["processing_mode"] == "local_processing":
                active_local_instances += 1
                key = (
                    int(item["gpu_units_allocated"]),
                    int(item["batch_size_allocated"]),
                    round(float(item["gpu_frequency_scale"]), 4),
                )
                local_config_counter[key] += 1
                source_counter[item["resource_config_source"]] += 1
                batch_counter[int(item["batch_size_allocated"])] += 1
                gpu_counter[int(item["gpu_units_allocated"])] += 1
                rail_counter[round(float(item["gpu_frequency_scale"]), 4)] += 1
            elif item["processing_mode"] == "cloud_offloaded":
                active_cloud_instances += 1
    return {
        "slot_count": len(slots),
        "all_local_slots": sum(1 for slot in slots if slot["cloud_pair_count"] == 0 and slot["local_pair_count"] > 0),
        "positive_energy_queue_slots": positive_energy_slots,
        "positive_delay_queue_slots": positive_delay_slots,
        "active_local_instances": active_local_instances,
        "active_cloud_instances": active_cloud_instances,
        "max_queue_energy_input_j": max(max_queue_energy_inputs) if max_queue_energy_inputs else 0.0,
        "mean_max_queue_energy_input_j": mean(max_queue_energy_inputs) if max_queue_energy_inputs else 0.0,
        "max_queue_delay_input_ms": max(max_queue_delay_inputs) if max_queue_delay_inputs else 0.0,
        "mean_max_queue_delay_input_ms": mean(max_queue_delay_inputs) if max_queue_delay_inputs else 0.0,
        "local_config_distribution": {
            f"gpu={key[0]},batch={key[1]},f_gpu={key[2]:.4f}": value
            for key, value in sorted(local_config_counter.items())
        },
        "resource_config_source_distribution": dict(sorted(source_counter.items())),
        "batch_distribution": {str(k): v for k, v in sorted(batch_counter.items())},
        "gpu_units_distribution": {str(k): v for k, v in sorted(gpu_counter.items())},
        "gpu_frequency_distribution": {f"{k:.4f}": v for k, v in sorted(rail_counter.items())},
    }


def _markdown(payload: dict) -> str:
    summary = payload["summary"]
    lines = [
        "# P5 Selected Resource and Queue Audit",
        "",
        f"- Meta: `{payload['meta_path']}`",
        f"- Raw comparison: `{payload.get('raw_path', '')}`",
        f"- Algorithm: `{payload['algorithm']}`",
        f"- Seed: {payload['seed']}",
        f"- Slots replayed: {summary['slot_count']}",
        f"- All-local slots: {summary['all_local_slots']}",
        f"- Positive energy-queue slots: {summary['positive_energy_queue_slots']}",
        f"- Positive delay-queue slots: {summary['positive_delay_queue_slots']}",
        (
            "- Max per-server energy queue input: "
            f"{summary['max_queue_energy_input_j']:.6f} J"
        ),
        (
            "- Max per-server delay queue input: "
            f"{summary['max_queue_delay_input_ms']:.6f} ms"
        ),
        "",
        "## Local Resource Config Distribution",
        "",
        "| config | count |",
        "|---|---:|",
    ]
    for key, value in summary["local_config_distribution"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend([
        "",
        "## Slot Queue Inputs",
        "",
        "| slot | local | cloud | max energy input J | energy threshold J | max delay input ms | delay threshold ms |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for slot in payload["slots"]:
        energy_rows = slot["queue_snapshot"]["energy"]
        delay_rows = slot["queue_snapshot"]["delay"]
        max_energy = max((row["energy_j"] for row in energy_rows), default=0.0)
        energy_threshold = max((row["threshold_j"] for row in energy_rows), default=0.0)
        max_delay = max((row["delay_ms"] for row in delay_rows), default=0.0)
        delay_threshold = max((row["threshold_ms"] for row in delay_rows), default=0.0)
        lines.append(
            f"| {slot['slot']} | {slot['local_pair_count']} | {slot['cloud_pair_count']} | "
            f"{max_energy:.6f} | {energy_threshold:.6f} | "
            f"{max_delay:.6f} | {delay_threshold:.6f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", required=True, type=Path)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--algorithm", default="LyHAM-CO")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--markdown", required=True, type=Path)
    args = parser.parse_args()

    meta_path = args.meta.resolve()
    raw_path = args.raw.resolve() if args.raw else None
    meta_payload = json.loads(meta_path.read_text(encoding="utf-8-sig"))
    config = _load_config(meta_payload)
    config.algorithms = (args.algorithm,)
    config.seeds = (int(args.seed),)

    set_all_random_seeds(args.seed)
    base_system = create_ablation_system(args.seed, config)
    workload_trace = build_workload_trace(base_system, config.time_slots)
    system_state = copy.deepcopy(base_system)
    slow_policy, fast_controller = get_algorithm_policies(args.algorithm)
    system_state.current_algorithm = str(args.algorithm)
    system_state.current_slow_policy = str(slow_policy)
    system_state.current_fast_controller = str(fast_controller)

    raw_rows = _raw_rows(raw_path) if raw_path else []
    slow_context_ready = False
    temporal_metric_history = []
    slots = []
    for slot, arrivals in enumerate(workload_trace):
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

        row = {
            "slot": int(slot),
            "delay_ms": float(result.delay_ms),
            "energy_j": float(result.energy_j),
            "cost": float(result.cost),
            "avg_y": float(result.avg_y),
            "avg_z": float(result.avg_z),
            "local_pair_count": int(result.local_pair_count),
            "cloud_pair_count": int(result.cloud_pair_count),
            "selected_candidate_source": str(result.selected_candidate_source),
            "selected_by": str(result.selected_by_dpp_or_claim_band),
            "resource_snapshot": _resource_snapshot(system_state),
            "queue_snapshot": _queue_snapshot(system_state),
            "raw_delta": {},
        }
        if slot < len(raw_rows):
            raw_row = raw_rows[slot]
            for field in ("delay_ms", "energy_j", "cost", "avg_y", "avg_z"):
                row["raw_delta"][field] = float(row[field] - _float(raw_row.get(field), 0.0))
        slots.append(row)

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

    payload = {
        "meta_path": str(meta_path),
        "raw_path": str(raw_path) if raw_path else "",
        "algorithm": str(args.algorithm),
        "seed": int(args.seed),
        "summary": _summarize(slots),
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
