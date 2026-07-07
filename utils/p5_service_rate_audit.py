"""Compute local service-rate components for P5 local feasibility traces."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import mean


REPO_ROOT = Path(__file__).resolve().parents[1]
UTILS_DIR = REPO_ROOT / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from ablation_config import AblationExperimentConfig  # noqa: E402
from run_ablation import create_ablation_system  # noqa: E402


def _coerce_config_value(key: str, value):
    if isinstance(value, list) and (
        key.endswith("_range") or key in {"cloud_f_pre_rails"}
    ):
        return tuple(value)
    return value


def _load_config(meta_payload: dict) -> AblationExperimentConfig:
    config = AblationExperimentConfig()
    for key, value in (meta_payload.get("config") or {}).items():
        if hasattr(config, key):
            setattr(config, key, _coerce_config_value(key, value))
    return config


def _service_components(record: dict, server) -> dict:
    arrival_rate = float(record.get("arrival_rate", 0.0))
    input_tokens = float(record.get("input_tokens", 0.0))
    output_tokens = float(record.get("output_tokens", 0.0))
    max_gpu_units = int(record.get("max_gpu_units", 0))

    effective_input_tokens = 0.7 * 1000.0 + 0.3 * input_tokens
    effective_output_tokens = 0.7 * 200.0 + 0.3 * output_tokens
    prefill_speed = float(getattr(server, "prefill_speed_tokens_per_sec", 0.0))
    decode_speed = float(getattr(server, "decode_speed_tokens_per_sec", 0.0))
    base_processing_ms = (
        effective_input_tokens / max(prefill_speed, 1e-9)
        + effective_output_tokens / max(decode_speed, 1e-9)
    ) * 1000.0
    single_gpu_service_rate = 1000.0 / max(base_processing_ms, 1e-9)

    max_batch_size = int(max(1, getattr(server, "max_batch_size", 1)))
    batch_window_s = 0.05
    optimal_batch_size = min(
        max_batch_size,
        max(1, int(math.ceil(arrival_rate * batch_window_s))),
    )
    batch_multiplier = 1.0 + (optimal_batch_size - 1) * 0.8

    gpu_service_rates = []
    for gpu_units in range(1, max_gpu_units + 1):
        parallel_efficiency = 0.95 ** (gpu_units - 1) if gpu_units > 1 else 1.0
        total_service_rate = (
            single_gpu_service_rate
            * batch_multiplier
            * float(gpu_units)
            * parallel_efficiency
        )
        rho = (
            arrival_rate / total_service_rate
            if total_service_rate > 0
            else float("inf")
        )
        gpu_service_rates.append(
            {
                "gpu_units": int(gpu_units),
                "parallel_efficiency": float(parallel_efficiency),
                "total_service_rate_req_s": float(total_service_rate),
                "rho": float(rho),
                "stable": bool(arrival_rate < total_service_rate),
            }
        )

    best = gpu_service_rates[-1] if gpu_service_rates else {
        "total_service_rate_req_s": 0.0,
        "rho": float("inf"),
        "stable": False,
    }
    return {
        "server_id": str(record.get("server_id", "")),
        "flow_id": str(record.get("flow_id", "")),
        "resource_hint": str(record.get("resource_hint", "")),
        "config_count": int(record.get("config_count", 0)),
        "arrival_rate_req_s": float(arrival_rate),
        "input_tokens": float(input_tokens),
        "output_tokens": float(output_tokens),
        "prefill_speed_tokens_per_sec": float(prefill_speed),
        "decode_speed_tokens_per_sec": float(decode_speed),
        "max_batch_size": int(max_batch_size),
        "max_gpu_units": int(max_gpu_units),
        "effective_input_tokens": float(effective_input_tokens),
        "effective_output_tokens": float(effective_output_tokens),
        "base_processing_ms": float(base_processing_ms),
        "single_gpu_service_rate_req_s": float(single_gpu_service_rate),
        "optimal_batch_size": int(optimal_batch_size),
        "batch_multiplier": float(batch_multiplier),
        "total_service_rate_at_max_gpu_req_s": float(
            best["total_service_rate_req_s"]
        ),
        "rho_at_max_gpu": float(best["rho"]),
        "stable_at_max_gpu": bool(best["stable"]),
        "gpu_service_rates": gpu_service_rates,
    }


def _summarize(rows: list[dict]) -> dict:
    empty = [row for row in rows if int(row["config_count"]) == 0]
    success = [row for row in rows if int(row["config_count"]) > 0]
    stable_empty = [row for row in empty if row["stable_at_max_gpu"]]
    overloaded_empty = [row for row in empty if not row["stable_at_max_gpu"]]
    rho_values = [float(row["rho_at_max_gpu"]) for row in rows]
    empty_rho_values = [float(row["rho_at_max_gpu"]) for row in empty]
    return {
        "record_count": len(rows),
        "empty_config_count": len(empty),
        "success_config_count": len(success),
        "empty_stable_at_max_gpu_count": len(stable_empty),
        "empty_overloaded_at_max_gpu_count": len(overloaded_empty),
        "rho_at_max_gpu_min": min(rho_values) if rho_values else None,
        "rho_at_max_gpu_mean": mean(rho_values) if rho_values else None,
        "rho_at_max_gpu_max": max(rho_values) if rho_values else None,
        "empty_rho_at_max_gpu_min": min(empty_rho_values) if empty_rho_values else None,
        "empty_rho_at_max_gpu_mean": (
            mean(empty_rho_values) if empty_rho_values else None
        ),
        "empty_rho_at_max_gpu_max": max(empty_rho_values) if empty_rho_values else None,
        "successful_rows": success,
        "lowest_rho_empty_rows": sorted(
            empty, key=lambda row: float(row["rho_at_max_gpu"])
        )[:8],
        "highest_rho_empty_rows": sorted(
            empty, key=lambda row: float(row["rho_at_max_gpu"]), reverse=True
        )[:8],
    }


def _markdown(summary: dict, trace_path: Path, meta_path: Path) -> str:
    lines = [
        "# P5 Local Service-Rate Component Audit",
        "",
        f"- Trace: `{trace_path.as_posix()}`",
        f"- Meta: `{meta_path.as_posix()}`",
        f"- Records: {summary['record_count']}",
        f"- Empty local enumerations: {summary['empty_config_count']}",
        f"- Successful local enumerations: {summary['success_config_count']}",
        (
            "- Empty rows stable at max GPU: "
            f"{summary['empty_stable_at_max_gpu_count']}"
        ),
        (
            "- Empty rows overloaded at max GPU: "
            f"{summary['empty_overloaded_at_max_gpu_count']}"
        ),
        (
            "- Empty rho range at max GPU: "
            f"{summary['empty_rho_at_max_gpu_min']:.3f} - "
            f"{summary['empty_rho_at_max_gpu_max']:.3f}"
            if summary["empty_rho_at_max_gpu_min"] is not None
            else "- Empty rho range at max GPU: n/a"
        ),
        "",
        "## Successful Rows",
        "",
        "| server | flow | hint | arrival | gpu | base ms | service req/s | rho | configs |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["successful_rows"]:
        lines.append(_row_markdown(row))

    lines.extend([
        "",
        "## Lowest-Rho Empty Rows",
        "",
        "| server | flow | hint | arrival | gpu | base ms | service req/s | rho | configs |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in summary["lowest_rho_empty_rows"]:
        lines.append(_row_markdown(row))

    lines.extend([
        "",
        "## Highest-Rho Empty Rows",
        "",
        "| server | flow | hint | arrival | gpu | base ms | service req/s | rho | configs |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in summary["highest_rho_empty_rows"]:
        lines.append(_row_markdown(row))
    lines.append("")
    return "\n".join(lines)


def _row_markdown(row: dict) -> str:
    hint = row["resource_hint"] or "-"
    return (
        f"| {row['server_id']} | {row['flow_id']} | {hint} | "
        f"{row['arrival_rate_req_s']:.3f} | {row['max_gpu_units']} | "
        f"{row['base_processing_ms']:.2f} | "
        f"{row['total_service_rate_at_max_gpu_req_s']:.3f} | "
        f"{row['rho_at_max_gpu']:.3f} | {row['config_count']} |"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--meta", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--markdown", required=True, type=Path)
    args = parser.parse_args()

    trace_path = args.trace.resolve()
    meta_path = args.meta.resolve()
    trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
    meta_payload = json.loads(meta_path.read_text(encoding="utf-8"))
    config = _load_config(meta_payload)
    system_state = create_ablation_system(args.seed, config)

    rows = []
    for record in trace_payload.get("records", []):
        server = system_state.edge_servers[str(record.get("server_id", ""))]
        rows.append(_service_components(record, server))

    summary = _summarize(rows)
    payload = {
        "run_id": trace_payload.get("run_id", ""),
        "config_hash": trace_payload.get("config_hash", ""),
        "seed": int(args.seed),
        "trace_path": str(trace_path),
        "meta_path": str(meta_path),
        "summary": summary,
        "records": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    args.markdown.write_text(
        _markdown(summary, trace_path, meta_path),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
