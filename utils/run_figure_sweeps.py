"""
论文绘图参数扫描入口
文件负责把现有绘图横坐标逐点送入统一实验 pipeline，并导出绘图长表。
工程边界：本文件只编排运行和汇总，不手工改数值，不覆盖 canonical 主实验表。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, List, Tuple

from ablation_config import AblationExperimentConfig, NORMAL_MAIN_ALGORITHMS, get_project_root
from ablation_figure_experiments import build_figure_experiment_configs
from figure_data_export import FIGURE_DATA_FIELDS
from run_ablation import run_ablation_experiment


PARAMETER_SWEEP_METRICS: Tuple[Tuple[str, str, str], ...] = (
    ("Average Response Delay", "delay_mean", "ms"),
    ("Total Energy Consumption", "energy_mean", "J"),
    ("Service-Chain Operational Cost", "cost_mean", "cost_unit"),
)

V_SWEEP_METRICS: Tuple[Tuple[str, str, str], ...] = (
    ("Average Virtual Energy Queue", "avg_y_mean", "queue_length"),
    ("Average Virtual Delay Queue", "avg_z_mean", "queue_length"),
    ("Total Energy Consumption", "energy_mean", "J"),
    ("Service-Chain Operational Cost", "cost_mean", "cost_unit"),
    ("Average Response Delay", "delay_mean", "ms"),
)

SWEEP_X_NAMES = {
    "chain_length": "Length of Service Chains",
    "arrival_rate": "Average Arrival Rate of Requests",
    "edge_nodes": "Number of Edge Nodes",
    "V": "Control Parameter V",
}


def _clean_number_text(value: object) -> str:
    """输出绘图横坐标文本，避免整数被写成 5.0。"""
    try:
        numeric = float(value)
        if numeric.is_integer():
            return str(int(numeric))
    except Exception:
        pass
    return str(value)


def _x_value_for_config(config: AblationExperimentConfig) -> str:
    """按绘图代码横坐标格式生成当前扫描点标签。"""
    sweep_name = str(getattr(config, "figure_sweep_name", ""))
    if sweep_name == "chain_length":
        low, high = getattr(config, "chain_length_range", (0, 0))
        return f"[{int(low)},{int(high)}]"
    return _clean_number_text(getattr(config, "figure_sweep_value", ""))


def _safe_token(value: str) -> str:
    """把扫描值转成可用于 run_id 的短 token。"""
    chars = []
    for char in str(value):
        if char.isalnum() or char in {"-", "_"}:
            chars.append(char)
        else:
            chars.append("_")
    token = "".join(chars).strip("_")
    return token or "value"


def _read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _write_figure_rows(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIGURE_DATA_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIGURE_DATA_FIELDS})


def _summary_path(base_dir: Path, run_id: str) -> Path:
    return base_dir / "summary" / f"ablation_summary_{run_id}.csv"


def _is_aggregate_row(row: Dict[str, str]) -> bool:
    try:
        return int(float(row.get("seed", ""))) == -1
    except Exception:
        return False


def _is_valid_row(row: Dict[str, str]) -> bool:
    return str(row.get("valid", "")).strip().lower() in {"true", "1", "yes"}


def _metric_value(row: Dict[str, str], column: str) -> float:
    return float(row.get(column, "nan"))


def summary_rows_to_figure_rows(summary_rows: Iterable[Dict[str, str]],
                                config: AblationExperimentConfig,
                                run_id: str,
                                time_slots: int,
                                source: str = "figure_sweep_pipeline_run") -> List[Dict[str, object]]:
    """把单个扫描点的 summary 聚合行转为绘图长表行。"""
    sweep_name = str(getattr(config, "figure_sweep_name", ""))
    if sweep_name not in SWEEP_X_NAMES:
        raise ValueError(f"unsupported figure sweep: {sweep_name}")

    x_value = _x_value_for_config(config)
    metrics = V_SWEEP_METRICS if sweep_name == "V" else PARAMETER_SWEEP_METRICS
    rows = [
        row for row in summary_rows
        if _is_aggregate_row(row) and _is_valid_row(row)
    ]
    if sweep_name == "V":
        rows = [row for row in rows if row.get("algorithm") == "LyHAM-CO"]
    else:
        rows = [row for row in rows if row.get("algorithm") in NORMAL_MAIN_ALGORITHMS]

    figure_rows: List[Dict[str, object]] = []
    for row in rows:
        for metric, column, unit in metrics:
            figure_rows.append({
                "figure_group": sweep_name,
                "metric": metric,
                "x_name": SWEEP_X_NAMES[sweep_name],
                "x_value": x_value,
                "algorithm": row.get("algorithm", ""),
                "value": _metric_value(row, column),
                "unit": unit,
                "source": source,
                "run_id": run_id,
                "seed": "mean",
                "time_slots": int(time_slots),
                "notes": (
                    "Formal sweep point from unified pipeline; "
                    "summary seed=-1 aggregate; canonical summary/table export is blocked for figure sweeps."
                ),
            })
    return figure_rows


def _existing_completed_keys(rows: Iterable[Dict[str, str]]) -> set[Tuple[str, str]]:
    return {
        (str(row.get("figure_group", "")), str(row.get("x_value", "")))
        for row in rows
        if str(row.get("source", "")) == "figure_sweep_pipeline_run"
    }


def _read_batch_meta(meta_path: Path) -> Dict[str, object]:
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _config_resume_fingerprint(config: AblationExperimentConfig, config_index: int, x_value: str) -> str:
    """构造绘图断点续跑指纹，避免不同配置复用旧横坐标输出。"""
    payload = {
        "index": int(config_index),
        "sweep": str(getattr(config, "figure_sweep_name", "")),
        "x_value": str(x_value),
        "config": config.to_dict(),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _existing_completed_fingerprints(meta_payload: Dict[str, object]) -> set[Tuple[str, str, str]]:
    if not isinstance(meta_payload, dict):
        return set()
    keys = set()
    resume_items = list(meta_payload.get("completed", []))
    for item in meta_payload.get("skipped", []):
        if isinstance(item, dict) and item.get("reason") == "completed_config_fingerprint_exists":
            resume_items.append(item)
    for item in resume_items:
        if not isinstance(item, dict):
            continue
        fingerprint = str(item.get("config_fingerprint", ""))
        if not fingerprint:
            continue
        keys.add((str(item.get("sweep", "")), str(item.get("x_value", "")), fingerprint))
    return keys


def _default_out_path(batch_id: str) -> Path:
    return get_project_root() / "绘图代码_单位校准" / "data" / f"figure_sweep_metrics_{batch_id}.csv"


def _write_batch_meta(meta_path: Path, payload: Dict[str, object]) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_figure_sweep_batch(args) -> Dict[str, object]:
    """运行一批绘图扫描点，并在每个点结束后增量写出 CSV 和 meta。"""
    batch_id = args.batch_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_start_clock = perf_counter()
    out_path = Path(args.out) if args.out else _default_out_path(batch_id)
    meta_path = Path(args.meta) if args.meta else out_path.with_suffix(".meta.json")
    rows: List[Dict[str, object]] = [] if args.no_resume else list(_read_rows(out_path))
    previous_meta = {} if args.no_resume else _read_batch_meta(meta_path)
    completed_output_keys = set() if args.no_resume else _existing_completed_keys(rows)
    completed_fingerprint_keys = set() if args.no_resume else _existing_completed_fingerprints(previous_meta)

    configs = build_figure_experiment_configs()
    if args.sweep_name:
        configs = [config for config in configs if config.figure_sweep_name == args.sweep_name]
    indexed_configs = list(enumerate(configs))
    if args.start_index:
        indexed_configs = indexed_configs[int(args.start_index):]
    if args.max_configs is not None:
        indexed_configs = indexed_configs[:int(args.max_configs)]

    manifest: Dict[str, object] = {
        "batch_id": batch_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "resume_enabled": not bool(args.no_resume),
        "output_csv": str(out_path),
        "meta_path": str(meta_path),
        "requested_count": len(indexed_configs),
        "completed_count": 0,
        "skipped_count": 0,
        "failed_count": 0,
        "elapsed_seconds": 0.0,
        "average_completed_config_seconds": 0.0,
        "completed": [],
        "skipped": [],
        "failed": [],
    }

    def write_manifest() -> None:
        manifest["completed_count"] = len(manifest["completed"])
        manifest["skipped_count"] = len(manifest["skipped"])
        manifest["failed_count"] = len(manifest["failed"])
        manifest["elapsed_seconds"] = round(perf_counter() - batch_start_clock, 3)
        elapsed_values = [
            float(item.get("elapsed_seconds", 0.0))
            for item in manifest["completed"]
            if item.get("elapsed_seconds", "") != ""
        ]
        manifest["average_completed_config_seconds"] = (
            round(sum(elapsed_values) / len(elapsed_values), 3)
            if elapsed_values else 0.0
        )
        _write_batch_meta(meta_path, manifest)

    write_manifest()
    print(json.dumps({"event": "batch_start", "batch_id": batch_id, "configs": len(indexed_configs), "out": str(out_path)}, ensure_ascii=False), flush=True)

    for local_index, (config_index, config) in enumerate(indexed_configs, start=1):
        if args.seeds:
            config.seeds = list(args.seeds)
        if args.time_slots is not None:
            config.time_slots = int(args.time_slots)
        if args.slow_epoch_slots is not None:
            config.slow_epoch_slots = int(args.slow_epoch_slots)
        if args.output_dir:
            config.output_dir = str(args.output_dir)
        if getattr(args, "algorithms", None):
            config.algorithms = list(args.algorithms)

        x_value = _x_value_for_config(config)
        sweep_key = (str(config.figure_sweep_name), x_value)
        config_fingerprint = _config_resume_fingerprint(config, config_index, x_value)
        fingerprint_key = (sweep_key[0], sweep_key[1], config_fingerprint)
        if sweep_key in completed_output_keys and fingerprint_key in completed_fingerprint_keys:
            manifest["skipped"].append({
                "index": config_index,
                "sweep": sweep_key[0],
                "x_value": sweep_key[1],
                "config_fingerprint": config_fingerprint,
                "progress_percent": round(local_index * 100.0 / max(len(indexed_configs), 1), 2),
                "reason": "completed_config_fingerprint_exists",
            })
            write_manifest()
            print(json.dumps({"event": "skip_completed", "index": config_index, "sweep": sweep_key[0], "x_value": sweep_key[1]}, ensure_ascii=False), flush=True)
            continue

        run_id = f"figure_{batch_id}_{config_index:02d}_{config.figure_sweep_name}_{_safe_token(x_value)}"
        config_start_clock = perf_counter()
        config_started_at = datetime.now().isoformat(timespec="seconds")
        start_progress = round((local_index - 1) * 100.0 / max(len(indexed_configs), 1), 2)
        print(json.dumps({
            "event": "config_start",
            "index": config_index,
            "progress_percent": start_progress,
            "sweep": config.figure_sweep_name,
            "x_value": x_value,
            "run_id": run_id,
            "seeds": list(config.seeds),
            "time_slots": int(config.time_slots),
        }, ensure_ascii=False), flush=True)

        try:
            result = run_ablation_experiment(config=config, run_id=run_id, silent=not args.verbose)
            base_dir = Path(result["base_dir"])
            summary_rows = _read_rows(_summary_path(base_dir, run_id))
            new_rows = summary_rows_to_figure_rows(summary_rows, config, run_id, int(config.time_slots))
            rows = [
                row for row in rows
                if (str(row.get("figure_group", "")), str(row.get("x_value", ""))) != sweep_key
            ]
            rows.extend(new_rows)
            _write_figure_rows(out_path, rows)
            completed_output_keys.add(sweep_key)
            completed_fingerprint_keys.add(fingerprint_key)
            finish_progress = round(local_index * 100.0 / max(len(indexed_configs), 1), 2)
            manifest["completed"].append({
                "index": config_index,
                "sweep": sweep_key[0],
                "x_value": sweep_key[1],
                "run_id": run_id,
                "config_fingerprint": config_fingerprint,
                "row_count": len(new_rows),
                "base_dir": str(base_dir),
                "summary_path": str(_summary_path(base_dir, run_id)),
                "meta_path": str(result.get("meta_path", "")),
                "started_at": config_started_at,
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "elapsed_seconds": round(perf_counter() - config_start_clock, 3),
                "progress_percent": finish_progress,
            })
            write_manifest()
            print(json.dumps({
                "event": "config_complete",
                "index": config_index,
                "progress_percent": finish_progress,
                "sweep": sweep_key[0],
                "x_value": sweep_key[1],
                "run_id": run_id,
                "exported_rows": len(new_rows),
                "out": str(out_path),
            }, ensure_ascii=False), flush=True)
        except Exception as exc:
            manifest["failed"].append({
                "index": config_index,
                "sweep": sweep_key[0],
                "x_value": sweep_key[1],
                "run_id": run_id,
                "config_fingerprint": config_fingerprint,
                "error": repr(exc),
                "started_at": config_started_at,
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "elapsed_seconds": round(perf_counter() - config_start_clock, 3),
                "progress_percent": start_progress,
            })
            write_manifest()
            raise

    write_manifest()
    print(json.dumps({"event": "batch_complete", "batch_id": batch_id, "out": str(out_path), "meta": str(meta_path)}, ensure_ascii=False), flush=True)
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(description="Run unified-pipeline data generation for 18 paper figures.")
    parser.add_argument("--batch-id", default=None)
    parser.add_argument("--sweep-name", choices=["chain_length", "arrival_rate", "edge_nodes", "V"], default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-configs", type=int, default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--time-slots", type=int, default=None)
    parser.add_argument("--slow-epoch-slots", type=int, default=None)
    parser.add_argument("--algorithms", nargs="+", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--meta", type=Path, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    run_figure_sweep_batch(parse_args())


if __name__ == "__main__":
    main()
