"""
论文绘图数据导出工具
从统一复现实验raw CSV导出绘图代码需要的长表数据，不手工生成数值。
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


FIGURE_DATA_FIELDS = [
    "figure_group",
    "metric",
    "x_name",
    "x_value",
    "algorithm",
    "value",
    "unit",
    "source",
    "run_id",
    "seed",
    "time_slots",
    "notes",
]

TIME_SERIES_METRICS = {
    "Average Response Delay": ("delay_ms", "ms"),
    "Total Energy Consumption": ("energy_j", "J"),
    "Service-Chain Operational Cost": ("cost", "cost_unit"),
}

QUEUE_METRICS = {
    "Average Virtual Energy Queue": ("avg_y", "queue_length"),
    "Average Virtual Delay Queue": ("avg_z", "queue_length"),
}


def _float_value(row: Dict[str, str], column: str) -> float:
    try:
        return float(row.get(column, ""))
    except Exception:
        return 0.0


def _int_value(row: Dict[str, str], column: str) -> int:
    try:
        return int(float(row.get(column, "")))
    except Exception:
        return 0


def _read_raw_rows(raw_dir: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in sorted(raw_dir.glob("*/*_per_slot.csv")):
        with path.open(newline="", encoding="utf-8-sig") as f:
            rows.extend(csv.DictReader(f))
    return [row for row in rows if str(row.get("status", "")).lower() in {"ok", "true", ""}]


def _time_frame_value(slot: int, max_slot: int, time_frame_max: int) -> int:
    if max_slot <= 0:
        return 0
    return int(round(float(slot) * float(time_frame_max) / float(max_slot)))


def _mean_by_key(rows: Iterable[Dict[str, str]], value_column: str,
                 key_columns: Tuple[str, ...]) -> Dict[Tuple[str, ...], float]:
    values: Dict[Tuple[str, ...], List[float]] = defaultdict(list)
    for row in rows:
        key = tuple(str(row.get(column, "")) for column in key_columns)
        values[key].append(_float_value(row, value_column))
    return {
        key: sum(items) / max(len(items), 1)
        for key, items in values.items()
    }


def _ratio_bucket(row: Dict[str, str]) -> str:
    cloud = max(_int_value(row, "cloud_pair_count"), 0)
    local = max(_int_value(row, "local_pair_count"), 0)
    total = max(cloud + local, 1)
    cloud_tenth = int(round(float(cloud) * 10.0 / float(total)))
    cloud_tenth = max(1, min(9, cloud_tenth))
    return f"{cloud_tenth}:{10 - cloud_tenth}"


def _write_rows(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIGURE_DATA_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def export_formal_run_figure_csv(base_dir: Path | str, run_id: str, out_path: Path | str,
                                 time_frame_max: int = 10000) -> Dict[str, object]:
    """导出formal主实验可直接支撑的绘图长表。

    输入是 `run_ablation.py` 生成的 raw/run_id 目录；输出覆盖平均时序图、
    虚拟队列图和卸载比率图。卸载比率按当前绘图代码的Cloud:Local十分位标签聚合。
    """
    base_dir = Path(base_dir)
    out_path = Path(out_path)
    raw_dir = base_dir / "raw" / run_id
    if not raw_dir.exists():
        raise FileNotFoundError(f"raw run directory not found: {raw_dir}")

    raw_rows = _read_raw_rows(raw_dir)
    if not raw_rows:
        raise ValueError(f"no raw rows found in: {raw_dir}")

    max_slot = max(_int_value(row, "slot") for row in raw_rows)
    time_slots = max_slot + 1
    rows: List[Dict[str, object]] = []

    for metric, (column, unit) in TIME_SERIES_METRICS.items():
        grouped = _mean_by_key(raw_rows, column, ("algorithm", "slot"))
        for (algorithm, slot_text), value in sorted(grouped.items(), key=lambda item: (item[0][0], int(item[0][1]))):
            slot = int(slot_text)
            rows.append({
                "figure_group": "time_series",
                "metric": metric,
                "x_name": "Time Frames",
                "x_value": _time_frame_value(slot, max_slot, time_frame_max),
                "algorithm": algorithm,
                "value": value,
                "unit": unit,
                "source": "formal_pipeline_run",
                "run_id": run_id,
                "seed": "mean",
                "time_slots": time_slots,
                "notes": "Averaged across formal seeds from unified raw per-slot CSV.",
            })

    for metric, (column, unit) in QUEUE_METRICS.items():
        grouped = _mean_by_key(raw_rows, column, ("algorithm", "slot"))
        for (algorithm, slot_text), value in sorted(grouped.items(), key=lambda item: (item[0][0], int(item[0][1]))):
            slot = int(slot_text)
            rows.append({
                "figure_group": "virtual_queue",
                "metric": metric,
                "x_name": "Time Frames",
                "x_value": _time_frame_value(slot, max_slot, time_frame_max),
                "algorithm": algorithm,
                "value": value,
                "unit": unit,
                "source": "formal_pipeline_run",
                "run_id": run_id,
                "seed": "mean",
                "time_slots": time_slots,
                "notes": "Averaged across formal seeds from unified raw per-slot CSV.",
            })

    lyham_rows = [row for row in raw_rows if row.get("algorithm") == "LyHAM-CO"]
    for row in lyham_rows:
        row["_ratio_bucket"] = _ratio_bucket(row)
    for metric, (column, unit) in TIME_SERIES_METRICS.items():
        grouped = _mean_by_key(lyham_rows, column, ("_ratio_bucket",))
        for (ratio,), value in sorted(grouped.items(), key=lambda item: int(item[0][0].split(":")[0])):
            rows.append({
                "figure_group": "offloading_ratio",
                "metric": metric,
                "x_name": "Offloading Ratio (Cloud:Local)",
                "x_value": ratio,
                "algorithm": "LyHAM-CO",
                "value": value,
                "unit": unit,
                "source": "formal_pipeline_run",
                "run_id": run_id,
                "seed": "mean",
                "time_slots": time_slots,
                "notes": "LyHAM-CO formal raw rows binned by cloud/local pair-count tenth to match plotting axis.",
            })

    _write_rows(out_path, rows)
    return {
        "run_id": run_id,
        "row_count": len(rows),
        "out_path": str(out_path),
        "raw_row_count": len(raw_rows),
        "time_slots": time_slots,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export formal raw data for paper figure CSVs.")
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--time-frame-max", type=int, default=10000)
    args = parser.parse_args()
    result = export_formal_run_figure_csv(args.base_dir, args.run_id, args.out, args.time_frame_max)
    print(result)


if __name__ == "__main__":
    main()
