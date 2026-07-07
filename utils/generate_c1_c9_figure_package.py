"""Generate PNG-only C1-C9 sensitivity figure packages from verified sweep CSVs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


ALGORITHMS: Sequence[str] = (
    "LyHAM-CO",
    "GMDA-RMPR-Myopic",
    "PDRS-Myopic",
    "FFD-Myopic",
)

COLORS: Dict[str, str] = {
    "LyHAM-CO": "#4C72B0",
    "GMDA-RMPR-Myopic": "#DD8452",
    "FFD-Myopic": "#55A868",
    "PDRS-Myopic": "#C44E52",
}

STYLES: Dict[str, Tuple[str, str]] = {
    "LyHAM-CO": ("-", "o"),
    "GMDA-RMPR-Myopic": ("--", "x"),
    "FFD-Myopic": (":", "s"),
    "PDRS-Myopic": ("-.", "P"),
}

MetricSpec = Tuple[str, str, str]

GROUP_SPECS: Dict[str, Dict[str, object]] = {
    "chain_length": {
        "x_values": ["[2,4]", "[3,5]", "[4,6]", "[5,7]", "[6,8]"],
        "x_label": "Length of Service Chains",
        "metrics": (
            ("Average Response Delay", "delay_chain_unit", "Average Response Delay (ms)"),
            ("Total Energy Consumption", "energy_chain_unit", "Total Energy Consumption (J)"),
            ("Service-Chain Operational Cost", "cost_chain_unit", "Service-Chain Operational Cost"),
        ),
    },
    "arrival_rate": {
        "x_values": ["5", "6", "7", "8", "9", "10", "11", "12"],
        "x_label": "Average Arrival Rate of Requests",
        "metrics": (
            ("Average Response Delay", "arrival_ratedelay_unit", "Average Response Delay (ms)"),
            ("Total Energy Consumption", "arrival_rateenergy_unit", "Total Energy Consumption (J)"),
            ("Service-Chain Operational Cost", "arrival_ratecost_unit", "Service-Chain Operational Cost"),
        ),
    },
    "edge_nodes": {
        "x_values": ["20", "25", "30", "35", "40", "45"],
        "x_label": "Number of Edge Nodes",
        "metrics": (
            ("Average Response Delay", "edgedelay_unit", "Average Response Delay (ms)"),
            ("Total Energy Consumption", "edgeenergy_unit", "Total Energy Consumption (J)"),
            ("Service-Chain Operational Cost", "edgecost_unit", "Service-Chain Operational Cost"),
        ),
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _float_value(value: object) -> float:
    try:
        parsed = float(value)
    except Exception as exc:
        raise ValueError(f"expected numeric value, got {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"expected finite numeric value, got {value!r}")
    return parsed


def _record(path: Path, kind: str, root: Path) -> Dict[str, object]:
    return {
        "kind": kind,
        "name": path.name,
        "path": str(path),
        "relative_path": str(path.relative_to(root)),
        "length": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _copy_into_data_dir(path: Path, data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / path.name
    if path.resolve() != target.resolve():
        shutil.copy2(path, target)
    return target


def _aggregate_rows(rows: Iterable[Dict[str, str]]) -> Dict[Tuple[str, str, str, str], float]:
    values: Dict[Tuple[str, str, str, str], List[float]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("figure_group", ""),
            row.get("metric", ""),
            row.get("x_value", ""),
            row.get("algorithm", ""),
        )
        values[key].append(_float_value(row.get("value", "")))
    return {key: sum(items) / len(items) for key, items in values.items()}


def _validate_complete_grid(rows: List[Dict[str, str]]) -> Dict[Tuple[str, str, str, str], float]:
    values = _aggregate_rows(rows)
    missing: List[str] = []
    for group, group_spec in GROUP_SPECS.items():
        x_values = group_spec["x_values"]
        metrics = group_spec["metrics"]
        assert isinstance(x_values, list)
        assert isinstance(metrics, tuple)
        for metric, _, _ in metrics:
            for x_value in x_values:
                for algorithm in ALGORITHMS:
                    if (group, metric, x_value, algorithm) not in values:
                        missing.append(f"{group}/{metric}/{x_value}/{algorithm}")
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(f"C1-C9 figure CSV is missing required cells: {preview}")
    return values


def _plot_metric(
    values: Dict[Tuple[str, str, str, str], float],
    group: str,
    metric: str,
    stem: str,
    xlabel: str,
    ylabel: str,
    x_values: Sequence[str],
    out_dir: Path,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_facecolor("white")
    ax.grid(True, linestyle="-", linewidth=1.2, color="#E5E5E5", zorder=0)
    x_positions = list(range(len(x_values)))

    for algorithm in ALGORITHMS:
        linestyle, marker = STYLES[algorithm]
        ax.plot(
            x_positions,
            [values[(group, metric, x_value, algorithm)] for x_value in x_values],
            color=COLORS[algorithm],
            linestyle=linestyle,
            linewidth=4,
            marker=marker,
            markersize=12,
            markeredgewidth=0,
            label=algorithm,
            zorder=3,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(list(x_values))
    ax.set_xlabel(xlabel, fontsize=24, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=24, labelpad=10)
    ax.tick_params(axis="both", which="major", labelsize=24, width=1.5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.legend(loc="upper left", fontsize=20, frameon=True, fancybox=True)
    fig.tight_layout()

    path = out_dir / f"{stem}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _advantage_summary(values: Dict[Tuple[str, str, str, str], float]) -> Dict[str, object]:
    records: List[Dict[str, object]] = []
    mean_values: List[float] = []
    point_values: List[float] = []
    for group, group_spec in GROUP_SPECS.items():
        x_values = group_spec["x_values"]
        metrics = group_spec["metrics"]
        assert isinstance(x_values, list)
        assert isinstance(metrics, tuple)
        for metric, _, _ in metrics:
            for baseline in [item for item in ALGORITHMS if item != "LyHAM-CO"]:
                improvements = []
                for x_value in x_values:
                    lyham = values[(group, metric, x_value, "LyHAM-CO")]
                    base = values[(group, metric, x_value, baseline)]
                    improvement = ((base - lyham) / base) * 100.0 if base else 0.0
                    improvements.append(improvement)
                    point_values.append(improvement)
                mean_improvement = sum(improvements) / len(improvements)
                mean_values.append(mean_improvement)
                records.append(
                    {
                        "group": group,
                        "metric": metric,
                        "baseline": baseline,
                        "mean_percent": mean_improvement,
                        "min_point_percent": min(improvements),
                        "max_point_percent": max(improvements),
                    }
                )
    return {
        "mean_min_percent": min(mean_values) if mean_values else None,
        "mean_max_percent": max(mean_values) if mean_values else None,
        "point_min_percent": min(point_values) if point_values else None,
        "point_max_percent": max(point_values) if point_values else None,
        "records": records,
    }


def _group_summary(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    summaries: List[Dict[str, object]] = []
    for group, group_spec in GROUP_SPECS.items():
        group_rows = [row for row in rows if row.get("figure_group") == group]
        x_values = group_spec["x_values"]
        metrics = group_spec["metrics"]
        assert isinstance(x_values, list)
        assert isinstance(metrics, tuple)
        summaries.append(
            {
                "group": group,
                "row_count": len(group_rows),
                "x_values": x_values,
                "algorithms": list(ALGORITHMS),
                "metrics": [metric for metric, _, _ in metrics],
            }
        )
    return summaries


def generate_c1_c9_package(
    figure_csv: Path | str,
    validation_manifest: Path | str,
    output_dir: Path | str,
    run_id: str,
    decision: str,
) -> Path:
    figure_csv = Path(figure_csv)
    validation_manifest = Path(validation_manifest)
    output_dir = Path(output_dir)
    figure_dir = output_dir / "figures"
    data_dir = output_dir / "figure_data"
    figure_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_csv(figure_csv)
    if not rows:
        raise ValueError(f"figure CSV has no rows: {figure_csv}")
    values = _validate_complete_grid(rows)
    validation = json.loads(validation_manifest.read_text(encoding="utf-8-sig"))

    figure_paths: List[Path] = []
    for group, group_spec in GROUP_SPECS.items():
        x_values = group_spec["x_values"]
        metrics = group_spec["metrics"]
        x_label = group_spec["x_label"]
        assert isinstance(x_values, list)
        assert isinstance(metrics, tuple)
        assert isinstance(x_label, str)
        for metric, stem, ylabel in metrics:
            figure_paths.append(
                _plot_metric(
                    values=values,
                    group=group,
                    metric=metric,
                    stem=stem,
                    xlabel=x_label,
                    ylabel=ylabel,
                    x_values=x_values,
                    out_dir=figure_dir,
                )
            )

    copied_figure_csv = _copy_into_data_dir(figure_csv, data_dir)
    copied_validation_manifest = _copy_into_data_dir(validation_manifest, data_dir)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "package": str(output_dir),
        "figure_format": "png-only",
        "figure_count": len(figure_paths),
        "row_count": len(rows),
        "source_figure_csv": str(figure_csv),
        "source_figure_csv_sha256": sha256_file(figure_csv),
        "source_validation_manifest": str(validation_manifest),
        "source_validation_manifest_sha256": sha256_file(validation_manifest),
        "copied_figure_csv": str(copied_figure_csv),
        "copied_figure_csv_sha256": sha256_file(copied_figure_csv),
        "copied_validation_manifest": str(copied_validation_manifest),
        "copied_validation_manifest_sha256": sha256_file(copied_validation_manifest),
        "decision": decision,
        "raw_csv_count": int(validation.get("raw_csv_count", 0)),
        "raw_row_count": int(validation.get("raw_row_count", 0)),
        "failed_raw_rows": int(validation.get("failed_raw_rows", 0)),
        "bad_key_numeric_cells": int(validation.get("bad_key_numeric_cells", 0)),
        "source_manifest_figure_row_count": validation.get("figure_row_count"),
        "source_manifest_figure_csv_sha256": validation.get("figure_csv_sha256"),
        "template_match_notes": (
            "C1-C9 PNGs keep the manuscript/unit-calibrated stems, 10x7 inch canvas, 300 dpi, "
            "thick line-marker style, white background, upper-left legend, and grid style. "
            "Static hard-coded arrays are replaced by current unified-pipeline formal sweep rows."
        ),
        "groups": _group_summary(rows),
        "advantage_summary": _advantage_summary(values),
        "figures": [_record(path, "figure", output_dir) for path in sorted(figure_paths)],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate PNG-only C1-C9 sensitivity figures.")
    parser.add_argument("--figure-csv", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--decision", required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    manifest_path = generate_c1_c9_package(
        figure_csv=args.figure_csv,
        validation_manifest=args.validation_manifest,
        output_dir=args.output_dir,
        run_id=args.run_id,
        decision=args.decision,
    )
    print(json.dumps({"manifest": str(manifest_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
