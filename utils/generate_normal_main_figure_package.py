"""Generate PNG-only normal-main figure and table packages from verified CSVs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


MetricSpec = Tuple[str, str, str]

TIME_SERIES_SPECS: Sequence[MetricSpec] = (
    ("Average Response Delay", "average_response_delay_1000_slots", "Average Response Delay (ms)"),
    ("Total Energy Consumption", "total_energy_consumption_1000_slots", "Total Energy Consumption (J)"),
    ("Service-Chain Operational Cost", "service_chain_operational_cost_1000_slots", "Service-Chain Operational Cost"),
)

QUEUE_SPECS: Sequence[MetricSpec] = (
    ("Average Virtual Energy Queue", "average_virtual_energy_queue_1000_slots", "Average Virtual Energy Queue"),
    ("Average Virtual Delay Queue", "average_virtual_delay_queue_1000_slots", "Average Virtual Delay Queue"),
)

OFFLOADING_SPECS: Sequence[MetricSpec] = (
    ("Average Response Delay", "offloading_ratio_average_response_delay", "Average Response Delay (ms)"),
    ("Total Energy Consumption", "offloading_ratio_total_energy_consumption", "Total Energy Consumption (J)"),
    ("Service-Chain Operational Cost", "offloading_ratio_service_chain_operational_cost", "Service-Chain Operational Cost"),
)

SUMMARY_FIELDS: Sequence[str] = (
    "algorithm",
    "delay_mean",
    "energy_mean",
    "cost_mean",
    "avg_y_mean",
    "avg_z_mean",
    "decision_time_mean_ms",
    "decision_time_p95_ms",
    "valid_seed_count",
    "formal_gate_passed",
    "mechanism_gate_passed",
    "claim_supported",
)


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
        return float(value)
    except Exception:
        return 0.0


def _sort_x_key(value: str) -> Tuple[int, float | str]:
    try:
        return (0, float(value))
    except Exception:
        if ":" in value:
            try:
                return (1, float(value.split(":", 1)[0]))
            except Exception:
                pass
        return (2, value)


def _metric_rows(rows: Iterable[Dict[str, str]], figure_group: str, metric: str) -> List[Dict[str, str]]:
    return [
        row
        for row in rows
        if row.get("figure_group") == figure_group and row.get("metric") == metric
    ]


def _plot_line_metric(rows: List[Dict[str, str]], metric: str, stem: str, ylabel: str, out_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_algorithm: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        by_algorithm.setdefault(row.get("algorithm", ""), []).append(row)

    plt.figure(figsize=(6.0, 3.8))
    for algorithm in sorted(by_algorithm):
        series = sorted(by_algorithm[algorithm], key=lambda item: _sort_x_key(str(item.get("x_value", ""))))
        plt.plot(
            [_float_value(row.get("x_value", "")) for row in series],
            [_float_value(row.get("value", "")) for row in series],
            linewidth=1.6 if algorithm == "LyHAM-CO" else 1.2,
            label=algorithm,
        )
    plt.xlabel(rows[0].get("x_name", "Time Frames") if rows else "Time Frames")
    plt.ylabel(ylabel)
    plt.title(metric)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    path = out_dir / f"{stem}.png"
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def _plot_offloading_metric(rows: List[Dict[str, str]], metric: str, stem: str, ylabel: str, out_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = sorted(rows, key=lambda item: _sort_x_key(str(item.get("x_value", ""))))
    x_labels = [str(row.get("x_value", "")) for row in series]
    values = [_float_value(row.get("value", "")) for row in series]
    plt.figure(figsize=(6.0, 3.8))
    plt.bar(x_labels, values, width=0.65, color="#3B6EA8")
    plt.xlabel(rows[0].get("x_name", "Offloading Ratio (Cloud:Local)") if rows else "Offloading Ratio")
    plt.ylabel(ylabel)
    plt.title(metric)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    path = out_dir / f"{stem}.png"
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def _write_summary_tables(summary_csv: Path, table_dir: Path) -> List[Path]:
    table_dir.mkdir(parents=True, exist_ok=True)
    rows = [row for row in _read_csv(summary_csv) if str(row.get("seed", "")) == "-1"]
    rows = sorted(rows, key=lambda item: item.get("algorithm", ""))

    summary_table = table_dir / "normal_main_1000_summary_table.csv"
    with summary_table.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})

    latex_table = table_dir / "normal_main_1000_table.tex"
    compact_table = table_dir / "normal_main_1000_compact_table.tex"
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Algorithm & Delay & Energy & Cost & Avg. $Y$ & Avg. $Z$ & Claim \\",
        r"\midrule",
    ]
    compact_lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Algorithm & Delay & Energy & Cost \\",
        r"\midrule",
    ]
    for row in rows:
        algorithm = row.get("algorithm", "")
        delay = _float_value(row.get("delay_mean", ""))
        energy = _float_value(row.get("energy_mean", ""))
        cost = _float_value(row.get("cost_mean", ""))
        avg_y = _float_value(row.get("avg_y_mean", ""))
        avg_z = _float_value(row.get("avg_z_mean", ""))
        claim = row.get("claim_supported", "")
        lines.append(
            f"{algorithm} & {delay:.3f} & {energy:.3f} & {cost:.3f} & {avg_y:.3f} & {avg_z:.3f} & {claim} \\\\"
        )
        compact_lines.append(f"{algorithm} & {delay:.3f} & {energy:.3f} & {cost:.3f} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    compact_lines.extend([r"\bottomrule", r"\end{tabular}"])
    latex_table.write_text("\n".join(lines) + "\n", encoding="utf-8")
    compact_table.write_text("\n".join(compact_lines) + "\n", encoding="utf-8")
    return [summary_table, latex_table, compact_table]


def _record(path: Path, kind: str, root: Path) -> Dict[str, object]:
    return {
        "kind": kind,
        "name": path.name,
        "path": str(path),
        "relative_path": str(path.relative_to(root)),
        "length": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def generate_normal_main_package(
    figure_csv: Path | str,
    summary_csv: Path | str,
    output_dir: Path | str,
    run_id: str,
    queue_stability_claim: str,
) -> Path:
    figure_csv = Path(figure_csv)
    summary_csv = Path(summary_csv)
    output_dir = Path(output_dir)
    figure_dir = output_dir / "figures"
    table_dir = output_dir / "tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_csv(figure_csv)
    figure_paths: List[Path] = []
    for metric, stem, ylabel in TIME_SERIES_SPECS:
        metric_rows = _metric_rows(rows, "time_series", metric)
        if metric_rows:
            figure_paths.append(_plot_line_metric(metric_rows, metric, stem, ylabel, figure_dir))
    for metric, stem, ylabel in QUEUE_SPECS:
        metric_rows = _metric_rows(rows, "virtual_queue", metric)
        if metric_rows:
            figure_paths.append(_plot_line_metric(metric_rows, metric, stem, ylabel, figure_dir))
    for metric, stem, ylabel in OFFLOADING_SPECS:
        metric_rows = _metric_rows(rows, "offloading_ratio", metric)
        if metric_rows:
            figure_paths.append(_plot_offloading_metric(metric_rows, metric, stem, ylabel, figure_dir))

    table_paths = _write_summary_tables(summary_csv, table_dir)
    data_dir = output_dir / "figure_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    data_copy = data_dir / figure_csv.name
    if figure_csv.resolve() != data_copy.resolve():
        data_copy.write_bytes(figure_csv.read_bytes())

    time_slots = ""
    if rows:
        time_slots = str(rows[0].get("time_slots", ""))
    x_values = [row.get("x_value", "") for row in rows if row.get("figure_group") == "time_series"]
    numeric_x = [_float_value(value) for value in x_values if value != ""]
    x_axis = ""
    if numeric_x:
        x_axis = f"{min(numeric_x):.0f}..{max(numeric_x):.0f}"

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "package": str(output_dir),
        "figure_format": "png-only",
        "figure_count": len(figure_paths),
        "table_count": len(table_paths),
        "source_figure_data_csv": str(figure_csv),
        "source_figure_data_sha256": sha256_file(figure_csv),
        "source_summary_csv": str(summary_csv),
        "source_summary_sha256": sha256_file(summary_csv),
        "copied_figure_data_csv": str(data_copy),
        "copied_figure_data_sha256": sha256_file(data_copy),
        "time_slots": time_slots,
        "x_axis": x_axis,
        "queue_stability_claim": queue_stability_claim,
        "figures": [_record(path, "figure", output_dir) for path in sorted(figure_paths)],
        "tables": [_record(path, "table", output_dir) for path in sorted(table_paths)],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate PNG-only normal-main figures and tables.")
    parser.add_argument("--figure-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--queue-stability-claim", required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    manifest_path = generate_normal_main_package(
        figure_csv=args.figure_csv,
        summary_csv=args.summary_csv,
        output_dir=args.output_dir,
        run_id=args.run_id,
        queue_stability_claim=args.queue_stability_claim,
    )
    print(json.dumps({"manifest": str(manifest_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
