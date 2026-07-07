"""Generate a PNG-only manuscript-style V sensitivity figure package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import struct
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


V_METRIC_STYLES: Sequence[Tuple[str, str, str, str, str]] = (
    ("Average Virtual Energy Queue", "Average Virtual Energy Queue", "#4C72B0", "--", "^"),
    ("Average Virtual Delay Queue", "Average Virtual Latency Queue", "#9AC9DB", "--", "v"),
    ("Total Energy Consumption", "Total Energy Consumption (J)", "#C44E52", "-.", "o"),
    ("Service-Chain Operational Cost", "Service-Chain Operational Cost", "#DD8452", "-.", "s"),
    ("Average Response Delay", "Average Response Delay (ms)", "#55A868", ":", "D"),
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


def _png_dimensions(path: Path) -> Tuple[int, int]:
    with path.open("rb") as f:
        header = f.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"not a valid PNG template: {path}")
    return struct.unpack(">II", header[16:24])


def _float_value(value: object) -> float:
    try:
        return float(value)
    except Exception as exc:
        raise ValueError(f"expected numeric value, got {value!r}") from exc


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


def _load_v_series(rows: Iterable[Dict[str, str]], algorithm: str) -> Dict[str, Dict[float, float]]:
    values: Dict[str, Dict[float, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get("figure_group") != "V":
            continue
        if row.get("algorithm") != algorithm:
            continue
        metric = row.get("metric", "")
        if metric not in {item[0] for item in V_METRIC_STYLES}:
            continue
        v_value = _float_value(row.get("x_value", ""))
        values[metric][v_value].append(_float_value(row.get("value", "")))

    series: Dict[str, Dict[float, float]] = {}
    for metric, by_v in values.items():
        series[metric] = {
            v_value: sum(items) / max(len(items), 1)
            for v_value, items in by_v.items()
        }

    required = [item[0] for item in V_METRIC_STYLES]
    missing = [metric for metric in required if metric not in series]
    if missing:
        raise ValueError(f"missing V metrics for {algorithm}: {missing}")
    expected_v = sorted(series[required[0]])
    if not expected_v:
        raise ValueError(f"no V rows found for {algorithm}")
    for metric in required[1:]:
        if sorted(series[metric]) != expected_v:
            raise ValueError(f"metric {metric} has mismatched V values")
    return series


def _plot_v_sensitivity(
    series: Dict[str, Dict[float, float]],
    output_path: Path,
    template_dimensions: Tuple[int, int],
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import host_subplot
    import mpl_toolkits.axisartist as AA

    width, height = template_dimensions
    dpi = 300 if width >= 1800 else 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    host = host_subplot(111, axes_class=AA.Axes)
    plt.subplots_adjust(left=0.15, right=0.75, bottom=0.18, top=0.90)

    par1 = host.get_aux_axes(viewlim_mode=None, sharex=host)
    par2 = host.get_aux_axes(viewlim_mode=None, sharex=host)
    par3 = host.get_aux_axes(viewlim_mode=None, sharex=host)
    par4 = host.get_aux_axes(viewlim_mode=None, sharex=host)
    axes = [host, par1, par2, par3, par4]

    host.axis["left"].set_label("Average Virtual Energy Queue")
    par1.axis["left"] = par1.new_fixed_axis(loc="left", offset=(-60, 0))
    par2.axis["right"] = par2.new_fixed_axis(loc="right", offset=(5, 0))
    par3.axis["right"] = par3.new_fixed_axis(loc="right", offset=(60, 0))
    par4.axis["right"] = par4.new_fixed_axis(loc="right", offset=(115, 0))
    par1.axis["right"].toggle(all=False)
    par1.axis["bottom"].toggle(all=False)
    par1.axis["top"].toggle(all=False)
    par2.axis["left"].toggle(all=False)
    par3.axis["left"].toggle(all=False)
    par4.axis["left"].toggle(all=False)

    v_values = sorted(next(iter(series.values())))
    lines = []
    for axis, (metric, label, color, linestyle, marker) in zip(axes, V_METRIC_STYLES):
        line, = axis.plot(
            v_values,
            [series[metric][v_value] for v_value in v_values],
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            marker=marker,
            markersize=6,
            label=label,
        )
        lines.append(line)

    host.set_xlabel("Control Parameter V", fontsize=11)
    host.set_xscale("log")
    host.set_xticks(v_values)
    host.set_xticklabels([str(int(value)) if float(value).is_integer() else str(value) for value in v_values])
    host.grid(True, which="major", linestyle="--", alpha=0.5, color="gray")

    def style_axis(axis, label: str, position: str) -> None:
        axis.axis[position].label.set_color("black")
        axis.axis[position].major_ticklabels.set_color("black")
        axis.axis[position].major_ticks.set_color("black")
        axis.axis[position].line.set_color("black")
        axis.axis[position].set_label(label)

    style_axis(host, V_METRIC_STYLES[0][1], "left")
    style_axis(par1, V_METRIC_STYLES[1][1], "left")
    style_axis(par2, V_METRIC_STYLES[2][1], "right")
    style_axis(par3, V_METRIC_STYLES[3][1], "right")
    style_axis(par4, V_METRIC_STYLES[4][1], "right")
    host.legend(lines, [line.get_label() for line in lines], loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=3, fontsize=9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def generate_v_template_package(
    figure_csv: Path | str,
    batch_meta: Path | str,
    template_png: Path | str,
    output_dir: Path | str,
    run_id: str,
    decision: str,
    algorithm: str = "LyHAM-CO",
) -> Path:
    figure_csv = Path(figure_csv)
    batch_meta = Path(batch_meta)
    template_png = Path(template_png)
    output_dir = Path(output_dir)
    figure_dir = output_dir / "figures"
    data_dir = output_dir / "figure_data"
    figure_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_csv(figure_csv)
    if not rows:
        raise ValueError(f"figure CSV has no rows: {figure_csv}")
    series = _load_v_series(rows, algorithm=algorithm)
    template_dimensions = _png_dimensions(template_png)
    figure_path = _plot_v_sensitivity(
        series=series,
        output_path=figure_dir / "v_sensitivity_template.png",
        template_dimensions=template_dimensions,
    )

    copied_figure_csv = _copy_into_data_dir(figure_csv, data_dir)
    copied_batch_meta = _copy_into_data_dir(batch_meta, data_dir)
    v_values = sorted(next(iter(series.values())))
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "package": str(output_dir),
        "figure_format": "png-only",
        "figure_count": 1,
        "algorithm": algorithm,
        "source_figure_csv": str(figure_csv),
        "source_figure_csv_sha256": sha256_file(figure_csv),
        "source_batch_meta": str(batch_meta),
        "source_batch_meta_sha256": sha256_file(batch_meta),
        "copied_figure_csv": str(copied_figure_csv),
        "copied_figure_csv_sha256": sha256_file(copied_figure_csv),
        "copied_batch_meta": str(copied_batch_meta),
        "copied_batch_meta_sha256": sha256_file(copied_batch_meta),
        "template_png": str(template_png),
        "template_png_sha256": sha256_file(template_png),
        "template_dimensions": {"width": template_dimensions[0], "height": template_dimensions[1]},
        "template_match_notes": (
            "Single multi-axis log-x V plot follows the manuscript V raster/local generator style; "
            "final figure uses unified-pipeline V sweep rows, not hard-coded static arrays."
        ),
        "decision": decision,
        "v_values": v_values,
        "metrics": [item[0] for item in V_METRIC_STYLES],
        "series": {
            metric: [{"V": v_value, "value": values[v_value]} for v_value in v_values]
            for metric, values in series.items()
        },
        "figures": [_record(figure_path, "figure", output_dir)],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a PNG-only manuscript-style V sensitivity package.")
    parser.add_argument("--figure-csv", type=Path, required=True)
    parser.add_argument("--batch-meta", type=Path, required=True)
    parser.add_argument("--template-png", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--decision", required=True)
    parser.add_argument("--algorithm", default="LyHAM-CO")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    manifest_path = generate_v_template_package(
        figure_csv=args.figure_csv,
        batch_meta=args.batch_meta,
        template_png=args.template_png,
        output_dir=args.output_dir,
        run_id=args.run_id,
        decision=args.decision,
        algorithm=args.algorithm,
    )
    print(json.dumps({"manifest": str(manifest_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
