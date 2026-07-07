"""Generate a PNG-only manuscript-style LR figure package from assembled LR CSVs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import struct
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


LR_COLORS: Sequence[str] = ("#0000FF", "#008000", "#FF8C00", "#FF0000", "#9467BD", "#8C564B")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _float_value(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _boolish(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _png_dimensions(path: Path) -> Tuple[int, int]:
    with path.open("rb") as f:
        header = f.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"not a valid PNG template: {path}")
    return struct.unpack(">II", header[16:24])


def _format_lr(value: object) -> str:
    lr = _float_value(value)
    if lr == 0:
        return str(value)
    return f"{lr:.5g}"


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


def _series_summary(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    by_lr: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        by_lr.setdefault(row.get("lr_label", ""), []).append(row)

    summaries: List[Dict[str, object]] = []
    for lr_label, series in sorted(by_lr.items(), key=lambda item: _float_value(item[1][0].get("lr", 0))):
        series = sorted(series, key=lambda row: int(_float_value(row.get("epoch", 0))))
        first = series[0]
        last = series[-1]
        tail = series[-10:]
        summaries.append(
            {
                "lr_label": lr_label,
                "lr": _float_value(first.get("lr", 0)),
                "epoch_count": len(series),
                "first_epoch": int(_float_value(first.get("epoch", 0))),
                "last_epoch": int(_float_value(last.get("epoch", 0))),
                "initial_train_loss": _float_value(first.get("train_loss", 0)),
                "final_train_loss": _float_value(last.get("train_loss", 0)),
                "final_validation_loss": _float_value(last.get("validation_loss", 0)),
                "tail_actor_collapse_count": sum(1 for row in tail if _boolish(row.get("actor_collapse_detected", ""))),
                "tail_instability_count": sum(1 for row in tail if _boolish(row.get("instability_detected", ""))),
                "checkpoint_path": last.get("checkpoint_path", ""),
                "checkpoint_sha256": last.get("checkpoint_sha256", ""),
                "train_meta_path": last.get("train_meta_path", ""),
                "train_meta_sha256": last.get("train_meta_sha256", ""),
            }
        )
    return summaries


def _plot_lr_training_loss(
    training_rows: List[Dict[str, str]],
    output_path: Path,
    template_dimensions: Tuple[int, int],
    x_label: str,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    width, height = template_dimensions
    dpi = 96
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)

    by_lr: Dict[str, List[Dict[str, str]]] = {}
    for row in training_rows:
        by_lr.setdefault(row.get("lr_label", ""), []).append(row)

    for index, (lr_label, rows) in enumerate(sorted(by_lr.items(), key=lambda item: _float_value(item[1][0].get("lr", 0)))):
        rows = sorted(rows, key=lambda row: int(_float_value(row.get("epoch", 0))))
        ax.plot(
            [int(_float_value(row.get("epoch", 0))) for row in rows],
            [_float_value(row.get("train_loss", 0)) for row in rows],
            color=LR_COLORS[index % len(LR_COLORS)],
            linewidth=2.6,
            label=f"Learning Rate = {_format_lr(rows[0].get('lr', lr_label))}",
        )

    ax.set_xlabel(x_label, fontsize=18, fontweight="bold")
    ax.set_ylabel("Training Loss", fontsize=18, fontweight="bold")
    ax.tick_params(axis="both", labelsize=14, width=1.0)
    ax.grid(True, color="#D9D9D9", linewidth=1.0, alpha=0.55)
    for spine in ax.spines.values():
        spine.set_color("#808080")
        spine.set_linewidth(1.2)
    legend = ax.legend(loc="upper right", fontsize=14, frameon=True)
    legend.get_frame().set_edgecolor("#CCCCCC")
    legend.get_frame().set_linewidth(0.8)
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.16, top=0.98)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def generate_lr_template_package(
    training_csv: Path | str,
    sensitivity_csv: Path | str,
    sweep_manifest: Path | str,
    template_png: Path | str,
    output_dir: Path | str,
    run_id: str,
    convergence_decision: str,
    x_label: str = "Training Epoch",
) -> Path:
    training_csv = Path(training_csv)
    sensitivity_csv = Path(sensitivity_csv)
    sweep_manifest = Path(sweep_manifest)
    template_png = Path(template_png)
    output_dir = Path(output_dir)
    figure_dir = output_dir / "figures"
    data_dir = output_dir / "figure_data"
    figure_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    training_rows = _read_csv(training_csv)
    if not training_rows:
        raise ValueError(f"training CSV has no rows: {training_csv}")
    template_dimensions = _png_dimensions(template_png)

    figure_path = _plot_lr_training_loss(
        training_rows=training_rows,
        output_path=figure_dir / "learning_rate_training_loss_template.png",
        template_dimensions=template_dimensions,
        x_label=x_label,
    )

    copied_training = _copy_into_data_dir(training_csv, data_dir)
    copied_sensitivity = _copy_into_data_dir(sensitivity_csv, data_dir)
    copied_sweep_manifest = _copy_into_data_dir(sweep_manifest, data_dir)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "package": str(output_dir),
        "figure_format": "png-only",
        "figure_count": 1,
        "source_training_csv": str(training_csv),
        "source_training_csv_sha256": sha256_file(training_csv),
        "source_sensitivity_csv": str(sensitivity_csv),
        "source_sensitivity_csv_sha256": sha256_file(sensitivity_csv),
        "source_lr_sweep_manifest": str(sweep_manifest),
        "source_lr_sweep_manifest_sha256": sha256_file(sweep_manifest),
        "copied_training_csv": str(copied_training),
        "copied_training_csv_sha256": sha256_file(copied_training),
        "copied_sensitivity_csv": str(copied_sensitivity),
        "copied_sensitivity_csv_sha256": sha256_file(copied_sensitivity),
        "copied_lr_sweep_manifest": str(copied_sweep_manifest),
        "copied_lr_sweep_manifest_sha256": sha256_file(copied_sweep_manifest),
        "template_png": str(template_png),
        "template_png_sha256": sha256_file(template_png),
        "template_dimensions": {"width": template_dimensions[0], "height": template_dimensions[1]},
        "template_match_notes": (
            "Canvas dimensions and single-panel legend/grid/line style follow the manuscript LR raster; "
            "x-axis label is the actual source index from the assembled training CSV."
        ),
        "x_label": x_label,
        "convergence_decision": convergence_decision,
        "lr_series": _series_summary(training_rows),
        "figures": [_record(figure_path, "figure", output_dir)],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a PNG-only manuscript-style LR figure package.")
    parser.add_argument("--training-csv", type=Path, required=True)
    parser.add_argument("--sensitivity-csv", type=Path, required=True)
    parser.add_argument("--sweep-manifest", type=Path, required=True)
    parser.add_argument("--template-png", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--convergence-decision", required=True)
    parser.add_argument("--x-label", default="Training Epoch")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    manifest_path = generate_lr_template_package(
        training_csv=args.training_csv,
        sensitivity_csv=args.sensitivity_csv,
        sweep_manifest=args.sweep_manifest,
        template_png=args.template_png,
        output_dir=args.output_dir,
        run_id=args.run_id,
        convergence_decision=args.convergence_decision,
        x_label=args.x_label,
    )
    print(json.dumps({"manifest": str(manifest_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
