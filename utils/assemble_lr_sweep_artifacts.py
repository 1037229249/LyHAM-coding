"""Assemble formal learning-rate sweep artifacts from verified run outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


DEFAULT_LR_SPECS: Sequence[Tuple[str, float]] = (
    ("lr_1e-5", 1e-5),
    ("lr_3e-5", 3e-5),
    ("lr_1e-4", 1e-4),
    ("lr_3e-4", 3e-4),
    ("lr_1e-3", 1e-3),
)

TRAINING_FIELDS = [
    "lr_label",
    "lr",
    "epoch",
    "train_loss",
    "train_config_hash",
    "training_dataset_hash",
    "sample_count",
    "final_train_loss",
    "min_train_loss",
    "min_train_loss_epoch",
    "last5_loss_delta",
    "label_quality_gate_passed",
    "formal_seed_excluded",
    "gradient_l2_norm",
    "gradient_nonfinite_count",
    "parameter_nonfinite_count",
    "logit_nonfinite_count",
    "probability_mean",
    "predicted_positive_ratio",
    "target_positive_ratio",
    "actor_collapse_detected",
    "instability_detected",
    "validation_loss",
    "validation_logit_nonfinite_count",
    "checkpoint_path",
    "checkpoint_sha256",
    "train_meta_path",
    "train_meta_sha256",
]

SENSITIVITY_FIELDS = [
    "lr_label",
    "lr",
    "algorithm",
    "delay_mean",
    "energy_mean",
    "cost_mean",
    "decision_time_mean_ms",
    "claim_supported",
    "formal_gate_passed",
    "mechanism_gate_passed",
    "run_id",
    "config_hash",
    "training_dataset_hash",
    "train_config_hash",
    "initial_model_hash",
    "final_model_hash",
    "online_entries",
    "label_quality_gate_passed",
    "formal_seed_excluded",
    "raw_csv_count",
    "raw_rows",
    "raw_bad_key_numeric_cells",
    "checkpoint_path",
    "checkpoint_sha256",
    "train_meta_path",
    "train_meta_sha256",
    "summary_path",
    "summary_sha256",
    "eval_meta_path",
    "eval_meta_sha256",
    "progress_path",
    "progress_sha256",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _single_file(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected exactly one {pattern} in {directory}, found {len(matches)}")
    return matches[0]


def _select_summary_csv(directory: Path) -> Path:
    canonical = directory / "ablation_summary.csv"
    if canonical.exists():
        return canonical
    matches = sorted(directory.glob("ablation_summary*.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one ablation summary CSV in {directory}, found {len(matches)}")
    return matches[0]


def _resolve_lr_spec(default_base_dir: Path, lr_spec: Sequence[Any]) -> Tuple[str, float, Path]:
    if len(lr_spec) == 2:
        lr_label, lr_value = lr_spec
        return str(lr_label), float(lr_value), default_base_dir
    if len(lr_spec) == 3:
        lr_label, lr_value, artifact_root = lr_spec
        return str(lr_label), float(lr_value), Path(artifact_root)
    raise ValueError(f"LR spec must be (label, value) or (label, value, artifact_root), got {lr_spec!r}")


def _is_resume_artifact(path: Path) -> bool:
    return path.name.startswith("pair_uac_actor_resume_")


def _last_epoch_from_meta(path: Path) -> int:
    try:
        meta = _load_json(path)
    except Exception:
        return -1
    epochs = [int(item.get("epoch", 0)) for item in meta.get("loss_history", []) if "epoch" in item]
    return max(epochs) if epochs else int(meta.get("train_config", {}).get("epochs", 0) or 0)


def _select_final_train_meta(directory: Path) -> Path:
    matches = sorted(path for path in directory.glob("*.meta.json") if not _is_resume_artifact(path))
    if not matches:
        raise FileNotFoundError(f"expected at least one final training meta in {directory}")
    if len(matches) == 1:
        return matches[0]
    return max(matches, key=lambda path: (_last_epoch_from_meta(path), path.stat().st_mtime, path.name))


def _checkpoint_for_train_meta(train_meta_path: Path) -> Path:
    if not train_meta_path.name.endswith(".meta.json"):
        raise ValueError(f"unexpected training meta name: {train_meta_path}")
    checkpoint_name = train_meta_path.name[: -len(".meta.json")] + ".pth"
    checkpoint_path = train_meta_path.with_name(checkpoint_name)
    if checkpoint_path.exists():
        return checkpoint_path
    matches = sorted(path for path in train_meta_path.parent.glob("*.pth") if not _is_resume_artifact(path))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected checkpoint matching {train_meta_path.name} or one final *.pth in {train_meta_path.parent}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _count_raw_rows_and_bad_cells(raw_dir: Path) -> Dict[str, int]:
    raw_files = sorted(raw_dir.rglob("*.csv")) if raw_dir.exists() else []
    rows = 0
    bad = 0
    numeric_columns = {"delay", "energy", "cost", "queue_backlog", "decision_time", "offloading_ratio"}
    for path in raw_files:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows += 1
                for column in numeric_columns.intersection(row.keys()):
                    value = row.get(column, "")
                    try:
                        number = float(value)
                    except (TypeError, ValueError):
                        bad += 1
                        continue
                    if number != number or number in (float("inf"), float("-inf")):
                        bad += 1
    return {
        "raw_csv_count": len(raw_files),
        "raw_rows": rows,
        "raw_bad_key_numeric_cells": bad,
    }


def _loss_summary(loss_history: List[Dict]) -> Dict:
    losses = [float(item["loss"]) for item in loss_history]
    min_loss = min(losses)
    min_index = losses.index(min_loss)
    if len(losses) >= 5:
        last5_delta = losses[-1] - losses[-5]
    elif len(losses) >= 2:
        last5_delta = losses[-1] - losses[0]
    else:
        last5_delta = 0.0
    return {
        "min_train_loss": min_loss,
        "min_train_loss_epoch": int(loss_history[min_index]["epoch"]),
        "last5_loss_delta": last5_delta,
    }


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_aggregate_rows(summary_path: Path) -> List[Dict]:
    with summary_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return [row for row in reader if str(row.get("seed")) == "-1"]


def _make_figures(output_dir: Path, training_rows: List[Dict], sensitivity_rows: List[Dict]) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_paths: List[Path] = []

    by_lr: Dict[str, List[Dict]] = {}
    for row in training_rows:
        by_lr.setdefault(row["lr_label"], []).append(row)

    plt.figure(figsize=(6.0, 3.8))
    for lr_label, rows in sorted(by_lr.items(), key=lambda item: float(item[1][0]["lr"])):
        rows = sorted(rows, key=lambda item: int(item["epoch"]))
        plt.plot(
            [int(row["epoch"]) for row in rows],
            [float(row["train_loss"]) for row in rows],
            marker="o",
            linewidth=1.5,
            markersize=3,
            label=lr_label.replace("lr_", ""),
        )
    plt.xlabel("Epoch")
    plt.ylabel("Pairwise training loss")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Learning rate", fontsize=8)
    plt.tight_layout()
    for ext in ("png", "pdf", "svg"):
        path = figure_dir / f"lr_training_loss_curve.{ext}"
        plt.savefig(path, dpi=300)
        figure_paths.append(path)
    plt.close()

    metrics = [
        ("cost_mean", "Average cost"),
        ("delay_mean", "Average delay"),
        ("energy_mean", "Average energy"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4), sharex=True)
    algorithms = sorted({row["algorithm"] for row in sensitivity_rows})
    for ax, (metric, label) in zip(axes, metrics):
        for algorithm in algorithms:
            rows = [row for row in sensitivity_rows if row["algorithm"] == algorithm]
            rows = sorted(rows, key=lambda item: float(item["lr"]))
            ax.plot(
                [float(row["lr"]) for row in rows],
                [float(row[metric]) for row in rows],
                marker="o",
                linewidth=1.5,
                markersize=3,
                label=algorithm,
            )
        ax.set_xscale("log")
        ax.set_xlabel("Learning rate")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        path = figure_dir / f"lr_sensitivity_metrics.{ext}"
        fig.savefig(path, dpi=300)
        figure_paths.append(path)
    plt.close(fig)

    return figure_paths


def assemble_lr_sweep(
    base_dir: Path | str,
    output_dir: Path | str | None = None,
    lr_specs: Sequence[Tuple[str, float]] = DEFAULT_LR_SPECS,
    make_figures: bool = True,
) -> Dict[str, Path]:
    base_dir = Path(base_dir)
    output_dir = Path(output_dir) if output_dir is not None else base_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    training_rows: List[Dict] = []
    sensitivity_rows: List[Dict] = []
    manifest_rows: List[Dict] = []

    resolved_specs = [_resolve_lr_spec(base_dir, lr_spec) for lr_spec in lr_specs]

    for lr_label, lr_value, artifact_root in resolved_specs:
        model_dir = artifact_root / "models" / lr_label
        eval_summary_dir = artifact_root / "eval" / lr_label / "summary"
        raw_dir = artifact_root / "eval" / lr_label / "raw"

        train_meta_path = _select_final_train_meta(model_dir)
        checkpoint_path = _checkpoint_for_train_meta(train_meta_path)
        summary_path = _select_summary_csv(eval_summary_dir)
        eval_meta_path = _single_file(eval_summary_dir, "ablation_run_meta_*.json")
        progress_path = _single_file(eval_summary_dir, "ablation_run_progress_*.json")

        train_meta = _load_json(train_meta_path)
        eval_meta = _load_json(eval_meta_path)
        progress = _load_json(progress_path)
        raw_stats = _count_raw_rows_and_bad_cells(raw_dir)

        train_meta_hash = sha256_file(train_meta_path)
        checkpoint_hash = sha256_file(checkpoint_path)
        summary_hash = sha256_file(summary_path)
        eval_meta_hash = sha256_file(eval_meta_path)
        progress_hash = sha256_file(progress_path)

        loss_history = list(train_meta.get("loss_history", []))
        loss_stats = _loss_summary(loss_history)
        for item in loss_history:
            row = {
                "lr_label": lr_label,
                "lr": lr_value,
                "epoch": int(item["epoch"]),
                "train_loss": float(item["loss"]),
                "train_config_hash": train_meta.get("train_config_hash", ""),
                "training_dataset_hash": train_meta.get("training_dataset_hash", ""),
                "sample_count": train_meta.get("sample_count", ""),
                "final_train_loss": train_meta.get("final_train_loss", ""),
                "label_quality_gate_passed": train_meta.get("label_quality_gate_passed", ""),
                "formal_seed_excluded": train_meta.get("formal_seed_excluded", ""),
                "gradient_l2_norm": item.get("gradient_l2_norm", ""),
                "gradient_nonfinite_count": item.get("gradient_nonfinite_count", ""),
                "parameter_nonfinite_count": item.get("parameter_nonfinite_count", ""),
                "logit_nonfinite_count": item.get("logit_nonfinite_count", ""),
                "probability_mean": item.get("probability_mean", ""),
                "predicted_positive_ratio": item.get("predicted_positive_ratio", ""),
                "target_positive_ratio": item.get("target_positive_ratio", ""),
                "actor_collapse_detected": item.get("actor_collapse_detected", ""),
                "instability_detected": item.get("instability_detected", ""),
                "validation_loss": item.get("validation_loss", ""),
                "validation_logit_nonfinite_count": item.get("validation_logit_nonfinite_count", ""),
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_hash,
                "train_meta_path": str(train_meta_path),
                "train_meta_sha256": train_meta_hash,
                **loss_stats,
            }
            training_rows.append(row)

        aggregate_rows = _read_aggregate_rows(summary_path)
        for aggregate in aggregate_rows:
            sensitivity_rows.append(
                {
                    "lr_label": lr_label,
                    "lr": lr_value,
                    "algorithm": aggregate.get("algorithm", ""),
                    "delay_mean": aggregate.get("delay_mean", ""),
                    "energy_mean": aggregate.get("energy_mean", ""),
                    "cost_mean": aggregate.get("cost_mean", ""),
                    "decision_time_mean_ms": aggregate.get("decision_time_mean_ms", ""),
                    "claim_supported": aggregate.get("claim_supported", ""),
                    "formal_gate_passed": aggregate.get("formal_gate_passed", ""),
                    "mechanism_gate_passed": aggregate.get("mechanism_gate_passed", ""),
                    "run_id": eval_meta.get("run_id", ""),
                    "config_hash": eval_meta.get("config_hash", ""),
                    "training_dataset_hash": eval_meta.get("training_dataset_hash", train_meta.get("training_dataset_hash", "")),
                    "train_config_hash": eval_meta.get("train_config_hash", train_meta.get("train_config_hash", "")),
                    "initial_model_hash": eval_meta.get("initial_model_hash", ""),
                    "final_model_hash": eval_meta.get("final_model_hash", ""),
                    "online_entries": len(eval_meta.get("uac_online_model_diagnostics", {}).get("entries", [])),
                    "label_quality_gate_passed": train_meta.get("label_quality_gate_passed", ""),
                    "formal_seed_excluded": train_meta.get("formal_seed_excluded", ""),
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_sha256": checkpoint_hash,
                    "train_meta_path": str(train_meta_path),
                    "train_meta_sha256": train_meta_hash,
                    "summary_path": str(summary_path),
                    "summary_sha256": summary_hash,
                    "eval_meta_path": str(eval_meta_path),
                    "eval_meta_sha256": eval_meta_hash,
                    "progress_path": str(progress_path),
                    "progress_sha256": progress_hash,
                    **raw_stats,
                }
            )

        manifest_rows.append(
            {
                "lr_label": lr_label,
                "lr": lr_value,
                "artifact_root": str(artifact_root),
                "run_id": eval_meta.get("run_id", ""),
                "config_hash": eval_meta.get("config_hash", ""),
                "train_config_hash": train_meta.get("train_config_hash", ""),
                "training_dataset_hash": train_meta.get("training_dataset_hash", ""),
                "sample_count": train_meta.get("sample_count", ""),
                "label_quality_gate_passed": train_meta.get("label_quality_gate_passed", ""),
                "formal_seed_excluded": train_meta.get("formal_seed_excluded", ""),
                "final_train_loss": train_meta.get("final_train_loss", ""),
                "min_train_loss": loss_stats["min_train_loss"],
                "min_train_loss_epoch": loss_stats["min_train_loss_epoch"],
                "last5_loss_delta": loss_stats["last5_loss_delta"],
                "progress_status": progress.get("status", ""),
                "completed_count": len(progress.get("completed", [])),
                "failed_count": len(progress.get("failed", [])),
                "raw_csv_count": raw_stats["raw_csv_count"],
                "raw_rows": raw_stats["raw_rows"],
                "raw_bad_key_numeric_cells": raw_stats["raw_bad_key_numeric_cells"],
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_hash,
                "train_meta_path": str(train_meta_path),
                "train_meta_sha256": train_meta_hash,
                "summary_path": str(summary_path),
                "summary_sha256": summary_hash,
                "eval_meta_path": str(eval_meta_path),
                "eval_meta_sha256": eval_meta_hash,
                "progress_path": str(progress_path),
                "progress_sha256": progress_hash,
            }
        )

    training_csv = output_dir / "lr_training_curve_data.csv"
    sensitivity_csv = output_dir / "lr_sensitivity_metrics.csv"
    manifest_json = output_dir / "lr_sweep_manifest.json"

    _write_csv(training_csv, TRAINING_FIELDS, training_rows)
    _write_csv(sensitivity_csv, SENSITIVITY_FIELDS, sensitivity_rows)

    figure_paths = _make_figures(output_dir, training_rows, sensitivity_rows) if make_figures else []
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "base_dir": str(base_dir),
        "output_dir": str(output_dir),
        "lr_specs": [
            {"lr_label": label, "lr": lr, "artifact_root": str(artifact_root)}
            for label, lr, artifact_root in resolved_specs
        ],
        "training_curve_csv": str(training_csv),
        "sensitivity_metrics_csv": str(sensitivity_csv),
        "figure_paths": [str(path) for path in figure_paths],
        "lr_rows": manifest_rows,
    }
    manifest_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    outputs = {
        "training_curve_csv": training_csv,
        "sensitivity_metrics_csv": sensitivity_csv,
        "manifest_json": manifest_json,
    }
    if figure_paths:
        outputs["figures_dir"] = output_dir / "figures"
    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Assemble formal LR sweep artifacts.")
    parser.add_argument("--base-dir", default="_test_artifacts/p4_lr_sweep_20260628")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-figures", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    outputs = assemble_lr_sweep(
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        make_figures=not args.no_figures,
    )
    print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


