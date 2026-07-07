import csv
import json
import shutil
import sys
import unittest
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
UTILS_DIR = REPO_DIR / "utils"


def _workspace_test_tmp(name: str) -> Path:
    path = REPO_DIR / "_test_artifacts" / name
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


class LrSweepArtifactsTest(unittest.TestCase):
    def test_assemble_lr_sweep_exports_training_and_sensitivity_rows(self):
        sys.path.insert(0, str(UTILS_DIR))

        from assemble_lr_sweep_artifacts import assemble_lr_sweep

        base_dir = _workspace_test_tmp("lr_sweep_artifacts_unit")
        model_dir = base_dir / "models" / "lr_1e-5"
        summary_dir = base_dir / "eval" / "lr_1e-5" / "summary"
        raw_dir = base_dir / "eval" / "lr_1e-5" / "raw" / "run_a" / "LyHAM-CO"
        model_dir.mkdir(parents=True)
        summary_dir.mkdir(parents=True)
        raw_dir.mkdir(parents=True)

        checkpoint = model_dir / "pair_uac_actor_test.pth"
        checkpoint.write_bytes(b"checkpoint")
        (model_dir / "pair_uac_actor_resume_identity_data123.pth").write_bytes(b"resume checkpoint")
        meta = {
            "train_config_hash": "cfg123",
            "training_dataset_hash": "data123",
            "sample_count": 4,
            "label_quality_gate_passed": True,
            "formal_seed_excluded": True,
            "train_config": {"lr": 1e-5, "epochs": 2},
            "loss_history": [
                {
                    "epoch": 1,
                    "loss": 3.0,
                    "lr": 1e-5,
                    "gradient_l2_norm": 0.5,
                    "gradient_nonfinite_count": 0,
                    "parameter_nonfinite_count": 0,
                    "logit_nonfinite_count": 0,
                    "probability_mean": 0.45,
                    "predicted_positive_ratio": 0.25,
                    "target_positive_ratio": 0.5,
                    "actor_collapse_detected": False,
                    "instability_detected": False,
                    "validation_loss": 2.8,
                    "validation_logit_nonfinite_count": 0,
                },
                {
                    "epoch": 2,
                    "loss": 2.5,
                    "lr": 1e-5,
                    "gradient_l2_norm": 0.4,
                    "gradient_nonfinite_count": 0,
                    "parameter_nonfinite_count": 0,
                    "logit_nonfinite_count": 0,
                    "probability_mean": 0.47,
                    "predicted_positive_ratio": 0.5,
                    "target_positive_ratio": 0.5,
                    "actor_collapse_detected": False,
                    "instability_detected": False,
                    "validation_loss": 2.8,
                    "validation_logit_nonfinite_count": 0,
                },
            ],
            "final_train_loss": 2.5,
        }
        (model_dir / "pair_uac_actor_test.meta.json").write_text(
            json.dumps(meta), encoding="utf-8"
        )

        with (summary_dir / "ablation_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "algorithm",
                    "seed",
                    "delay_mean",
                    "energy_mean",
                    "cost_mean",
                    "decision_time_mean_ms",
                    "claim_supported",
                    "formal_gate_passed",
                    "mechanism_gate_passed",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "algorithm": "LyHAM-CO",
                    "seed": "-1",
                    "delay_mean": "10.0",
                    "energy_mean": "2.0",
                    "cost_mean": "20.0",
                    "decision_time_mean_ms": "3.0",
                    "claim_supported": "True",
                    "formal_gate_passed": "True",
                    "mechanism_gate_passed": "True",
                }
            )
        (summary_dir / "ablation_run_meta_20260628_test.json").write_text(
            json.dumps(
                {
                    "run_id": "run_a",
                    "config_hash": "evalcfg",
                    "initial_model_hash": "init",
                    "final_model_hash": "final",
                    "uac_online_model_diagnostics": {"entries": [1, 2]},
                }
            ),
            encoding="utf-8",
        )
        (summary_dir / "ablation_run_progress_20260628_test.json").write_text(
            json.dumps({"status": "complete", "completed": [1], "failed": []}),
            encoding="utf-8",
        )
        with (raw_dir / "seed_38_per_slot.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["slot", "delay", "energy", "cost"])
            writer.writeheader()
            writer.writerow({"slot": "0", "delay": "1.0", "energy": "0.2", "cost": "2.0"})
            writer.writerow({"slot": "1", "delay": "1.1", "energy": "0.3", "cost": "2.1"})

        outputs = assemble_lr_sweep(
            base_dir,
            output_dir=base_dir / "assembled",
            lr_specs=[("lr_1e-5", 1e-5)],
            make_figures=False,
        )

        with outputs["training_curve_csv"].open(encoding="utf-8", newline="") as f:
            training_rows = list(csv.DictReader(f))
        with outputs["sensitivity_metrics_csv"].open(encoding="utf-8", newline="") as f:
            sensitivity_rows = list(csv.DictReader(f))
        manifest = json.loads(outputs["manifest_json"].read_text(encoding="utf-8"))

        self.assertEqual(len(training_rows), 2)
        self.assertEqual(training_rows[0]["lr_label"], "lr_1e-5")
        self.assertEqual(training_rows[-1]["train_loss"], "2.5")
        self.assertIn("gradient_l2_norm", training_rows[0])
        self.assertEqual(training_rows[0]["gradient_l2_norm"], "0.5")
        self.assertEqual(training_rows[0]["gradient_nonfinite_count"], "0")
        self.assertEqual(training_rows[0]["parameter_nonfinite_count"], "0")
        self.assertEqual(training_rows[0]["logit_nonfinite_count"], "0")
        self.assertEqual(training_rows[0]["probability_mean"], "0.45")
        self.assertEqual(training_rows[0]["predicted_positive_ratio"], "0.25")
        self.assertEqual(training_rows[0]["target_positive_ratio"], "0.5")
        self.assertEqual(training_rows[0]["actor_collapse_detected"], "False")
        self.assertEqual(training_rows[0]["instability_detected"], "False")
        self.assertIn("validation_loss", training_rows[0])
        self.assertEqual(training_rows[0]["validation_loss"], "2.8")
        self.assertEqual(training_rows[0]["validation_logit_nonfinite_count"], "0")
        self.assertEqual(len(sensitivity_rows), 1)
        self.assertEqual(sensitivity_rows[0]["algorithm"], "LyHAM-CO")
        self.assertEqual(sensitivity_rows[0]["raw_csv_count"], "1")
        self.assertEqual(sensitivity_rows[0]["raw_rows"], "2")
        self.assertEqual(sensitivity_rows[0]["label_quality_gate_passed"], "True")
        self.assertEqual(Path(sensitivity_rows[0]["checkpoint_path"]).name, "pair_uac_actor_test.pth")
        self.assertEqual(manifest["lr_rows"][0]["run_id"], "run_a")

    def test_assemble_lr_sweep_accepts_per_lr_roots_and_named_summary(self):
        sys.path.insert(0, str(UTILS_DIR))

        from assemble_lr_sweep_artifacts import assemble_lr_sweep

        base_dir = _workspace_test_tmp("lr_sweep_multi_root_unit")

        def write_lr(root: Path, lr_label: str, lr_value: float, run_id: str) -> None:
            model_dir = root / "models" / lr_label
            summary_dir = root / "eval" / lr_label / "summary"
            raw_dir = root / "eval" / lr_label / "raw" / run_id / "LyHAM-CO"
            model_dir.mkdir(parents=True)
            summary_dir.mkdir(parents=True)
            raw_dir.mkdir(parents=True)

            (model_dir / f"{lr_label}.pth").write_bytes(f"checkpoint-{lr_label}".encode("utf-8"))
            meta = {
                "train_config_hash": f"cfg-{lr_label}",
                "training_dataset_hash": "dataset",
                "sample_count": 8,
                "label_quality_gate_passed": True,
                "formal_seed_excluded": True,
                "train_config": {"lr": lr_value, "epochs": 2},
                "loss_history": [
                    {
                        "epoch": 1,
                        "loss": 3.0,
                        "lr": lr_value,
                        "gradient_l2_norm": 0.5,
                        "gradient_nonfinite_count": 0,
                        "parameter_nonfinite_count": 0,
                        "logit_nonfinite_count": 0,
                        "probability_mean": 0.45,
                        "predicted_positive_ratio": 0.25,
                        "target_positive_ratio": 0.5,
                        "actor_collapse_detected": False,
                        "instability_detected": False,
                        "validation_loss": 2.8,
                        "validation_logit_nonfinite_count": 0,
                    },
                    {
                        "epoch": 2,
                        "loss": 2.4,
                        "lr": lr_value,
                        "gradient_l2_norm": 0.4,
                        "gradient_nonfinite_count": 0,
                        "parameter_nonfinite_count": 0,
                        "logit_nonfinite_count": 0,
                        "probability_mean": 0.47,
                        "predicted_positive_ratio": 0.5,
                        "target_positive_ratio": 0.5,
                        "actor_collapse_detected": False,
                        "instability_detected": False,
                        "validation_loss": 2.7,
                        "validation_logit_nonfinite_count": 0,
                    },
                ],
                "final_train_loss": 2.4,
            }
            (model_dir / f"{lr_label}.meta.json").write_text(json.dumps(meta), encoding="utf-8")

            with (summary_dir / f"ablation_summary_{run_id}.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "algorithm",
                        "seed",
                        "delay_mean",
                        "energy_mean",
                        "cost_mean",
                        "decision_time_mean_ms",
                        "claim_supported",
                        "formal_gate_passed",
                        "mechanism_gate_passed",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "algorithm": "LyHAM-CO",
                        "seed": "-1",
                        "delay_mean": "10.0",
                        "energy_mean": "2.0",
                        "cost_mean": "20.0",
                        "decision_time_mean_ms": "3.0",
                        "claim_supported": "True",
                        "formal_gate_passed": "True",
                        "mechanism_gate_passed": "True",
                    }
                )
            (summary_dir / f"ablation_run_meta_{run_id}.json").write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "config_hash": "evalcfg",
                        "initial_model_hash": "init",
                        "final_model_hash": "final",
                        "uac_online_model_diagnostics": {"entries": [1]},
                    }
                ),
                encoding="utf-8",
            )
            (summary_dir / f"ablation_run_progress_{run_id}.json").write_text(
                json.dumps({"status": "complete", "completed": [1], "failed": []}),
                encoding="utf-8",
            )
            with (raw_dir / "seed_38_per_slot.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["slot", "delay", "energy", "cost"])
                writer.writeheader()
                writer.writerow({"slot": "0", "delay": "1.0", "energy": "0.2", "cost": "2.0"})

        root_a = base_dir / "root_a"
        root_b = base_dir / "root_b"
        write_lr(root_a, "lr_1e-3", 1e-3, "run_a")
        write_lr(root_b, "lr_5e-3", 5e-3, "run_b")

        outputs = assemble_lr_sweep(
            base_dir / "unused_default",
            output_dir=base_dir / "assembled",
            lr_specs=[("lr_1e-3", 1e-3, root_a), ("lr_5e-3", 5e-3, root_b)],
            make_figures=False,
        )

        with outputs["sensitivity_metrics_csv"].open(encoding="utf-8", newline="") as f:
            sensitivity_rows = list(csv.DictReader(f))
        manifest = json.loads(outputs["manifest_json"].read_text(encoding="utf-8"))

        self.assertEqual([row["run_id"] for row in sensitivity_rows], ["run_a", "run_b"])
        self.assertEqual(manifest["lr_specs"][0]["artifact_root"], str(root_a))
        self.assertEqual(manifest["lr_specs"][1]["artifact_root"], str(root_b))
        self.assertEqual(manifest["lr_rows"][0]["artifact_root"], str(root_a))
        self.assertTrue(sensitivity_rows[0]["summary_path"].endswith("ablation_summary_run_a.csv"))

    def test_generate_lr_template_package_writes_png_only_with_template_provenance(self):
        sys.path.insert(0, str(UTILS_DIR))

        from generate_lr_template_figure_package import generate_lr_template_package

        base_dir = _workspace_test_tmp("lr_template_package_unit")
        training_csv = base_dir / "lr_training_curve_data.csv"
        sensitivity_csv = base_dir / "lr_sensitivity_metrics.csv"
        sweep_manifest = base_dir / "lr_sweep_manifest.json"
        template_png = base_dir / "learning_rate_plot.png"
        out_dir = base_dir / "package"

        with training_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "lr_label",
                    "lr",
                    "epoch",
                    "train_loss",
                    "validation_loss",
                    "actor_collapse_detected",
                    "instability_detected",
                    "checkpoint_path",
                    "checkpoint_sha256",
                    "train_meta_path",
                    "train_meta_sha256",
                ],
            )
            writer.writeheader()
            for lr_label, lr_value, losses in [
                ("lr_1e-3", "0.001", [2.60, 2.50, 2.45]),
                ("lr_3e-4", "0.0003", [2.60, 2.58, 2.56]),
            ]:
                for epoch, loss in enumerate(losses, start=1):
                    writer.writerow(
                        {
                            "lr_label": lr_label,
                            "lr": lr_value,
                            "epoch": epoch,
                            "train_loss": loss,
                            "validation_loss": loss + 0.1,
                            "actor_collapse_detected": "False",
                            "instability_detected": "False",
                            "checkpoint_path": f"{lr_label}.pth",
                            "checkpoint_sha256": f"{lr_label}-ckpt",
                            "train_meta_path": f"{lr_label}.meta.json",
                            "train_meta_sha256": f"{lr_label}-meta",
                        }
                    )

        with sensitivity_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["lr_label", "lr", "algorithm", "delay_mean", "energy_mean", "cost_mean"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "lr_label": "lr_1e-3",
                    "lr": "0.001",
                    "algorithm": "LyHAM-CO",
                    "delay_mean": "10.0",
                    "energy_mean": "2.0",
                    "cost_mean": "20.0",
                }
            )
        sweep_manifest.write_text(json.dumps({"lr_specs": [{"lr_label": "lr_1e-3", "lr": 0.001}]}), encoding="utf-8")

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(906 / 96, 581 / 96), dpi=96)
        fig.savefig(template_png)
        plt.close(fig)

        manifest_path = generate_lr_template_package(
            training_csv=training_csv,
            sensitivity_csv=sensitivity_csv,
            sweep_manifest=sweep_manifest,
            template_png=template_png,
            output_dir=out_dir,
            run_id="lr_unit",
            convergence_decision="unit-test candidate",
        )

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        pngs = sorted((out_dir / "figures").glob("*.png"))
        all_figure_files = sorted((out_dir / "figures").glob("*.*"))

        self.assertEqual(len(pngs), 1)
        self.assertEqual(all_figure_files, pngs)
        self.assertEqual(manifest["figure_format"], "png-only")
        self.assertEqual(manifest["template_dimensions"], {"width": 906, "height": 581})
        self.assertEqual(manifest["figure_count"], 1)
        self.assertEqual(manifest["figures"][0]["name"], "learning_rate_training_loss_template.png")
        self.assertTrue((out_dir / "figure_data" / training_csv.name).exists())
        self.assertIn("source_training_csv_sha256", manifest)
        self.assertIn("source_sensitivity_csv_sha256", manifest)
        self.assertIn("source_lr_sweep_manifest_sha256", manifest)


if __name__ == "__main__":
    unittest.main()





