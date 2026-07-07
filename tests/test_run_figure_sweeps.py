import csv
import shutil
import sys
import types
import unittest
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
UTILS_DIR = REPO_DIR / "utils"


def _workspace_test_tmp(name: str) -> Path:
    path = REPO_DIR / "_test_artifacts" / name
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


class RunFigureSweepsTest(unittest.TestCase):
    def test_algorithm_override_limits_v_batch_to_requested_algorithm(self):
        sys.path.insert(0, str(UTILS_DIR))

        import run_figure_sweeps

        base_dir = _workspace_test_tmp("run_figure_sweeps_algorithm_override_unit")
        captured_algorithms = []

        def fake_run_ablation_experiment(config, run_id, silent=True):
            captured_algorithms.append(list(config.algorithms))
            summary_dir = Path(config.output_dir) / "summary"
            summary_dir.mkdir(parents=True, exist_ok=True)
            summary_path = summary_dir / f"ablation_summary_{run_id}.csv"
            with summary_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "algorithm",
                        "seed",
                        "valid",
                        "delay_mean",
                        "energy_mean",
                        "cost_mean",
                        "avg_y_mean",
                        "avg_z_mean",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "algorithm": "LyHAM-CO",
                        "seed": "-1",
                        "valid": "True",
                        "delay_mean": "10",
                        "energy_mean": "2",
                        "cost_mean": "20",
                        "avg_y_mean": "3",
                        "avg_z_mean": "4",
                    }
                )
            return {"base_dir": str(config.output_dir), "meta_path": str(summary_dir / "meta.json")}

        original_runner = run_figure_sweeps.run_ablation_experiment
        run_figure_sweeps.run_ablation_experiment = fake_run_ablation_experiment
        try:
            args = types.SimpleNamespace(
                batch_id="v_algorithm_override_unit",
                sweep_name="V",
                start_index=0,
                max_configs=1,
                seeds=[38],
                time_slots=5,
                slow_epoch_slots=None,
                output_dir=base_dir / "results",
                out=base_dir / "v_metrics.csv",
                meta=base_dir / "v_metrics.meta.json",
                no_resume=True,
                verbose=False,
                algorithms=["LyHAM-CO"],
            )
            run_figure_sweeps.run_figure_sweep_batch(args)
        finally:
            run_figure_sweeps.run_ablation_experiment = original_runner

        self.assertEqual(captured_algorithms, [["LyHAM-CO"]])
        with (base_dir / "v_metrics.csv").open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 5)
        self.assertEqual({row["algorithm"] for row in rows}, {"LyHAM-CO"})


if __name__ == "__main__":
    unittest.main()
