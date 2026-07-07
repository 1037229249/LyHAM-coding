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


class NormalMainFigurePackageTest(unittest.TestCase):
    def test_generator_writes_png_only_figures_tables_and_manifest(self):
        sys.path.insert(0, str(UTILS_DIR))

        from generate_normal_main_figure_package import generate_normal_main_package

        base_dir = _workspace_test_tmp("normal_main_figure_package_unit")
        figure_csv = base_dir / "figure_data.csv"
        summary_csv = base_dir / "summary.csv"
        out_dir = base_dir / "package"

        figure_fields = [
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
        rows = []
        for algorithm, offset in [("LyHAM-CO", 0.0), ("FFD-Myopic", 1.0)]:
            for slot in [0, 1, 2]:
                for metric, unit in [
                    ("Average Response Delay", "ms"),
                    ("Total Energy Consumption", "J"),
                    ("Service-Chain Operational Cost", "cost_unit"),
                    ("Average Virtual Energy Queue", "queue_length"),
                    ("Average Virtual Delay Queue", "queue_length"),
                ]:
                    rows.append(
                        {
                            "figure_group": "virtual_queue" if "Queue" in metric else "time_series",
                            "metric": metric,
                            "x_name": "Time Frames",
                            "x_value": str(slot),
                            "algorithm": algorithm,
                            "value": str(slot + 1.0 + offset),
                            "unit": unit,
                            "source": "unit",
                            "run_id": "run_unit",
                            "seed": "mean",
                            "time_slots": "3",
                            "notes": "",
                        }
                    )
        for metric, unit in [
            ("Average Response Delay", "ms"),
            ("Total Energy Consumption", "J"),
            ("Service-Chain Operational Cost", "cost_unit"),
        ]:
            rows.append(
                {
                    "figure_group": "offloading_ratio",
                    "metric": metric,
                    "x_name": "Offloading Ratio (Cloud:Local)",
                    "x_value": "5:5",
                    "algorithm": "LyHAM-CO",
                    "value": "1.5",
                    "unit": unit,
                    "source": "unit",
                    "run_id": "run_unit",
                    "seed": "mean",
                    "time_slots": "3",
                    "notes": "",
                }
            )

        with figure_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=figure_fields)
            writer.writeheader()
            writer.writerows(rows)

        with summary_csv.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "algorithm",
                    "seed",
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
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "algorithm": "LyHAM-CO",
                    "seed": "-1",
                    "delay_mean": "10",
                    "energy_mean": "2",
                    "cost_mean": "20",
                    "avg_y_mean": "3",
                    "avg_z_mean": "4",
                    "decision_time_mean_ms": "5",
                    "decision_time_p95_ms": "6",
                    "valid_seed_count": "2",
                    "formal_gate_passed": "True",
                    "mechanism_gate_passed": "False",
                    "claim_supported": "True",
                }
            )

        manifest_path = generate_normal_main_package(
            figure_csv=figure_csv,
            summary_csv=summary_csv,
            output_dir=out_dir,
            run_id="run_unit",
            queue_stability_claim="unit-test queue gate",
        )

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        pngs = sorted((out_dir / "figures").glob("*.png"))
        non_pngs = sorted((out_dir / "figures").glob("*.*"))
        table_paths = sorted((out_dir / "tables").glob("*"))

        self.assertEqual(len(pngs), 8)
        self.assertEqual(non_pngs, pngs)
        self.assertEqual(manifest["figure_format"], "png-only")
        self.assertEqual(manifest["run_id"], "run_unit")
        self.assertEqual(manifest["figure_count"], 8)
        self.assertTrue((out_dir / "tables" / "normal_main_1000_summary_table.csv").exists())
        self.assertTrue((out_dir / "tables" / "normal_main_1000_compact_table.tex").exists())
        self.assertGreaterEqual(len(table_paths), 2)
        self.assertTrue(all(record["name"].endswith(".png") for record in manifest["figures"]))
        self.assertIn("source_figure_data_sha256", manifest)
        self.assertIn("source_summary_sha256", manifest)


if __name__ == "__main__":
    unittest.main()
