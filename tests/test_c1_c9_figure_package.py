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


class C1C9FigurePackageTest(unittest.TestCase):
    def test_generate_c1_c9_package_writes_png_only_manifest(self):
        sys.path.insert(0, str(UTILS_DIR))

        from generate_c1_c9_figure_package import generate_c1_c9_package

        base_dir = _workspace_test_tmp("c1_c9_figure_package_unit")
        figure_csv = base_dir / "formal_parameter_sweep_figure_metrics.csv"
        validation_manifest = base_dir / "c1_c9_validation_manifest.json"
        out_dir = base_dir / "package"

        groups = {
            "edge_nodes": {
                "x_name": "Number of Edge Nodes",
                "x_values": ["20", "25", "30", "35", "40", "45"],
            },
            "chain_length": {
                "x_name": "Length of Service Chains",
                "x_values": ["[2,4]", "[3,5]", "[4,6]", "[5,7]", "[6,8]"],
            },
            "arrival_rate": {
                "x_name": "Average Arrival Rate of Requests",
                "x_values": ["5", "6", "7", "8", "9", "10", "11", "12"],
            },
        }
        metrics = [
            ("Average Response Delay", "ms", 300.0),
            ("Total Energy Consumption", "J", 12.0),
            ("Service-Chain Operational Cost", "cost_unit", 700.0),
        ]
        algorithms = [
            ("LyHAM-CO", 0.78),
            ("GMDA-RMPR-Myopic", 1.00),
            ("PDRS-Myopic", 1.05),
            ("FFD-Myopic", 1.10),
        ]

        fieldnames = [
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
        for group, spec in groups.items():
            for x_index, x_value in enumerate(spec["x_values"]):
                for metric, unit, base_value in metrics:
                    for algorithm, multiplier in algorithms:
                        rows.append(
                            {
                                "figure_group": group,
                                "metric": metric,
                                "x_name": spec["x_name"],
                                "x_value": x_value,
                                "algorithm": algorithm,
                                "value": str((base_value + x_index * 7.0) * multiplier),
                                "unit": unit,
                                "source": "figure_sweep_pipeline_run",
                                "run_id": f"{group}_{x_index}",
                                "seed": "mean",
                                "time_slots": "100",
                                "notes": "unit",
                            }
                        )

        with figure_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        validation_manifest.write_text(
            json.dumps(
                {
                    "run_id": "c1_c9_unit",
                    "figure_row_count": len(rows),
                    "raw_csv_count": 380,
                    "raw_row_count": 38000,
                    "failed_raw_rows": 0,
                    "bad_key_numeric_cells": 0,
                }
            ),
            encoding="utf-8",
        )

        manifest_path = generate_c1_c9_package(
            figure_csv=figure_csv,
            validation_manifest=validation_manifest,
            output_dir=out_dir,
            run_id="c1_c9_unit",
            decision="unit-test C1-C9 package",
        )

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        pngs = sorted((out_dir / "figures").glob("*.png"))
        non_pngs = sorted((out_dir / "figures").glob("*.*"))

        self.assertEqual(manifest["figure_format"], "png-only")
        self.assertEqual(manifest["figure_count"], 9)
        self.assertEqual(manifest["row_count"], 228)
        self.assertEqual(manifest["raw_csv_count"], 380)
        self.assertEqual(manifest["raw_row_count"], 38000)
        self.assertEqual(manifest["failed_raw_rows"], 0)
        self.assertEqual(manifest["bad_key_numeric_cells"], 0)
        self.assertEqual(len(pngs), 9)
        self.assertEqual(non_pngs, pngs)
        self.assertEqual(
            {path.name for path in pngs},
            {
                "delay_chain_unit.png",
                "energy_chain_unit.png",
                "cost_chain_unit.png",
                "arrival_ratedelay_unit.png",
                "arrival_rateenergy_unit.png",
                "arrival_ratecost_unit.png",
                "edgedelay_unit.png",
                "edgeenergy_unit.png",
                "edgecost_unit.png",
            },
        )
        self.assertTrue(all(record["name"].endswith(".png") for record in manifest["figures"]))
        self.assertTrue((out_dir / "figure_data" / figure_csv.name).exists())
        self.assertTrue((out_dir / "figure_data" / validation_manifest.name).exists())
        self.assertIn("source_figure_csv_sha256", manifest)
        self.assertIn("source_validation_manifest_sha256", manifest)
        self.assertIn("advantage_summary", manifest)


if __name__ == "__main__":
    unittest.main()
