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


class VTemplatePackageTest(unittest.TestCase):
    def test_generate_v_template_package_writes_single_png_with_template_provenance(self):
        sys.path.insert(0, str(UTILS_DIR))

        from generate_v_template_figure_package import generate_v_template_package

        base_dir = _workspace_test_tmp("v_template_package_unit")
        figure_csv = base_dir / "v_figure_data.csv"
        batch_meta = base_dir / "v_figure_data.meta.json"
        template_png = base_dir / "V.png"
        out_dir = base_dir / "package"

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
        metrics = [
            ("Average Virtual Energy Queue", "queue_length", [8, 28, 55, 85, 155, 205, 245, 285]),
            ("Average Virtual Delay Queue", "queue_length", [55, 155, 310, 560, 810, 1010, 1160, 1310]),
            ("Total Energy Consumption", "J", [7.10, 5.95, 5.30, 4.85, 4.65, 4.60, 4.58, 4.56]),
            ("Service-Chain Operational Cost", "cost_unit", [560, 520, 480, 445, 438, 432, 428, 426]),
            ("Average Response Delay", "ms", [388, 402, 422, 456, 535, 620, 710, 755]),
        ]
        v_values = [1, 5, 10, 20, 50, 100, 200, 500]
        rows = []
        for metric, unit, values in metrics:
            for v_value, value in zip(v_values, values):
                rows.append(
                    {
                        "figure_group": "V",
                        "metric": metric,
                        "x_name": "Control Parameter V",
                        "x_value": str(v_value),
                        "algorithm": "LyHAM-CO",
                        "value": str(value),
                        "unit": unit,
                        "source": "figure_sweep_pipeline_run",
                        "run_id": f"v_unit_{v_value}",
                        "seed": "mean",
                        "time_slots": "100",
                        "notes": "unit",
                    }
                )
        with figure_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        batch_meta.write_text(json.dumps({"completed_count": 8, "failed_count": 0}), encoding="utf-8")

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(12, 5), dpi=100)
        fig.savefig(template_png)
        plt.close(fig)

        manifest_path = generate_v_template_package(
            figure_csv=figure_csv,
            batch_meta=batch_meta,
            template_png=template_png,
            output_dir=out_dir,
            run_id="v_unit",
            decision="unit-test V package",
        )

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        pngs = sorted((out_dir / "figures").glob("*.png"))
        non_pngs = sorted((out_dir / "figures").glob("*.*"))

        self.assertEqual(manifest["figure_format"], "png-only")
        self.assertEqual(manifest["figure_count"], 1)
        self.assertEqual(manifest["template_dimensions"], {"width": 1200, "height": 500})
        self.assertEqual(manifest["v_values"], [float(value) for value in v_values])
        self.assertEqual(set(manifest["metrics"]), {metric for metric, _, _ in metrics})
        self.assertEqual(len(pngs), 1)
        self.assertEqual(non_pngs, pngs)
        self.assertTrue(manifest["figures"][0]["name"].endswith(".png"))
        self.assertIn("source_figure_csv_sha256", manifest)
        self.assertIn("source_batch_meta_sha256", manifest)
        self.assertIn("template_png_sha256", manifest)
        self.assertTrue((out_dir / "figure_data" / figure_csv.name).exists())
        self.assertTrue((out_dir / "figure_data" / batch_meta.name).exists())


if __name__ == "__main__":
    unittest.main()
