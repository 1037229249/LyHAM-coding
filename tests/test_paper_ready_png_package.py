import json
import shutil
import sys
import unittest
from pathlib import Path

from PIL import Image


REPO_DIR = Path(__file__).resolve().parents[1]
UTILS_DIR = REPO_DIR / "utils"


def _workspace_test_tmp(name: str) -> Path:
    path = REPO_DIR / "_test_artifacts" / name
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_nonblank_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (2, 1), "white")
    image.putpixel((0, 0), (0, 0, 0))
    image.save(path)


def _source_package(root: Path, key: str, figure_name: str) -> Path:
    package = root / key
    figures = package / "figures"
    data = package / "figure_data"
    _write_nonblank_png(figures / figure_name)
    data.mkdir(parents=True, exist_ok=True)
    (data / f"{key}_metrics.csv").write_text("metric,value\ncost,1\n", encoding="utf-8")
    manifest = {
        "package": str(package),
        "figure_format": "png-only",
        "figure_count": 1,
        "figures": [{"name": figure_name}],
    }
    (package / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return package


class PaperReadyPngPackageTest(unittest.TestCase):
    def test_assemble_package_copies_pngs_data_and_manifest(self):
        sys.path.insert(0, str(UTILS_DIR))

        from assemble_paper_ready_png_package import assemble_paper_ready_package

        base_dir = _workspace_test_tmp("paper_ready_png_package_unit")
        normal = _source_package(base_dir, "normal", "average_response_delay_1000_slots.png")
        c1_c9 = _source_package(base_dir, "c1_c9", "edgedelay_unit.png")
        v = _source_package(base_dir, "v", "v_sensitivity_template.png")
        lr = _source_package(base_dir, "lr", "learning_rate_training_loss_template.png")
        out_dir = base_dir / "paper_ready"

        manifest_path = assemble_paper_ready_package(
            source_packages={
                "normal": normal,
                "c1_c9": c1_c9,
                "v": v,
                "lr": lr,
            },
            output_dir=out_dir,
            decision="unit-test final package",
        )

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        pngs = sorted((out_dir / "figures").glob("*"))
        data_files = sorted((out_dir / "figure_data").glob("*"))

        self.assertEqual(manifest["figure_format"], "png-only")
        self.assertEqual(manifest["figure_count"], 4)
        self.assertTrue(manifest["figures_dir_png_only"])
        self.assertTrue(manifest["all_figures_nonblank"])
        self.assertEqual(pngs, sorted((out_dir / "figures").glob("*.png")))
        self.assertTrue((out_dir / "README.md").exists())
        self.assertIn("unit-test final package", (out_dir / "README.md").read_text(encoding="utf-8"))
        self.assertIn("normal", manifest["source_packages"])
        self.assertIn("normal_source_package_manifest.json", {path.name for path in data_files})
        self.assertIn("normal_normal_metrics.csv", {path.name for path in data_files})
        self.assertIn("Fig-A1_average_response_delay_1000_slots.png", {path.name for path in pngs})
        self.assertTrue(all(record["name"].endswith(".png") for record in manifest["figures"]))

    def test_assemble_package_rejects_non_png_source_figure(self):
        sys.path.insert(0, str(UTILS_DIR))

        from assemble_paper_ready_png_package import assemble_paper_ready_package

        base_dir = _workspace_test_tmp("paper_ready_png_package_non_png_unit")
        normal = _source_package(base_dir, "normal", "average_response_delay_1000_slots.png")
        (normal / "figures" / "bad.svg").write_text("<svg></svg>", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "non-PNG"):
            assemble_paper_ready_package(
                source_packages={"normal": normal},
                output_dir=base_dir / "paper_ready",
                decision="reject non-png",
            )


if __name__ == "__main__":
    unittest.main()
