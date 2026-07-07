"""Assemble a reproducible PNG-only paper-ready figure package."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Dict, Mapping

from PIL import Image


FIGURE_NAME_MAP: Dict[str, str] = {
    "average_response_delay_1000_slots.png": "Fig-A1_average_response_delay_1000_slots.png",
    "total_energy_consumption_1000_slots.png": "Fig-A2_total_energy_consumption_1000_slots.png",
    "service_chain_operational_cost_1000_slots.png": "Fig-A3_service_chain_operational_cost_1000_slots.png",
    "average_virtual_energy_queue_1000_slots.png": "Fig-Q1_average_virtual_energy_queue_1000_slots.png",
    "average_virtual_delay_queue_1000_slots.png": "Fig-Q2_average_virtual_delay_queue_1000_slots.png",
    "offloading_ratio_average_response_delay.png": "Fig-O1_offloading_ratio_average_response_delay.png",
    "offloading_ratio_total_energy_consumption.png": "Fig-O2_offloading_ratio_total_energy_consumption.png",
    "offloading_ratio_service_chain_operational_cost.png": "Fig-O3_offloading_ratio_service_chain_operational_cost.png",
    "edgedelay_unit.png": "Fig-C1_edge_nodes_average_response_delay.png",
    "edgeenergy_unit.png": "Fig-C2_edge_nodes_total_energy_consumption.png",
    "edgecost_unit.png": "Fig-C3_edge_nodes_service_chain_operational_cost.png",
    "delay_chain_unit.png": "Fig-C4_chain_length_average_response_delay.png",
    "energy_chain_unit.png": "Fig-C5_chain_length_total_energy_consumption.png",
    "cost_chain_unit.png": "Fig-C6_chain_length_service_chain_operational_cost.png",
    "arrival_ratedelay_unit.png": "Fig-C7_arrival_rate_average_response_delay.png",
    "arrival_rateenergy_unit.png": "Fig-C8_arrival_rate_total_energy_consumption.png",
    "arrival_ratecost_unit.png": "Fig-C9_arrival_rate_service_chain_operational_cost.png",
    "v_sensitivity_template.png": "Fig-V1_control_parameter_v_sensitivity.png",
    "learning_rate_training_loss_template.png": "Fig-LR1_learning_rate_training_loss_template.png",
}


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _image_record(path: Path, root: Path, source_key: str, source_path: Path) -> Dict[str, object]:
    with Image.open(path) as image:
        extrema = image.convert("RGB").getextrema()
        nonblank = any(low != high for low, high in extrema)
        width, height = image.size
    return {
        "name": path.name,
        "source_package": source_key,
        "source_path": str(source_path),
        "path": str(path),
        "relative_path": str(path.relative_to(root)),
        "length": path.stat().st_size,
        "sha256": sha256_file(path),
        "dimensions": {"width": width, "height": height},
        "nonblank": nonblank,
    }


def _data_record(path: Path, root: Path, source_key: str, source_path: Path) -> Dict[str, object]:
    return {
        "name": path.name,
        "source_package": source_key,
        "source_path": str(source_path),
        "path": str(path),
        "relative_path": str(path.relative_to(root)),
        "length": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _load_source_manifest(package: Path) -> Dict[str, object]:
    manifest = package / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"source package manifest not found: {manifest}")
    return json.loads(manifest.read_text(encoding="utf-8"))


def _final_figure_name(source_key: str, source_name: str) -> str:
    return FIGURE_NAME_MAP.get(source_name, f"{source_key}_{source_name}")


def _copy_source_package(
    source_key: str,
    source_package: Path,
    output_dir: Path,
    figures_dir: Path,
    data_dir: Path,
) -> tuple[Dict[str, object], list[Dict[str, object]], list[Dict[str, object]]]:
    source_package = Path(source_package)
    manifest_path = source_package / "manifest.json"
    manifest = _load_source_manifest(source_package)
    source_figures = source_package / "figures"
    if not source_figures.exists():
        raise FileNotFoundError(f"source figures directory not found: {source_figures}")

    figure_files = sorted(path for path in source_figures.iterdir() if path.is_file())
    non_png = [path.name for path in figure_files if path.suffix.lower() != ".png"]
    if non_png:
        raise ValueError(f"source package {source_key} contains non-PNG figure files: {non_png}")

    figure_records: list[Dict[str, object]] = []
    for source_figure in figure_files:
        target = figures_dir / _final_figure_name(source_key, source_figure.name)
        if target.exists():
            raise FileExistsError(f"duplicate final figure name: {target.name}")
        shutil.copy2(source_figure, target)
        figure_records.append(_image_record(target, output_dir, source_key, source_figure))

    data_records: list[Dict[str, object]] = []
    target_manifest = data_dir / f"{source_key}_source_package_manifest.json"
    shutil.copy2(manifest_path, target_manifest)
    data_records.append(_data_record(target_manifest, output_dir, source_key, manifest_path))

    source_data_dir = source_package / "figure_data"
    if source_data_dir.exists():
        for source_data in sorted(path for path in source_data_dir.rglob("*") if path.is_file()):
            target = data_dir / f"{source_key}_{source_data.name}"
            if target.exists():
                raise FileExistsError(f"duplicate final figure_data name: {target.name}")
            shutil.copy2(source_data, target)
            data_records.append(_data_record(target, output_dir, source_key, source_data))

    source_record = {
        "path": str(source_package),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "figure_count": len(figure_files),
        "figure_format": manifest.get("figure_format", ""),
    }
    return source_record, figure_records, data_records


def assemble_paper_ready_package(
    source_packages: Mapping[str, Path | str],
    output_dir: Path | str,
    decision: str,
) -> Path:
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    figures_dir = output_dir / "figures"
    data_dir = output_dir / "figure_data"
    figures_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    source_records: Dict[str, Dict[str, object]] = {}
    figure_records: list[Dict[str, object]] = []
    data_records: list[Dict[str, object]] = []
    for source_key, source_package in source_packages.items():
        source_record, source_figures, source_data = _copy_source_package(
            source_key=str(source_key),
            source_package=Path(source_package),
            output_dir=output_dir,
            figures_dir=figures_dir,
            data_dir=data_dir,
        )
        source_records[str(source_key)] = source_record
        figure_records.extend(source_figures)
        data_records.extend(source_data)

    final_files = [path for path in figures_dir.iterdir() if path.is_file()]
    figures_dir_png_only = all(path.suffix.lower() == ".png" for path in final_files)
    all_figures_nonblank = all(bool(record["nonblank"]) for record in figure_records)

    readme = output_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Paper-Ready PNG Figure Package",
                "",
                f"Decision: {decision}",
                "",
                f"Figure count: {len(figure_records)}",
                "Figures directory contains PNG files only.",
                "CSV/JSON provenance is stored under figure_data/.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "package": str(output_dir),
        "decision": decision,
        "figure_format": "png-only",
        "figure_count": len(figure_records),
        "figures_dir_png_only": figures_dir_png_only,
        "all_figures_nonblank": all_figures_nonblank,
        "source_packages": source_records,
        "figures": sorted(figure_records, key=lambda item: str(item["name"])),
        "figure_data": sorted(data_records, key=lambda item: str(item["name"])),
        "readme_sha256": sha256_file(readme),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if not figures_dir_png_only:
        raise ValueError("final figures directory is not PNG-only")
    if not all_figures_nonblank:
        raise ValueError("one or more final PNG figures are blank")
    return manifest_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Assemble a PNG-only paper-ready figure package.")
    parser.add_argument("--normal-package", type=Path, required=True)
    parser.add_argument("--c1-c9-package", type=Path, required=True)
    parser.add_argument("--v-package", type=Path, required=True)
    parser.add_argument("--lr-package", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--decision", required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    manifest_path = assemble_paper_ready_package(
        source_packages={
            "normal": args.normal_package,
            "c1_c9": args.c1_c9_package,
            "v": args.v_package,
            "lr": args.lr_package,
        },
        output_dir=args.output_dir,
        decision=args.decision,
    )
    print(json.dumps({"manifest": str(manifest_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
