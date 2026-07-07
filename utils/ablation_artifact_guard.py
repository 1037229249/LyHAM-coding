"""
实验产物清理守卫
只删除不可引用的C4/主实验产物，删除前写manifest便于审计。
"""
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List


def _safe_run_id_from_meta(path: Path) -> str:
    stem = path.stem
    prefix = "ablation_run_meta_"
    return stem[len(prefix):] if stem.startswith(prefix) else stem


def _load_meta(path: Path) -> Dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_load_error": str(exc)}


def scan_runs(base_dir: Path) -> Dict[str, Dict]:
    """扫描raw/summary/meta，按run_id归并产物"""
    base_dir = Path(base_dir).resolve()
    runs: Dict[str, Dict] = {}
    raw_root = base_dir / "raw"
    if raw_root.exists():
        for raw_dir in raw_root.iterdir():
            if raw_dir.is_dir():
                runs.setdefault(raw_dir.name, {"run_id": raw_dir.name})["raw_dir"] = str(raw_dir)

    summary_root = base_dir / "summary"
    if summary_root.exists():
        for meta_path in summary_root.glob("ablation_run_meta_*.json"):
            run_id = _safe_run_id_from_meta(meta_path)
            row = runs.setdefault(run_id, {"run_id": run_id})
            row["meta_path"] = str(meta_path)
            row["meta"] = _load_meta(meta_path)
        for summary_path in summary_root.glob("ablation_summary_*.csv"):
            run_id = summary_path.stem.replace("ablation_summary_", "", 1)
            runs.setdefault(run_id, {"run_id": run_id})["summary_path"] = str(summary_path)
    return runs


def classify_run(run_record: Dict) -> str:
    """判断run是否可引用"""
    meta = run_record.get("meta")
    if not meta:
        return "missing_meta"
    if meta.get("_load_error"):
        return "invalid_meta"
    # 当前正文可引用口径以run_ablation写出的canonical gate为准。
    # 早期formal结果没有energy-hard门禁，即使main/claim为true，也只能算历史draft。
    if bool(meta.get("canonical_export_allowed")):
        return "formal_valid"
    if "raw_dir" in run_record and ("meta_path" not in run_record or "summary_path" not in run_record):
        return "partial"
    return "draft"


def _assert_inside(base_dir: Path, target: Path) -> Path:
    """删除前确认路径没有越界"""
    base_dir = base_dir.resolve()
    target = target.resolve()
    if target == base_dir or base_dir not in target.parents:
        raise ValueError(f"拒绝删除输出目录外路径: {target}")
    return target


def _delete_path(path: Path):
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _latest_diagnostic_run_id(runs: Dict[str, Dict]) -> str:
    """选择最近一轮不可引用诊断run，供清理时保留根因证据。"""
    candidates = []
    for run_id, record in runs.items():
        if classify_run(record) == "formal_valid":
            continue
        if "raw_dir" not in record or "meta_path" not in record or "summary_path" not in record:
            continue
        paths = [Path(record["meta_path"]), Path(record["summary_path"]), Path(record["raw_dir"])]
        newest_mtime = max(path.stat().st_mtime for path in paths if path.exists())
        candidates.append((newest_mtime, str(run_id)))
    if not candidates:
        return ""
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[-1][1]


def _collect_delete_items(base_dir: Path, runs: Dict[str, Dict],
                          keep_latest_diagnostic: bool = False) -> List[Dict]:
    """收集不可引用产物；可保留最近诊断run作为根因证据。"""
    items: List[Dict] = []
    keep_run_id = _latest_diagnostic_run_id(runs) if keep_latest_diagnostic else ""
    for run_id, record in sorted(runs.items()):
        status = classify_run(record)
        if status == "formal_valid" or str(run_id) == keep_run_id:
            continue
        reason = f"non_citable_{status}"
        for key in ["raw_dir", "meta_path", "summary_path"]:
            value = record.get(key)
            if value:
                items.append({"run_id": run_id, "path": value, "reason": reason})

    # draft表不具备正文引用资格，清理时统一删除。
    draft_table = base_dir / "tables" / "ablation_table_draft.tex"
    if draft_table.exists():
        items.append({"run_id": "", "path": str(draft_table), "reason": "draft_latex_table"})

    formal_exists = any(classify_run(record) == "formal_valid" for record in runs.values())
    if not formal_exists:
        for path in [
            base_dir / "summary" / "ablation_summary.csv",
            base_dir / "tables" / "ablation_table.tex",
            base_dir / "tables" / "solver_benchmark_table.tex",
        ]:
            if path.exists():
                items.append({"run_id": "", "path": str(path), "reason": "stale_canonical_without_valid_meta"})
    return items


def delete_non_citable_runs(base_dir: Path, dry_run: bool = False,
                            keep_latest_diagnostic: bool = False) -> Path:
    """写manifest并删除不可引用run；必要时保留最近诊断run。"""
    base_dir = Path(base_dir).resolve()
    runs = scan_runs(base_dir)
    items = _collect_delete_items(base_dir, runs, keep_latest_diagnostic=keep_latest_diagnostic)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_path = base_dir / "summary" / f"cleanup_manifest_{timestamp}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "base_dir": str(base_dir),
        "dry_run": bool(dry_run),
        "keep_latest_diagnostic": bool(keep_latest_diagnostic),
        "kept_latest_diagnostic_run_id": _latest_diagnostic_run_id(runs) if keep_latest_diagnostic else "",
        "items": items,
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if not dry_run:
        for item in items:
            target = _assert_inside(base_dir, Path(item["path"]))
            if target == manifest_path:
                continue
            _delete_path(target)
    return manifest_path



