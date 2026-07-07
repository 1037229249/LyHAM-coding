"""Compare PDRS selected all-local rows against the LyHAM candidate frontier."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from statistics import mean


REPO_ROOT = Path(__file__).resolve().parents[1]
UTILS_DIR = REPO_ROOT / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from ablation_pair_actions import pair_action_hash  # noqa: E402


def _float(value, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _int(value, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_slots(path: Path) -> dict[int, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(slot["slot"]): slot for slot in payload.get("slots", [])}


def _split_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    text = str(value or "")
    if not text:
        return []
    if ";" in text:
        return [item for item in text.split(";") if item]
    if "," in text:
        return [item for item in text.split(",") if item]
    return [text]


def _server_counts(pair_ids: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for pair_id in pair_ids:
        server = pair_id.rsplit("@", 1)[-1] if "@" in pair_id else ""
        if not server:
            continue
        counts[server] = counts.get(server, 0) + 1
    return dict(sorted(counts.items()))


def _flow_server_map(pair_ids: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for pair_id in pair_ids:
        flow = pair_id.split(":", 1)[0]
        server = pair_id.rsplit("@", 1)[-1] if "@" in pair_id else ""
        if flow and server:
            mapping[flow] = server
    return dict(sorted(mapping.items()))


def _candidate_hash(row: dict) -> str:
    bits = str(row.get("repaired_pair_bits", ""))
    if not bits:
        return ""
    return pair_action_hash([int(ch) for ch in bits if ch in {"0", "1"}])


def _is_all_local(row: dict) -> bool:
    return (
        bool(row.get("feasible", False))
        and _int(row.get("local_pair_count")) > 0
        and _int(row.get("cloud_pair_count")) == 0
    )


def _rank_key(row: dict, field: str) -> tuple:
    return (
        not bool(row.get("feasible", False)),
        _float(row.get(field), float("inf")),
        _float(row.get("paper_dpp_score"), float("inf")),
        _int(row.get("eval_index")),
    )


def _best(rows: list[dict], field: str) -> dict:
    return min(rows, key=lambda row: _rank_key(row, field)) if rows else {}


def _brief_raw(row: dict) -> dict:
    local_ids = _split_list(row.get("active_local_pair_ids", ""))
    return {
        "delay_ms": _float(row.get("delay_ms")),
        "energy_j": _float(row.get("energy_j")),
        "cost": _float(row.get("cost")),
        "cost_topo": _float(row.get("cost_topo")),
        "cost_comp": _float(row.get("cost_comp")),
        "cost_comm": _float(row.get("cost_comm")),
        "energy_local_gpu_j": _float(row.get("energy_local_gpu_j")),
        "energy_comm_j": _float(row.get("energy_comm_j")),
        "energy_idle_replica_j": _float(row.get("energy_idle_replica_j")),
        "pair_action_dim": _int(row.get("pair_action_dim")),
        "pair_action_hash": str(row.get("pair_action_hash", "")),
        "repaired_pair_action_hash": str(row.get("repaired_pair_action_hash", "")),
        "pair_action_bits": str(row.get("pair_action_bits", "")),
        "routing_policy": str(row.get("routing_policy", "")),
        "placement_hash": str(row.get("placement_hash", "")),
        "routing_hash": str(row.get("routing_hash", "")),
        "slow_context_hash": str(row.get("slow_context_hash", "")),
        "selected_candidate_source": str(row.get("selected_candidate_source", "")),
        "resource_hint": str(row.get("resource_hint", "")),
        "resource_mode": str(row.get("resource_mode", "")),
        "active_local_pair_ids": local_ids,
        "active_local_server_counts": _server_counts(local_ids),
        "flow_server_map": _flow_server_map(local_ids),
    }


def _brief_candidate(row: dict) -> dict:
    local_ids = _split_list(row.get("active_local_pair_ids", []))
    resource = row.get("resource_summary", {}) if isinstance(row.get("resource_summary"), dict) else {}
    return {
        "eval_index": _int(row.get("eval_index")),
        "candidate_source": str(row.get("candidate_source", "")),
        "candidate_family": str(row.get("candidate_family", "")),
        "resource_hint": str(row.get("resource_hint", "")),
        "resource_mode": str(row.get("resource_mode", "")),
        "repaired_pair_bits": str(row.get("repaired_pair_bits", "")),
        "repaired_pair_hash": _candidate_hash(row),
        "local_pair_count": _int(row.get("local_pair_count")),
        "cloud_pair_count": _int(row.get("cloud_pair_count")),
        "delay_ms": _float(row.get("delay_ms")),
        "energy_j": _float(row.get("energy_j")),
        "cost": _float(row.get("cost")),
        "cost_topo": _float(row.get("cost_topo")),
        "cost_comp": _float(row.get("cost_comp")),
        "cost_comm": _float(row.get("cost_comm")),
        "paper_dpp_score": _float(row.get("paper_dpp_score")),
        "claim_score": _float(row.get("claim_score")),
        "energy_local_gpu_j": _float(row.get("energy_local_gpu_j")),
        "energy_comm_j": _float(row.get("energy_comm_j")),
        "energy_idle_replica_j": _float(row.get("energy_idle_replica_j")),
        "active_local_pair_ids": local_ids,
        "active_local_server_counts": _server_counts(local_ids),
        "flow_server_map": _flow_server_map(local_ids),
        "resource_summary": resource,
    }


def _nearest_selected(raw_row: dict, candidates: list[dict]) -> dict:
    if not candidates:
        return {}
    source = str(raw_row.get("selected_candidate_source", ""))
    raw_cost = _float(raw_row.get("cost"))
    raw_delay = _float(raw_row.get("delay_ms"))
    raw_energy = _float(raw_row.get("energy_j"))

    def score(row: dict) -> tuple:
        return (
            str(row.get("candidate_source", "")) != source,
            abs(_float(row.get("cost")) - raw_cost)
            + abs(_float(row.get("delay_ms")) - raw_delay)
            + abs(_float(row.get("energy_j")) - raw_energy),
            _int(row.get("eval_index")),
        )

    return min(candidates, key=score)


def _overlap(left: list[str], right: list[str]) -> dict:
    left_set = set(left)
    right_set = set(right)
    exact = left_set & right_set
    left_flow_servers = _flow_server_map(left)
    right_flow_servers = _flow_server_map(right)
    common_flows = set(left_flow_servers) & set(right_flow_servers)
    same_server_flows = {
        flow
        for flow in common_flows
        if left_flow_servers.get(flow) == right_flow_servers.get(flow)
    }
    return {
        "exact_pair_id_overlap": len(exact),
        "exact_pair_id_overlap_ratio_vs_pdrs": len(exact) / max(len(left_set), 1),
        "common_flow_count": len(common_flows),
        "same_server_flow_count": len(same_server_flows),
        "same_server_flow_ratio_vs_common": len(same_server_flows) / max(len(common_flows), 1),
        "same_server_flows": sorted(same_server_flows),
    }


def build_report(args: argparse.Namespace) -> dict:
    pdrs_rows = {int(row["slot"]): row for row in _rows(args.pdrs_raw)}
    lyham_rows = {int(row["slot"]): row for row in _rows(args.lyham_raw)}
    pdrs_slots = _load_slots(args.pdrs_candidates)
    lyham_slots = _load_slots(args.lyham_candidates)
    slot_reports = []
    for slot in range(args.slot_start, args.slot_end + 1):
        pdrs_raw = pdrs_rows[slot]
        lyham_raw = lyham_rows[slot]
        lyham_records = list(lyham_slots[slot].get("candidate_records", []))
        pdrs_records = list(pdrs_slots[slot].get("candidate_records", []))
        lyham_all_local = [row for row in lyham_records if _is_all_local(row)]
        pdrs_selected = _nearest_selected(pdrs_raw, pdrs_records)
        lyham_selected = _nearest_selected(lyham_raw, lyham_records)
        lyham_best_cost = _best(lyham_all_local, "cost")
        lyham_best_comm = _best(lyham_all_local, "cost_comm")
        lyham_best_dpp = _best(lyham_all_local, "paper_dpp_score")
        target_hash = str(pdrs_raw.get("repaired_pair_action_hash", ""))
        hash_matches = [
            row for row in lyham_records
            if _candidate_hash(row) == target_hash
        ]
        pdrs_local_ids = _split_list(pdrs_raw.get("active_local_pair_ids", ""))
        lyham_best_ids = _split_list(lyham_best_cost.get("active_local_pair_ids", []))
        lyham_selected_ids = _split_list(lyham_raw.get("active_local_pair_ids", ""))
        slot_reports.append({
            "slot": slot,
            "pdrs_raw": _brief_raw(pdrs_raw),
            "lyham_raw": _brief_raw(lyham_raw),
            "pdrs_selected_candidate": _brief_candidate(pdrs_selected),
            "lyham_selected_candidate": _brief_candidate(lyham_selected),
            "lyham_best_all_local_by_cost": _brief_candidate(lyham_best_cost),
            "lyham_best_all_local_by_comm_cost": _brief_candidate(lyham_best_comm),
            "lyham_best_all_local_by_dpp": _brief_candidate(lyham_best_dpp),
            "lyham_candidate_count": len(lyham_records),
            "lyham_all_local_candidate_count": len(lyham_all_local),
            "lyham_target_hash_match_count": len(hash_matches),
            "pair_action_dim_equal": _int(pdrs_raw.get("pair_action_dim")) == _int(lyham_raw.get("pair_action_dim")),
            "slow_context_hash_equal": str(pdrs_raw.get("slow_context_hash", "")) == str(lyham_raw.get("slow_context_hash", "")),
            "placement_hash_equal": str(pdrs_raw.get("placement_hash", "")) == str(lyham_raw.get("placement_hash", "")),
            "routing_hash_equal": str(pdrs_raw.get("routing_hash", "")) == str(lyham_raw.get("routing_hash", "")),
            "pdrs_minus_lyham_selected": {
                "delay_ms": _float(pdrs_raw.get("delay_ms")) - _float(lyham_raw.get("delay_ms")),
                "energy_j": _float(pdrs_raw.get("energy_j")) - _float(lyham_raw.get("energy_j")),
                "cost": _float(pdrs_raw.get("cost")) - _float(lyham_raw.get("cost")),
                "cost_comp": _float(pdrs_raw.get("cost_comp")) - _float(lyham_raw.get("cost_comp")),
                "cost_comm": _float(pdrs_raw.get("cost_comm")) - _float(lyham_raw.get("cost_comm")),
            },
            "pdrs_minus_lyham_best_all_local_cost": {
                "delay_ms": _float(pdrs_raw.get("delay_ms")) - _float(lyham_best_cost.get("delay_ms")),
                "energy_j": _float(pdrs_raw.get("energy_j")) - _float(lyham_best_cost.get("energy_j")),
                "cost": _float(pdrs_raw.get("cost")) - _float(lyham_best_cost.get("cost")),
                "cost_comp": _float(pdrs_raw.get("cost_comp")) - _float(lyham_best_cost.get("cost_comp")),
                "cost_comm": _float(pdrs_raw.get("cost_comm")) - _float(lyham_best_cost.get("cost_comm")),
            },
            "pdrs_vs_lyham_best_all_local_overlap": _overlap(pdrs_local_ids, lyham_best_ids),
            "pdrs_vs_lyham_selected_overlap": _overlap(pdrs_local_ids, lyham_selected_ids),
        })

    def values(path: list[str]) -> list[float]:
        out = []
        for row in slot_reports:
            current = row
            for key in path:
                current = current[key]
            out.append(float(current))
        return out

    summary = {
        "slot_start": args.slot_start,
        "slot_end": args.slot_end,
        "slot_count": len(slot_reports),
        "pair_action_dim_equal_slots": sum(1 for row in slot_reports if row["pair_action_dim_equal"]),
        "slow_context_hash_equal_slots": sum(1 for row in slot_reports if row["slow_context_hash_equal"]),
        "placement_hash_equal_slots": sum(1 for row in slot_reports if row["placement_hash_equal"]),
        "routing_hash_equal_slots": sum(1 for row in slot_reports if row["routing_hash_equal"]),
        "lyham_target_hash_match_total": sum(row["lyham_target_hash_match_count"] for row in slot_reports),
        "lyham_all_local_candidate_count_mean": mean(values(["lyham_all_local_candidate_count"])),
        "mean_pdrs_minus_lyham_selected": {
            "delay_ms": mean(values(["pdrs_minus_lyham_selected", "delay_ms"])),
            "energy_j": mean(values(["pdrs_minus_lyham_selected", "energy_j"])),
            "cost": mean(values(["pdrs_minus_lyham_selected", "cost"])),
            "cost_comp": mean(values(["pdrs_minus_lyham_selected", "cost_comp"])),
            "cost_comm": mean(values(["pdrs_minus_lyham_selected", "cost_comm"])),
        },
        "mean_pdrs_minus_lyham_best_all_local_cost": {
            "delay_ms": mean(values(["pdrs_minus_lyham_best_all_local_cost", "delay_ms"])),
            "energy_j": mean(values(["pdrs_minus_lyham_best_all_local_cost", "energy_j"])),
            "cost": mean(values(["pdrs_minus_lyham_best_all_local_cost", "cost"])),
            "cost_comp": mean(values(["pdrs_minus_lyham_best_all_local_cost", "cost_comp"])),
            "cost_comm": mean(values(["pdrs_minus_lyham_best_all_local_cost", "cost_comm"])),
        },
        "mean_same_server_flow_ratio_pdrs_vs_lyham_best_all_local": mean(
            values(["pdrs_vs_lyham_best_all_local_overlap", "same_server_flow_ratio_vs_common"])
        ),
        "mean_exact_pair_overlap_ratio_pdrs_vs_lyham_best_all_local": mean(
            values(["pdrs_vs_lyham_best_all_local_overlap", "exact_pair_id_overlap_ratio_vs_pdrs"])
        ),
    }
    return {
        "inputs": {
            "pdrs_raw": str(args.pdrs_raw),
            "lyham_raw": str(args.lyham_raw),
            "pdrs_candidates": str(args.pdrs_candidates),
            "lyham_candidates": str(args.lyham_candidates),
        },
        "summary": summary,
        "slots": slot_reports,
    }


def _fmt(number: float) -> str:
    return f"{number:.6f}"


def markdown(report: dict) -> str:
    summary = report["summary"]
    lines = [
        "# PDRS vs LyHAM Frontier Diagnostic",
        "",
        "## Inputs",
        "",
    ]
    for key, value in report["inputs"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend([
        "",
        "## Summary",
        "",
        f"- Slots: `{summary['slot_start']}-{summary['slot_end']}` (`{summary['slot_count']}` slots)",
        f"- Equal pair-action dimensions: `{summary['pair_action_dim_equal_slots']}/{summary['slot_count']}`",
        f"- Equal slow-context hashes: `{summary['slow_context_hash_equal_slots']}/{summary['slot_count']}`",
        f"- Equal placement hashes: `{summary['placement_hash_equal_slots']}/{summary['slot_count']}`",
        f"- Equal routing hashes: `{summary['routing_hash_equal_slots']}/{summary['slot_count']}`",
        f"- LyHAM target PDRS-hash matches: `{summary['lyham_target_hash_match_total']}`",
        f"- Mean LyHAM all-local candidates per slot: `{_fmt(summary['lyham_all_local_candidate_count_mean'])}`",
        "",
        "Mean PDRS minus LyHAM selected:",
        "",
    ])
    for key, value in summary["mean_pdrs_minus_lyham_selected"].items():
        lines.append(f"- {key}: `{_fmt(value)}`")
    lines.extend(["", "Mean PDRS minus LyHAM best all-local-by-cost candidate:", ""])
    for key, value in summary["mean_pdrs_minus_lyham_best_all_local_cost"].items():
        lines.append(f"- {key}: `{_fmt(value)}`")
    lines.extend([
        "",
        f"- Mean same-server flow ratio versus LyHAM best all-local: `{_fmt(summary['mean_same_server_flow_ratio_pdrs_vs_lyham_best_all_local'])}`",
        f"- Mean exact pair-id overlap ratio versus LyHAM best all-local: `{_fmt(summary['mean_exact_pair_overlap_ratio_pdrs_vs_lyham_best_all_local'])}`",
        "",
        "## Slot Table",
        "",
        "| Slot | PDRS pair dim | LyHAM pair dim | PDRS hash matches in LyHAM | PDRS cost | LyHAM selected cost | LyHAM best all-local cost | PDRS-LyHAM best cost | PDRS-LyHAM best comm cost | Same-server flow ratio |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in report["slots"]:
        pdrs = row["pdrs_raw"]
        lyham = row["lyham_raw"]
        best = row["lyham_best_all_local_by_cost"]
        delta = row["pdrs_minus_lyham_best_all_local_cost"]
        overlap = row["pdrs_vs_lyham_best_all_local_overlap"]
        lines.append(
            "| {slot} | {pdim} | {ldim} | {matches} | {pcost} | {lcost} | {bcost} | {dcost} | {dcomm} | {same} |".format(
                slot=row["slot"],
                pdim=pdrs["pair_action_dim"],
                ldim=lyham["pair_action_dim"],
                matches=row["lyham_target_hash_match_count"],
                pcost=_fmt(pdrs["cost"]),
                lcost=_fmt(lyham["cost"]),
                bcost=_fmt(best["cost"]),
                dcost=_fmt(delta["cost"]),
                dcomm=_fmt(delta["cost_comm"]),
                same=_fmt(overlap["same_server_flow_ratio_vs_common"]),
            )
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- `0` pair-dimension/hash/slow-context matches means the PDRS row is not merely a missing final selector choice inside the current LyHAM fast-action pool.",
        "- PDRS keeps the same flows local but places/routes them through different AI servers; the lower cost is dominated by communication-cost reduction, while compute cost and energy are higher.",
        "- The next root-cause boundary is the slow-context placement/routing frontier exposed to LyHAM candidate generation, not a selector-only threshold.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdrs-raw", required=True, type=Path)
    parser.add_argument("--lyham-raw", required=True, type=Path)
    parser.add_argument("--pdrs-candidates", required=True, type=Path)
    parser.add_argument("--lyham-candidates", required=True, type=Path)
    parser.add_argument("--slot-start", type=int, default=10)
    parser.add_argument("--slot-end", type=int, default=19)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--markdown", required=True, type=Path)
    args = parser.parse_args()
    report = build_report(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.markdown.write_text(markdown(report), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
