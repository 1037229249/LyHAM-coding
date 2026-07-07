"""
消融实验导出工具
统一导出raw CSV、summary CSV和紧凑LaTeX表格。
"""
import csv
from math import isfinite
from pathlib import Path
from typing import Iterable, List

from ablation_metrics import AlgorithmSummary, SlotResult
from ablation_config import ABLATION_MAIN_ALGORITHMS, ABLATION_SOLVER_ALGORITHMS, NORMAL_MAIN_ALGORITHMS


RAW_FIELDS = [
    "slot", "seed", "algorithm", "slow_policy", "fast_controller", "status",
    "failure_reason", "delay_ms", "energy_j", "cost", "avg_y", "avg_z",
    "dpp_score", "legacy_reward", "feasible", "local_count", "cloud_count",
    "forced_cloud_count", "decision_time_ms", "slow_context_reused", "model_path",
    "paper_dpp_score", "scaled_energy_sum", "scaled_delay_burden_sum",
    "candidate_count", "selected_candidate_rank", "replay_written",
    "online_update_step", "solver_gap_vs_lycd", "p95_decision_time_ms",
    "routing_entropy", "slow_profile_reused", "retry_count",
    "action_dim", "pair_action_dim", "routing_policy",
    "cost_topo", "cost_comp", "cost_comm", "cost_component_consistent",
    "active_ai_energy_j", "system_active_ai_chain_energy_j", "energy_scope",
    "formal_gate_passed", "model_mutated",
    "v_cost_term", "energy_queue_term", "delay_queue_term",
    "best_local_score", "best_cloud_score", "score_gap_local_vs_cloud",
    "local_candidate_feasible_count", "candidate_diversity",
    "all_cloud_candidate_count", "all_local_candidate_count",
    "routing_metric_consumed", "routing_delay_consumed",
    "routing_probability_mass", "mechanism_gate_passed",
    "energy_local_gpu_j", "energy_cloud_compute_j", "energy_comm_j",
    "energy_idle_replica_j", "action_hash", "pair_action_hash",
    "repaired_pair_action_hash", "best_energy_candidate_source",
    "best_energy_candidate_hash", "best_energy_candidate_energy_j",
    "best_energy_candidate_dpp_score", "best_energy_candidate_energy_gap",
    "best_energy_candidate_dpp_gap",
    "original_pair_action_hash", "repair_changed_pair_count", "repair_changed_ratio",
    "pair_action_bits", "selected_candidate_source",
    "active_local_pair_indices", "active_local_pair_ids", "active_local_pair_signature",
    "active_cloud_pair_indices", "active_cloud_pair_ids", "active_cloud_pair_signature",
    "local_pair_count", "cloud_pair_count", "repaired_candidate_diversity",
    "repaired_source_distribution", "best_repaired_uac_vs_myopic_dpp_gap",
    "uac_selected_source", "repaired_hamming_vs_reference",
    "reference_pair_action_bits", "reference_pair_action_hash",
    "reference_pair_action_source", "placement_hash", "routing_hash", "slow_context_hash",
    "claim_score", "is_pareto_candidate", "dpp_band_passed",
    "selected_by_dpp_or_claim_band",
    "per_pair_delta_delay", "per_pair_delta_energy", "per_pair_delta_cost",
    "resource_hint", "resource_queue_aware", "resource_queue_scale", "resource_mode",
    "resource_hint_collapsed",
    "candidate_source_score_summary", "candidate_source_family_count",
    "predicted_avg_y", "predicted_avg_z",
    "post_update_queue_drift_term", "post_update_queue_pressure_term",
    "post_update_queue_delta_term", "post_update_energy_queue_delta_term",
    "post_update_delay_queue_delta_term", "post_update_queue_drift_enabled",
    "tail_risk_candidate_source", "tail_risk_candidate_hash",
    "tail_risk_candidate_delay_ms", "tail_risk_candidate_energy_j",
    "tail_risk_candidate_cost", "tail_risk_candidate_claim_score",
    "tail_risk_candidate_dpp_score", "tail_risk_candidate_predicted_avg_y",
    "tail_risk_candidate_predicted_avg_z",
    "tail_risk_candidate_post_update_queue_drift_term",
    "tail_risk_candidate_upper_excess_score",
    "tail_risk_selected_upper_excess_score",
    "tail_risk_candidate_upper_excess_improvement",
    "tail_risk_best_relief_source", "tail_risk_best_relief_hash",
    "tail_risk_best_relief_delay_ms", "tail_risk_best_relief_energy_j",
    "tail_risk_best_relief_cost", "tail_risk_best_relief_claim_score",
    "tail_risk_best_relief_dpp_score", "tail_risk_best_relief_predicted_avg_y",
    "tail_risk_best_relief_predicted_avg_z",
    "tail_risk_best_relief_post_update_queue_drift_term",
    "tail_risk_best_relief_upper_excess_score",
    "tail_risk_best_relief_improvement",
    "tail_risk_best_relief_reject_reason",
    "energy_relief_candidate_source", "energy_relief_candidate_hash",
    "energy_relief_candidate_delay_ms", "energy_relief_candidate_energy_j",
    "energy_relief_candidate_cost", "energy_relief_candidate_claim_score",
    "energy_relief_candidate_dpp_score", "energy_relief_candidate_predicted_avg_y",
    "energy_relief_candidate_predicted_avg_z",
    "energy_relief_candidate_post_update_queue_drift_term",
    "energy_relief_candidate_energy_gain_j",
    "energy_relief_candidate_delay_regret_ms",
    "energy_relief_candidate_cost_regret",
    "energy_relief_candidate_dpp_regret",
    "energy_relief_best_lower_source", "energy_relief_best_lower_hash",
    "energy_relief_best_lower_delay_ms", "energy_relief_best_lower_energy_j",
    "energy_relief_best_lower_cost", "energy_relief_best_lower_claim_score",
    "energy_relief_best_lower_dpp_score", "energy_relief_best_lower_predicted_avg_y",
    "energy_relief_best_lower_predicted_avg_z",
    "energy_relief_best_lower_post_update_queue_drift_term",
    "energy_relief_best_lower_energy_gain_j",
    "energy_relief_best_lower_delay_regret_ms",
    "energy_relief_best_lower_cost_regret",
    "energy_relief_best_lower_dpp_regret",
    "energy_relief_best_lower_reject_reason",
]


SUMMARY_FIELDS = [
    "algorithm", "seed", "n_valid_slots", "n_failed_slots", "delay_mean",
    "delay_std", "energy_mean", "energy_std", "cost_mean", "cost_std",
    "avg_y_mean", "avg_y_std", "avg_z_mean", "avg_z_std", "dpp_score_mean",
    "dpp_score_std", "decision_time_mean_ms", "decision_time_p95_ms",
    "feasible_ratio", "valid", "paper_dpp_score_mean", "paper_dpp_score_std",
    "scaled_energy_sum_mean", "scaled_energy_sum_std",
    "scaled_delay_burden_sum_mean", "scaled_delay_burden_sum_std",
    "candidate_count_mean", "selected_candidate_rank_mean",
    "solver_gap_vs_lycd_mean", "routing_entropy_mean", "retry_count_mean",
    "action_dim_mean", "pair_action_dim_mean", "cost_topo_mean",
    "cost_comp_mean", "cost_comm_mean", "active_ai_energy_mean",
    "system_active_ai_chain_energy_mean", "cost_component_consistent_ratio",
    "formal_gate_passed", "valid_seed_count", "config_hash",
    "v_cost_term_mean", "energy_queue_term_mean", "delay_queue_term_mean",
    "best_local_score_mean", "best_cloud_score_mean",
    "score_gap_local_vs_cloud_mean", "local_candidate_feasible_count_mean",
    "candidate_diversity_mean", "all_cloud_ratio",
    "all_cloud_candidate_count_mean", "all_local_candidate_count_mean",
    "routing_metric_consumed_ratio", "routing_delay_consumed_ratio",
    "routing_probability_mass_mean", "mechanism_gate_passed",
    "claim_supported", "energy_scope_gate_passed",
    "energy_local_gpu_mean", "energy_cloud_compute_mean",
    "energy_comm_mean", "energy_idle_replica_mean",
    "local_pair_count_mean", "cloud_pair_count_mean",
    "repaired_candidate_diversity_mean", "best_repaired_uac_vs_myopic_dpp_gap_mean",
    "best_energy_candidate_energy_mean", "best_energy_candidate_dpp_mean",
    "best_energy_candidate_energy_gap_mean", "best_energy_candidate_dpp_gap_mean",
    "uac_selected_source_ratio", "repaired_hamming_vs_reference_mean",
    "selected_action_diversity_mean", "uac_source_family_count_mean"
]


def _write_csv(path: Path, fieldnames: List[str], rows: Iterable[dict]) -> Path:
    """写入CSV文件"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: _sanitize_csv_value(row.get(key, ""))
                for key in fieldnames
            })
    return path


def _sanitize_csv_value(value):
    """CSV里不写NaN，失败项留空由status和failure_reason解释"""
    if isinstance(value, float) and not isfinite(value):
        return ""
    return value


def _sanitize_csv_row(row: dict) -> dict:
    """清理CSV行"""
    return {key: _sanitize_csv_value(row.get(key, "")) for key in row.keys()}


def export_raw_slot_results(base_dir: Path, run_id: str, algorithm: str, seed: int,
                            slot_results: Iterable[SlotResult]) -> Path:
    """导出单算法单seed的时隙级结果"""
    safe_algorithm = algorithm.replace("/", "_")
    path = base_dir / "raw" / run_id / safe_algorithm / f"seed_{seed}_per_slot.csv"
    return _write_csv(path, RAW_FIELDS, [row.to_dict() for row in slot_results])


def export_summary_csv(base_dir: Path, summaries: Iterable[AlgorithmSummary],
                       run_id: str = None, canonical: bool = True) -> Path:
    """导出汇总CSV"""
    filename = "ablation_summary.csv" if canonical else f"ablation_summary_{run_id}.csv"
    path = base_dir / "summary" / filename
    return _write_csv(path, SUMMARY_FIELDS, [row.to_dict() for row in summaries])


def _format_mean_std(mean_value: float, std_value: float) -> str:
    """格式化均值和标准差"""
    try:
        if mean_value != mean_value:
            return "--"
        return f"{mean_value:.4f} $\\pm$ {std_value:.4f}"
    except Exception:
        return "--"


def _format_plain(value: float) -> str:
    """格式化单个数值"""
    try:
        if value != value or not isfinite(float(value)):
            return "--"
        return f"{float(value):.4f}"
    except Exception:
        return "--"


def export_latex_table(base_dir: Path, summaries: Iterable[AlgorithmSummary],
                       formal_gate_passed: bool = False,
                       claim_supported: bool = False,
                       include_energy_claim: bool = False,
                       experiment_type: str = "c4_ablation",
                       canonical_allowed: bool = True) -> Path:
    """导出紧凑消融LaTeX表"""
    citable = bool(formal_gate_passed and claim_supported and include_energy_claim and canonical_allowed)
    if experiment_type == "normal_main":
        filename = "normal_main_table.tex" if citable else "normal_main_table_draft.tex"
        table_algorithms = NORMAL_MAIN_ALGORITHMS
    else:
        filename = "ablation_table.tex" if citable else "ablation_table_draft.tex"
        table_algorithms = ABLATION_MAIN_ALGORITHMS
    path = base_dir / "tables" / filename
    path.parent.mkdir(parents=True, exist_ok=True)

    summaries = list(summaries)
    aggregate_rows = [
        row for row in summaries
        if row.seed == -1 and row.algorithm in table_algorithms
    ]
    table_rows = aggregate_rows or [
        row for row in summaries
        if row.algorithm in table_algorithms
    ]
    row_map = {
        row.algorithm: row for row in table_rows
    }
    rows = [
        row_map[algorithm] for algorithm in table_algorithms
        if algorithm in row_map
    ]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        (
            ("\\caption{Normal-main comparison results under the same reproducible simulation pipeline.}"
             if experiment_type == "normal_main" else
             "\\caption{Energy-hard ablation results under the same simulation pipeline.}")
            if citable else
            "\\caption{Draft ablation results. Formal, claim, and energy-scope gates are not all passed; do not cite this table as final data.}"
        ),
        "\\label{tab:ablation}",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Algorithm & Average Response Delay (ms) & Total Energy Consumption (J) & Service-Chain Operational Cost & Average Virtual Energy Queue & Average Virtual Delay Queue \\\\",
        "\\midrule",
    ]
    for summary in rows:
        lines.append(
            f"{summary.algorithm} & "
            f"{_format_mean_std(summary.delay_mean, summary.delay_std)} & "
            f"{_format_mean_std(summary.energy_mean, summary.energy_std)} & "
            f"{_format_mean_std(summary.cost_mean, summary.cost_std)} & "
            f"{_format_mean_std(summary.avg_y_mean, summary.avg_y_std)} & "
            f"{_format_mean_std(summary.avg_z_mean, summary.avg_z_std)} \\\\"
        )
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def export_solver_benchmark_table(base_dir: Path, summaries: Iterable[AlgorithmSummary]) -> Path:
    """导出UAC求解器对比表"""
    path = base_dir / "tables" / "solver_benchmark_table.tex"
    path.parent.mkdir(parents=True, exist_ok=True)

    row_map = {
        row.algorithm: row for row in summaries
        if row.seed == -1 and row.algorithm in ABLATION_SOLVER_ALGORITHMS
    }
    rows = [
        row_map[algorithm] for algorithm in ABLATION_SOLVER_ALGORITHMS
        if algorithm in row_map
    ]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Solver benchmark under the same slow-layer context.}",
        "\\label{tab:solver_benchmark}",
        "\\begin{tabular}{lcccccccc}",
        "\\toprule",
        "Algorithm & DPP score & gap vs. GSLA-LyCD & avg decision time & P95 decision time & feasible ratio & Delay & Energy & Cost \\\\",
        "\\midrule",
    ]
    for summary in rows:
        lines.append(
            f"{summary.algorithm} & "
            f"{_format_mean_std(summary.paper_dpp_score_mean, summary.paper_dpp_score_std)} & "
            f"{_format_plain(summary.solver_gap_vs_lycd_mean)} & "
            f"{_format_plain(summary.decision_time_mean_ms)} & "
            f"{_format_plain(summary.decision_time_p95_ms)} & "
            f"{_format_plain(summary.feasible_ratio)} & "
            f"{_format_plain(summary.delay_mean)} & "
            f"{_format_plain(summary.energy_mean)} & "
            f"{_format_plain(summary.cost_mean)} \\\\"
        )
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path





