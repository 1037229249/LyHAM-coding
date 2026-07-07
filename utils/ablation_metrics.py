"""
消融实验指标结构
失败时隙只记录原因，不写入均值，避免旧代码中0值污染统计结果。
"""
from dataclasses import asdict, dataclass
from math import isfinite
from statistics import mean, stdev
from typing import Dict, Iterable, List


@dataclass
class SlotResult:
    """单个时隙的实验结果"""
    slot: int
    seed: int
    algorithm: str
    slow_policy: str
    fast_controller: str
    status: str
    failure_reason: str
    delay_ms: float
    energy_j: float
    cost: float
    avg_y: float
    avg_z: float
    dpp_score: float
    legacy_reward: float
    feasible: bool
    local_count: int
    cloud_count: int
    forced_cloud_count: int
    decision_time_ms: float
    slow_context_reused: bool
    model_path: str
    paper_dpp_score: float = 0.0
    scaled_energy_sum: float = 0.0
    scaled_delay_burden_sum: float = 0.0
    candidate_count: int = 1
    selected_candidate_rank: int = 0
    replay_written: bool = False
    online_update_step: int = 0
    solver_gap_vs_lycd: float = 0.0
    p95_decision_time_ms: float = 0.0
    routing_entropy: float = 0.0
    slow_profile_reused: bool = False
    retry_count: int = 0
    action_dim: int = 0
    pair_action_dim: int = 0
    routing_policy: str = ""
    cost_topo: float = 0.0
    cost_comp: float = 0.0
    cost_comm: float = 0.0
    cost_component_consistent: bool = True
    active_ai_energy_j: float = 0.0
    system_active_ai_chain_energy_j: float = 0.0
    energy_scope: str = "active_ai_chain_engineering"
    formal_gate_passed: bool = False
    model_mutated: bool = False
    v_cost_term: float = 0.0
    energy_queue_term: float = 0.0
    delay_queue_term: float = 0.0
    best_local_score: float = float("inf")
    best_cloud_score: float = float("inf")
    score_gap_local_vs_cloud: float = float("inf")
    local_candidate_feasible_count: int = 0
    candidate_diversity: float = 0.0
    all_cloud_candidate_count: int = 0
    all_local_candidate_count: int = 0
    routing_metric_consumed: bool = False
    routing_delay_consumed: bool = False
    routing_probability_mass: float = 0.0
    mechanism_gate_passed: bool = False
    energy_local_gpu_j: float = 0.0
    energy_cloud_compute_j: float = 0.0
    energy_comm_j: float = 0.0
    energy_idle_replica_j: float = 0.0
    action_hash: str = ""
    pair_action_hash: str = ""
    repaired_pair_action_hash: str = ""
    original_pair_action_hash: str = ""
    repair_changed_pair_count: int = 0
    repair_changed_ratio: float = 0.0
    pair_action_bits: str = ""
    selected_candidate_source: str = ""
    active_local_pair_indices: str = ""
    active_local_pair_ids: str = ""
    active_local_pair_signature: str = ""
    active_cloud_pair_indices: str = ""
    active_cloud_pair_ids: str = ""
    active_cloud_pair_signature: str = ""
    local_pair_count: int = 0
    cloud_pair_count: int = 0
    repaired_candidate_diversity: float = 0.0
    repaired_source_distribution: str = ""
    best_repaired_uac_vs_myopic_dpp_gap: float = 0.0
    best_energy_candidate_source: str = ""
    best_energy_candidate_hash: str = ""
    best_energy_candidate_energy_j: float = 0.0
    best_energy_candidate_dpp_score: float = 0.0
    best_energy_candidate_energy_gap: float = 0.0
    best_energy_candidate_dpp_gap: float = 0.0
    uac_selected_source: bool = False
    repaired_hamming_vs_reference: float = 0.0
    reference_pair_action_bits: str = ""
    reference_pair_action_hash: str = ""
    reference_pair_action_source: str = ""
    selected_action_diversity: float = 0.0
    uac_source_family_count: float = 0.0
    placement_hash: str = ""
    routing_hash: str = ""
    slow_context_hash: str = ""
    claim_score: float = 0.0
    is_pareto_candidate: bool = False
    dpp_band_passed: bool = False
    selected_by_dpp_or_claim_band: str = "dpp"
    per_pair_delta_delay: str = ""
    per_pair_delta_energy: str = ""
    per_pair_delta_cost: str = ""
    resource_hint: str = ""
    resource_queue_aware: bool = True
    resource_queue_scale: float = 1.0
    resource_mode: str = ""
    resource_hint_collapsed: bool = False
    candidate_source_score_summary: str = ""
    candidate_source_family_count: float = 0.0
    predicted_avg_y: float = 0.0
    predicted_avg_z: float = 0.0
    post_update_queue_drift_term: float = 0.0
    post_update_queue_pressure_term: float = 0.0
    post_update_queue_delta_term: float = 0.0
    post_update_energy_queue_delta_term: float = 0.0
    post_update_delay_queue_delta_term: float = 0.0
    post_update_queue_drift_enabled: bool = False
    tail_risk_candidate_source: str = ""
    tail_risk_candidate_hash: str = ""
    tail_risk_candidate_delay_ms: float = 0.0
    tail_risk_candidate_energy_j: float = 0.0
    tail_risk_candidate_cost: float = 0.0
    tail_risk_candidate_claim_score: float = 0.0
    tail_risk_candidate_dpp_score: float = 0.0
    tail_risk_candidate_predicted_avg_y: float = 0.0
    tail_risk_candidate_predicted_avg_z: float = 0.0
    tail_risk_candidate_post_update_queue_drift_term: float = 0.0
    tail_risk_candidate_upper_excess_score: float = 0.0
    tail_risk_selected_upper_excess_score: float = 0.0
    tail_risk_candidate_upper_excess_improvement: float = 0.0
    tail_risk_best_relief_source: str = ""
    tail_risk_best_relief_hash: str = ""
    tail_risk_best_relief_delay_ms: float = 0.0
    tail_risk_best_relief_energy_j: float = 0.0
    tail_risk_best_relief_cost: float = 0.0
    tail_risk_best_relief_claim_score: float = 0.0
    tail_risk_best_relief_dpp_score: float = 0.0
    tail_risk_best_relief_predicted_avg_y: float = 0.0
    tail_risk_best_relief_predicted_avg_z: float = 0.0
    tail_risk_best_relief_post_update_queue_drift_term: float = 0.0
    tail_risk_best_relief_upper_excess_score: float = 0.0
    tail_risk_best_relief_improvement: float = 0.0
    tail_risk_best_relief_reject_reason: str = ""
    energy_relief_candidate_source: str = ""
    energy_relief_candidate_hash: str = ""
    energy_relief_candidate_delay_ms: float = 0.0
    energy_relief_candidate_energy_j: float = 0.0
    energy_relief_candidate_cost: float = 0.0
    energy_relief_candidate_claim_score: float = 0.0
    energy_relief_candidate_dpp_score: float = 0.0
    energy_relief_candidate_predicted_avg_y: float = 0.0
    energy_relief_candidate_predicted_avg_z: float = 0.0
    energy_relief_candidate_post_update_queue_drift_term: float = 0.0
    energy_relief_candidate_energy_gain_j: float = 0.0
    energy_relief_candidate_delay_regret_ms: float = 0.0
    energy_relief_candidate_cost_regret: float = 0.0
    energy_relief_candidate_dpp_regret: float = 0.0
    energy_relief_best_lower_source: str = ""
    energy_relief_best_lower_hash: str = ""
    energy_relief_best_lower_delay_ms: float = 0.0
    energy_relief_best_lower_energy_j: float = 0.0
    energy_relief_best_lower_cost: float = 0.0
    energy_relief_best_lower_claim_score: float = 0.0
    energy_relief_best_lower_dpp_score: float = 0.0
    energy_relief_best_lower_predicted_avg_y: float = 0.0
    energy_relief_best_lower_predicted_avg_z: float = 0.0
    energy_relief_best_lower_post_update_queue_drift_term: float = 0.0
    energy_relief_best_lower_energy_gain_j: float = 0.0
    energy_relief_best_lower_delay_regret_ms: float = 0.0
    energy_relief_best_lower_cost_regret: float = 0.0
    energy_relief_best_lower_dpp_regret: float = 0.0
    energy_relief_best_lower_reject_reason: str = ""

    def to_dict(self) -> Dict:
        """转换为CSV行"""
        return asdict(self)

    @property
    def is_valid(self) -> bool:
        """判断该时隙是否可参与均值统计"""
        values = [
            self.delay_ms,
            self.energy_j,
            self.cost,
            self.avg_y,
            self.avg_z,
            self.dpp_score,
            self.paper_dpp_score,
        ]
        return self.status == "ok" and self.feasible and all(isfinite(float(v)) for v in values)


@dataclass
class AlgorithmSummary:
    """算法汇总结果"""
    algorithm: str
    seed: int
    n_valid_slots: int
    n_failed_slots: int
    delay_mean: float
    delay_std: float
    energy_mean: float
    energy_std: float
    cost_mean: float
    cost_std: float
    avg_y_mean: float
    avg_y_std: float
    avg_z_mean: float
    avg_z_std: float
    dpp_score_mean: float
    dpp_score_std: float
    decision_time_mean_ms: float
    decision_time_p95_ms: float
    feasible_ratio: float
    valid: bool
    paper_dpp_score_mean: float = 0.0
    paper_dpp_score_std: float = 0.0
    scaled_energy_sum_mean: float = 0.0
    scaled_energy_sum_std: float = 0.0
    scaled_delay_burden_sum_mean: float = 0.0
    scaled_delay_burden_sum_std: float = 0.0
    candidate_count_mean: float = 1.0
    selected_candidate_rank_mean: float = 0.0
    solver_gap_vs_lycd_mean: float = 0.0
    routing_entropy_mean: float = 0.0
    retry_count_mean: float = 0.0
    action_dim_mean: float = 0.0
    pair_action_dim_mean: float = 0.0
    cost_topo_mean: float = 0.0
    cost_comp_mean: float = 0.0
    cost_comm_mean: float = 0.0
    active_ai_energy_mean: float = 0.0
    system_active_ai_chain_energy_mean: float = 0.0
    cost_component_consistent_ratio: float = 0.0
    formal_gate_passed: bool = False
    valid_seed_count: int = 0
    config_hash: str = ""
    v_cost_term_mean: float = 0.0
    energy_queue_term_mean: float = 0.0
    delay_queue_term_mean: float = 0.0
    best_local_score_mean: float = 0.0
    best_cloud_score_mean: float = 0.0
    score_gap_local_vs_cloud_mean: float = 0.0
    local_candidate_feasible_count_mean: float = 0.0
    candidate_diversity_mean: float = 0.0
    all_cloud_ratio: float = 0.0
    all_cloud_candidate_count_mean: float = 0.0
    all_local_candidate_count_mean: float = 0.0
    routing_metric_consumed_ratio: float = 0.0
    routing_delay_consumed_ratio: float = 0.0
    routing_probability_mass_mean: float = 0.0
    mechanism_gate_passed: bool = False
    claim_supported: bool = False
    energy_scope_gate_passed: bool = False
    energy_local_gpu_mean: float = 0.0
    energy_cloud_compute_mean: float = 0.0
    energy_comm_mean: float = 0.0
    energy_idle_replica_mean: float = 0.0
    local_pair_count_mean: float = 0.0
    cloud_pair_count_mean: float = 0.0
    repaired_candidate_diversity_mean: float = 0.0
    best_repaired_uac_vs_myopic_dpp_gap_mean: float = 0.0
    best_energy_candidate_energy_mean: float = 0.0
    best_energy_candidate_dpp_mean: float = 0.0
    best_energy_candidate_energy_gap_mean: float = 0.0
    best_energy_candidate_dpp_gap_mean: float = 0.0
    uac_selected_source_ratio: float = 0.0
    repaired_hamming_vs_reference_mean: float = 0.0
    selected_action_diversity_mean: float = 0.0
    uac_source_family_count_mean: float = 0.0

    def to_dict(self) -> Dict:
        """转换为CSV行"""
        return asdict(self)


def _safe_mean(values: List[float]) -> float:
    return mean(values) if values else float("nan")


def _safe_finite_mean(values: List[float]) -> float:
    finite_values = [float(value) for value in values if isfinite(float(value))]
    return mean(finite_values) if finite_values else float("nan")


def _safe_std(values: List[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def _safe_p95(values: List[float]) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * 0.95))
    return ordered[index]


def summarize_slot_results(slot_results: Iterable[SlotResult]) -> AlgorithmSummary:
    """汇总单个算法单个seed的时隙结果"""
    rows = list(slot_results)
    if not rows:
        return AlgorithmSummary(
            algorithm="",
            seed=-1,
            n_valid_slots=0,
            n_failed_slots=0,
            delay_mean=float("nan"),
            delay_std=0.0,
            energy_mean=float("nan"),
            energy_std=0.0,
            cost_mean=float("nan"),
            cost_std=0.0,
            avg_y_mean=float("nan"),
            avg_y_std=0.0,
            avg_z_mean=float("nan"),
            avg_z_std=0.0,
            dpp_score_mean=float("nan"),
            dpp_score_std=0.0,
            decision_time_mean_ms=float("nan"),
            decision_time_p95_ms=float("nan"),
            feasible_ratio=0.0,
            valid=False,
            paper_dpp_score_mean=float("nan"),
            paper_dpp_score_std=0.0,
            scaled_energy_sum_mean=float("nan"),
            scaled_energy_sum_std=0.0,
            scaled_delay_burden_sum_mean=float("nan"),
            scaled_delay_burden_sum_std=0.0,
            candidate_count_mean=0.0,
            selected_candidate_rank_mean=0.0,
            solver_gap_vs_lycd_mean=0.0,
            routing_entropy_mean=0.0,
            retry_count_mean=0.0,
            action_dim_mean=0.0,
            pair_action_dim_mean=0.0,
            cost_topo_mean=float("nan"),
            cost_comp_mean=float("nan"),
            cost_comm_mean=float("nan"),
            active_ai_energy_mean=float("nan"),
            system_active_ai_chain_energy_mean=float("nan"),
            cost_component_consistent_ratio=0.0,
            formal_gate_passed=False,
            valid_seed_count=0,
            config_hash="",
        )

    valid_rows = [row for row in rows if row.is_valid]
    failed_rows = [row for row in rows if not row.is_valid]
    decision_times = [row.decision_time_ms for row in valid_rows]
    feasible_count = sum(1 for row in rows if row.feasible and row.status == "ok")
    all_cloud_count = sum(1 for row in valid_rows if row.cloud_count > 0 and row.local_count == 0)
    energy_values = [float(row.energy_j) for row in valid_rows]
    energy_component_series = [
        [float(row.energy_local_gpu_j) for row in valid_rows],
        [float(row.energy_cloud_compute_j) for row in valid_rows],
        [float(row.energy_comm_j) for row in valid_rows],
        [float(row.energy_idle_replica_j) for row in valid_rows],
    ]
    energy_component_traceable = any(
        any(abs(value) > 1e-12 for value in series)
        for series in energy_component_series
    )
    energy_component_changes = any(
        len(series) >= 2 and max(series) - min(series) > 1e-9
        for series in energy_component_series
    )
    action_hashes = {
        row.pair_action_hash or row.action_hash
        for row in valid_rows
        if row.pair_action_hash or row.action_hash
    }
    routing_consumed = any(
        row.routing_metric_consumed or row.routing_delay_consumed
        for row in valid_rows
    )
    energy_scope_gate = (
        len(energy_values) >= 2 and
        max(energy_values) - min(energy_values) > 1e-9 and
        energy_component_traceable and
        (energy_component_changes or len(action_hashes) > 1 or routing_consumed)
    )

    return AlgorithmSummary(
        algorithm=rows[0].algorithm,
        seed=rows[0].seed,
        n_valid_slots=len(valid_rows),
        n_failed_slots=len(failed_rows),
        delay_mean=_safe_mean([row.delay_ms for row in valid_rows]),
        delay_std=_safe_std([row.delay_ms for row in valid_rows]),
        energy_mean=_safe_mean([row.energy_j for row in valid_rows]),
        energy_std=_safe_std([row.energy_j for row in valid_rows]),
        cost_mean=_safe_mean([row.cost for row in valid_rows]),
        cost_std=_safe_std([row.cost for row in valid_rows]),
        avg_y_mean=_safe_mean([row.avg_y for row in valid_rows]),
        avg_y_std=_safe_std([row.avg_y for row in valid_rows]),
        avg_z_mean=_safe_mean([row.avg_z for row in valid_rows]),
        avg_z_std=_safe_std([row.avg_z for row in valid_rows]),
        dpp_score_mean=_safe_mean([row.dpp_score for row in valid_rows]),
        dpp_score_std=_safe_std([row.dpp_score for row in valid_rows]),
        decision_time_mean_ms=_safe_mean(decision_times),
        decision_time_p95_ms=_safe_p95(decision_times),
        feasible_ratio=feasible_count / max(len(rows), 1),
        valid=len(valid_rows) == len(rows) and len(valid_rows) > 0,
        paper_dpp_score_mean=_safe_mean([row.paper_dpp_score for row in valid_rows]),
        paper_dpp_score_std=_safe_std([row.paper_dpp_score for row in valid_rows]),
        scaled_energy_sum_mean=_safe_mean([row.scaled_energy_sum for row in valid_rows]),
        scaled_energy_sum_std=_safe_std([row.scaled_energy_sum for row in valid_rows]),
        scaled_delay_burden_sum_mean=_safe_mean([row.scaled_delay_burden_sum for row in valid_rows]),
        scaled_delay_burden_sum_std=_safe_std([row.scaled_delay_burden_sum for row in valid_rows]),
        candidate_count_mean=_safe_mean([row.candidate_count for row in valid_rows]),
        selected_candidate_rank_mean=_safe_mean([row.selected_candidate_rank for row in valid_rows]),
        solver_gap_vs_lycd_mean=_safe_mean([row.solver_gap_vs_lycd for row in valid_rows]),
        routing_entropy_mean=_safe_mean([row.routing_entropy for row in valid_rows]),
        retry_count_mean=_safe_mean([row.retry_count for row in valid_rows]),
        action_dim_mean=_safe_mean([row.action_dim for row in valid_rows]),
        pair_action_dim_mean=_safe_mean([row.pair_action_dim for row in valid_rows]),
        cost_topo_mean=_safe_mean([row.cost_topo for row in valid_rows]),
        cost_comp_mean=_safe_mean([row.cost_comp for row in valid_rows]),
        cost_comm_mean=_safe_mean([row.cost_comm for row in valid_rows]),
        active_ai_energy_mean=_safe_mean([row.active_ai_energy_j for row in valid_rows]),
        system_active_ai_chain_energy_mean=_safe_mean([row.system_active_ai_chain_energy_j for row in valid_rows]),
        cost_component_consistent_ratio=(
            sum(1 for row in valid_rows if row.cost_component_consistent) / max(len(valid_rows), 1)
        ),
        formal_gate_passed=all(row.formal_gate_passed for row in valid_rows) if valid_rows else False,
        valid_seed_count=1 if len(valid_rows) == len(rows) and len(valid_rows) > 0 else 0,
        v_cost_term_mean=_safe_mean([row.v_cost_term for row in valid_rows]),
        energy_queue_term_mean=_safe_mean([row.energy_queue_term for row in valid_rows]),
        delay_queue_term_mean=_safe_mean([row.delay_queue_term for row in valid_rows]),
        best_local_score_mean=_safe_finite_mean([row.best_local_score for row in valid_rows]),
        best_cloud_score_mean=_safe_finite_mean([row.best_cloud_score for row in valid_rows]),
        score_gap_local_vs_cloud_mean=_safe_finite_mean([row.score_gap_local_vs_cloud for row in valid_rows]),
        local_candidate_feasible_count_mean=_safe_mean([row.local_candidate_feasible_count for row in valid_rows]),
        candidate_diversity_mean=_safe_mean([row.candidate_diversity for row in valid_rows]),
        all_cloud_ratio=all_cloud_count / max(len(valid_rows), 1),
        all_cloud_candidate_count_mean=_safe_mean([row.all_cloud_candidate_count for row in valid_rows]),
        all_local_candidate_count_mean=_safe_mean([row.all_local_candidate_count for row in valid_rows]),
        routing_metric_consumed_ratio=(
            sum(1 for row in valid_rows if row.routing_metric_consumed) / max(len(valid_rows), 1)
        ),
        routing_delay_consumed_ratio=(
            sum(1 for row in valid_rows if row.routing_delay_consumed) / max(len(valid_rows), 1)
        ),
        routing_probability_mass_mean=_safe_mean([row.routing_probability_mass for row in valid_rows]),
        mechanism_gate_passed=(
            len(valid_rows) > 0 and
            (all_cloud_count / max(len(valid_rows), 1)) < 0.95 and
            sum(1 for row in valid_rows if row.routing_metric_consumed) / max(len(valid_rows), 1) > 0.0 and
            sum(1 for row in valid_rows if row.routing_delay_consumed) / max(len(valid_rows), 1) > 0.0
        ),
        energy_scope_gate_passed=energy_scope_gate,
        energy_local_gpu_mean=_safe_mean([row.energy_local_gpu_j for row in valid_rows]),
        energy_cloud_compute_mean=_safe_mean([row.energy_cloud_compute_j for row in valid_rows]),
        energy_comm_mean=_safe_mean([row.energy_comm_j for row in valid_rows]),
        energy_idle_replica_mean=_safe_mean([row.energy_idle_replica_j for row in valid_rows]),
        local_pair_count_mean=_safe_mean([row.local_pair_count for row in valid_rows]),
        cloud_pair_count_mean=_safe_mean([row.cloud_pair_count for row in valid_rows]),
        repaired_candidate_diversity_mean=_safe_mean([row.repaired_candidate_diversity for row in valid_rows]),
        best_repaired_uac_vs_myopic_dpp_gap_mean=_safe_finite_mean([row.best_repaired_uac_vs_myopic_dpp_gap for row in valid_rows]),
        best_energy_candidate_energy_mean=_safe_finite_mean([row.best_energy_candidate_energy_j for row in valid_rows]),
        best_energy_candidate_dpp_mean=_safe_finite_mean([row.best_energy_candidate_dpp_score for row in valid_rows]),
        best_energy_candidate_energy_gap_mean=_safe_finite_mean([row.best_energy_candidate_energy_gap for row in valid_rows]),
        best_energy_candidate_dpp_gap_mean=_safe_finite_mean([row.best_energy_candidate_dpp_gap for row in valid_rows]),
        uac_selected_source_ratio=(
            sum(1 for row in valid_rows if row.uac_selected_source) / max(len(valid_rows), 1)
        ),
        repaired_hamming_vs_reference_mean=_safe_finite_mean([
            row.repaired_hamming_vs_reference for row in valid_rows
        ]),
        selected_action_diversity_mean=_safe_finite_mean([
            row.selected_action_diversity for row in valid_rows
        ]),
        uac_source_family_count_mean=_safe_finite_mean([
            row.uac_source_family_count for row in valid_rows
        ]),
    )


def aggregate_algorithm_summaries(summaries: Iterable[AlgorithmSummary]) -> List[AlgorithmSummary]:
    """按算法聚合多个seed的汇总结果"""
    groups: Dict[str, List[AlgorithmSummary]] = {}
    for summary in summaries:
        groups.setdefault(summary.algorithm, []).append(summary)

    aggregated = []
    for algorithm, rows in groups.items():
        valid_rows = [row for row in rows if row.valid]
        if not valid_rows:
            aggregated.append(AlgorithmSummary(
                algorithm=algorithm,
                seed=-1,
                n_valid_slots=0,
                n_failed_slots=sum(row.n_failed_slots for row in rows),
                delay_mean=float("nan"),
                delay_std=0.0,
                energy_mean=float("nan"),
                energy_std=0.0,
                cost_mean=float("nan"),
                cost_std=0.0,
                avg_y_mean=float("nan"),
                avg_y_std=0.0,
                avg_z_mean=float("nan"),
                avg_z_std=0.0,
                dpp_score_mean=float("nan"),
                dpp_score_std=0.0,
                decision_time_mean_ms=float("nan"),
                decision_time_p95_ms=float("nan"),
                feasible_ratio=0.0,
                valid=False,
                paper_dpp_score_mean=float("nan"),
                paper_dpp_score_std=0.0,
                scaled_energy_sum_mean=float("nan"),
                scaled_energy_sum_std=0.0,
                scaled_delay_burden_sum_mean=float("nan"),
                scaled_delay_burden_sum_std=0.0,
                candidate_count_mean=0.0,
                selected_candidate_rank_mean=0.0,
                solver_gap_vs_lycd_mean=0.0,
                routing_entropy_mean=0.0,
                retry_count_mean=0.0,
                action_dim_mean=0.0,
                pair_action_dim_mean=0.0,
                cost_topo_mean=float("nan"),
                cost_comp_mean=float("nan"),
                cost_comm_mean=float("nan"),
                active_ai_energy_mean=float("nan"),
                system_active_ai_chain_energy_mean=float("nan"),
                cost_component_consistent_ratio=0.0,
                formal_gate_passed=False,
                valid_seed_count=0,
                config_hash="",
            ))
            continue

        aggregated.append(AlgorithmSummary(
            algorithm=algorithm,
            seed=-1,
            n_valid_slots=sum(row.n_valid_slots for row in valid_rows),
            n_failed_slots=sum(row.n_failed_slots for row in rows),
            delay_mean=_safe_mean([row.delay_mean for row in valid_rows]),
            delay_std=_safe_std([row.delay_mean for row in valid_rows]),
            energy_mean=_safe_mean([row.energy_mean for row in valid_rows]),
            energy_std=_safe_std([row.energy_mean for row in valid_rows]),
            cost_mean=_safe_mean([row.cost_mean for row in valid_rows]),
            cost_std=_safe_std([row.cost_mean for row in valid_rows]),
            avg_y_mean=_safe_mean([row.avg_y_mean for row in valid_rows]),
            avg_y_std=_safe_std([row.avg_y_mean for row in valid_rows]),
            avg_z_mean=_safe_mean([row.avg_z_mean for row in valid_rows]),
            avg_z_std=_safe_std([row.avg_z_mean for row in valid_rows]),
            dpp_score_mean=_safe_mean([row.dpp_score_mean for row in valid_rows]),
            dpp_score_std=_safe_std([row.dpp_score_mean for row in valid_rows]),
            decision_time_mean_ms=_safe_mean([row.decision_time_mean_ms for row in valid_rows]),
            decision_time_p95_ms=_safe_p95([row.decision_time_p95_ms for row in valid_rows]),
            feasible_ratio=_safe_mean([row.feasible_ratio for row in valid_rows]),
            valid=len(valid_rows) == len(rows),
            paper_dpp_score_mean=_safe_mean([row.paper_dpp_score_mean for row in valid_rows]),
            paper_dpp_score_std=_safe_std([row.paper_dpp_score_mean for row in valid_rows]),
            scaled_energy_sum_mean=_safe_mean([row.scaled_energy_sum_mean for row in valid_rows]),
            scaled_energy_sum_std=_safe_std([row.scaled_energy_sum_mean for row in valid_rows]),
            scaled_delay_burden_sum_mean=_safe_mean([row.scaled_delay_burden_sum_mean for row in valid_rows]),
            scaled_delay_burden_sum_std=_safe_std([row.scaled_delay_burden_sum_mean for row in valid_rows]),
            candidate_count_mean=_safe_mean([row.candidate_count_mean for row in valid_rows]),
            selected_candidate_rank_mean=_safe_mean([row.selected_candidate_rank_mean for row in valid_rows]),
            solver_gap_vs_lycd_mean=_safe_mean([row.solver_gap_vs_lycd_mean for row in valid_rows]),
            routing_entropy_mean=_safe_mean([row.routing_entropy_mean for row in valid_rows]),
            retry_count_mean=_safe_mean([row.retry_count_mean for row in valid_rows]),
            action_dim_mean=_safe_mean([row.action_dim_mean for row in valid_rows]),
            pair_action_dim_mean=_safe_mean([row.pair_action_dim_mean for row in valid_rows]),
            cost_topo_mean=_safe_mean([row.cost_topo_mean for row in valid_rows]),
            cost_comp_mean=_safe_mean([row.cost_comp_mean for row in valid_rows]),
            cost_comm_mean=_safe_mean([row.cost_comm_mean for row in valid_rows]),
            active_ai_energy_mean=_safe_mean([row.active_ai_energy_mean for row in valid_rows]),
            system_active_ai_chain_energy_mean=_safe_mean([row.system_active_ai_chain_energy_mean for row in valid_rows]),
            cost_component_consistent_ratio=_safe_mean([row.cost_component_consistent_ratio for row in valid_rows]),
            formal_gate_passed=False,
            valid_seed_count=len(valid_rows),
            v_cost_term_mean=_safe_mean([row.v_cost_term_mean for row in valid_rows]),
            energy_queue_term_mean=_safe_mean([row.energy_queue_term_mean for row in valid_rows]),
            delay_queue_term_mean=_safe_mean([row.delay_queue_term_mean for row in valid_rows]),
            best_local_score_mean=_safe_finite_mean([row.best_local_score_mean for row in valid_rows]),
            best_cloud_score_mean=_safe_finite_mean([row.best_cloud_score_mean for row in valid_rows]),
            score_gap_local_vs_cloud_mean=_safe_finite_mean([row.score_gap_local_vs_cloud_mean for row in valid_rows]),
            local_candidate_feasible_count_mean=_safe_mean([
                row.local_candidate_feasible_count_mean for row in valid_rows
            ]),
            candidate_diversity_mean=_safe_mean([row.candidate_diversity_mean for row in valid_rows]),
            all_cloud_ratio=_safe_mean([row.all_cloud_ratio for row in valid_rows]),
            all_cloud_candidate_count_mean=_safe_mean([
                row.all_cloud_candidate_count_mean for row in valid_rows
            ]),
            all_local_candidate_count_mean=_safe_mean([
                row.all_local_candidate_count_mean for row in valid_rows
            ]),
            routing_metric_consumed_ratio=_safe_mean([
                row.routing_metric_consumed_ratio for row in valid_rows
            ]),
            routing_delay_consumed_ratio=_safe_mean([
                row.routing_delay_consumed_ratio for row in valid_rows
            ]),
            routing_probability_mass_mean=_safe_mean([
                row.routing_probability_mass_mean for row in valid_rows
            ]),
            mechanism_gate_passed=all(row.mechanism_gate_passed for row in valid_rows),
            energy_scope_gate_passed=all(row.energy_scope_gate_passed for row in valid_rows),
            energy_local_gpu_mean=_safe_mean([row.energy_local_gpu_mean for row in valid_rows]),
            energy_cloud_compute_mean=_safe_mean([row.energy_cloud_compute_mean for row in valid_rows]),
            energy_comm_mean=_safe_mean([row.energy_comm_mean for row in valid_rows]),
            energy_idle_replica_mean=_safe_mean([row.energy_idle_replica_mean for row in valid_rows]),
            local_pair_count_mean=_safe_mean([row.local_pair_count_mean for row in valid_rows]),
            cloud_pair_count_mean=_safe_mean([row.cloud_pair_count_mean for row in valid_rows]),
            repaired_candidate_diversity_mean=_safe_mean([row.repaired_candidate_diversity_mean for row in valid_rows]),
            best_repaired_uac_vs_myopic_dpp_gap_mean=_safe_finite_mean([row.best_repaired_uac_vs_myopic_dpp_gap_mean for row in valid_rows]),
            best_energy_candidate_energy_mean=_safe_finite_mean([row.best_energy_candidate_energy_mean for row in valid_rows]),
            best_energy_candidate_dpp_mean=_safe_finite_mean([row.best_energy_candidate_dpp_mean for row in valid_rows]),
            best_energy_candidate_energy_gap_mean=_safe_finite_mean([row.best_energy_candidate_energy_gap_mean for row in valid_rows]),
            best_energy_candidate_dpp_gap_mean=_safe_finite_mean([row.best_energy_candidate_dpp_gap_mean for row in valid_rows]),
            uac_selected_source_ratio=_safe_mean([row.uac_selected_source_ratio for row in valid_rows]),
            repaired_hamming_vs_reference_mean=_safe_finite_mean([
                row.repaired_hamming_vs_reference_mean for row in valid_rows
            ]),
            selected_action_diversity_mean=_safe_finite_mean([
                row.selected_action_diversity_mean for row in valid_rows
            ]),
            uac_source_family_count_mean=_safe_finite_mean([
                row.uac_source_family_count_mean for row in valid_rows
            ]),
        ))

    return aggregated





