"""
消融实验统一配置
集中管理论文算法命名、默认场景、随机种子和实验导出路径。
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


ABLATION_MAIN_ALGORITHMS = ["LyHAM-CO", "GSLA-Myopic", "FFD-UAC"]
ABLATION_BASELINE_ALGORITHMS = ["PDRS-Myopic", "FFD-Myopic", "Random-Myopic", "LoadAware-Myopic"]
ABLATION_SOLVER_ALGORITHMS = ["LyHAM-CO", "GSLA-LyCD", "GSLA-Myopic"]
FROZEN_UAC_ALGORITHMS = {"LyHAM-CO-Frozen"}
NORMAL_MAIN_ALGORITHMS = ["LyHAM-CO", "GMDA-RMPR-Myopic", "PDRS-Myopic", "FFD-Myopic"]
NORMAL_MAIN_BASELINE_ALGORITHMS = ["GMDA-RMPR-Myopic", "PDRS-Myopic", "FFD-Myopic"]
DIAGNOSTIC_ALGORITHMS = ["LoadAware-Myopic", "Random-Myopic", "GSLA-Myopic", "FFD-UAC"]
UAC_DO_ALGORITHMS = {"LyHAM-CO", "FFD-UAC", "LyHAM-CO-Frozen"}


def get_utils_dir() -> Path:
    """获取utils目录"""
    return Path(__file__).resolve().parent


def get_project_root() -> Path:
    """获取当前实验工程根目录"""
    return get_utils_dir().parents[2]


def get_default_model_path() -> Path:
    """获取默认模型路径，避免旧代码中的硬编码盘符"""
    return get_utils_dir().parent / "Training results" / "Training results1" / "trained_ai_offloading_model_1.2.pth"


def get_default_pair_actor_model_path() -> Path:
    """获取最新可验证pair actor checkpoint，用于formal在线UAC-DO。"""
    pair_dir = get_utils_dir().parent / "Training results" / "PairUAC"
    if not pair_dir.exists():
        return pair_dir / "missing_pair_actor.pth"
    candidates = []
    for meta_path in pair_dir.glob("pair_uac_actor_*.meta.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        checkpoint = Path(str(meta.get("checkpoint_path", "")))
        if not checkpoint.exists():
            checkpoint = meta_path.with_suffix("").with_suffix(".pth")
        if (
            bool(meta.get("pair_actor", False)) and
            bool(meta.get("formal_seed_excluded", False)) and
            not bool(meta.get("model_mutated", False)) and
            checkpoint.exists()
        ):
            candidates.append((meta_path.stat().st_mtime, checkpoint))
    if candidates:
        candidates.sort(key=lambda item: (item[0], str(item[1])))
        return candidates[-1][1]
    fallback = sorted(pair_dir.glob("pair_uac_actor_*.pth"), key=lambda path: path.stat().st_mtime)
    return fallback[-1] if fallback else pair_dir / "missing_pair_actor.pth"


def get_default_output_dir() -> Path:
    """获取消融实验默认输出目录"""
    return get_project_root() / "实验结果" / "消融实验"


@dataclass
class AblationExperimentConfig:
    """消融实验配置"""
    seeds: List[int] = field(default_factory=lambda: [38, 39, 40, 41, 42])
    time_slots: int = 100
    slot_duration_s: int = 1
    slow_epoch_slots: int = 20
    traditional_nodes: int = 20
    ai_nodes: int = 10
    request_flow_count: int = 12
    chain_length_range: Tuple[int, int] = (4, 6)
    arrival_range_req_s: Tuple[float, float] = (5.0, 12.0)
    input_tokens_range: Tuple[int, int] = (128, 512)
    output_tokens_range: Tuple[int, int] = (32, 128)
    V: float = 20.0
    model_path: str = field(default_factory=lambda: str(get_default_model_path()))
    output_dir: str = field(default_factory=lambda: str(get_default_output_dir()))
    algorithms: List[str] = field(default_factory=lambda: list(ABLATION_MAIN_ALGORITHMS))
    include_solver_benchmark: bool = False
    myopic_randomness_rate: float = 0.0
    fixed_arrival_rate: float = None
    uac_candidate_count: int = 10
    uac_gumbel_count: int = 4
    uac_candidate_mechanism: str = "legacy"
    uac_compact_pair_repair_limit: int = 20
    uac_compact_frontier_width: int = 4
    myopic_candidate_limit: int = 64
    enable_online_update: bool = False
    online_update_interval: int = 1
    online_update_batch_size: int = 16
    online_replay_capacity: int = 512
    online_learning_rate: float = 1e-4
    lycd_max_rounds: int = 3
    random_max_retries: int = 5
    omega_energy: float = 1.0
    omega_delay: float = 1.0
    energy_claim_omega_energy: float = 10.0
    energy_claim_omega_delay: float = 4.0
    energy_claim_threshold_scale: float = 0.75
    queue_energy_threshold_j: Optional[float] = None
    queue_delay_threshold_ms: Optional[float] = None
    energy_hard_dpp_tolerance_ratio: float = 0.03
    energy_hard_claim_escape_dpp_ratio: float = 2.0
    energy_hard_claim_escape_min_score_gain: float = 0.20
    energy_hard_claim_escape_min_delay_gain: float = 0.15
    queue_pressure_delay_guard_min_normalized_gain: float = 0.5
    queue_pressure_delay_guard_temporal_max_delay_regret_ms: float = 20.0
    queue_pressure_delay_guard_dpp_regret_ratio: float = 0.35
    queue_pressure_delay_guard_max_energy_regret_j: float = 3.0
    queue_pressure_delay_guard_max_cost_regret: float = 120.0
    queue_pressure_delay_guard_max_predicted_y_regret: float = 8.0
    queue_pressure_delay_guard_max_predicted_z_regret: float = 0.5
    queue_pressure_delay_guard_max_queue_drift_regret: float = 15.0
    queue_pressure_delay_guard_energy_pressure_min_current_avg_y: float = 8.0
    queue_pressure_delay_guard_energy_pressure_max_predicted_y_regret: float = 0.25
    queue_pressure_delay_guard_energy_pressure_max_energy_queue_delta_regret: float = 0.0
    queue_pressure_delay_guard_energy_pressure_max_queue_drift_regret: float = 5.0
    queue_pressure_delay_guard_severe_delay_ms: float = 360.0
    queue_pressure_delay_guard_severe_dpp_regret_ratio: float = 0.55
    queue_pressure_delay_guard_severe_max_predicted_z_regret: float = 1.0
    queue_pressure_delay_guard_severe_max_queue_drift_regret: float = 80.0
    queue_pressure_delay_guard_severe_max_energy_queue_delta_regret: float = 80.0
    queue_pressure_dpp_min_queue_to_v_ratio: float = 0.25
    queue_pressure_energy_guard_enabled: bool = False
    queue_pressure_energy_guard_min_energy_gain_j: float = 1.0
    queue_pressure_energy_guard_max_delay_regret_ms: float = 20.0
    queue_pressure_energy_guard_dpp_improvement_delay_slack_ratio: float = 0.05
    queue_pressure_energy_guard_max_cost_regret: float = 80.0
    queue_pressure_energy_guard_dpp_slack_ratio: float = 0.0
    queue_pressure_energy_guard_strict_dpp_regret_ratio: float = 0.0
    queue_pressure_energy_guard_min_current_avg_y: float = 8.0
    queue_pressure_energy_guard_max_predicted_y_regret: float = 0.25
    queue_pressure_energy_guard_max_energy_queue_delta_regret: float = 0.0
    queue_pressure_energy_guard_max_queue_drift_regret: float = 5.0
    queue_pressure_energy_guard_dpp_improvement_queue_drift_offset_ratio: float = 1.0
    queue_pressure_energy_guard_min_queue_relief: float = 0.05
    queue_pressure_energy_guard_require_queue_relief: bool = True
    queue_pressure_resource_variant_min_current_avg_y: float = 8.0
    queue_pressure_resource_variant_disable_queue_unaware: bool = True
    queue_pressure_resource_variant_emit_full_queue_relief: bool = False
    energy_claim_stability_guard_enabled: bool = False
    energy_claim_stability_min_candidates: int = 3
    energy_claim_stability_regret_ratio: float = 0.08
    energy_claim_stability_regret_min: float = 0.10
    energy_claim_stability_min_score_gain: float = 0.20
    energy_claim_stability_claim_weight: float = 0.05
    energy_claim_stability_queue_drift_weight: float = 0.0
    energy_claim_stability_latency_queue_weight: float = 0.0
    energy_claim_stability_delay_target_ratio: float = 0.90
    energy_claim_stability_energy_target_ratio: float = 1.85
    energy_claim_temporal_guard_enabled: bool = False
    energy_claim_temporal_guard_window: int = 20
    energy_claim_temporal_guard_min_history: int = 5
    energy_claim_temporal_guard_regret_ratio: float = 0.12
    energy_claim_temporal_guard_regret_min: float = 0.10
    energy_claim_temporal_guard_dpp_regret_ratio: float = 0.10
    energy_claim_temporal_guard_min_score_gain: float = 0.20
    energy_claim_temporal_guard_claim_weight: float = 0.05
    energy_claim_temporal_guard_source_switch_weight: float = 0.0
    energy_claim_temporal_guard_local_cloud_weight: float = 0.0
    energy_claim_temporal_guard_local_cloud_scale: float = 4.0
    energy_claim_temporal_guard_latency_queue_weight: float = 0.0
    energy_claim_temporal_guard_energy_override_enabled: bool = False
    energy_claim_temporal_guard_energy_override_min_gain_j: float = 1.0
    energy_claim_temporal_guard_energy_override_max_delay_regret_ms: float = 35.0
    energy_claim_temporal_guard_energy_override_max_cost_regret: float = 40.0
    energy_claim_temporal_guard_energy_override_max_queue_regret: float = 0.50
    post_update_queue_drift_enabled: bool = False
    post_update_queue_drift_weight: float = 0.0
    energy_ref_j: float = 2.0
    delay_ref_ms: float = 50.0
    experiment_type: str = "c4_ablation"
    include_energy_claim: bool = False
    energy_claim_reference_scaled: bool = False
    uac_pair_repair_limit: int = 24
    uac_hamming_threshold: float = 0.05
    uac_source_ratio_threshold: float = 0.20
    strict_pair_actor_required: bool = False
    claim_baselines: List[str] = field(default_factory=lambda: list(NORMAL_MAIN_BASELINE_ALGORITHMS))
    claim_metric_set: List[str] = field(default_factory=lambda: ["delay", "energy", "cost"])
    scenario_profile: str = "default"
    scenario_profile_frozen: bool = False
    ai_gpu_units_range: Tuple[int, int] = (1, 4)
    ai_max_batch_size: int = 32
    ai_prefill_speed_range: Tuple[float, float] = (500000.0, 500000.0)
    ai_decode_speed_range: Tuple[float, float] = (30000.0, 30000.0)
    cloud_latency_base_range: Tuple[float, float] = (50.0, 50.0)
    cloud_bandwidth_range: Tuple[float, float] = (100.0, 100.0)
    cloud_f_pre_rails: Tuple[float, ...] = (0.25, 0.40, 0.55, 0.70, 0.85, 1.00)
    cloud_remote_energy_factor: float = 1.0
    gsla_uac_cloud_cost_factor: float = 1.0
    gsla_uac_routing_cost_factor: float = 1.0
    gsla_uac_cloud_latency_factor: float = 1.0
    gsla_uac_cloud_energy_factor: float = 1.0
    non_gsla_uac_cloud_latency_factor: float = 1.0
    non_gsla_uac_cloud_energy_factor: float = 1.0
    gsla_uac_local_latency_factor: float = 1.0
    gsla_uac_local_energy_factor: float = 1.0
    non_gsla_uac_local_latency_factor: float = 1.0
    non_gsla_uac_local_energy_factor: float = 1.0
    claim_delay_ref_ms: float = 100.0
    claim_energy_ref_j: float = 2.0
    claim_cost_ref: float = 400.0
    claim_improvement_margin: float = 0.0
    figure_sweep_name: str = ""
    figure_sweep_value: str = ""

    def resolve_model_path(self) -> Path:
        """解析模型路径"""
        return Path(self.model_path).expanduser().resolve()

    def resolve_output_dir(self) -> Path:
        """解析输出目录"""
        return Path(self.output_dir).expanduser().resolve()

    def is_uac_algorithm(self, algorithm: str) -> bool:
        """判断算法是否需要训练模型"""
        return algorithm in UAC_DO_ALGORITHMS

    def to_dict(self) -> dict:
        """转换为日志友好的字典"""
        return {
            "seeds": list(self.seeds),
            "time_slots": self.time_slots,
            "slot_duration_s": self.slot_duration_s,
            "slow_epoch_slots": self.slow_epoch_slots,
            "traditional_nodes": self.traditional_nodes,
            "ai_nodes": self.ai_nodes,
            "request_flow_count": self.request_flow_count,
            "chain_length_range": self.chain_length_range,
            "arrival_range_req_s": self.arrival_range_req_s,
            "input_tokens_range": self.input_tokens_range,
            "output_tokens_range": self.output_tokens_range,
            "V": self.V,
            "model_path": str(self.resolve_model_path()),
            "output_dir": str(self.resolve_output_dir()),
            "algorithms": list(self.algorithms),
            "include_solver_benchmark": self.include_solver_benchmark,
            "myopic_randomness_rate": self.myopic_randomness_rate,
            "fixed_arrival_rate": self.fixed_arrival_rate,
            "uac_candidate_count": self.uac_candidate_count,
            "uac_gumbel_count": self.uac_gumbel_count,
            "uac_candidate_mechanism": self.uac_candidate_mechanism,
            "uac_compact_pair_repair_limit": self.uac_compact_pair_repair_limit,
            "uac_compact_frontier_width": self.uac_compact_frontier_width,
            "myopic_candidate_limit": self.myopic_candidate_limit,
            "enable_online_update": self.enable_online_update,
            "online_update_interval": self.online_update_interval,
            "online_update_batch_size": self.online_update_batch_size,
            "online_replay_capacity": self.online_replay_capacity,
            "online_learning_rate": self.online_learning_rate,
            "lycd_max_rounds": self.lycd_max_rounds,
            "random_max_retries": self.random_max_retries,
            "omega_energy": self.omega_energy,
            "omega_delay": self.omega_delay,
            "energy_claim_omega_energy": self.energy_claim_omega_energy,
            "energy_claim_omega_delay": self.energy_claim_omega_delay,
            "energy_claim_threshold_scale": self.energy_claim_threshold_scale,
            "queue_energy_threshold_j": self.queue_energy_threshold_j,
            "queue_delay_threshold_ms": self.queue_delay_threshold_ms,
            "energy_hard_dpp_tolerance_ratio": self.energy_hard_dpp_tolerance_ratio,
            "energy_hard_claim_escape_dpp_ratio": self.energy_hard_claim_escape_dpp_ratio,
            "energy_hard_claim_escape_min_score_gain": self.energy_hard_claim_escape_min_score_gain,
            "energy_hard_claim_escape_min_delay_gain": self.energy_hard_claim_escape_min_delay_gain,
            "queue_pressure_delay_guard_min_normalized_gain": self.queue_pressure_delay_guard_min_normalized_gain,
            "queue_pressure_delay_guard_temporal_max_delay_regret_ms": self.queue_pressure_delay_guard_temporal_max_delay_regret_ms,
            "queue_pressure_delay_guard_dpp_regret_ratio": self.queue_pressure_delay_guard_dpp_regret_ratio,
            "queue_pressure_delay_guard_max_energy_regret_j": self.queue_pressure_delay_guard_max_energy_regret_j,
            "queue_pressure_delay_guard_max_cost_regret": self.queue_pressure_delay_guard_max_cost_regret,
            "queue_pressure_delay_guard_max_predicted_y_regret": self.queue_pressure_delay_guard_max_predicted_y_regret,
            "queue_pressure_delay_guard_max_predicted_z_regret": self.queue_pressure_delay_guard_max_predicted_z_regret,
            "queue_pressure_delay_guard_max_queue_drift_regret": self.queue_pressure_delay_guard_max_queue_drift_regret,
            "queue_pressure_delay_guard_energy_pressure_min_current_avg_y": self.queue_pressure_delay_guard_energy_pressure_min_current_avg_y,
            "queue_pressure_delay_guard_energy_pressure_max_predicted_y_regret": self.queue_pressure_delay_guard_energy_pressure_max_predicted_y_regret,
            "queue_pressure_delay_guard_energy_pressure_max_energy_queue_delta_regret": self.queue_pressure_delay_guard_energy_pressure_max_energy_queue_delta_regret,
            "queue_pressure_delay_guard_energy_pressure_max_queue_drift_regret": self.queue_pressure_delay_guard_energy_pressure_max_queue_drift_regret,
            "queue_pressure_delay_guard_severe_delay_ms": self.queue_pressure_delay_guard_severe_delay_ms,
            "queue_pressure_delay_guard_severe_dpp_regret_ratio": self.queue_pressure_delay_guard_severe_dpp_regret_ratio,
            "queue_pressure_delay_guard_severe_max_predicted_z_regret": self.queue_pressure_delay_guard_severe_max_predicted_z_regret,
            "queue_pressure_delay_guard_severe_max_queue_drift_regret": self.queue_pressure_delay_guard_severe_max_queue_drift_regret,
            "queue_pressure_delay_guard_severe_max_energy_queue_delta_regret": self.queue_pressure_delay_guard_severe_max_energy_queue_delta_regret,
            "queue_pressure_dpp_min_queue_to_v_ratio": self.queue_pressure_dpp_min_queue_to_v_ratio,
            "queue_pressure_energy_guard_enabled": self.queue_pressure_energy_guard_enabled,
            "queue_pressure_energy_guard_min_energy_gain_j": self.queue_pressure_energy_guard_min_energy_gain_j,
            "queue_pressure_energy_guard_max_delay_regret_ms": self.queue_pressure_energy_guard_max_delay_regret_ms,
            "queue_pressure_energy_guard_dpp_improvement_delay_slack_ratio": self.queue_pressure_energy_guard_dpp_improvement_delay_slack_ratio,
            "queue_pressure_energy_guard_max_cost_regret": self.queue_pressure_energy_guard_max_cost_regret,
            "queue_pressure_energy_guard_dpp_slack_ratio": self.queue_pressure_energy_guard_dpp_slack_ratio,
            "queue_pressure_energy_guard_strict_dpp_regret_ratio": self.queue_pressure_energy_guard_strict_dpp_regret_ratio,
            "queue_pressure_energy_guard_min_current_avg_y": self.queue_pressure_energy_guard_min_current_avg_y,
            "queue_pressure_energy_guard_max_predicted_y_regret": self.queue_pressure_energy_guard_max_predicted_y_regret,
            "queue_pressure_energy_guard_max_energy_queue_delta_regret": self.queue_pressure_energy_guard_max_energy_queue_delta_regret,
            "queue_pressure_energy_guard_max_queue_drift_regret": self.queue_pressure_energy_guard_max_queue_drift_regret,
            "queue_pressure_energy_guard_dpp_improvement_queue_drift_offset_ratio": self.queue_pressure_energy_guard_dpp_improvement_queue_drift_offset_ratio,
            "queue_pressure_energy_guard_min_queue_relief": self.queue_pressure_energy_guard_min_queue_relief,
            "queue_pressure_energy_guard_require_queue_relief": self.queue_pressure_energy_guard_require_queue_relief,
            "queue_pressure_resource_variant_min_current_avg_y": self.queue_pressure_resource_variant_min_current_avg_y,
            "queue_pressure_resource_variant_disable_queue_unaware": self.queue_pressure_resource_variant_disable_queue_unaware,
            "queue_pressure_resource_variant_emit_full_queue_relief": self.queue_pressure_resource_variant_emit_full_queue_relief,
            "energy_claim_stability_guard_enabled": self.energy_claim_stability_guard_enabled,
            "energy_claim_stability_min_candidates": self.energy_claim_stability_min_candidates,
            "energy_claim_stability_regret_ratio": self.energy_claim_stability_regret_ratio,
            "energy_claim_stability_regret_min": self.energy_claim_stability_regret_min,
            "energy_claim_stability_min_score_gain": self.energy_claim_stability_min_score_gain,
            "energy_claim_stability_claim_weight": self.energy_claim_stability_claim_weight,
            "energy_claim_stability_queue_drift_weight": self.energy_claim_stability_queue_drift_weight,
            "energy_claim_stability_latency_queue_weight": self.energy_claim_stability_latency_queue_weight,
            "energy_claim_stability_delay_target_ratio": self.energy_claim_stability_delay_target_ratio,
            "energy_claim_stability_energy_target_ratio": self.energy_claim_stability_energy_target_ratio,
            "energy_claim_temporal_guard_enabled": self.energy_claim_temporal_guard_enabled,
            "energy_claim_temporal_guard_window": self.energy_claim_temporal_guard_window,
            "energy_claim_temporal_guard_min_history": self.energy_claim_temporal_guard_min_history,
            "energy_claim_temporal_guard_regret_ratio": self.energy_claim_temporal_guard_regret_ratio,
            "energy_claim_temporal_guard_regret_min": self.energy_claim_temporal_guard_regret_min,
            "energy_claim_temporal_guard_dpp_regret_ratio": self.energy_claim_temporal_guard_dpp_regret_ratio,
            "energy_claim_temporal_guard_min_score_gain": self.energy_claim_temporal_guard_min_score_gain,
            "energy_claim_temporal_guard_claim_weight": self.energy_claim_temporal_guard_claim_weight,
            "energy_claim_temporal_guard_source_switch_weight": self.energy_claim_temporal_guard_source_switch_weight,
            "energy_claim_temporal_guard_local_cloud_weight": self.energy_claim_temporal_guard_local_cloud_weight,
            "energy_claim_temporal_guard_local_cloud_scale": self.energy_claim_temporal_guard_local_cloud_scale,
            "energy_claim_temporal_guard_latency_queue_weight": self.energy_claim_temporal_guard_latency_queue_weight,
            "energy_claim_temporal_guard_energy_override_enabled": self.energy_claim_temporal_guard_energy_override_enabled,
            "energy_claim_temporal_guard_energy_override_min_gain_j": self.energy_claim_temporal_guard_energy_override_min_gain_j,
            "energy_claim_temporal_guard_energy_override_max_delay_regret_ms": self.energy_claim_temporal_guard_energy_override_max_delay_regret_ms,
            "energy_claim_temporal_guard_energy_override_max_cost_regret": self.energy_claim_temporal_guard_energy_override_max_cost_regret,
            "energy_claim_temporal_guard_energy_override_max_queue_regret": self.energy_claim_temporal_guard_energy_override_max_queue_regret,
            "post_update_queue_drift_enabled": self.post_update_queue_drift_enabled,
            "post_update_queue_drift_weight": self.post_update_queue_drift_weight,
            "energy_ref_j": self.energy_ref_j,
            "delay_ref_ms": self.delay_ref_ms,
            "experiment_type": self.experiment_type,
            "include_energy_claim": self.include_energy_claim,
            "energy_claim_reference_scaled": self.energy_claim_reference_scaled,
            "uac_pair_repair_limit": self.uac_pair_repair_limit,
            "uac_hamming_threshold": self.uac_hamming_threshold,
            "uac_source_ratio_threshold": self.uac_source_ratio_threshold,
            "strict_pair_actor_required": self.strict_pair_actor_required,
            "normal_main_baselines": list(NORMAL_MAIN_BASELINE_ALGORITHMS),
            "claim_baselines": list(self.claim_baselines),
            "claim_metric_set": list(self.claim_metric_set),
            "scenario_profile": self.scenario_profile,
            "scenario_profile_frozen": self.scenario_profile_frozen,
            "ai_gpu_units_range": self.ai_gpu_units_range,
            "ai_max_batch_size": self.ai_max_batch_size,
            "ai_prefill_speed_range": self.ai_prefill_speed_range,
            "ai_decode_speed_range": self.ai_decode_speed_range,
            "cloud_latency_base_range": self.cloud_latency_base_range,
            "cloud_bandwidth_range": self.cloud_bandwidth_range,
            "cloud_f_pre_rails": tuple(self.cloud_f_pre_rails),
            "cloud_remote_energy_factor": self.cloud_remote_energy_factor,
            "gsla_uac_cloud_cost_factor": self.gsla_uac_cloud_cost_factor,
            "gsla_uac_routing_cost_factor": self.gsla_uac_routing_cost_factor,
            "gsla_uac_cloud_latency_factor": self.gsla_uac_cloud_latency_factor,
            "gsla_uac_cloud_energy_factor": self.gsla_uac_cloud_energy_factor,
            "non_gsla_uac_cloud_latency_factor": self.non_gsla_uac_cloud_latency_factor,
            "non_gsla_uac_cloud_energy_factor": self.non_gsla_uac_cloud_energy_factor,
            "gsla_uac_local_latency_factor": self.gsla_uac_local_latency_factor,
            "gsla_uac_local_energy_factor": self.gsla_uac_local_energy_factor,
            "non_gsla_uac_local_latency_factor": self.non_gsla_uac_local_latency_factor,
            "non_gsla_uac_local_energy_factor": self.non_gsla_uac_local_energy_factor,
            "claim_delay_ref_ms": self.claim_delay_ref_ms,
            "claim_energy_ref_j": self.claim_energy_ref_j,
            "claim_cost_ref": self.claim_cost_ref,
            "claim_improvement_margin": self.claim_improvement_margin,
            "figure_sweep_name": self.figure_sweep_name,
            "figure_sweep_value": self.figure_sweep_value,
        }











