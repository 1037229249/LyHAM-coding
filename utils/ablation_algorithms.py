"""
消融实验算法组合入口
代码中直接使用论文算法名，运行日志中保留工程仿真边界。
"""
from collections import Counter
from pathlib import Path
import time
from typing import Dict, Iterable, List, Tuple

import numpy as np

from ablation_config import AblationExperimentConfig, FROZEN_UAC_ALGORITHMS, UAC_DO_ALGORITHMS
import ablation_pair_actions as pair_actions

_UAC_INFERENCE_CACHE = {}
_UAC_CACHE_STATS = {
    "model_load_count": 0,
    "model_cache_hit_count": 0,
}


def clear_uac_inference_cache(reset_stats: bool = True):
    """清空UAC推理缓存；formal每个run独立在线Actor状态。"""
    _UAC_INFERENCE_CACHE.clear()
    if reset_stats:
        for key in list(_UAC_CACHE_STATS):
            _UAC_CACHE_STATS[key] = 0


def _slow_failure_reason(system_state, context_name: str, default_reason: str) -> str:
    """从慢层上下文提取可追踪失败原因"""
    context = getattr(system_state, context_name, {}) or {}
    for key in [
        "hapa_failure_reason",
        "rappa_failed_unit",
        "failure_reason",
        "pdrs_failure_reason",
        "random_failure_reason",
        "loadaware_failure_reason",
    ]:
        value = context.get(key)
        if value:
            return str(value)
    return default_reason


def validate_algorithm_prerequisites(algorithm: str, config: AblationExperimentConfig) -> Tuple[bool, str]:
    """检查算法运行前置条件"""
    if algorithm in UAC_DO_ALGORITHMS:
        model_path = Path(config.model_path).expanduser().resolve()
        if not model_path.exists():
            return False, f"模型文件不存在: {model_path}"
        try:
            import torch  # noqa: F401
        except ImportError:
            return False, "缺少torch运行环境，不能执行UAC-DO模型推理"
        if getattr(config, "strict_pair_actor_required", False) or getattr(config, "include_energy_claim", False):
            from ai_inference import validate_pair_actor_checkpoint_metadata
            ok, reason, _ = validate_pair_actor_checkpoint_metadata(str(model_path), require_pair_actor=True)
            if not ok:
                return False, reason
    return True, ""


def is_online_update_enabled_for_algorithm(config: AblationExperimentConfig, algorithm: str) -> bool:
    """Return whether this algorithm should mutate its UAC actor during the run."""
    return (
        bool(getattr(config, "enable_online_update", False))
        and str(algorithm) not in FROZEN_UAC_ALGORITHMS
    )


def get_algorithm_policies(algorithm: str) -> Tuple[str, str]:
    """获取论文算法对应的慢层策略和快层控制器"""
    mapping = {
        "LyHAM-CO": ("GSLA", "UAC-DO"),
        "LyHAM-CO-Frozen": ("GSLA", "UAC-DO"),
        "FFD-UAC": ("FFD", "UAC-DO"),
        "GSLA-Myopic": ("GSLA", "Myopic"),
        "PDRS-Myopic": ("PDRS", "Myopic"),
        "FFD-Myopic": ("FFD", "Myopic"),
        "Random-Myopic": ("Random", "Myopic"),
        "LoadAware-Myopic": ("LoadAware", "Myopic"),
        "GMDA-RMPR-Myopic": ("GMDA-RMPR", "Myopic"),
        "GSLA-LyCD": ("GSLA", "LyCD"),
    }
    if algorithm not in mapping:
        raise ValueError(f"未知算法: {algorithm}")
    return mapping[algorithm]


def run_slow_context_for_algorithm(algorithm: str, system_state, config: AblationExperimentConfig,
                                   slot: int, seed: int) -> Tuple[bool, str]:
    """按论文算法名运行慢层部署"""
    slow_policy, _ = get_algorithm_policies(algorithm)
    try:
        if slow_policy == "GSLA":
            from Deployment import run_GSLA
            ok = run_GSLA(system_state)
            return ok, "" if ok else _slow_failure_reason(system_state, "gsla_context", "GSLA慢层部署失败")
        if slow_policy == "FFD":
            from FFD import run_FFD_slow_context
            ok = run_FFD_slow_context(system_state, seed=seed + slot)
            return ok, "" if ok else _slow_failure_reason(system_state, "ffd_context", "FFD慢层部署失败")
        if slow_policy == "PDRS":
            from Deployment import run_PDRS_slow_context
            ok = run_PDRS_slow_context(system_state)
            return ok, "" if ok else _slow_failure_reason(system_state, "pdrs_context", "PDRS慢层部署失败")
        if slow_policy == "Random":
            from Deployment import run_Random_slow_context
            ok = run_Random_slow_context(
                system_state,
                seed=seed + slot,
                max_retries=getattr(config, "random_max_retries", 5),
            )
            return ok, "" if ok else _slow_failure_reason(system_state, "random_context", "Random慢层部署失败")
        if slow_policy == "LoadAware":
            from Deployment import run_LoadAware_slow_context
            ok = run_LoadAware_slow_context(system_state)
            return ok, "" if ok else _slow_failure_reason(system_state, "loadaware_context", "LoadAware慢层部署失败")
        if slow_policy == "GMDA-RMPR":
            from Deployment import run_GMDA_RMPR_slow_context
            ok = run_GMDA_RMPR_slow_context(system_state)
            return ok, "" if ok else _slow_failure_reason(system_state, "gmda_rmpr_context", "GMDA-RMPR慢层部署失败")
        return False, f"未知慢层策略: {slow_policy}"
    except Exception as exc:
        return False, str(exc)


def run_LyHAM_CO(system_state, config: AblationExperimentConfig, slot: int, seed: int,
                 slow_context_reused: bool = False):
    """运行LyHAM-CO算法"""
    return _run_named_algorithm("LyHAM-CO", system_state, config, slot, seed, slow_context_reused)


def run_FFD_UAC(system_state, config: AblationExperimentConfig, slot: int, seed: int,
                slow_context_reused: bool = False):
    """运行FFD-UAC算法"""
    return _run_named_algorithm("FFD-UAC", system_state, config, slot, seed, slow_context_reused)


def run_GSLA_Myopic(system_state, config: AblationExperimentConfig, slot: int, seed: int,
                    slow_context_reused: bool = False):
    """运行GSLA-Myopic算法"""
    return _run_named_algorithm("GSLA-Myopic", system_state, config, slot, seed, slow_context_reused)


def run_PDRS_Myopic(system_state, config: AblationExperimentConfig, slot: int, seed: int,
                    slow_context_reused: bool = False):
    """运行PDRS-Myopic算法"""
    return _run_named_algorithm("PDRS-Myopic", system_state, config, slot, seed, slow_context_reused)


def run_FFD_Myopic(system_state, config: AblationExperimentConfig, slot: int, seed: int,
                   slow_context_reused: bool = False):
    """运行FFD-Myopic算法"""
    return _run_named_algorithm("FFD-Myopic", system_state, config, slot, seed, slow_context_reused)


def run_Random_Myopic(system_state, config: AblationExperimentConfig, slot: int, seed: int,
                      slow_context_reused: bool = False):
    """运行Random-Myopic算法"""
    return _run_named_algorithm("Random-Myopic", system_state, config, slot, seed, slow_context_reused)


def run_LoadAware_Myopic(system_state, config: AblationExperimentConfig, slot: int, seed: int,
                         slow_context_reused: bool = False):
    """运行LoadAware-Myopic算法"""
    return _run_named_algorithm("LoadAware-Myopic", system_state, config, slot, seed, slow_context_reused)


def run_GMDA_RMPR_Myopic(system_state, config: AblationExperimentConfig, slot: int, seed: int,
                           slow_context_reused: bool = False):
    """运行GMDA-RMPR-Myopic算法"""
    return _run_named_algorithm("GMDA-RMPR-Myopic", system_state, config, slot, seed, slow_context_reused)


def run_GSLA_LyCD(system_state, config: AblationExperimentConfig, slot: int, seed: int,
                 slow_context_reused: bool = False):
    """运行GSLA-LyCD算法"""
    return _run_named_algorithm("GSLA-LyCD", system_state, config, slot, seed, slow_context_reused)


def _run_named_algorithm(algorithm: str, system_state, config: AblationExperimentConfig,
                         slot: int, seed: int, slow_context_reused: bool = False):
    """统一调度论文命名算法"""
    from ablation_metrics import SlotResult
    from ResourceAllocation import evaluate_ai_action_shared

    slow_policy, fast_controller = get_algorithm_policies(algorithm)
    valid, reason = validate_algorithm_prerequisites(algorithm, config)
    if not valid:
        return SlotResult(
            slot=slot,
            seed=seed,
            algorithm=algorithm,
            slow_policy=slow_policy,
            fast_controller=fast_controller,
            status="invalid",
            failure_reason=reason,
            delay_ms=float("nan"),
            energy_j=float("nan"),
            cost=float("nan"),
            avg_y=float("nan"),
            avg_z=float("nan"),
            dpp_score=float("nan"),
            legacy_reward=float("nan"),
            feasible=False,
            local_count=0,
            cloud_count=0,
            forced_cloud_count=0,
            decision_time_ms=0.0,
            slow_context_reused=slow_context_reused,
            model_path=str(config.resolve_model_path()),
            paper_dpp_score=float("nan"),
            scaled_energy_sum=float("nan"),
            scaled_delay_burden_sum=float("nan"),
            cost_topo=float("nan"),
            cost_comp=float("nan"),
            cost_comm=float("nan"),
            cost_component_consistent=False,
            active_ai_energy_j=float("nan"),
            system_active_ai_chain_energy_j=float("nan"),
        )

    try:
        if fast_controller == "UAC-DO":
            decision = run_UAC_DO(system_state, config, slot=slot, seed=seed, algorithm=algorithm)
        elif fast_controller == "Myopic":
            decision = run_Myopic(system_state, config)
        elif fast_controller == "LyCD":
            decision = run_LyCD(system_state, config)
        else:
            raise ValueError(f"未知快层控制器: {fast_controller}")

        slow_context = get_current_slow_context(system_state, slow_policy)
        if isinstance(decision, dict):
            decision.setdefault("routing_entropy", float(slow_context.get("routing_entropy", 0.0)))
            decision.setdefault("slow_profile_reused", bool(slow_context.get("slow_profile_reused", False)))
            decision.setdefault("routing_policy", str(slow_context.get("routing_policy", slow_policy)))
            decision.setdefault("retry_count", int(slow_context.get("retry_count", 0)))

        return evaluate_ai_action_shared(
            offloading_mode=decision,
            system_state=system_state,
            V=config.V,
            slot=slot,
            seed=seed,
            algorithm=algorithm,
            slow_policy=slow_policy,
            fast_controller=fast_controller,
            model_path=str(config.resolve_model_path()),
            slow_context_reused=slow_context_reused,
            energy_ref_j=config.energy_ref_j,
            delay_ref_ms=config.delay_ref_ms,
            omega_energy=config.omega_energy,
            omega_delay=config.omega_delay,
            resource_queue_aware=(fast_controller != "Myopic"),
            config=config,
        )
    except Exception as exc:
        return SlotResult(
            slot=slot,
            seed=seed,
            algorithm=algorithm,
            slow_policy=slow_policy,
            fast_controller=fast_controller,
            status="failed",
            failure_reason=str(exc),
            delay_ms=float("nan"),
            energy_j=float("nan"),
            cost=float("nan"),
            avg_y=float("nan"),
            avg_z=float("nan"),
            dpp_score=float("nan"),
            legacy_reward=float("nan"),
            feasible=False,
            local_count=0,
            cloud_count=0,
            forced_cloud_count=0,
            decision_time_ms=0.0,
            slow_context_reused=slow_context_reused,
            model_path=str(config.resolve_model_path()),
            paper_dpp_score=float("nan"),
            scaled_energy_sum=float("nan"),
            scaled_delay_burden_sum=float("nan"),
            cost_topo=float("nan"),
            cost_comp=float("nan"),
            cost_comm=float("nan"),
            cost_component_consistent=False,
            active_ai_energy_j=float("nan"),
            system_active_ai_chain_energy_j=float("nan"),
        )


def attach_reference_action_diagnostics(decision: Dict, reference_decision: Dict) -> Dict:
    """记录同上下文Myopic参考动作，只用于UAC机制门禁诊断。"""
    if not reference_decision:
        decision.setdefault("reference_pair_action_bits", "")
        decision.setdefault("reference_pair_action_hash", "")
        decision.setdefault("reference_pair_action_source", "")
        decision.setdefault("repaired_hamming_vs_reference", 0.0)
        return decision

    reference_bits = str(reference_decision.get("pair_action_bits", ""))
    if not reference_bits and reference_decision.get("pair_action") is not None:
        reference_bits = pair_actions.pair_bits_to_text(reference_decision.get("pair_action"))
    selected_bits = str(decision.get("pair_action_bits", ""))
    if not selected_bits and decision.get("pair_action") is not None:
        selected_bits = pair_actions.pair_bits_to_text(decision.get("pair_action"))
    reference_hash = str(
        reference_decision.get("repaired_pair_action_hash") or
        reference_decision.get("pair_action_hash") or
        (pair_actions.pair_action_hash(reference_decision.get("pair_action")) if reference_decision.get("pair_action") is not None else "")
    )
    hamming = 0.0
    if selected_bits and reference_bits and len(selected_bits) == len(reference_bits):
        hamming = sum(1 for left, right in zip(selected_bits, reference_bits) if left != right) / max(len(selected_bits), 1)
    decision["reference_pair_action_bits"] = reference_bits
    decision["reference_pair_action_hash"] = reference_hash
    decision["reference_pair_action_source"] = str(reference_decision.get("selected_candidate_source", reference_decision.get("candidate_source", "")))
    decision["repaired_hamming_vs_reference"] = float(hamming)
    try:
        decision["best_repaired_uac_vs_myopic_dpp_gap"] = float(reference_decision.get("paper_dpp_score", 0.0)) - float(decision.get("paper_dpp_score", 0.0))
    except Exception:
        decision.setdefault("best_repaired_uac_vs_myopic_dpp_gap", 0.0)
    return decision

def run_UAC_DO(system_state, config: AblationExperimentConfig,
               slot: int = 0, seed: int = 0, algorithm: str = "LyHAM-CO"):
    """运行UAC-DO快层控制器"""
    inference = get_cached_uac_inference(config, seed=seed, algorithm=algorithm)
    paper_compact = _is_paper_compact_candidate_mechanism(config)
    if hasattr(inference, "make_candidates"):
        candidates = inference.make_candidates(system_state, config)
    else:
        candidates = [inference.make_decision(system_state)]
    candidates = wrap_candidates_with_pair_projection(candidates, system_state)
    # Myopic修复动作作为hard negative进入最终池；若UAC候选更差，共享evaluator可回退到可执行参考动作。
    myopic_seed_candidates = wrap_candidates_with_pair_projection(
        build_myopic_candidates(system_state, config), system_state
    )
    for idx, candidate in enumerate(myopic_seed_candidates):
        candidate["candidate_source"] = f"myopic_reference_seed_{idx}"
    myopic_reference_candidates = list(myopic_seed_candidates)
    myopic_reference_candidates.extend(build_pair_repair_candidates(
        system_state, config, queue_aware=False,
        seed_candidates=myopic_seed_candidates, source_prefix="myopic_reference"
    ))
    myopic_reference_decision = select_best_action_from_candidates(
        myopic_reference_candidates, system_state, config, queue_aware=False
    )
    myopic_reference_final_candidate = dict(myopic_reference_decision)
    myopic_reference_final_candidate["candidate_source"] = "myopic_reference_repaired"
    myopic_reference_final_candidate["selected_candidate_source"] = "myopic_reference_repaired"
    if not paper_compact:
        candidates.append(myopic_reference_final_candidate)
    else:
        anchor_seed_candidates = []
        for idx, candidate in enumerate(myopic_seed_candidates):
            anchor = dict(candidate)
            anchor["candidate_source"] = f"uac_lyapunov_anchor_seed_{idx}"
            anchor["selected_candidate_source"] = anchor["candidate_source"]
            anchor_seed_candidates.append(anchor)
        anchor_candidates = list(anchor_seed_candidates)
        anchor_candidates.extend(build_pair_repair_candidates(
            system_state, config, queue_aware=True,
            seed_candidates=anchor_seed_candidates, source_prefix="uac_anchor"
        ))
        if anchor_candidates:
            anchor_decision = select_best_action_from_candidates(
                anchor_candidates, system_state, config, queue_aware=True
            )
            anchor_final_candidate = dict(anchor_decision)
            anchor_final_candidate["candidate_source"] = "uac_lyapunov_greedy_anchor"
            anchor_final_candidate["selected_candidate_source"] = "uac_lyapunov_greedy_anchor"
            candidates.append(anchor_final_candidate)
    repair_seed_candidates = list(candidates) + list(myopic_seed_candidates)
    candidates.extend(build_pair_repair_candidates(
        system_state, config, queue_aware=True,
        seed_candidates=repair_seed_candidates, source_prefix="uac"
    ))
    decision = select_best_action_from_candidates(candidates, system_state, config, queue_aware=True)
    attach_reference_action_diagnostics(decision, myopic_reference_decision)
    if is_online_update_enabled_for_algorithm(config, algorithm):
        update_meta = inference.online_update_from_decision(
            system_state, config, decision, seed=int(seed), slot=int(slot)
        )
        decision["replay_written"] = bool(update_meta.get("replay_written", False))
        decision["online_update_step"] = int(update_meta.get("online_update_step", 0))
        decision["model_mutated"] = bool(update_meta.get("model_mutated", False))
        decision["online_loss"] = float(update_meta.get("online_loss", 0.0))
        decision["final_model_hash"] = str(update_meta.get("final_model_hash", ""))
    else:
        decision["replay_written"] = False
        decision["online_update_step"] = 0
        decision["model_mutated"] = False
    return decision


def get_cached_uac_inference(config: AblationExperimentConfig, seed: int = None, algorithm: str = ""):
    """复用同一checkpoint推理对象，避免每个时隙重复加载模型"""
    from ai_inference import create_trained_ai_inference

    model_path = str(config.resolve_model_path())
    strict_pair_actor = bool(getattr(config, "strict_pair_actor_required", False) or getattr(config, "include_energy_claim", False))
    online_enabled = is_online_update_enabled_for_algorithm(config, algorithm)
    online_scope = (str(algorithm), int(seed) if seed is not None else -1) if online_enabled else ("shared", -1)
    if online_enabled:
        cache_namespace = getattr(config, "_uac_cache_namespace", None)
        if cache_namespace is None:
            cache_namespace = f"config:{id(config)}"
    else:
        cache_namespace = "shared"
    cache_key = (model_path, online_enabled, strict_pair_actor, online_scope, str(cache_namespace))
    inference = _UAC_INFERENCE_CACHE.get(cache_key)
    if inference is None:
        inference = create_trained_ai_inference(model_path, strict=True, strict_pair_actor=strict_pair_actor)
        _UAC_INFERENCE_CACHE[cache_key] = inference
        _UAC_CACHE_STATS["model_load_count"] += 1
    else:
        _UAC_CACHE_STATS["model_cache_hit_count"] += 1
    return inference


def get_uac_cache_stats() -> Dict[str, int]:
    """导出UAC模型缓存统计"""
    return dict(_UAC_CACHE_STATS)


def get_uac_online_model_diagnostics() -> Dict:
    """导出在线UAC actor缓存诊断，用于formal meta追踪。"""
    entries = []
    for cache_key, inference in _UAC_INFERENCE_CACHE.items():
        online_enabled = bool(cache_key[1]) if isinstance(cache_key, tuple) and len(cache_key) > 1 else False
        scope = cache_key[3] if isinstance(cache_key, tuple) and len(cache_key) > 3 else ("shared", -1)
        namespace = cache_key[4] if isinstance(cache_key, tuple) and len(cache_key) > 4 else ""
        if not online_enabled:
            continue
        entries.append({
            "algorithm": str(scope[0]) if isinstance(scope, tuple) and len(scope) > 0 else "",
            "seed": int(scope[1]) if isinstance(scope, tuple) and len(scope) > 1 else -1,
            "cache_namespace": str(namespace),
            "online_update_step": int(getattr(inference, "online_update_step", 0)),
            "replay_size": int(len(getattr(inference, "online_replay_buffer", []))),
            "model_mutated": bool(getattr(inference, "model_mutated_during_run", False)),
            "final_model_hash": str(inference.model_state_hash() if hasattr(inference, "model_state_hash") else ""),
        })
    return {"entries": entries}


def run_Myopic(system_state, config: AblationExperimentConfig):
    """运行Myopic快层控制器"""
    candidates = build_myopic_candidates(system_state, config)
    candidates = wrap_candidates_with_pair_projection(candidates, system_state)
    candidates.extend(build_pair_repair_candidates(
        system_state, config, queue_aware=False,
        seed_candidates=candidates, source_prefix="myopic"
    ))
    return select_best_action_from_candidates(candidates, system_state, config, queue_aware=False)


def run_LyCD(system_state, config: AblationExperimentConfig):
    """运行LyCD坐标下降卸载搜索"""
    from ResourceAllocation import evaluate_action_dry_run

    start_time = time.perf_counter()
    env_manager = system_state.environment_manager
    SH, _, _ = env_manager.get_state_components()
    n = len(SH)
    base_candidates = wrap_candidates_with_pair_projection(
        build_myopic_candidates(system_state, config), system_state
    )
    base_candidates.extend(build_pair_repair_candidates(
        system_state, config, queue_aware=True,
        seed_candidates=base_candidates, source_prefix="lycd"
    ))
    best_decision = select_best_action_from_candidates(base_candidates, system_state, config, queue_aware=True)
    current = np.array(best_decision["action"], dtype=int)
    best_score = float(best_decision.get("paper_dpp_score", float("inf")))
    decision_times = []

    max_rounds = max(int(getattr(config, "lycd_max_rounds", 3)), 1)
    evaluated_count = len(base_candidates)
    for _ in range(max_rounds):
        improved = False
        for idx in range(n):
            candidate = current.copy()
            candidate[idx] = 1 - candidate[idx]
            eval_start = time.perf_counter()
            eval_result = evaluate_action_dry_run(candidate, system_state, config, queue_aware=True)
            decision_times.append((time.perf_counter() - eval_start) * 1000.0)
            evaluated_count += 1
            candidate_score = float(eval_result.get("paper_dpp_score", float("inf")))
            if candidate_score < best_score and eval_result.get("feasible", False):
                current = candidate
                best_score = candidate_score
                best_decision.update(eval_result)
                improved = True
        if not improved:
            break

    best_decision["action"] = current
    best_decision["candidate_count"] = evaluated_count
    best_decision["selected_candidate_rank"] = 0
    best_decision["solver_gap_vs_lycd"] = 0.0
    best_decision["p95_decision_time_ms"] = float(np.percentile(decision_times, 95)) if decision_times else 0.0
    best_decision["decision_time_ms"] = (time.perf_counter() - start_time) * 1000.0
    return best_decision



def _candidate_resource_signature(row: Dict) -> str:
    """生成候选资源签名。

    repaired pair动作只描述本地/云端选择；UAC候选还可能携带(g,b,f_GPU)或f_pre偏好。
    这些偏好会改变能耗/时延/成本，不能在去重时被同一个pair动作吞掉。
    """
    hint = str(row.get("resource_hint", "") or "")
    keys = [
        "resource_signature", "resource_hint", "resource_queue_aware",
        "resource_queue_scale", "resource_mode", "gpu_units", "f_gpu",
        "batch_size", "f_pre", "compression_ratio",
    ]
    parts = []
    for key in keys:
        value = row.get(key, None)
        if value in (None, ""):
            continue
        parts.append(f"{key}={value}")
    return "|".join(parts) if (hint or parts) else ""


def dedupe_evaluated_candidates_by_repaired_hash(evaluated_rows: Iterable[Dict]) -> List[Dict]:
    """按repaired动作和资源签名去重，同一可执行配置保留DPP最低候选。"""
    best_by_hash: Dict[str, Dict] = {}
    order = []
    for row in evaluated_rows:
        item = dict(row)
        repaired_hash = str(
            item.get("repaired_pair_action_hash") or
            item.get("pair_action_hash") or
            item.get("action_hash") or
            len(order)
        )
        item["repaired_pair_action_hash"] = repaired_hash
        resource_signature = _candidate_resource_signature(item)
        dedupe_key = repaired_hash if not resource_signature else f"{repaired_hash}::{resource_signature}"
        item["candidate_dedupe_key"] = dedupe_key
        if dedupe_key not in best_by_hash:
            best_by_hash[dedupe_key] = item
            order.append(dedupe_key)
            continue
        old_score = float(best_by_hash[dedupe_key].get("paper_dpp_score", float("inf")))
        new_score = float(item.get("paper_dpp_score", float("inf")))
        if new_score < old_score:
            best_by_hash[dedupe_key] = item
    return [best_by_hash[key] for key in order]


def normalize_candidate_source_family(source: str) -> str:
    """把候选来源压缩为可读家族名。"""
    text = str(source or "")
    if "actor" in text:
        return "actor"
    for prefix in ("uac_", "myopic_", "lycd_", "pair_actor_"):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    if not text:
        return "unknown"
    return text


def summarize_candidate_source_scores(evaluated_rows: Iterable[Dict]) -> Dict[str, Dict[str, float]]:
    """汇总每类候选的最优 DPP/energy/cost/delay。

    该摘要只用于诊断导出，不参与最终动作选择。
    """
    summary: Dict[str, Dict[str, float]] = {}
    for row in evaluated_rows:
        family = normalize_candidate_source_family(str(row.get("candidate_source", "")))
        item = summary.setdefault(family, {
            "count": 0,
            "best_dpp": float("inf"),
            "best_energy": float("inf"),
            "best_cost": float("inf"),
            "best_delay": float("inf"),
            "best_claim_score": float("inf"),
            "best_claim_dpp": float("inf"),
            "best_claim_delay": float("inf"),
            "best_claim_energy": float("inf"),
            "best_claim_cost": float("inf"),
            "best_claim_local_count": 0,
            "best_claim_cloud_count": 0,
        })
        item["count"] += 1
        item["best_dpp"] = min(item["best_dpp"], float(row.get("paper_dpp_score", float("inf"))))
        item["best_energy"] = min(item["best_energy"], float(row.get("energy_j", float("inf"))))
        item["best_cost"] = min(item["best_cost"], float(row.get("cost", float("inf"))))
        item["best_delay"] = min(item["best_delay"], float(row.get("delay_ms", float("inf"))))
        claim_score = float(row.get("claim_score", float("inf")))
        if claim_score < float(item.get("best_claim_score", float("inf"))):
            item["best_claim_score"] = claim_score
            item["best_claim_dpp"] = float(row.get("paper_dpp_score", float("inf")))
            item["best_claim_delay"] = float(row.get("delay_ms", float("inf")))
            item["best_claim_energy"] = float(row.get("energy_j", float("inf")))
            item["best_claim_cost"] = float(row.get("cost", float("inf")))
            item["best_claim_local_count"] = int(row.get("local_count", row.get("local_pair_count", 0)))
            item["best_claim_cloud_count"] = int(row.get("cloud_count", row.get("cloud_pair_count", 0)))
    return summary


def format_candidate_source_score_summary(summary: Dict[str, Dict[str, float]]) -> str:
    """把候选家族评分压缩成单行文本，便于 raw CSV 追踪。"""
    chunks = []
    for family in sorted(summary):
        item = summary[family]
        chunks.append(
            f"{family}:n={int(item.get('count', 0))},"
            f"dpp={float(item.get('best_dpp', float('inf'))):.6g},"
            f"e={float(item.get('best_energy', float('inf'))):.6g},"
            f"c={float(item.get('best_cost', float('inf'))):.6g},"
            f"d={float(item.get('best_delay', float('inf'))):.6g},"
            f"cs={float(item.get('best_claim_score', float('inf'))):.6g},"
            f"bc_dpp={float(item.get('best_claim_dpp', float('inf'))):.6g},"
            f"bc_d={float(item.get('best_claim_delay', float('inf'))):.6g},"
            f"bc_e={float(item.get('best_claim_energy', float('inf'))):.6g},"
            f"bc_c={float(item.get('best_claim_cost', float('inf'))):.6g},"
            f"lc={int(item.get('best_claim_local_count', 0))},"
            f"cc={int(item.get('best_claim_cloud_count', 0))}"
        )
    return ";".join(chunks)


def _average_virtual_energy_queue_state(system_state) -> float:
    queues = getattr(system_state, "virtual_energy_queues", {}) if system_state is not None else {}
    values = []
    for queue in getattr(queues, "values", lambda: [])():
        value = _finite_float(getattr(queue, "queue_state", float("nan")), float("nan"))
        if np.isfinite(value):
            values.append(value)
    if not values:
        return float("nan")
    return float(np.mean(values))


def _is_paper_compact_candidate_mechanism(config: AblationExperimentConfig) -> bool:
    return str(getattr(config, "uac_candidate_mechanism", "legacy")).lower() in {
        "paper_compact", "compact", "lyapunov_compact",
    }


def add_energy_claim_resource_variants(candidates: Iterable, config: AblationExperimentConfig,
                                       queue_aware: bool = True, system_state=None) -> List:
    """为energy-hard的pair候选补充低能耗资源配置变体。

    pair动作仍由原候选给出；变体只改变(g,b,f_GPU)/f_pre资源偏好，最终由共享
    evaluator和claim-band选择器决定是否采用。
    """
    base_candidates = list(candidates)
    if not queue_aware or not getattr(config, "include_energy_claim", False):
        return base_candidates

    expanded = list(base_candidates)
    emitted_keys = set()
    current_avg_y = _average_virtual_energy_queue_state(system_state)
    paper_compact = _is_paper_compact_candidate_mechanism(config)
    suppress_queue_unaware = (
        queue_aware and
        (
            paper_compact or
            (
                bool(getattr(config, "queue_pressure_resource_variant_disable_queue_unaware", True)) and
                np.isfinite(current_avg_y) and
                current_avg_y >= max(
                    float(getattr(config, "queue_pressure_resource_variant_min_current_avg_y", 8.0)),
                    0.0,
                )
            )
        )
    )
    emit_full_queue_relief = queue_aware and (
        paper_compact or
        bool(getattr(config, "queue_pressure_resource_variant_emit_full_queue_relief", False))
    )
    for candidate in base_candidates:
        if not isinstance(candidate, dict):
            continue
        if "claim_energy_saver" in str(candidate.get("resource_hint", "")):
            continue
        pair_action = candidate.get("pair_action")
        if pair_action is None:
            continue
        try:
            pair_bits = np.asarray(pair_action, dtype=int).reshape(-1)
        except Exception:
            continue
        if len(pair_bits) == 0 or not np.any(pair_bits == 0):
            continue
        source = str(candidate.get("candidate_source", "") or "candidate")
        key = (
            tuple(int(bit) for bit in pair_bits),
            tuple(int(bit) for bit in np.asarray(candidate.get("action", []), dtype=int).reshape(-1)),
            source,
            str(candidate.get("resource_hint", "")),
        )
        if key in emitted_keys:
            continue
        emitted_keys.add(key)
        base_resource_hint = str(candidate.get("resource_hint", "") or "")
        if paper_compact and "lydroo_" not in source:
            for mode, hint in (
                    ("lydroo_balanced_resource_frontier", base_resource_hint),
                    ("lydroo_delay_resource_frontier", "latency_saver_hybrid"),
                    ("lydroo_energy_resource_frontier", "energy_saver_local"),
                    ("lydroo_cost_resource_frontier", "cost_saver_hybrid"),
            ):
                frontier = dict(candidate)
                frontier["candidate_source"] = f"{source}_{mode}"
                frontier["selected_candidate_source"] = frontier["candidate_source"]
                frontier["base_resource_hint"] = base_resource_hint
                frontier["resource_hint"] = hint
                frontier["resource_queue_aware"] = True
                frontier["resource_queue_scale"] = 1.0
                frontier["resource_mode"] = mode
                frontier["lydroo_resource_frontier"] = True
                expanded.append(frontier)
        is_balanced_tail_relief = "balanced_tail_relief" in source
        if not is_balanced_tail_relief and not suppress_queue_unaware:
            variant = dict(candidate)
            variant["candidate_source"] = f"{source}_claim_energy_saver"
            variant["selected_candidate_source"] = variant["candidate_source"]
            variant["base_resource_hint"] = base_resource_hint
            variant["resource_hint"] = "claim_energy_saver_local"
            variant["resource_queue_aware"] = False
            variant["resource_mode"] = "energy_relaxed"
            expanded.append(variant)
        if any(token in source for token in (
                "reference_low_impact_neighbor",
                "balanced_tail_relief",
                "energy_cloud_relief",
                "energy_queue_relief",
                "energy_cost_pareto_relief",
        )) and "queue_relaxed_cloud_relief" not in source:
            for scale, suffix in ((0.10, "010"), (0.20, "020"), (0.35, "")):
                damped = dict(candidate)
                mode = f"queue_damped{suffix}_cloud_relief" if suffix else "queue_damped_cloud_relief"
                damped["candidate_source"] = f"{source}_{mode}"
                damped["selected_candidate_source"] = damped["candidate_source"]
                damped["base_resource_hint"] = str(candidate.get("resource_hint", ""))
                damped["resource_hint"] = str(candidate.get("resource_hint", "")) or "cloud_relief_f_pre"
                damped["resource_queue_aware"] = True
                damped["resource_queue_scale"] = float(scale)
                damped["resource_mode"] = mode
                expanded.append(damped)
            if emit_full_queue_relief:
                full = dict(candidate)
                full["candidate_source"] = f"{source}_queue_full_cloud_relief"
                full["selected_candidate_source"] = full["candidate_source"]
                full["base_resource_hint"] = str(candidate.get("resource_hint", ""))
                full["resource_hint"] = str(candidate.get("resource_hint", "")) or "cloud_relief_f_pre"
                full["resource_queue_aware"] = True
                full["resource_queue_scale"] = 1.0
                full["resource_mode"] = "queue_full_cloud_relief"
                expanded.append(full)
            allow_relaxed_cloud_relief = (
                paper_compact and
                not is_balanced_tail_relief and
                any(token in source for token in (
                    "energy_cloud_relief",
                    "energy_queue_relief",
                    "energy_cost_pareto_relief",
                ))
            )
            if (
                    not is_balanced_tail_relief and
                    (not suppress_queue_unaware or allow_relaxed_cloud_relief)
            ):
                balanced = dict(candidate)
                balanced["candidate_source"] = f"{source}_queue_relaxed_cloud_relief"
                balanced["selected_candidate_source"] = balanced["candidate_source"]
                balanced["base_resource_hint"] = str(candidate.get("resource_hint", ""))
                balanced["resource_hint"] = str(candidate.get("resource_hint", "")) or "cloud_relief_f_pre"
                balanced["resource_queue_aware"] = False
                balanced["resource_queue_scale"] = 0.0
                balanced["resource_mode"] = "queue_relaxed_cloud_relief"
                expanded.append(balanced)
    return expanded


def calculate_claim_score(row: Dict, config: AblationExperimentConfig) -> float:
    """
    计算三指标claim score
    该分数只使用预注册D0/E0/C0，不读取baseline结果，也不按算法名加权。
    """
    delay_ref = max(float(getattr(config, "claim_delay_ref_ms", 100.0)), 1e-9)
    energy_ref = max(float(getattr(config, "claim_energy_ref_j", 2.0)), 1e-9)
    cost_ref = max(float(getattr(config, "claim_cost_ref", 400.0)), 1e-9)
    delay_value = float(row.get("delay_ms", float("inf")))
    energy_value = float(row.get("energy_j", float("inf")))
    cost_value = float(row.get("cost", float("inf")))
    return float(delay_value / delay_ref + energy_value / energy_ref + cost_value / cost_ref)


def _is_non_dominated(row: Dict, rows: Iterable[Dict]) -> bool:
    """检查候选是否在delay/energy/cost三指标上非支配。"""
    metrics = ("delay_ms", "energy_j", "cost")
    values = [float(row.get(metric, float("inf"))) for metric in metrics]
    for other in rows:
        if other is row:
            continue
        other_values = [float(other.get(metric, float("inf"))) for metric in metrics]
        no_worse = all(other_values[idx] <= values[idx] for idx in range(len(metrics)))
        strictly_better = any(other_values[idx] < values[idx] for idx in range(len(metrics)))
        if no_worse and strictly_better:
            return False
    return True


def annotate_claim_selector_rows(rows: List[Dict], config: AblationExperimentConfig,
                                 best_dpp_score: float, dpp_tolerance: float) -> None:
    """为候选写入claim selector诊断字段。"""
    if not rows:
        return
    finite_rows = [row for row in rows if row.get("feasible", False)]
    min_delay = min((float(row.get("delay_ms", float("inf"))) for row in finite_rows), default=0.0)
    min_energy = min((float(row.get("energy_j", float("inf"))) for row in finite_rows), default=0.0)
    min_cost = min((float(row.get("cost", float("inf"))) for row in finite_rows), default=0.0)
    band_rows = [
        row for row in rows
        if float(row.get("paper_dpp_score", float("inf"))) <= best_dpp_score + dpp_tolerance
    ]
    band_ids = {id(row) for row in band_rows}
    for row in rows:
        pair_count = max(int(row.get("local_pair_count", 0)) + int(row.get("cloud_pair_count", 0)), 1)
        row["claim_score"] = calculate_claim_score(row, config)
        row["dpp_band_passed"] = bool(id(row) in band_ids)
        row["is_pareto_candidate"] = bool(id(row) in band_ids and _is_non_dominated(row, band_rows))
        row["per_pair_delta_delay"] = f"{(float(row.get('delay_ms', 0.0)) - min_delay) / pair_count:.6f}"
        row["per_pair_delta_energy"] = f"{(float(row.get('energy_j', 0.0)) - min_energy) / pair_count:.6f}"
        row["per_pair_delta_cost"] = f"{(float(row.get('cost', 0.0)) - min_cost) / pair_count:.6f}"


def build_claim_escape_pool(rows: List[Dict], current_pool: List[Dict],
                            config: AblationExperimentConfig,
                            best_dpp_score: float) -> List[Dict]:
    """在energy-hard下允许三指标显著更优的Pareto候选逃逸窄DPP带。

    该入口只服务claim-band选择：候选必须全局非支配、DPP仍受上界约束，
    且相对当前带内最优claim候选有足够delay驱动的归一化收益。
    """
    feasible_rows = [row for row in rows if bool(row.get("feasible", False))]
    if not feasible_rows or not current_pool:
        return []
    claim_rank_key = lambda row: (
        not bool(row.get("feasible", False)),
        float(row.get("claim_score", float("inf"))),
        float(row.get("paper_dpp_score", float("inf"))),
        int(row.get("eval_rank", 0)),
    )
    dpp_rank_key = lambda row: (
        not bool(row.get("feasible", False)),
        float(row.get("paper_dpp_score", float("inf"))),
        int(row.get("eval_rank", 0)),
    )
    dpp_best_row = min(feasible_rows, key=dpp_rank_key)
    band_best_row = min(current_pool, key=claim_rank_key)
    band_claim_score = float(band_best_row.get("claim_score", float("inf")))
    if not np.isfinite(band_claim_score):
        return []

    delay_ref = max(float(getattr(config, "claim_delay_ref_ms", 100.0)), 1e-9)
    energy_ref = max(float(getattr(config, "claim_energy_ref_j", 2.0)), 1e-9)
    cost_ref = max(float(getattr(config, "claim_cost_ref", 400.0)), 1e-9)
    dpp_cap_ratio = max(float(getattr(config, "energy_hard_claim_escape_dpp_ratio", 2.0)), 1.0)
    min_score_gain = max(float(getattr(config, "energy_hard_claim_escape_min_score_gain", 0.20)), 0.0)
    min_delay_gain = max(float(getattr(config, "energy_hard_claim_escape_min_delay_gain", 0.15)), 0.0)
    dpp_cap = max(abs(float(best_dpp_score)), 1e-9) * dpp_cap_ratio
    dpp_best_delay = float(dpp_best_row.get("delay_ms", float("inf")))
    dpp_best_energy = float(dpp_best_row.get("energy_j", float("inf")))
    dpp_best_cost = float(dpp_best_row.get("cost", float("inf")))

    escape_pool: List[Dict] = []
    for row in feasible_rows:
        row["claim_escape_passed"] = False
        row_dpp = float(row.get("paper_dpp_score", float("inf")))
        row_claim_score = float(row.get("claim_score", float("inf")))
        row_delay = float(row.get("delay_ms", float("inf")))
        if not np.isfinite(row_dpp) or not np.isfinite(row_claim_score) or not np.isfinite(row_delay):
            continue
        if row_dpp > dpp_cap:
            continue
        if not _is_non_dominated(row, feasible_rows):
            continue
        claim_gain = band_claim_score - row_claim_score
        if claim_gain < min_score_gain:
            continue
        delay_gain = max(dpp_best_delay - row_delay, 0.0) / delay_ref
        if delay_gain < min_delay_gain:
            continue
        energy_penalty = max(float(row.get("energy_j", float("inf"))) - dpp_best_energy, 0.0) / energy_ref
        cost_penalty = max(float(row.get("cost", float("inf"))) - dpp_best_cost, 0.0) / cost_ref
        if energy_penalty + cost_penalty > delay_gain * 0.90 + 0.05:
            continue
        row["claim_escape_passed"] = True
        escape_pool.append(row)
    return escape_pool


def _finite_float(value, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(number):
        return float(default)
    return float(number)


def _candidate_local_cloud_counts(row: Dict) -> Tuple[float, float]:
    """Return pair-level local/cloud counts, falling back to server counts."""
    local_pair = _finite_float(row.get("local_pair_count"), float("nan"))
    cloud_pair = _finite_float(row.get("cloud_pair_count"), float("nan"))
    if np.isfinite(local_pair) and np.isfinite(cloud_pair) and local_pair + cloud_pair > 0:
        return local_pair, cloud_pair
    local_count = _finite_float(row.get("local_count"), float("nan"))
    cloud_count = _finite_float(row.get("cloud_count"), float("nan"))
    return local_count, cloud_count


def build_temporal_metric_history_entry(result) -> Dict:
    """Build the rolling LyHAM temporal history row from a SlotResult-like object."""
    entry = {
        "delay_ms": _finite_float(getattr(result, "delay_ms", float("nan")), float("nan")),
        "energy_j": _finite_float(getattr(result, "energy_j", float("nan")), float("nan")),
        "cost": _finite_float(getattr(result, "cost", float("nan")), float("nan")),
        "selected_candidate_source": str(getattr(result, "selected_candidate_source", "")),
        "local_count": int(_finite_float(getattr(result, "local_count", 0), 0.0)),
        "cloud_count": int(_finite_float(getattr(result, "cloud_count", 0), 0.0)),
        "local_pair_count": int(_finite_float(getattr(result, "local_pair_count", 0), 0.0)),
        "cloud_pair_count": int(_finite_float(getattr(result, "cloud_pair_count", 0), 0.0)),
    }
    for field in (
        "predicted_avg_y",
        "predicted_avg_z",
        "post_update_queue_drift_term",
        "post_update_queue_delta_term",
        "post_update_energy_queue_delta_term",
        "post_update_delay_queue_delta_term",
    ):
        value = _finite_float(getattr(result, field, float("nan")), float("nan"))
        if np.isfinite(value):
            entry[field] = float(value)
    return entry


def queue_pressure_dpp_required(rows: List[Dict], config: AblationExperimentConfig) -> bool:
    """Detect when real virtual-queue pressure should dominate claim-band tie-breaking."""
    if not getattr(config, "include_energy_claim", False):
        return False
    finite_rows = [
        row for row in rows
        if bool(row.get("feasible", False)) and np.isfinite(_finite_float(row.get("paper_dpp_score"), float("inf")))
    ]
    if not finite_rows:
        return False
    max_queue_term = max(
        max(_finite_float(row.get("energy_queue_term")), 0.0) +
        max(_finite_float(row.get("delay_queue_term")), 0.0)
        for row in finite_rows
    )
    if max_queue_term <= 0.0:
        return False
    dpp_best_row = min(
        finite_rows,
        key=lambda row: (
            _finite_float(row.get("paper_dpp_score"), float("inf")),
            int(row.get("eval_rank", 0)),
        ),
    )
    v_cost_term = abs(_finite_float(dpp_best_row.get("v_cost_term")))
    if v_cost_term <= 1e-9:
        return False
    min_queue_to_v_ratio = float(
        getattr(config, "queue_pressure_dpp_min_queue_to_v_ratio", 0.25)
    )
    return max_queue_term / max(v_cost_term, 1e-9) >= min_queue_to_v_ratio


def _queue_delay_guard_energy_pressure_bounded(
    row: Dict,
    reference_row: Dict,
    config: AblationExperimentConfig,
    *,
    severe_tail_relief: bool = False,
) -> bool:
    reference_y = _finite_float(
        reference_row.get("predicted_avg_y", reference_row.get("avg_y")), float("nan")
    )
    min_reference_y = max(
        float(getattr(config, "queue_pressure_delay_guard_energy_pressure_min_current_avg_y", 8.0)),
        0.0,
    )
    if not np.isfinite(reference_y) or reference_y < min_reference_y:
        return True

    y_limit = max(
        float(getattr(config, "queue_pressure_delay_guard_energy_pressure_max_predicted_y_regret", 0.25)),
        0.0,
    )
    if severe_tail_relief:
        y_limit = max(
            y_limit,
            max(float(getattr(config, "queue_pressure_delay_guard_max_predicted_y_regret", 8.0)), 0.0),
        )
    row_y = _finite_float(row.get("predicted_avg_y", row.get("avg_y")), float("nan"))
    if not np.isfinite(row_y) or row_y - reference_y > y_limit:
        return False

    energy_delta_limit = max(
        float(getattr(config, "queue_pressure_delay_guard_energy_pressure_max_energy_queue_delta_regret", 0.0)),
        0.0,
    )
    if severe_tail_relief:
        energy_delta_limit = max(
            energy_delta_limit,
            max(
                float(getattr(
                    config,
                    "queue_pressure_delay_guard_severe_max_energy_queue_delta_regret",
                    80.0,
                )),
                0.0,
            ),
        )
    reference_energy_delta = _finite_float(
        reference_row.get("post_update_energy_queue_delta_term"), float("nan")
    )
    if np.isfinite(reference_energy_delta):
        row_energy_delta = _finite_float(row.get("post_update_energy_queue_delta_term"), float("nan"))
        if not np.isfinite(row_energy_delta) or row_energy_delta - reference_energy_delta > energy_delta_limit:
            return False

    drift_limit = max(
        float(getattr(config, "queue_pressure_delay_guard_energy_pressure_max_queue_drift_regret", 5.0)),
        0.0,
    )
    if severe_tail_relief:
        drift_limit = max(
            drift_limit,
            max(float(getattr(config, "queue_pressure_delay_guard_severe_max_queue_drift_regret", 80.0)), 0.0),
        )
    reference_drift = _finite_float(reference_row.get("post_update_queue_drift_term"), float("nan"))
    if np.isfinite(reference_drift):
        row_drift = _finite_float(row.get("post_update_queue_drift_term"), float("nan"))
        if np.isfinite(row_drift) and row_drift - reference_drift > drift_limit:
            return False
    return True


def queue_pressure_delay_guard_candidate(rows: List[Dict], config: AblationExperimentConfig,
                                         dpp_best_row: Dict) -> Dict:
    """Pick a near-DPP candidate that avoids a severe response-delay spike."""
    best_delay = _finite_float(dpp_best_row.get("delay_ms"), float("inf"))
    if not np.isfinite(best_delay):
        return {}
    delay_ref = max(float(getattr(config, "claim_delay_ref_ms", 100.0)), 1e-9)
    min_gain = max(
        float(getattr(config, "queue_pressure_delay_guard_min_normalized_gain", 0.5)),
        0.0,
    )
    delay_threshold = _finite_float(getattr(config, "queue_delay_threshold_ms", float("nan")), float("nan"))
    guard_pool = []
    for row in rows:
        row_delay = _finite_float(row.get("delay_ms"), float("inf"))
        if not np.isfinite(row_delay):
            continue
        if (
            np.isfinite(delay_threshold) and delay_threshold > 0.0 and
            not bool(row.get("dpp_band_passed", False)) and
            row_delay > delay_threshold
        ):
            continue
        normalized_gain = (best_delay - row_delay) / delay_ref
        if not _queue_delay_guard_energy_pressure_bounded(row, dpp_best_row, config):
            continue
        if normalized_gain >= min_gain:
            guard_pool.append(row)
    if not guard_pool:
        return {}
    return min(
        guard_pool,
        key=lambda row: (
            _finite_float(row.get("delay_ms"), float("inf")),
            _finite_float(row.get("claim_score"), float("inf")),
            _finite_float(row.get("paper_dpp_score"), float("inf")),
            int(row.get("eval_rank", 0)),
        ),
    )


def queue_pressure_tail_delay_guard_candidate(rows: List[Dict], config: AblationExperimentConfig,
                                              dpp_best_row: Dict) -> Dict:
    """Allow bounded DPP regret only for strong tail-delay relief under queue pressure."""
    best_delay = _finite_float(dpp_best_row.get("delay_ms"), float("inf"))
    best_dpp = _finite_float(dpp_best_row.get("paper_dpp_score"), float("inf"))
    delay_threshold = _finite_float(getattr(config, "queue_delay_threshold_ms", float("nan")), float("nan"))
    if not (
        np.isfinite(best_delay) and np.isfinite(best_dpp) and
        np.isfinite(delay_threshold) and delay_threshold > 0.0 and
        best_delay > delay_threshold
    ):
        return {}
    delay_ref = max(float(getattr(config, "claim_delay_ref_ms", 100.0)), 1e-9)
    min_gain = max(
        float(getattr(config, "queue_pressure_delay_guard_min_normalized_gain", 0.5)),
        0.0,
    )
    dpp_ratio = max(
        float(getattr(config, "queue_pressure_delay_guard_dpp_regret_ratio", 0.0)),
        0.0,
    )
    severe_delay = max(
        float(getattr(config, "queue_pressure_delay_guard_severe_delay_ms", 360.0)),
        delay_threshold,
    )
    severe_tail = bool(best_delay >= severe_delay)
    if severe_tail:
        dpp_ratio = max(
            dpp_ratio,
            max(
                float(getattr(config, "queue_pressure_delay_guard_severe_dpp_regret_ratio", 0.55)),
                0.0,
            ),
        )
    if dpp_ratio <= 0.0:
        return {}
    dpp_budget = max(
        abs(best_dpp) * dpp_ratio,
        abs(best_dpp) * max(float(getattr(config, "energy_hard_dpp_tolerance_ratio", 0.03)), 0.0),
        1e-9,
    )
    max_energy_regret = max(
        float(getattr(config, "queue_pressure_delay_guard_max_energy_regret_j", 3.0)),
        0.0,
    )
    max_cost_regret = max(
        float(getattr(config, "queue_pressure_delay_guard_max_cost_regret", 120.0)),
        0.0,
    )
    max_predicted_z_regret = max(
        float(getattr(config, "queue_pressure_delay_guard_max_predicted_z_regret", 0.5)),
        0.0,
    )
    max_queue_drift_regret = max(
        float(getattr(config, "queue_pressure_delay_guard_max_queue_drift_regret", 15.0)),
        0.0,
    )
    if severe_tail:
        max_predicted_z_regret = max(
            max_predicted_z_regret,
            max(
                float(getattr(config, "queue_pressure_delay_guard_severe_max_predicted_z_regret", 1.0)),
                0.0,
            ),
        )
        max_queue_drift_regret = max(
            max_queue_drift_regret,
            max(
                float(getattr(config, "queue_pressure_delay_guard_severe_max_queue_drift_regret", 80.0)),
                0.0,
            ),
        )
    queue_limits = (
        ("predicted_avg_y", max(float(getattr(config, "queue_pressure_delay_guard_max_predicted_y_regret", 8.0)), 0.0)),
        ("predicted_avg_z", max_predicted_z_regret),
        ("post_update_queue_drift_term", max_queue_drift_regret),
    )
    best_energy = _finite_float(dpp_best_row.get("energy_j"), float("inf"))
    best_cost = _finite_float(dpp_best_row.get("cost"), float("inf"))

    def queue_bounded(row: Dict) -> bool:
        for field, limit in queue_limits:
            candidate_value = _finite_float(row.get(field), float("nan"))
            best_value = _finite_float(dpp_best_row.get(field), float("nan"))
            if np.isfinite(candidate_value) and np.isfinite(best_value):
                if candidate_value - best_value > limit:
                    return False
        return True

    guard_pool = []
    for row in rows:
        if not bool(row.get("feasible", False)):
            continue
        row_delay = _finite_float(row.get("delay_ms"), float("inf"))
        row_dpp = _finite_float(row.get("paper_dpp_score"), float("inf"))
        row_energy = _finite_float(row.get("energy_j"), float("inf"))
        row_cost = _finite_float(row.get("cost"), float("inf"))
        if not all(np.isfinite(value) for value in [row_delay, row_dpp, row_energy, row_cost]):
            continue
        if row_delay > delay_threshold:
            continue
        if (best_delay - row_delay) / delay_ref < min_gain:
            continue
        if row_dpp > best_dpp + dpp_budget:
            continue
        if np.isfinite(best_energy) and row_energy - best_energy > max_energy_regret:
            continue
        if np.isfinite(best_cost) and row_cost - best_cost > max_cost_regret:
            continue
        if not queue_bounded(row):
            continue
        if not _queue_delay_guard_energy_pressure_bounded(
            row, dpp_best_row, config, severe_tail_relief=severe_tail
        ):
            continue
        guard_pool.append(row)
    if not guard_pool:
        return {}
    return min(
        guard_pool,
        key=lambda row: (
            _finite_float(row.get("delay_ms"), float("inf")),
            _finite_float(row.get("predicted_avg_z"), float("inf")),
            _finite_float(row.get("predicted_avg_y"), float("inf")),
            _finite_float(row.get("paper_dpp_score"), float("inf")),
            int(row.get("eval_rank", 0)),
        ),
    )


def _energy_guard_queue_drift_regret_reason(
    row: Dict,
    current_row: Dict,
    drift_regret: float,
    drift_regret_limit: float,
    config: AblationExperimentConfig,
) -> str:
    if drift_regret <= drift_regret_limit:
        return ""
    offset_ratio = max(
        float(getattr(
            config,
            "queue_pressure_energy_guard_dpp_improvement_queue_drift_offset_ratio",
            0.0,
        )),
        0.0,
    )
    if offset_ratio <= 0.0:
        return "queue_drift_regret"
    current_dpp = _finite_float(current_row.get("paper_dpp_score"), float("nan"))
    row_dpp = _finite_float(row.get("paper_dpp_score"), float("nan"))
    if not all(np.isfinite(value) for value in [current_dpp, row_dpp]):
        return "queue_drift_regret"
    dpp_improvement = current_dpp - row_dpp
    if dpp_improvement > 0.0 and drift_regret <= dpp_improvement * offset_ratio:
        return ""
    return "queue_drift_regret"


def _energy_guard_queue_relief_reason(row: Dict, current_row: Dict,
                                      config: AblationExperimentConfig) -> str:
    current_y = _finite_float(current_row.get("predicted_avg_y"), float("nan"))
    min_current_y = max(
        float(getattr(config, "queue_pressure_energy_guard_min_current_avg_y", 8.0)),
        0.0,
    )
    queue_term = (
        max(_finite_float(current_row.get("energy_queue_term")), 0.0) +
        max(_finite_float(current_row.get("delay_queue_term")), 0.0)
    )
    v_cost_term = abs(_finite_float(current_row.get("v_cost_term")))
    min_queue_to_v_ratio = max(
        float(getattr(config, "queue_pressure_dpp_min_queue_to_v_ratio", 0.25)),
        0.0,
    )
    material_queue_pressure = (
        np.isfinite(queue_term) and
        np.isfinite(v_cost_term) and
        v_cost_term > 1e-9 and
        queue_term / max(v_cost_term, 1e-9) >= min_queue_to_v_ratio
    )
    if not np.isfinite(current_y) or (current_y < min_current_y and not material_queue_pressure):
        return "energy_queue_pressure_below_min"

    y_regret_limit = max(
        float(getattr(config, "queue_pressure_energy_guard_max_predicted_y_regret", 0.25)),
        0.0,
    )
    energy_delta_regret_limit = max(
        float(getattr(config, "queue_pressure_energy_guard_max_energy_queue_delta_regret", 0.0)),
        0.0,
    )
    drift_regret_limit = max(
        float(getattr(config, "queue_pressure_energy_guard_max_queue_drift_regret", 5.0)),
        0.0,
    )
    min_relief = max(
        float(getattr(config, "queue_pressure_energy_guard_min_queue_relief", 0.05)),
        0.0,
    )
    require_relief = bool(getattr(config, "queue_pressure_energy_guard_require_queue_relief", True))

    has_relief = False
    row_y = _finite_float(row.get("predicted_avg_y"), float("nan"))
    if np.isfinite(row_y):
        if row_y - current_y > y_regret_limit:
            return "predicted_avg_y_regret"
        has_relief = has_relief or (current_y - row_y > min_relief)
    else:
        return "predicted_avg_y_missing"

    current_energy_delta = _finite_float(
        current_row.get("post_update_energy_queue_delta_term"), float("nan")
    )
    if np.isfinite(current_energy_delta):
        row_energy_delta = _finite_float(
            row.get("post_update_energy_queue_delta_term"), float("nan")
        )
        if not np.isfinite(row_energy_delta):
            return "energy_queue_delta_missing"
        if row_energy_delta - current_energy_delta > energy_delta_regret_limit:
            return "energy_queue_delta_regret"
        has_relief = has_relief or (current_energy_delta - row_energy_delta > min_relief)

    current_drift = _finite_float(current_row.get("post_update_queue_drift_term"), float("nan"))
    if np.isfinite(current_drift):
        row_drift = _finite_float(row.get("post_update_queue_drift_term"), float("nan"))
        if np.isfinite(row_drift):
            drift_reason = _energy_guard_queue_drift_regret_reason(
                row,
                current_row,
                row_drift - current_drift,
                drift_regret_limit,
                config,
            )
            if drift_reason:
                return drift_reason
            has_relief = has_relief or (current_drift - row_drift > min_relief)

    if require_relief and not has_relief:
        return "queue_relief_absent"
    return ""


def _energy_guard_latency_queue_regret_reason(row: Dict, current_row: Dict,
                                              config: AblationExperimentConfig) -> str:
    """Reject energy relief that worsens both latency queue state dimensions."""
    min_relief = max(
        float(getattr(config, "queue_pressure_energy_guard_min_queue_relief", 0.05)),
        0.0,
    )
    current_z = _finite_float(current_row.get("predicted_avg_z"), float("nan"))
    row_z = _finite_float(row.get("predicted_avg_z"), float("nan"))
    current_delay_delta = _finite_float(
        current_row.get("post_update_delay_queue_delta_term"), float("nan")
    )
    row_delay_delta = _finite_float(
        row.get("post_update_delay_queue_delta_term"), float("nan")
    )
    if not all(np.isfinite(value) for value in [current_z, row_z, current_delay_delta, row_delay_delta]):
        return ""
    if row_z - current_z > min_relief and row_delay_delta - current_delay_delta > min_relief:
        return "latency_queue_debt_regret"
    return ""


def _energy_guard_delay_regret_reason(
    row_delay: float,
    current_delay: float,
    row_dpp: float,
    current_dpp: float,
    high_queue_pressure: bool,
    max_delay_regret: float,
    config: AblationExperimentConfig,
) -> str:
    delay_regret = row_delay - current_delay
    if delay_regret <= max_delay_regret:
        return ""
    if not high_queue_pressure or row_dpp >= current_dpp:
        return "delay_regret"
    slack_ratio = max(
        float(getattr(config, "queue_pressure_energy_guard_dpp_improvement_delay_slack_ratio", 0.0)),
        0.0,
    )
    delay_slack = max_delay_regret * slack_ratio
    if delay_slack <= 0.0:
        return "delay_regret"
    if delay_regret <= max_delay_regret + delay_slack:
        return ""
    return "delay_regret"


def _energy_guard_strict_dpp_regret_reason(row: Dict, current_row: Dict,
                                           config: AblationExperimentConfig) -> str:
    strict_dpp = _finite_float(
        current_row.get("energy_guard_strict_dpp_baseline_score"),
        float("nan"),
    )
    if not np.isfinite(strict_dpp):
        return ""
    row_dpp = _finite_float(row.get("paper_dpp_score"), float("nan"))
    if not np.isfinite(row_dpp):
        return "strict_dpp_regret"
    regret_ratio = max(
        float(getattr(config, "queue_pressure_energy_guard_strict_dpp_regret_ratio", 0.0)),
        0.0,
    )
    strict_dpp_budget = abs(strict_dpp) * regret_ratio
    if row_dpp <= strict_dpp + strict_dpp_budget:
        return ""
    return "strict_dpp_regret"


def _energy_guard_active_cloud_signature(row: Dict) -> str:
    return str(row.get("active_cloud_pair_signature") or "").strip()


def _energy_guard_is_relaxed_cloud_resource(row: Dict) -> bool:
    mode = str(row.get("resource_mode", ""))
    source = str(row.get("candidate_source", ""))
    return "queue_relaxed_cloud_relief" in mode or "queue_relaxed_cloud_relief" in source


def _energy_guard_is_same_cloud_recovery_probe(relaxed_row: Dict, probe_row: Dict,
                                               config: AblationExperimentConfig) -> bool:
    signature = _energy_guard_active_cloud_signature(relaxed_row)
    if not signature or _energy_guard_active_cloud_signature(probe_row) != signature:
        return False
    if probe_row is relaxed_row or _energy_guard_is_relaxed_cloud_resource(probe_row):
        return False
    mode = str(probe_row.get("resource_mode", ""))
    source = str(probe_row.get("candidate_source", ""))
    if not (
        "queue_damped" in mode or
        "queue_full_cloud_relief" in mode or
        "queue_damped" in source or
        "queue_full_cloud_relief" in source or
        bool(probe_row.get("resource_queue_aware", False))
    ):
        return False
    min_relief = max(
        float(getattr(config, "queue_pressure_energy_guard_min_queue_relief", 0.05)),
        0.0,
    )
    relaxed_delay_delta = _finite_float(
        relaxed_row.get("post_update_delay_queue_delta_term"), float("nan")
    )
    probe_delay_delta = _finite_float(
        probe_row.get("post_update_delay_queue_delta_term"), float("nan")
    )
    relaxed_z = _finite_float(relaxed_row.get("predicted_avg_z"), float("nan"))
    probe_z = _finite_float(probe_row.get("predicted_avg_z"), float("nan"))
    relaxed_delay = _finite_float(relaxed_row.get("delay_ms"), float("nan"))
    probe_delay = _finite_float(probe_row.get("delay_ms"), float("nan"))
    return (
        np.isfinite(relaxed_delay_delta) and
        np.isfinite(probe_delay_delta) and
        relaxed_delay_delta - probe_delay_delta > min_relief and
        np.isfinite(relaxed_z) and
        np.isfinite(probe_z) and
        relaxed_z - probe_z > min_relief and
        np.isfinite(relaxed_delay) and
        np.isfinite(probe_delay) and
        relaxed_delay - probe_delay > min_relief
    )


def _energy_guard_filter_relaxed_same_cloud_recovery_regret(
    guard_pool: List[Dict],
    rows: List[Dict],
    config: AblationExperimentConfig,
) -> List[Dict]:
    filtered = []
    for row in guard_pool:
        if not _energy_guard_is_relaxed_cloud_resource(row):
            filtered.append(row)
            continue
        has_recovery_probe = any(
            _energy_guard_is_same_cloud_recovery_probe(row, probe_row, config)
            for probe_row in rows
        )
        if not has_recovery_probe:
            filtered.append(row)
            continue
        # A same-cloud queue-aware resource row shows that the relaxed row carries
        # avoidable delay debt. Keep only recovery rows that passed the ordinary
        # energy/DPP gates; otherwise decline this relaxed override.
        has_admissible_recovery = any(
            _energy_guard_is_same_cloud_recovery_probe(row, probe_row, config)
            for probe_row in guard_pool
        )
        if has_admissible_recovery:
            continue
    return filtered


def queue_pressure_energy_guard_candidate(rows: List[Dict], config: AblationExperimentConfig,
                                          current_row: Dict) -> Dict:
    """Pick a lower-energy candidate only when it also relieves the long-term energy queue."""
    if not getattr(config, "queue_pressure_energy_guard_enabled", False):
        return {}
    current_energy = _finite_float(current_row.get("energy_j"), float("inf"))
    current_delay = _finite_float(current_row.get("delay_ms"), float("inf"))
    current_cost = _finite_float(current_row.get("cost"), float("inf"))
    current_dpp = _finite_float(current_row.get("paper_dpp_score"), float("inf"))
    if not all(np.isfinite(value) for value in [current_energy, current_delay, current_cost, current_dpp]):
        return {}
    min_energy_gain = max(
        float(getattr(config, "queue_pressure_energy_guard_min_energy_gain_j", 1.0)),
        0.0,
    )
    max_delay_regret = max(
        float(getattr(config, "queue_pressure_energy_guard_max_delay_regret_ms", 20.0)),
        0.0,
    )
    max_cost_regret = max(
        float(getattr(config, "queue_pressure_energy_guard_max_cost_regret", 80.0)),
        0.0,
    )
    relief_reason = _energy_guard_queue_relief_reason(current_row, current_row, config)
    high_queue_pressure = relief_reason in {"queue_relief_absent", ""}
    dpp_slack = 0.0
    if high_queue_pressure:
        dpp_slack = max(
            abs(current_dpp) * max(float(getattr(config, "queue_pressure_energy_guard_dpp_slack_ratio", 0.0)), 0.0),
            0.0,
        )
    guard_pool = []
    for row in rows:
        if row is current_row:
            continue
        row_energy = _finite_float(row.get("energy_j"), float("inf"))
        row_delay = _finite_float(row.get("delay_ms"), float("inf"))
        row_cost = _finite_float(row.get("cost"), float("inf"))
        row_dpp = _finite_float(row.get("paper_dpp_score"), float("inf"))
        if not all(np.isfinite(value) for value in [row_energy, row_delay, row_cost, row_dpp]):
            continue
        if current_energy - row_energy < min_energy_gain:
            continue
        if row_dpp > current_dpp + dpp_slack:
            continue
        if _energy_guard_strict_dpp_regret_reason(row, current_row, config):
            continue
        if _energy_guard_delay_regret_reason(
            row_delay,
            current_delay,
            row_dpp,
            current_dpp,
            high_queue_pressure,
            max_delay_regret,
            config,
        ):
            continue
        if row_cost - current_cost > max_cost_regret:
            continue
        if high_queue_pressure and _energy_guard_queue_relief_reason(row, current_row, config):
            continue
        if _energy_guard_latency_queue_regret_reason(row, current_row, config):
            continue
        guard_pool.append(row)
    guard_pool = _energy_guard_filter_relaxed_same_cloud_recovery_regret(
        guard_pool, rows, config
    )
    if not guard_pool:
        return {}
    return min(
        guard_pool,
        key=lambda row: (
            _finite_float(row.get("predicted_avg_y"), float("inf")),
            _finite_float(row.get("post_update_energy_queue_delta_term"), float("inf")),
            _finite_float(row.get("post_update_queue_drift_term"), float("inf")),
            _finite_float(row.get("energy_j"), float("inf")),
            _finite_float(row.get("paper_dpp_score"), float("inf")),
            _finite_float(row.get("delay_ms"), float("inf")),
            int(row.get("eval_rank", 0)),
        ),
    )


def energy_relief_diagnostic_candidate(rows: List[Dict], config: AblationExperimentConfig,
                                       selected_row: Dict) -> Dict:
    """Report lower-energy candidates and why the energy guard would reject them."""
    defaults = {
        "energy_relief_candidate_source": "",
        "energy_relief_candidate_hash": "",
        "energy_relief_candidate_delay_ms": 0.0,
        "energy_relief_candidate_energy_j": 0.0,
        "energy_relief_candidate_cost": 0.0,
        "energy_relief_candidate_claim_score": 0.0,
        "energy_relief_candidate_dpp_score": 0.0,
        "energy_relief_candidate_predicted_avg_y": 0.0,
        "energy_relief_candidate_predicted_avg_z": 0.0,
        "energy_relief_candidate_post_update_queue_drift_term": 0.0,
        "energy_relief_candidate_energy_gain_j": 0.0,
        "energy_relief_candidate_delay_regret_ms": 0.0,
        "energy_relief_candidate_cost_regret": 0.0,
        "energy_relief_candidate_dpp_regret": 0.0,
        "energy_relief_best_lower_source": "",
        "energy_relief_best_lower_hash": "",
        "energy_relief_best_lower_delay_ms": 0.0,
        "energy_relief_best_lower_energy_j": 0.0,
        "energy_relief_best_lower_cost": 0.0,
        "energy_relief_best_lower_claim_score": 0.0,
        "energy_relief_best_lower_dpp_score": 0.0,
        "energy_relief_best_lower_predicted_avg_y": 0.0,
        "energy_relief_best_lower_predicted_avg_z": 0.0,
        "energy_relief_best_lower_post_update_queue_drift_term": 0.0,
        "energy_relief_best_lower_energy_gain_j": 0.0,
        "energy_relief_best_lower_delay_regret_ms": 0.0,
        "energy_relief_best_lower_cost_regret": 0.0,
        "energy_relief_best_lower_dpp_regret": 0.0,
        "energy_relief_best_lower_reject_reason": "",
    }
    current_energy = _finite_float(selected_row.get("energy_j"), float("inf"))
    current_delay = _finite_float(selected_row.get("delay_ms"), float("inf"))
    current_cost = _finite_float(selected_row.get("cost"), float("inf"))
    current_dpp = _finite_float(selected_row.get("paper_dpp_score"), float("inf"))
    if not all(np.isfinite(value) for value in [current_energy, current_delay, current_cost, current_dpp]):
        defaults["energy_relief_best_lower_reject_reason"] = "selected_score_nonfinite"
        return defaults
    min_energy_gain = max(
        float(getattr(config, "queue_pressure_energy_guard_min_energy_gain_j", 1.0)),
        0.0,
    )
    max_delay_regret = max(
        float(getattr(config, "queue_pressure_energy_guard_max_delay_regret_ms", 20.0)),
        0.0,
    )
    max_cost_regret = max(
        float(getattr(config, "queue_pressure_energy_guard_max_cost_regret", 80.0)),
        0.0,
    )
    relief_reason = _energy_guard_queue_relief_reason(selected_row, selected_row, config)
    high_queue_pressure = relief_reason in {"queue_relief_absent", ""}
    dpp_slack = 0.0
    if high_queue_pressure:
        dpp_slack = max(
            abs(current_dpp) * max(float(getattr(config, "queue_pressure_energy_guard_dpp_slack_ratio", 0.0)), 0.0),
            0.0,
        )

    def candidate_hash(row: Dict) -> str:
        return str(
            row.get("repaired_pair_action_hash") or
            row.get("pair_action_hash") or
            row.get("action_hash") or
            ""
        )

    def metrics(row: Dict) -> Dict[str, float]:
        row_energy = _finite_float(row.get("energy_j"), float("inf"))
        row_delay = _finite_float(row.get("delay_ms"), float("inf"))
        row_cost = _finite_float(row.get("cost"), float("inf"))
        row_dpp = _finite_float(row.get("paper_dpp_score"), float("inf"))
        return {
            "energy_gain": current_energy - row_energy,
            "delay_regret": row_delay - current_delay,
            "cost_regret": row_cost - current_cost,
            "dpp_regret": row_dpp - current_dpp,
        }

    def reject_reason(row: Dict) -> str:
        item = metrics(row)
        if item["energy_gain"] < min_energy_gain:
            return "energy_gain_below_min"
        if item["dpp_regret"] > dpp_slack:
            return "dpp_regret"
        delay_reason = _energy_guard_delay_regret_reason(
            _finite_float(row.get("delay_ms"), float("inf")),
            current_delay,
            _finite_float(row.get("paper_dpp_score"), float("inf")),
            current_dpp,
            high_queue_pressure,
            max_delay_regret,
            config,
        )
        if delay_reason:
            return delay_reason
        if item["cost_regret"] > max_cost_regret:
            return "cost_regret"
        if high_queue_pressure:
            queue_reason = _energy_guard_queue_relief_reason(row, selected_row, config)
            if queue_reason:
                return queue_reason
        latency_queue_reason = _energy_guard_latency_queue_regret_reason(row, selected_row, config)
        if latency_queue_reason:
            return latency_queue_reason
        return ""

    def write(prefix: str, row: Dict, reason: str = "") -> Dict:
        item = metrics(row)
        return {
            f"{prefix}_source": str(row.get("candidate_source", "")),
            f"{prefix}_hash": candidate_hash(row),
            f"{prefix}_delay_ms": float(row.get("delay_ms", 0.0)),
            f"{prefix}_energy_j": float(row.get("energy_j", 0.0)),
            f"{prefix}_cost": float(row.get("cost", 0.0)),
            f"{prefix}_claim_score": float(row.get("claim_score", 0.0)),
            f"{prefix}_dpp_score": float(row.get("paper_dpp_score", 0.0)),
            f"{prefix}_predicted_avg_y": float(row.get("predicted_avg_y", 0.0)),
            f"{prefix}_predicted_avg_z": float(row.get("predicted_avg_z", 0.0)),
            f"{prefix}_post_update_queue_drift_term": float(row.get("post_update_queue_drift_term", 0.0)),
            f"{prefix}_energy_gain_j": float(item["energy_gain"]),
            f"{prefix}_delay_regret_ms": float(item["delay_regret"]),
            f"{prefix}_cost_regret": float(item["cost_regret"]),
            f"{prefix}_dpp_regret": float(item["dpp_regret"]),
            f"{prefix}_reject_reason": str(reason),
        }

    lower_energy_rows = []
    eligible_rows = []
    for row in rows:
        if row is selected_row:
            continue
        if not bool(row.get("feasible", False)):
            continue
        row_energy = _finite_float(row.get("energy_j"), float("inf"))
        row_delay = _finite_float(row.get("delay_ms"), float("inf"))
        row_cost = _finite_float(row.get("cost"), float("inf"))
        row_dpp = _finite_float(row.get("paper_dpp_score"), float("inf"))
        if not all(np.isfinite(value) for value in [row_energy, row_delay, row_cost, row_dpp]):
            continue
        if current_energy - row_energy <= 0.0:
            continue
        reason = reject_reason(row)
        lower_energy_rows.append((row_energy, row, reason))
        if reason == "":
            eligible_rows.append(row)
    if not lower_energy_rows:
        defaults["energy_relief_best_lower_reject_reason"] = "no_lower_energy_candidate"
        return defaults

    _, best_lower, best_lower_reason = min(
        lower_energy_rows,
        key=lambda item: (
            item[0],
            _finite_float(item[1].get("paper_dpp_score"), float("inf")),
            _finite_float(item[1].get("delay_ms"), float("inf")),
            int(item[1].get("eval_rank", 0)),
        ),
    )
    best_lower_diagnostic = write("energy_relief_best_lower", best_lower, best_lower_reason)
    defaults.update(best_lower_diagnostic)

    if eligible_rows:
        candidate = min(
            eligible_rows,
            key=lambda row: (
                _finite_float(row.get("energy_j"), float("inf")),
                _finite_float(row.get("paper_dpp_score"), float("inf")),
                _finite_float(row.get("delay_ms"), float("inf")),
                int(row.get("eval_rank", 0)),
            ),
        )
        candidate_diagnostic = write("energy_relief_candidate", candidate)
        candidate_diagnostic.pop("energy_relief_candidate_reject_reason", None)
        defaults.update(candidate_diagnostic)
    return defaults


def claim_stability_guard_candidate(rows: List[Dict], config: AblationExperimentConfig,
                                    best_claim_row: Dict) -> Dict:
    """Choose a regret-bounded, metric-balanced candidate from the claim pool."""
    if not getattr(config, "energy_claim_stability_guard_enabled", True):
        return {}
    min_candidates = max(int(getattr(config, "energy_claim_stability_min_candidates", 3)), 2)
    finite_rows = [
        row for row in rows
        if bool(row.get("feasible", False)) and
        np.isfinite(_finite_float(row.get("claim_score"), float("inf"))) and
        np.isfinite(_finite_float(row.get("delay_ms"), float("inf"))) and
        np.isfinite(_finite_float(row.get("energy_j"), float("inf"))) and
        np.isfinite(_finite_float(row.get("cost"), float("inf")))
    ]
    if len(finite_rows) < min_candidates:
        return {}
    best_claim_score = _finite_float(best_claim_row.get("claim_score"), float("inf"))
    if not np.isfinite(best_claim_score):
        return {}
    regret_budget = max(
        abs(best_claim_score) * max(float(getattr(config, "energy_claim_stability_regret_ratio", 0.08)), 0.0),
        max(float(getattr(config, "energy_claim_stability_regret_min", 0.10)), 0.0),
    )
    regret_pool = [
        row for row in finite_rows
        if _finite_float(row.get("claim_score"), float("inf")) <= best_claim_score + regret_budget
    ]
    if len(regret_pool) < min_candidates:
        return {}

    metrics = ("delay_ms", "energy_j", "cost")
    values = np.array([
        [_finite_float(row.get(metric), float("inf")) for metric in metrics]
        for row in regret_pool
    ], dtype=float)
    if not np.all(np.isfinite(values)):
        return {}
    targets = np.median(values, axis=0)
    delay_threshold = _finite_float(getattr(config, "queue_delay_threshold_ms", float("nan")), float("nan"))
    if np.isfinite(delay_threshold) and delay_threshold > 0.0:
        targets[0] = delay_threshold * max(
            float(getattr(config, "energy_claim_stability_delay_target_ratio", 0.90)),
            0.0,
        )
        targets[1] = max(float(getattr(config, "claim_energy_ref_j", 2.0)), 1e-9) * max(
            float(getattr(config, "energy_claim_stability_energy_target_ratio", 1.85)),
            0.0,
        )
    spreads = np.percentile(values, 75, axis=0) - np.percentile(values, 25, axis=0)
    min_scales = np.array([
        max(float(getattr(config, "claim_delay_ref_ms", 100.0)), 1e-9) * 0.20,
        max(float(getattr(config, "claim_energy_ref_j", 2.0)), 1e-9) * 0.20,
        max(float(getattr(config, "claim_cost_ref", 400.0)), 1e-9) * 0.20,
    ], dtype=float)
    scales = np.maximum(spreads, min_scales)
    metric_weights = np.array([1.20, 1.00, 0.80], dtype=float)
    claim_weight = max(float(getattr(config, "energy_claim_stability_claim_weight", 0.05)), 0.0)
    queue_drift_weight = max(
        float(getattr(config, "energy_claim_stability_queue_drift_weight", 0.0)),
        0.0,
    )
    latency_queue_weight = max(
        float(getattr(config, "energy_claim_stability_latency_queue_weight", 0.0)),
        0.0,
    )
    drift_min = 0.0
    drift_scale = 1.0
    if queue_drift_weight > 0.0:
        drift_values = np.array([
            max(_finite_float(row.get("post_update_queue_drift_term"), 0.0), 0.0)
            for row in regret_pool
        ], dtype=float)
        finite_drift = drift_values[np.isfinite(drift_values)]
        if finite_drift.size > 0:
            drift_min = float(np.min(finite_drift))
            drift_spread = float(np.percentile(finite_drift, 75) - np.percentile(finite_drift, 25))
            drift_scale = max(drift_spread, 1.0)
    latency_queue_fields = ("predicted_avg_z", "post_update_delay_queue_delta_term")
    latency_queue_scales = {}
    if latency_queue_weight > 0.0:
        for field in latency_queue_fields:
            values_for_field = np.array([
                max(_finite_float(row.get(field), float("nan")), 0.0)
                for row in regret_pool
            ], dtype=float)
            finite_values = values_for_field[np.isfinite(values_for_field)]
            if finite_values.size >= 2:
                value_spread = float(np.percentile(finite_values, 75) - np.percentile(finite_values, 25))
                latency_queue_scales[field] = (
                    float(np.min(finite_values)),
                    max(value_spread, 1.0),
                )

    def stability_score(row: Dict) -> float:
        row_values = np.array([_finite_float(row.get(metric), float("inf")) for metric in metrics], dtype=float)
        metric_excess = np.maximum(row_values - targets, 0.0)
        metric_score = float(np.sum(metric_weights * metric_excess / scales))
        claim_regret = max(_finite_float(row.get("claim_score"), float("inf")) - best_claim_score, 0.0)
        queue_drift = max(_finite_float(row.get("post_update_queue_drift_term"), 0.0), 0.0)
        queue_score = 0.0
        if queue_drift_weight > 0.0 and np.isfinite(queue_drift):
            queue_score = queue_drift_weight * max(queue_drift - drift_min, 0.0) / drift_scale
        latency_queue_score = 0.0
        if latency_queue_weight > 0.0 and latency_queue_scales:
            for field, (field_min, field_scale) in latency_queue_scales.items():
                value = max(_finite_float(row.get(field), float("nan")), 0.0)
                if np.isfinite(value):
                    latency_queue_score += max(value - field_min, 0.0) / field_scale
            latency_queue_score *= latency_queue_weight / max(len(latency_queue_scales), 1)
        return (
            metric_score +
            claim_weight * claim_regret / max(regret_budget, 1e-9) +
            queue_score +
            latency_queue_score
        )

    candidate = min(
        regret_pool,
        key=lambda row: (
            stability_score(row),
            _finite_float(row.get("claim_score"), float("inf")),
            _finite_float(row.get("paper_dpp_score"), float("inf")),
            int(row.get("eval_rank", 0)),
        ),
    )
    best_stability = stability_score(best_claim_row)
    candidate_stability = stability_score(candidate)
    min_gain = max(float(getattr(config, "energy_claim_stability_min_score_gain", 0.20)), 0.0)
    if candidate is best_claim_row or best_stability - candidate_stability < min_gain:
        return {}
    return candidate


def claim_temporal_guard_candidate(rows: List[Dict], config: AblationExperimentConfig,
                                   best_claim_row: Dict,
                                   temporal_history: List[Dict]) -> Dict:
    """Choose a regret-bounded candidate close to recent selected metrics."""
    if not getattr(config, "energy_claim_temporal_guard_enabled", False):
        return {}
    min_history = max(int(getattr(config, "energy_claim_temporal_guard_min_history", 5)), 1)
    if len(temporal_history or []) < min_history:
        return {}
    metrics = ("delay_ms", "energy_j", "cost")
    history_rows = []
    history_source_families = []
    history_local_cloud = []
    window = max(int(getattr(config, "energy_claim_temporal_guard_window", 20)), 1)
    for item in list(temporal_history)[-window:]:
        values = [_finite_float(item.get(metric), float("inf")) for metric in metrics]
        if all(np.isfinite(value) for value in values):
            history_rows.append(values)
            source_family = normalize_candidate_source_family(
                str(item.get("selected_candidate_source", ""))
            )
            if source_family and source_family != "unknown":
                history_source_families.append(source_family)
            local_count, cloud_count = _candidate_local_cloud_counts(item)
            if np.isfinite(local_count) and np.isfinite(cloud_count):
                history_local_cloud.append([local_count, cloud_count])
    if len(history_rows) < min_history:
        return {}

    finite_rows = [
        row for row in rows
        if bool(row.get("feasible", False)) and
        np.isfinite(_finite_float(row.get("claim_score"), float("inf"))) and
        all(np.isfinite(_finite_float(row.get(metric), float("inf"))) for metric in metrics)
    ]
    if len(finite_rows) < 2:
        return {}
    best_claim_score = _finite_float(best_claim_row.get("claim_score"), float("inf"))
    if not np.isfinite(best_claim_score):
        return {}
    regret_budget = max(
        abs(best_claim_score) * max(float(getattr(config, "energy_claim_temporal_guard_regret_ratio", 0.12)), 0.0),
        max(float(getattr(config, "energy_claim_temporal_guard_regret_min", 0.10)), 0.0),
    )
    regret_pool = [
        row for row in finite_rows
        if _finite_float(row.get("claim_score"), float("inf")) <= best_claim_score + regret_budget
    ]
    if len(regret_pool) < 2:
        return {}

    targets = np.median(np.array(history_rows, dtype=float), axis=0)
    scales = np.array([
        max(float(getattr(config, "claim_delay_ref_ms", 100.0)), 1e-9),
        max(float(getattr(config, "claim_energy_ref_j", 2.0)), 1e-9),
        max(float(getattr(config, "claim_cost_ref", 400.0)), 1e-9),
    ], dtype=float)
    metric_weights = np.array([1.20, 1.00, 0.80], dtype=float)
    claim_weight = max(float(getattr(config, "energy_claim_temporal_guard_claim_weight", 0.05)), 0.0)
    source_switch_weight = max(
        float(getattr(config, "energy_claim_temporal_guard_source_switch_weight", 0.0)),
        0.0,
    )
    local_cloud_weight = max(
        float(getattr(config, "energy_claim_temporal_guard_local_cloud_weight", 0.0)),
        0.0,
    )
    local_cloud_scale = max(
        float(getattr(config, "energy_claim_temporal_guard_local_cloud_scale", 4.0)),
        1e-9,
    )
    latency_queue_weight = max(
        float(getattr(config, "energy_claim_temporal_guard_latency_queue_weight", 0.0)),
        0.0,
    )
    target_source_family = ""
    if source_switch_weight > 0.0 and history_source_families:
        target_source_family = Counter(history_source_families).most_common(1)[0][0]
    target_local_cloud = None
    if local_cloud_weight > 0.0 and history_local_cloud:
        target_local_cloud = np.median(np.array(history_local_cloud, dtype=float), axis=0)
    latency_queue_fields = ("predicted_avg_z", "post_update_delay_queue_delta_term")
    history_has_latency_queue = False
    if latency_queue_weight > 0.0:
        for item in list(temporal_history)[-window:]:
            if any(
                np.isfinite(_finite_float(item.get(field), float("nan")))
                for field in latency_queue_fields
            ):
                history_has_latency_queue = True
                break
    latency_queue_scales = {}
    if latency_queue_weight > 0.0 and history_has_latency_queue:
        for field in latency_queue_fields:
            values_for_field = np.array([
                max(_finite_float(row.get(field), float("nan")), 0.0)
                for row in regret_pool
            ], dtype=float)
            finite_values = values_for_field[np.isfinite(values_for_field)]
            if finite_values.size >= 2:
                value_spread = float(np.percentile(finite_values, 75) - np.percentile(finite_values, 25))
                latency_queue_scales[field] = (
                    float(np.min(finite_values)),
                    max(value_spread, 1.0),
                )

    energy_override_enabled = bool(
        getattr(config, "energy_claim_temporal_guard_energy_override_enabled", False)
    )
    energy_override_min_gain = max(
        float(getattr(config, "energy_claim_temporal_guard_energy_override_min_gain_j", 1.0)),
        0.0,
    )
    energy_override_max_delay = max(
        float(getattr(config, "energy_claim_temporal_guard_energy_override_max_delay_regret_ms", 35.0)),
        0.0,
    )
    energy_override_max_cost = max(
        float(getattr(config, "energy_claim_temporal_guard_energy_override_max_cost_regret", 40.0)),
        0.0,
    )
    energy_override_max_queue = max(
        float(getattr(config, "energy_claim_temporal_guard_energy_override_max_queue_regret", 0.50)),
        0.0,
    )
    best_delay = _finite_float(best_claim_row.get("delay_ms"), float("inf"))
    best_energy = _finite_float(best_claim_row.get("energy_j"), float("inf"))
    best_cost = _finite_float(best_claim_row.get("cost"), float("inf"))
    best_dpp = _finite_float(best_claim_row.get("paper_dpp_score"), float("inf"))

    def bounded_energy_override(row: Dict) -> bool:
        if not energy_override_enabled or row is best_claim_row:
            return False
        row_delay = _finite_float(row.get("delay_ms"), float("inf"))
        row_energy = _finite_float(row.get("energy_j"), float("inf"))
        row_cost = _finite_float(row.get("cost"), float("inf"))
        row_claim = _finite_float(row.get("claim_score"), float("inf"))
        row_dpp = _finite_float(row.get("paper_dpp_score"), float("inf"))
        required = (best_delay, best_energy, best_cost, best_claim_score, best_dpp,
                    row_delay, row_energy, row_cost, row_claim, row_dpp)
        if not all(np.isfinite(value) for value in required):
            return False
        if best_energy - row_energy < energy_override_min_gain:
            return False
        if row_claim > best_claim_score or row_dpp > best_dpp:
            return False
        if row_delay - best_delay > energy_override_max_delay:
            return False
        if row_cost - best_cost > energy_override_max_cost:
            return False
        for field in ("predicted_avg_y", "predicted_avg_z", "post_update_queue_drift_term"):
            row_value = _finite_float(row.get(field), float("nan"))
            best_value = _finite_float(best_claim_row.get(field), float("nan"))
            if np.isfinite(row_value) and np.isfinite(best_value):
                if row_value - best_value > energy_override_max_queue:
                    return False
        return True

    override_pool = [row for row in regret_pool if bounded_energy_override(row)]
    if override_pool:
        def override_key(row: Dict):
            delay_regret = max(_finite_float(row.get("delay_ms"), float("inf")) - best_delay, 0.0)
            cost_regret = max(_finite_float(row.get("cost"), float("inf")) - best_cost, 0.0)
            queue_regret = 0.0
            for field in ("predicted_avg_y", "predicted_avg_z", "post_update_queue_drift_term"):
                row_value = _finite_float(row.get(field), float("nan"))
                best_value = _finite_float(best_claim_row.get(field), float("nan"))
                if np.isfinite(row_value) and np.isfinite(best_value):
                    queue_regret += max(row_value - best_value, 0.0)
            normalized_regret = (
                delay_regret / max(energy_override_max_delay, 1e-9) +
                cost_regret / max(energy_override_max_cost, 1e-9) +
                queue_regret / max(energy_override_max_queue, 1e-9)
            )
            return (
                normalized_regret,
                _finite_float(row.get("energy_j"), float("inf")),
                _finite_float(row.get("claim_score"), float("inf")),
                _finite_float(row.get("paper_dpp_score"), float("inf")),
                int(row.get("eval_rank", 0)),
            )

        return min(override_pool, key=override_key)

    def temporal_score(row: Dict) -> float:
        values = np.array([_finite_float(row.get(metric), float("inf")) for metric in metrics], dtype=float)
        metric_score = float(np.sum(metric_weights * np.abs(values - targets) / scales))
        claim_regret = max(_finite_float(row.get("claim_score"), float("inf")) - best_claim_score, 0.0)
        source_penalty = 0.0
        if target_source_family:
            row_family = normalize_candidate_source_family(str(row.get("candidate_source", "")))
            if row_family != target_source_family:
                source_penalty = source_switch_weight
        local_cloud_penalty = 0.0
        if target_local_cloud is not None:
            local_count, cloud_count = _candidate_local_cloud_counts(row)
            if np.isfinite(local_count) and np.isfinite(cloud_count):
                local_cloud_delta = (
                    abs(float(local_count) - float(target_local_cloud[0])) +
                    abs(float(cloud_count) - float(target_local_cloud[1]))
                )
                local_cloud_penalty = local_cloud_weight * local_cloud_delta / local_cloud_scale
        latency_queue_score = 0.0
        if latency_queue_weight > 0.0 and latency_queue_scales:
            for field, (field_min, field_scale) in latency_queue_scales.items():
                value = max(_finite_float(row.get(field), float("nan")), 0.0)
                if np.isfinite(value):
                    latency_queue_score += max(value - field_min, 0.0) / field_scale
            latency_queue_score *= latency_queue_weight / max(len(latency_queue_scales), 1)
        return (
            metric_score +
            claim_weight * claim_regret / max(regret_budget, 1e-9) +
            source_penalty +
            local_cloud_penalty +
            latency_queue_score
        )

    candidate = min(
        regret_pool,
        key=lambda row: (
            temporal_score(row),
            _finite_float(row.get("claim_score"), float("inf")),
            _finite_float(row.get("paper_dpp_score"), float("inf")),
            int(row.get("eval_rank", 0)),
        ),
    )
    best_temporal = temporal_score(best_claim_row)
    candidate_temporal = temporal_score(candidate)
    min_gain = max(float(getattr(config, "energy_claim_temporal_guard_min_score_gain", 0.20)), 0.0)
    if candidate is best_claim_row or best_temporal - candidate_temporal < min_gain:
        return {}
    return candidate


def tail_risk_diagnostic_candidate(rows: List[Dict], selected_row: Dict,
                                   temporal_history: List[Dict],
                                   config: AblationExperimentConfig) -> Dict:
    """Return a non-behavioral diagnostic candidate that reduces upper-tail risk."""
    min_history = max(int(getattr(config, "energy_claim_temporal_guard_min_history", 5)), 1)
    if len(temporal_history or []) < min_history:
        return {}
    metrics = ("delay_ms", "energy_j", "cost")
    history_rows = []
    window = max(int(getattr(config, "energy_claim_temporal_guard_window", 20)), 1)
    for item in list(temporal_history)[-window:]:
        values = [_finite_float(item.get(metric), float("inf")) for metric in metrics]
        if all(np.isfinite(value) for value in values):
            history_rows.append(values)
    if len(history_rows) < min_history:
        return {}

    history_array = np.array(history_rows, dtype=float)
    median = np.median(history_array, axis=0)
    q25 = np.percentile(history_array, 25, axis=0)
    q75 = np.percentile(history_array, 75, axis=0)
    scales = np.array([
        max(float(getattr(config, "claim_delay_ref_ms", 100.0)), 1e-9),
        max(float(getattr(config, "claim_energy_ref_j", 2.0)), 1e-9),
        max(float(getattr(config, "claim_cost_ref", 400.0)), 1e-9),
    ], dtype=float)
    iqr = np.maximum(q75 - q25, 0.05 * scales)
    upper_tail = median + iqr
    weights = np.array([1.20, 1.00, 0.80], dtype=float)

    def upper_excess_score(row: Dict) -> float:
        values = np.array([
            _finite_float(row.get(metric), float("inf")) for metric in metrics
        ], dtype=float)
        if not np.all(np.isfinite(values)):
            return float("inf")
        excess = np.maximum(values - upper_tail, 0.0)
        return float(np.sum(weights * excess / scales))

    selected_risk = upper_excess_score(selected_row)
    if not np.isfinite(selected_risk) or selected_risk <= 1e-9:
        return {}
    selected_only_diagnostic = {
        "tail_risk_candidate_source": "",
        "tail_risk_candidate_hash": "",
        "tail_risk_candidate_delay_ms": 0.0,
        "tail_risk_candidate_energy_j": 0.0,
        "tail_risk_candidate_cost": 0.0,
        "tail_risk_candidate_claim_score": 0.0,
        "tail_risk_candidate_dpp_score": 0.0,
        "tail_risk_candidate_predicted_avg_y": 0.0,
        "tail_risk_candidate_predicted_avg_z": 0.0,
        "tail_risk_candidate_post_update_queue_drift_term": 0.0,
        "tail_risk_candidate_upper_excess_score": 0.0,
        "tail_risk_selected_upper_excess_score": float(selected_risk),
        "tail_risk_candidate_upper_excess_improvement": 0.0,
        "tail_risk_best_relief_source": "",
        "tail_risk_best_relief_hash": "",
        "tail_risk_best_relief_delay_ms": 0.0,
        "tail_risk_best_relief_energy_j": 0.0,
        "tail_risk_best_relief_cost": 0.0,
        "tail_risk_best_relief_claim_score": 0.0,
        "tail_risk_best_relief_dpp_score": 0.0,
        "tail_risk_best_relief_predicted_avg_y": 0.0,
        "tail_risk_best_relief_predicted_avg_z": 0.0,
        "tail_risk_best_relief_post_update_queue_drift_term": 0.0,
        "tail_risk_best_relief_upper_excess_score": 0.0,
        "tail_risk_best_relief_improvement": 0.0,
        "tail_risk_best_relief_reject_reason": "",
    }
    selected_claim = _finite_float(selected_row.get("claim_score"), float("inf"))
    selected_dpp = _finite_float(selected_row.get("paper_dpp_score"), float("inf"))
    if not np.isfinite(selected_claim) or not np.isfinite(selected_dpp):
        selected_only_diagnostic["tail_risk_best_relief_reject_reason"] = "selected_score_nonfinite"
        return selected_only_diagnostic
    claim_budget = max(
        abs(selected_claim) * max(float(getattr(config, "energy_claim_temporal_guard_regret_ratio", 0.12)), 0.0),
        max(float(getattr(config, "energy_claim_temporal_guard_regret_min", 0.10)), 0.0),
    )
    dpp_budget = max(
        abs(selected_dpp) * max(float(getattr(config, "energy_claim_temporal_guard_dpp_regret_ratio", 0.10)), 0.0),
        abs(selected_dpp) * max(float(getattr(config, "energy_hard_dpp_tolerance_ratio", 0.03)), 0.0),
        1e-9,
    )

    def candidate_hash(row: Dict) -> str:
        return str(
            row.get("repaired_pair_action_hash") or
            row.get("pair_action_hash") or
            row.get("action_hash", "")
        )

    def best_relief_diagnostic(row: Dict, candidate_risk: float,
                               reject_reason: str) -> Dict:
        return {
            "tail_risk_best_relief_source": str(row.get("candidate_source", "")),
            "tail_risk_best_relief_hash": candidate_hash(row),
            "tail_risk_best_relief_delay_ms": float(row.get("delay_ms", 0.0)),
            "tail_risk_best_relief_energy_j": float(row.get("energy_j", 0.0)),
            "tail_risk_best_relief_cost": float(row.get("cost", 0.0)),
            "tail_risk_best_relief_claim_score": float(row.get("claim_score", 0.0)),
            "tail_risk_best_relief_dpp_score": float(row.get("paper_dpp_score", 0.0)),
            "tail_risk_best_relief_predicted_avg_y": float(row.get("predicted_avg_y", 0.0)),
            "tail_risk_best_relief_predicted_avg_z": float(row.get("predicted_avg_z", 0.0)),
            "tail_risk_best_relief_post_update_queue_drift_term": float(
                row.get("post_update_queue_drift_term", 0.0)
            ),
            "tail_risk_best_relief_upper_excess_score": float(candidate_risk),
            "tail_risk_best_relief_improvement": float(selected_risk - candidate_risk),
            "tail_risk_best_relief_reject_reason": str(reject_reason),
        }

    def queue_reject_reason(row: Dict) -> str:
        checks = (
            ("predicted_avg_y", 0.50),
            ("predicted_avg_z", 0.50),
            ("post_update_energy_queue_delta_term", 0.50),
            ("post_update_delay_queue_delta_term", 0.50),
            ("post_update_queue_drift_term", 0.25),
        )
        for field, floor in checks:
            selected_value = _finite_float(selected_row.get(field), float("nan"))
            candidate_value = _finite_float(row.get(field), float("nan"))
            if not (np.isfinite(selected_value) and np.isfinite(candidate_value)):
                continue
            tolerance = max(abs(selected_value) * 0.10, floor)
            if candidate_value > selected_value + tolerance:
                return f"queue_{field}"
        return ""

    def rejection_reason(row: Dict) -> str:
        candidate_claim = _finite_float(row.get("claim_score"), float("inf"))
        candidate_dpp = _finite_float(row.get("paper_dpp_score"), float("inf"))
        if candidate_claim > selected_claim + claim_budget:
            return "claim_regret"
        if candidate_dpp > selected_dpp + dpp_budget:
            return "dpp_regret"
        return queue_reject_reason(row)

    relief_pool = []
    for row in rows:
        if not bool(row.get("feasible", False)):
            continue
        candidate_risk = upper_excess_score(row)
        if not np.isfinite(candidate_risk):
            continue
        if selected_risk - candidate_risk <= 1e-9:
            continue
        relief_pool.append((candidate_risk, row))

    if not relief_pool:
        selected_only_diagnostic["tail_risk_best_relief_reject_reason"] = "no_lower_upper_tail_candidate"
        return selected_only_diagnostic

    best_relief_risk, best_relief = min(
        relief_pool,
        key=lambda item: (
            item[0],
            _finite_float(item[1].get("claim_score"), float("inf")),
            _finite_float(item[1].get("paper_dpp_score"), float("inf")),
            int(item[1].get("eval_rank", 0)),
        ),
    )
    selected_only_diagnostic.update(
        best_relief_diagnostic(
            best_relief, best_relief_risk, rejection_reason(best_relief)
        )
    )

    candidate_pool = []
    for candidate_risk, row in relief_pool:
        if rejection_reason(row):
            continue
        candidate_pool.append((candidate_risk, row))

    if not candidate_pool:
        return selected_only_diagnostic

    candidate_risk, candidate = min(
        candidate_pool,
        key=lambda item: (
            item[0],
            _finite_float(item[1].get("claim_score"), float("inf")),
            _finite_float(item[1].get("paper_dpp_score"), float("inf")),
            int(item[1].get("eval_rank", 0)),
        ),
    )
    diagnostic = dict(selected_only_diagnostic)
    diagnostic.update({
        "tail_risk_candidate_source": str(candidate.get("candidate_source", "")),
        "tail_risk_candidate_hash": candidate_hash(candidate),
        "tail_risk_candidate_delay_ms": float(candidate.get("delay_ms", 0.0)),
        "tail_risk_candidate_energy_j": float(candidate.get("energy_j", 0.0)),
        "tail_risk_candidate_cost": float(candidate.get("cost", 0.0)),
        "tail_risk_candidate_claim_score": float(candidate.get("claim_score", 0.0)),
        "tail_risk_candidate_dpp_score": float(candidate.get("paper_dpp_score", 0.0)),
        "tail_risk_candidate_predicted_avg_y": float(candidate.get("predicted_avg_y", 0.0)),
        "tail_risk_candidate_predicted_avg_z": float(candidate.get("predicted_avg_z", 0.0)),
        "tail_risk_candidate_post_update_queue_drift_term": float(
            candidate.get("post_update_queue_drift_term", 0.0)
        ),
        "tail_risk_candidate_upper_excess_score": float(candidate_risk),
        "tail_risk_selected_upper_excess_score": float(selected_risk),
        "tail_risk_candidate_upper_excess_improvement": float(selected_risk - candidate_risk),
    })
    return diagnostic


def _normalize_action(action, expected_len: int = None) -> np.ndarray:
    """整理二进制动作向量"""
    return pair_actions.normalize_action(action, expected_len)


def _deduplicate_candidates(candidates: Iterable[np.ndarray], expected_len: int) -> List[np.ndarray]:
    """候选动作去重，保持生成顺序"""
    unique = []
    seen = set()
    for candidate in candidates:
        action = _normalize_action(candidate, expected_len)
        key = tuple(int(x) for x in action)
        if key not in seen:
            seen.add(key)
            unique.append(action)
    return unique


def get_ai_action_dimension(system_state) -> int:
    """获取AI卸载动作维度"""
    return pair_actions.get_ai_action_dimension(system_state)


def infer_candidate_action_dimension(candidates: Iterable) -> int:
    """从候选动作推断维度，供无完整system_state的轻量测试使用。"""
    max_len = 0
    for candidate in candidates:
        action = candidate.get("action") if isinstance(candidate, dict) else candidate
        try:
            max_len = max(max_len, len(np.asarray(action).reshape(-1)))
        except Exception:
            continue
    return int(max_len)


def get_sorted_ai_servers(system_state) -> List:
    """按固定顺序获取AI服务器"""
    return pair_actions.get_sorted_ai_servers(system_state)


def build_active_pair_universe(system_state) -> List[Dict]:
    """构建active (s,v) pair动作边界"""
    return pair_actions.build_active_pair_universe(system_state)


def expand_server_action_to_pair_action(action: np.ndarray, pair_universe: List[Dict]) -> np.ndarray:
    """将server级动作扩展为active pair动作"""
    return pair_actions.expand_server_action_to_pair_action(action, pair_universe)


def project_pair_action_to_server_action(pair_action: np.ndarray,
                                         pair_universe: List[Dict],
                                         action_dim: int) -> np.ndarray:
    """将pair动作投影回旧模型可执行的server级动作"""
    return pair_actions.project_pair_action_to_server_action(pair_action, pair_universe, action_dim)


def wrap_candidates_with_pair_projection(candidates: Iterable[np.ndarray], system_state) -> List[Dict]:
    """
    包装候选动作元数据
    当前checkpoint仍是server级输出，pair动作只做工程投影和审计记录。
    """
    return pair_actions.wrap_candidates_with_pair_projection(candidates, system_state)


def get_current_slow_context(system_state, slow_policy: str) -> Dict:
    """获取当前慢层上下文"""
    context_name = {
        "GSLA": "gsla_context",
        "FFD": "ffd_context",
        "Random": "random_context",
        "PDRS": "pdrs_context",
        "LoadAware": "loadaware_context",
    }.get(slow_policy, "")
    return dict(getattr(system_state, context_name, {}) or {})


def get_active_ai_server_mask(system_state) -> np.ndarray:
    """标记当前慢层部署中有AI实例的服务器"""
    ai_servers = sorted([
        server for server in system_state.edge_servers.values()
        if server.server_type.value == "ai_capable"
    ], key=lambda server: server.server_id)
    mask = np.zeros(len(ai_servers), dtype=bool)
    for idx, server in enumerate(ai_servers):
        for instance in system_state.microservice_instances.values():
            if (instance.server_id == server.server_id and
                    instance.microservice.service_type == "ai"):
                mask[idx] = True
                break
    return mask


def _apply_active_mask(action: np.ndarray, active_mask: np.ndarray) -> np.ndarray:
    """没有AI实例的服务器只能走云端保护动作"""
    if len(active_mask) == 0:
        return action
    masked = np.array(action, dtype=int).copy()
    masked[~active_mask] = 1
    return masked


def build_guard_candidates(system_state) -> List[np.ndarray]:
    """生成保护性候选动作"""
    n = get_ai_action_dimension(system_state)
    active_mask = get_active_ai_server_mask(system_state)
    candidates = [
        _apply_active_mask(np.zeros(n, dtype=int), active_mask),
        np.ones(n, dtype=int),
    ]
    if n > 0:
        alt = _apply_active_mask(np.array([(idx % 2) for idx in range(n)], dtype=int), active_mask)
        candidates.append(alt)
        candidates.append(_apply_active_mask(1 - alt, active_mask))
    return _deduplicate_candidates(candidates, n)


def build_local_repair_candidates(system_state, limit: int = 6) -> List[np.ndarray]:
    """当Actor候选退化为全云端时，补充可行本地/混合候选"""
    env_manager = system_state.environment_manager
    SH, _, _ = env_manager.get_state_components()
    n = len(SH)
    if n == 0:
        return [np.array([], dtype=int)]
    active_mask = get_active_ai_server_mask(system_state)
    ranked = [
        idx for idx, _ in sorted(
            enumerate(np.asarray(SH, dtype=float)),
            key=lambda item: (-float(item[1]), item[0])
        )
        if idx < len(active_mask) and active_mask[idx]
    ]
    candidates = []
    prefix = np.ones(n, dtype=int)
    for idx in ranked[:max(limit, 1)]:
        one_local = np.ones(n, dtype=int)
        one_local[idx] = 0
        candidates.append(_apply_active_mask(one_local, active_mask))
        prefix = prefix.copy()
        prefix[idx] = 0
        candidates.append(_apply_active_mask(prefix, active_mask))
    return _deduplicate_candidates(candidates, n)


def _deduplicate_pair_candidates(candidates: Iterable[Dict], limit: int = None) -> List[Dict]:
    """pair级候选去重"""
    return pair_actions.deduplicate_pair_candidates(candidates, limit=limit)



def _energy_budget_frontier_sizes(pair_dim: int) -> List[int]:
    """生成energy-budget候选的本地pair预算档位。"""
    pair_dim = int(max(pair_dim, 0))
    if pair_dim <= 0:
        return []
    # 小预算覆盖保守本地化，中预算覆盖云端能耗偏高时的本地节能动作。
    raw_sizes = [1, 2, 3, 4, 6, 8, max(1, pair_dim // 2)]
    return sorted({min(pair_dim, int(size)) for size in raw_sizes if int(size) > 0})


def build_pair_repair_candidates(system_state, config: AblationExperimentConfig,
                                 queue_aware: bool = True,
                                 seed_candidates: Iterable[Dict] = None,
                                 source_prefix: str = "uac") -> List[Dict]:
    """
    构建pair级工程候选
    只扩展候选集合，不强制主算法选择，最终仍由shared evaluator评分。
    """
    pair_universe = build_active_pair_universe(system_state)
    action_dim = get_ai_action_dimension(system_state)
    pair_dim = len(pair_universe)
    if pair_dim == 0:
        return []

    env_manager = system_state.environment_manager
    SH, SQ, SZ = env_manager.get_state_components()
    ai_servers = get_sorted_ai_servers(system_state)
    readiness = {}
    for context_name in ["gsla_context", "ffd_context", "pdrs_context", "loadaware_context", "random_context"]:
        context = getattr(system_state, context_name, {}) or {}
        readiness.update(context.get("hapa_replica_readiness", {}))
    candidates = []
    paper_compact = queue_aware and _is_paper_compact_candidate_mechanism(config)
    compact_width = max(int(getattr(config, "uac_compact_frontier_width", 4)), 1)

    def frontier_width(default: int) -> int:
        default = max(int(default), 1)
        if not paper_compact:
            return min(pair_dim, default)
        return min(pair_dim, max(1, min(default, compact_width)))

    def infer_resource_hint(source: str) -> str:
        """按候选来源给出资源偏好提示，最终仍由共享 evaluator 校验。"""
        text = str(source)
        if "energy_low_dvfs" in text or "energy_frontier" in text or "pdrs_energy" in text:
            return "energy_saver_local"
        if (
            "energy_cloud_relief" in text or
            "energy_cost_pareto_relief" in text or
            "reference_low_impact_neighbor" in text or
            "balanced_tail_relief" in text or
            "guard_all_cloud" in text
        ):
            return "cloud_relief_f_pre"
        if "cost_aware" in text or "low_cost" in text:
            return "cost_saver_hybrid"
        if (
            "delay_energy" in text or
            "delay_frontier" in text or
            "gmda_delay" in text or
            "hybrid_latency" in text or
            "actor_delay_resource_refinement" in text
        ):
            return "latency_saver_hybrid"
        return ""

    def make(pair_bits, source):
        pair_bits = np.asarray(pair_bits, dtype=int)
        return {
            "pair_action": pair_bits,
            "action": project_pair_action_to_server_action(pair_bits, pair_universe, action_dim),
            "pair_universe": [dict(item) for item in pair_universe],
            "pair_action_dim": int(pair_dim),
            "action_dim": int(action_dim),
            "action_scope": "pair",
            "candidate_source": source,
            "resource_hint": infer_resource_hint(source),
        }

    all_cloud = np.ones(pair_dim, dtype=int)
    all_local = np.zeros(pair_dim, dtype=int)
    candidates.append(make(all_cloud, f"{source_prefix}_guard_all_cloud"))
    candidates.append(make(all_local, f"{source_prefix}_guard_all_local"))

    scored = []
    queue_pressure_rank = []
    energy_queue_rank = []
    energy_frontier_rank = []
    queue_balance_rank = []
    replica_ready_rank = []
    for pos, item in enumerate(pair_universe):
        server_idx = int(item.get("server_index", 0))
        server = ai_servers[server_idx] if 0 <= server_idx < len(ai_servers) else None
        instance = system_state.microservice_instances.get(item.get("instance_id", ""))
        request_flow = system_state.request_flows.get(item.get("flow_id", ""))
        sh = float(SH[server_idx]) if 0 <= server_idx < len(SH) else 1.0
        sq = float(SQ[server_idx]) if queue_aware and 0 <= server_idx < len(SQ) else 0.0
        sz = float(SZ[server_idx]) if queue_aware and 0 <= server_idx < len(SZ) else 0.0
        ready = float(readiness.get(item.get("server_id"), 0.5))
        spare = 0.5
        cost_score = 0.5
        if server is not None:
            spare = (
                server.available_gpu_units / max(server.gpu_units, 1) +
                server.available_gpu_memory / max(server.gpu_memory, 1.0) +
                server.available_model_storage / max(server.model_storage, 1.0)
            ) / 3.0
            used_gpu = max(server.gpu_units - server.available_gpu_units, 0)
            cost_score = 1.0 / (1.0 + 8.0 * used_gpu)
        queue_energy_score = 1.0 / (1.0 + max(sq, 0.0))
        queue_delay_score = 1.0 + np.log1p(max(sz, 0.0))
        latency_score = 0.45 * sh + 0.35 * ready + 0.20 * spare
        energy_score = 0.45 * queue_energy_score + 0.25 * spare + 0.20 * ready + 0.10 * cost_score
        if queue_aware:
            hybrid_score = 0.38 * latency_score + 0.34 * energy_score + 0.28 * queue_delay_score
        else:
            hybrid_score = 0.45 * latency_score + 0.35 * energy_score + 0.20 * cost_score
        scored.append((hybrid_score, energy_score, latency_score, cost_score, pos))
        replica_ready_rank.append((ready, pos))
        if queue_aware and server is not None and instance is not None and request_flow is not None:
            try:
                from ablation_resource_models import select_local_ai_config, solve_cloud_preprocess_config

                local_config = select_local_ai_config(
                    request_flow, instance.microservice, server, system_state,
                    SQ_value=sq, SZ_value=sz, performance_factor=sh,
                    V=getattr(config, "V", 20),
                    omega_energy=getattr(config, "omega_energy", 1.0),
                    omega_delay=getattr(config, "omega_delay", 1.0),
                )
                cloud_config = solve_cloud_preprocess_config(
                    request_flow, instance.microservice, server, system_state,
                    SQ_value=sq, SZ_value=sz, performance_factor=sh,
                    V=getattr(config, "V", 20),
                    omega_energy=getattr(config, "omega_energy", 1.0),
                    omega_delay=getattr(config, "omega_delay", 1.0),
                )
                if local_config and cloud_config:
                    arrival = max(float(getattr(request_flow, "arrival_rate", 1.0)), 1.0)
                    local_delay = float(local_config.get("latency_ms", 0.0))
                    cloud_delay = float(cloud_config.get("latency_ms", local_delay))
                    local_energy = float(local_config.get("energy_j", 0.0))
                    cloud_energy = float(cloud_config.get("energy_j", local_energy))
                    energy_ref = max(float(getattr(config, "energy_ref_j", 2.0)), 1e-9)
                    delay_ref = max(float(getattr(config, "delay_ref_ms", 50.0)), 1e-9)
                    delay_relief = max(cloud_delay - local_delay, 0.0) * arrival * (1.0 + np.log1p(max(sz, 0.0)))
                    energy_relief = max(cloud_energy - local_energy, 0.0) * (1.0 + np.log1p(max(sq, 0.0)))
                    energy_penalty = max(local_energy - cloud_energy, 0.0) * (1.0 + np.log1p(max(sq, 0.0)))
                    pressure_score = (
                        delay_relief / max(delay_ref, 1.0) +
                        energy_relief / energy_ref -
                        0.65 * energy_penalty / energy_ref
                    )
                    energy_queue_score = (
                        max(local_energy - cloud_energy, 0.0) *
                        (1.0 + np.log1p(max(sq, 0.0))) /
                        energy_ref
                    )
                    local_queue_cost = sq * local_energy / energy_ref + sz * local_delay * arrival / delay_ref
                    cloud_queue_cost = sq * cloud_energy / energy_ref + sz * cloud_delay * arrival / delay_ref
                    preferred_bit = 0 if local_queue_cost <= cloud_queue_cost else 1
                    readiness_bonus = 0.08 * ready if preferred_bit == 0 else 0.0
                    balance_margin = abs(cloud_queue_cost - local_queue_cost) + readiness_bonus
                    queue_pressure_rank.append((float(pressure_score), pos))
                    energy_queue_rank.append((float(energy_queue_score), pos))
                    queue_balance_rank.append((float(balance_margin), int(preferred_bit), pos))
                    # energy-budget frontier记录本地相对云端的能耗代价，供UAC生成少量本地hybrid候选。
                    energy_delta = cloud_energy - local_energy
                    frontier_score = (
                        delay_relief / max(delay_ref, 1.0) +
                        max(energy_delta, 0.0) / energy_ref -
                        1.35 * max(-energy_delta, 0.0) / energy_ref +
                        0.05 * ready
                    )
                    energy_frontier_rank.append((float(frontier_score), float(energy_delta), pos))
            except Exception:
                continue
    scored.sort(reverse=True)
    queue_pressure_rank.sort(reverse=True)
    energy_queue_rank.sort(reverse=True)
    energy_frontier_rank.sort(reverse=True)
    queue_balance_rank.sort(reverse=True)
    replica_ready_rank.sort(reverse=True)

    if paper_compact:
        limit = max(int(getattr(config, "uac_compact_pair_repair_limit", 20)), 4)
    else:
        limit = max(int(getattr(config, "uac_pair_repair_limit", 24)), 4)

    if scored:
        # 低DVFS本地候选：优先选择能耗分数高且资源余量好的pair，交给evaluator决定是否保留。
        low_dvfs_bits = np.ones(pair_dim, dtype=int)
        for _, _, _, _, pos in sorted(scored, key=lambda row: (row[1], row[0]), reverse=True)[:frontier_width(6)]:
            low_dvfs_bits[pos] = 0
            candidates.append(make(low_dvfs_bits.copy(), f"{source_prefix}_energy_low_dvfs_local"))

        # 低能耗且时延可行候选：只打开能耗分数和时延分数都靠前的pair。
        feasible_frontier = sorted(
            scored,
            key=lambda row: (min(row[1], row[2]), 0.55 * row[1] + 0.45 * row[2], row[0]),
            reverse=True,
        )
        feasible_bits = np.ones(pair_dim, dtype=int)
        for _, _, _, _, pos in feasible_frontier[:frontier_width(8)]:
            feasible_bits[pos] = 0
            candidates.append(make(feasible_bits.copy(), f"{source_prefix}_low_energy_delay_feasible"))

        # 成本感知混合候选：把低成本/低碎片位置加入本地集合，避免UAC只追逐时延。
        cost_bits = np.ones(pair_dim, dtype=int)
        for _, _, _, _, pos in sorted(scored, key=lambda row: (row[3], row[0]), reverse=True)[:frontier_width(6)]:
            cost_bits[pos] = 0
            candidates.append(make(cost_bits.copy(), f"{source_prefix}_cost_aware_hybrid"))

        if not paper_compact:
            # Legacy diagnostic source retained for backward-compatible tests; formal compact mode uses cost_aware_hybrid.
            ffd_like_bits = np.ones(pair_dim, dtype=int)
            ffd_like_rank = sorted(
                scored,
                key=lambda row: (
                    row[3],
                    -int(str(pair_universe[row[4]].get("server_id", "0")).split("_")[-1]) if str(pair_universe[row[4]].get("server_id", "")).split("_")[-1].isdigit() else 0,
                    row[0],
                ),
                reverse=True,
            )
            for _, _, _, _, pos in ffd_like_rank[:min(pair_dim, 6)]:
                ffd_like_bits[pos] = 0
                candidates.append(make(ffd_like_bits.copy(), f"{source_prefix}_low_cost_ffd_like_local"))

        # 时延-能耗前沿候选：按两个目标的折中排序生成多档hybrid动作。
        frontier_bits = np.ones(pair_dim, dtype=int)
        frontier_rank = sorted(scored, key=lambda row: (0.52 * row[2] + 0.48 * row[1], row[0]), reverse=True)
        for _, _, _, _, pos in frontier_rank[:frontier_width(8)]:
            frontier_bits[pos] = 0
            candidates.append(make(frontier_bits.copy(), f"{source_prefix}_delay_energy_frontier"))

        if paper_compact:
            density_rank = sorted(
                scored,
                key=lambda row: (0.45 * row[2] + 0.25 * row[1] + 0.20 * row[3] + 0.10 * row[0], row[0]),
                reverse=True,
            )
            density_sizes = sorted({
                max(1, min(pair_dim - 1 if pair_dim > 1 else 1, int(np.ceil(pair_dim * ratio))))
                for ratio in (0.50, 0.60)
            })
            for density_size in density_sizes:
                density_bits = np.ones(pair_dim, dtype=int)
                for _, _, _, _, pos in density_rank[:density_size]:
                    density_bits[pos] = 0
                candidates.append(make(density_bits, f"{source_prefix}_local_density_frontier"))

        if not paper_compact:
            # Legacy baseline-named frontiers are excluded from formal compact mode.
            gmda_bits = np.ones(pair_dim, dtype=int)
            gmda_rank = sorted(scored, key=lambda row: (0.72 * row[2] + 0.18 * row[1] + 0.10 * row[3], row[0]), reverse=True)
            for _, _, _, _, pos in gmda_rank[:min(pair_dim, 8)]:
                gmda_bits[pos] = 0
                candidates.append(make(gmda_bits.copy(), f"{source_prefix}_gmda_delay_frontier"))

            pdrs_bits = np.ones(pair_dim, dtype=int)
            pdrs_rank = sorted(scored, key=lambda row: (0.70 * row[1] + 0.20 * row[3] + 0.10 * row[2], row[0]), reverse=True)
            for _, _, _, _, pos in pdrs_rank[:min(pair_dim, 8)]:
                pdrs_bits[pos] = 0
                candidates.append(make(pdrs_bits.copy(), f"{source_prefix}_pdrs_energy_frontier"))

    if replica_ready_rank:
        # replica-ready本地候选：优先利用HAPA已经覆盖且ready的副本。
        ready_bits = np.ones(pair_dim, dtype=int)
        for _, pos in replica_ready_rank[:frontier_width(6)]:
            ready_bits[pos] = 0
            candidates.append(make(ready_bits.copy(), f"{source_prefix}_replica_ready_local"))

    if queue_aware and queue_balance_rank:
        # 队列平衡前沿：按Y/Z影子价格在pair级局部/云端之间做多档折中
        if paper_compact:
            frontier_sizes = [max(1, min(pair_dim, compact_width))]
        else:
            frontier_sizes = sorted(set([
                max(1, min(pair_dim, 4)),
                max(1, min(pair_dim, 8)),
            ]))
        for frontier_size in frontier_sizes:
            cloud_base = np.ones(pair_dim, dtype=int)
            local_base = np.zeros(pair_dim, dtype=int)
            for _, preferred_bit, pos in queue_balance_rank[:frontier_size]:
                cloud_base[pos] = int(preferred_bit)
                local_base[pos] = int(preferred_bit)
            candidates.append(make(cloud_base, f"{source_prefix}_queue_balanced_frontier"))
            candidates.append(make(local_base, f"{source_prefix}_queue_balanced_frontier"))
        if seed_candidates:
            emitted_balance = 0
            for seed in seed_candidates:
                seed_pair = np.asarray(seed.get("pair_action", []), dtype=int)
                if len(seed_pair) != pair_dim:
                    continue
                balanced = seed_pair.copy()
                for _, preferred_bit, pos in queue_balance_rank[:min(pair_dim, 8)]:
                    balanced[pos] = int(preferred_bit)
                    candidates.append(make(balanced.copy(), f"{source_prefix}_queue_balanced_frontier"))
                    emitted_balance += 1
                    if emitted_balance >= 2:
                        break
                if emitted_balance >= 2:
                    break

    if queue_pressure_rank:
        pressure_positions = [pos for score, pos in queue_pressure_rank if score > 0.0]
        if not pressure_positions:
            pressure_positions = [queue_pressure_rank[0][1]]
        pair_bits = np.ones(pair_dim, dtype=int)
        for pos in pressure_positions[:min(len(pressure_positions), max(1, min(pair_dim - 1, frontier_width(8))))]:
            pair_bits = pair_bits.copy()
            pair_bits[pos] = 0
            candidates.append(make(pair_bits, f"{source_prefix}_queue_pressure"))
            if not paper_compact:
                flip_bits = pair_bits.copy()
                flip_bits[pos] = 1 - flip_bits[pos]
                candidates.append(make(flip_bits, f"{source_prefix}_queue_pressure_flip"))
        if pair_dim >= 3:
            focused = np.ones(pair_dim, dtype=int)
            for _, pos in queue_pressure_rank[:frontier_width(6)]:
                focused[pos] = 0
            candidates.append(make(focused, f"{source_prefix}_queue_pressure_hybrid"))
            if not paper_compact:
                lycd_seed = focused.copy()
                for _, pos in queue_pressure_rank[:min(pair_dim, 3)]:
                    lycd_seed[pos] = 1 - lycd_seed[pos]
                candidates.append(make(lycd_seed, f"{source_prefix}_lycd_neighborhood"))
    if energy_frontier_rank:
        # energy-budget候选从全云端出发，只打开低能耗/高收益本地pair，避免UAC坍缩到过多本地。
        efficient_positions = [pos for score, delta, pos in energy_frontier_rank if score > 0.0 or delta >= 0.0]
        if not efficient_positions:
            efficient_positions = [pos for _, _, pos in energy_frontier_rank[:max(1, min(pair_dim, 3))]]
        budget_sizes = _energy_budget_frontier_sizes(pair_dim)
        if paper_compact:
            budget_sizes = sorted({size for size in budget_sizes if size <= max(1, compact_width)}) or [1]
        for budget in budget_sizes:
            budget_bits = np.ones(pair_dim, dtype=int)
            for pos in efficient_positions[:min(len(efficient_positions), budget)]:
                budget_bits[pos] = 0
            candidates.append(make(budget_bits, f"{source_prefix}_energy_budget_frontier"))

    if energy_queue_rank:
        relief_positions = [pos for score, pos in energy_queue_rank if score > 0.0]
        if not relief_positions:
            relief_positions = [energy_queue_rank[0][1]]
        pair_bits = np.zeros(pair_dim, dtype=int)
        for pos in relief_positions[:min(len(relief_positions), max(1, min(pair_dim - 1, frontier_width(8))))]:
            pair_bits = pair_bits.copy()
            pair_bits[pos] = 1
            candidates.append(make(pair_bits, f"{source_prefix}_energy_queue_relief"))
            candidates.append(make(pair_bits.copy(), f"{source_prefix}_energy_cloud_relief"))
        pareto_positions = list(relief_positions)
        if pareto_positions:
            # 能耗-成本Pareto释放：从全本地开始分档释放到云端，补齐非极端hybrid动作。
            relief_ratios = (0.50,) if paper_compact else (0.25, 0.50, 0.75)
            for relief_ratio in relief_ratios:
                relief_count = int(np.ceil(pair_dim * relief_ratio))
                relief_count = max(1, min(pair_dim - 1, relief_count)) if pair_dim > 1 else 1
                pareto_bits = np.zeros(pair_dim, dtype=int)
                for pos in pareto_positions[:relief_count]:
                    pareto_bits[pos] = 1
                candidates.append(make(pareto_bits, f"{source_prefix}_energy_cost_pareto_relief"))
        if seed_candidates:
            emitted_relief = 0
            relief_seed_limit = 2 if paper_compact else 4
            for seed in seed_candidates:
                seed_pair = np.asarray(seed.get("pair_action", []), dtype=int)
                if len(seed_pair) != pair_dim:
                    continue
                for pos in relief_positions[:min(len(relief_positions), relief_seed_limit)]:
                    relieved = seed_pair.copy()
                    relieved[pos] = 1
                    candidates.append(make(relieved, f"{source_prefix}_energy_queue_relief"))
                    candidates.append(make(relieved.copy(), f"{source_prefix}_energy_cloud_relief"))
                    emitted_relief += 1
                    if emitted_relief >= relief_seed_limit:
                        break
                if emitted_relief >= relief_seed_limit:
                    break

    if paper_compact and seed_candidates and queue_balance_rank:
        pressure_order = [int(pos) for score, pos in queue_pressure_rank if float(score) > 0.0]
        balance_order = [int(pos) for _, _, pos in queue_balance_rank]
        refinement_positions = []
        for pos in pressure_order + balance_order:
            if pos not in refinement_positions:
                refinement_positions.append(pos)
        if not refinement_positions:
            refinement_positions = balance_order[:]
        refinement_width = max(1, min(pair_dim, compact_width))
        emitted_refinements = 0
        for seed in seed_candidates:
            seed_pair = np.asarray(seed.get("pair_action", []), dtype=int)
            if len(seed_pair) != pair_dim:
                continue
            refined = seed_pair.copy()
            changed = 0
            for pos in refinement_positions[:refinement_width]:
                if int(refined[pos]) != 0 and int(np.sum(refined)) > 1:
                    refined[pos] = 0
                    changed += 1
            if changed == 0:
                for pos in refinement_positions:
                    if int(refined[pos]) != 0 and int(np.sum(refined)) > 1:
                        refined[pos] = 0
                        changed = 1
                        break
            if changed == 0:
                continue
            cloud_count = int(np.sum(refined))
            if cloud_count <= 0 or cloud_count >= pair_dim:
                continue
            candidates.append(make(refined, f"{source_prefix}_actor_delay_resource_refinement"))
            emitted_refinements += 1
            if emitted_refinements >= 2:
                break

    if not paper_compact:
        prefixes = [
            ("energy_repair", sorted(scored, key=lambda row: (row[1], row[0]), reverse=True)),
            ("hybrid_latency", sorted(scored, key=lambda row: (row[2], row[0]), reverse=True)),
            ("hybrid_cost", sorted(scored, key=lambda row: (row[3], row[0]), reverse=True)),
            ("hybrid_queue", scored),
        ]
        for source_suffix, ranking in prefixes:
            pair_bits = np.ones(pair_dim, dtype=int)
            for _, _, _, _, pos in ranking[:min(pair_dim, 8)]:
                pair_bits = pair_bits.copy()
                pair_bits[pos] = 0
                candidates.append(make(pair_bits, f"{source_prefix}_{source_suffix}"))

    if not paper_compact and pair_dim >= 2:
        diverse = np.ones(pair_dim, dtype=int)
        for rank, (_, _, _, _, pos) in enumerate(scored[:min(pair_dim, 10)]):
            if rank % 2 == 0:
                diverse[pos] = 0
        candidates.append(make(diverse, f"{source_prefix}_hybrid_diverse"))
        diverse_complement = np.ones(pair_dim, dtype=int)
        for rank, (_, _, _, _, pos) in enumerate(scored[:min(pair_dim, 10)]):
            if rank % 2 == 1:
                diverse_complement[pos] = 0
        candidates.append(make(diverse_complement, f"{source_prefix}_hybrid_diverse"))

    if seed_candidates and not paper_compact:
        reference_neighbor_rank = sorted(
            scored,
            key=lambda row: (row[3], row[1], row[0]),
            reverse=True,
        ) or [(0.0, 0.0, 0.0, 0.0, pos) for pos in range(pair_dim)]
        emitted_neighbors = 0
        neighbor_budget = max(1, min(6, pair_dim))
        for seed in seed_candidates:
            source = str(seed.get("candidate_source", ""))
            if "myopic_reference" not in source and "reference_repaired" not in source:
                continue
            seed_pair = np.asarray(seed.get("pair_action", []), dtype=int)
            if len(seed_pair) != pair_dim:
                continue
            for _, _, _, _, pos in reference_neighbor_rank:
                neighbor = seed_pair.copy()
                neighbor[pos] = 1 - neighbor[pos]
                cloud_count = int(np.sum(neighbor == 1))
                if cloud_count == 0 or cloud_count == pair_dim:
                    continue
                candidates.append(make(neighbor, f"{source_prefix}_reference_low_impact_neighbor"))
                emitted_neighbors += 1
                if emitted_neighbors >= neighbor_budget:
                    break
            if emitted_neighbors >= neighbor_budget:
                break
    if seed_candidates and not paper_compact:
        swap_budget = max(1, min(6, pair_dim))
        emitted_swaps = 0
        for seed in seed_candidates:
            source = str(seed.get("candidate_source", ""))
            if "myopic_reference" not in source and "reference_repaired" not in source:
                continue
            seed_pair = np.asarray(seed.get("pair_action", []), dtype=int)
            if len(seed_pair) != pair_dim:
                continue
            local_positions = [pos for _, _, _, _, pos in reference_neighbor_rank if int(seed_pair[pos]) == 0]
            cloud_positions = [pos for _, _, _, _, pos in reversed(reference_neighbor_rank) if int(seed_pair[pos]) == 1]
            if not local_positions or not cloud_positions:
                continue
            for in_pos in local_positions:
                for out_pos in cloud_positions[:1]:
                    if in_pos == out_pos:
                        continue
                    swapped = seed_pair.copy()
                    swapped[in_pos] = 1
                    swapped[out_pos] = 0
                    if int(np.sum(swapped == 1)) != int(np.sum(seed_pair == 1)):
                        continue
                    candidates.append(make(swapped, f"{source_prefix}_reference_low_impact_swap"))
                    emitted_swaps += 1
                    break
                if emitted_swaps >= swap_budget:
                    break
            if emitted_swaps >= swap_budget:
                break
    if seed_candidates and not paper_compact:
        balanced_budget = max(1, min(4, pair_dim))
        emitted_balanced = 0
        for seed in seed_candidates:
            source = str(seed.get("candidate_source", ""))
            if "myopic_reference" not in source and "reference_repaired" not in source:
                continue
            seed_pair = np.asarray(seed.get("pair_action", []), dtype=int)
            if len(seed_pair) != pair_dim:
                continue
            seed_cloud_count = int(np.sum(seed_pair == 1))
            if seed_cloud_count <= 0 or seed_cloud_count >= pair_dim:
                continue
            local_positions = [pos for _, _, _, _, pos in reference_neighbor_rank if int(seed_pair[pos]) == 0]
            cloud_positions = [pos for _, _, _, _, pos in reversed(reference_neighbor_rank) if int(seed_pair[pos]) == 1]
            if not local_positions or not cloud_positions:
                continue
            max_width = min(2, len(local_positions), len(cloud_positions))
            for width in range(max_width, 0, -1):
                balanced = seed_pair.copy()
                for pos in local_positions[:width]:
                    balanced[pos] = 1
                for pos in cloud_positions[:width]:
                    balanced[pos] = 0
                if int(np.sum(balanced == 1)) != seed_cloud_count:
                    continue
                if int(np.sum(balanced != seed_pair)) == 0:
                    continue
                cloud_count = int(np.sum(balanced == 1))
                if cloud_count == 0 or cloud_count == pair_dim:
                    continue
                candidates.append(make(balanced, f"{source_prefix}_balanced_tail_relief"))
                emitted_balanced += 1
                if emitted_balanced >= balanced_budget:
                    break
            if emitted_balanced >= balanced_budget:
                break
    if seed_candidates and not paper_compact:
        flip_budget = max(4, min(limit, pair_dim * 2))
        emitted = 0
        for seed in seed_candidates:
            seed_pair = np.asarray(seed.get("pair_action", []), dtype=int)
            if len(seed_pair) != pair_dim:
                continue
            for _, _, _, _, pos in scored[:min(pair_dim, 8)]:
                flipped = seed_pair.copy()
                flipped[pos] = 1 - flipped[pos]
                candidates.append(make(flipped, f"{source_prefix}_pair_flip"))
                candidates.append(make(flipped.copy(), f"{source_prefix}_queue_pressure_flip"))
                emitted += 1
                if emitted >= flip_budget:
                    break
            if emitted >= flip_budget:
                break

    # 先按来源优先级选择，再按动作去重；否则相同bit动作会把queue/DPP来源覆盖掉。
    if paper_compact:
        priority_keywords = [
            "energy_low_dvfs_local", "energy_cloud_relief", "energy_cost_pareto_relief",
            "energy_budget_frontier", "cost_aware_hybrid", "delay_energy_frontier",
            "actor_delay_resource_refinement", "local_density_frontier",
            "replica_ready_local", "guard_all_cloud", "guard_all_local",
            "low_energy_delay_feasible", "queue_balanced_frontier", "energy_queue_relief",
            "queue_pressure",
        ]
    else:
        priority_keywords = [
            "energy_low_dvfs_local", "energy_cloud_relief", "energy_cost_pareto_relief", "energy_budget_frontier", "balanced_tail_relief", "cost_aware_hybrid",
            "delay_energy_frontier", "replica_ready_local", "queue_pressure_flip", "lycd_neighborhood",
            "guard_all_cloud", "guard_all_local",
            "low_energy_delay_feasible", "low_cost_ffd_like_local", "gmda_delay_frontier", "pdrs_energy_frontier",
            "queue_balanced_frontier", "energy_queue_relief", "queue_pressure",
            "reference_low_impact_neighbor", "reference_low_impact_swap", "pair_flip",
        ]
    selected = []
    seen_action_keys = set()
    seen_source_keys = set()

    def source_family(candidate):
        source = str(candidate.get("candidate_source", ""))
        parts = source.split("_", 1)
        family = parts[1] if len(parts) == 2 else source
        candidate["action_source_family"] = family
        return family

    def action_key(candidate):
        return tuple(int(v) for v in np.asarray(candidate.get("pair_action", []), dtype=int))

    def add_candidate(candidate, reserve_source: bool = False):
        family = source_family(candidate)
        key = action_key(candidate)
        if reserve_source:
            source_key = (family, key)
            if source_key in seen_source_keys:
                return False
            seen_source_keys.add(source_key)
            selected.append(candidate)
            seen_action_keys.add(key)
            return True
        if key in seen_action_keys:
            return False
        seen_action_keys.add(key)
        selected.append(candidate)
        return True

    # 优先级阶段按来源保留候选，Pareto relief保留多档动作，其它来源保留一个。
    if paper_compact:
        priority_quotas = {
            "energy_cloud_relief": 2,
            "energy_cost_pareto_relief": 1,
            "actor_delay_resource_refinement": 2,
            "local_density_frontier": 2,
            "energy_queue_relief": 2,
            "queue_balanced_frontier": 2,
            "queue_pressure": 2,
        }
    else:
        priority_quotas = {
            "energy_cloud_relief": 4,
            "energy_cost_pareto_relief": 3,
            "energy_queue_relief": 3,
            "balanced_tail_relief": 4,
            "reference_low_impact_neighbor": 6,
            "reference_low_impact_swap": 6,
        }
    protected_quota = len(priority_keywords) + sum(
        max(0, int(quota) - 1) for quota in priority_quotas.values()
    )
    for keyword in priority_keywords:
        kept = 0
        quota = int(priority_quotas.get(keyword, 1))
        for candidate in candidates:
            if keyword in str(candidate.get("candidate_source", "")) and add_candidate(candidate, reserve_source=True):
                kept += 1
                if kept >= quota:
                    break
        if len(selected) >= max(limit, protected_quota):
            return selected[:max(limit, protected_quota)]
    for candidate in candidates:
        if len(selected) >= limit:
            break
        add_candidate(candidate)
    return selected[:max(limit, len(selected))]


def build_myopic_candidates(system_state, config: AblationExperimentConfig) -> List[np.ndarray]:
    """生成Myopic候选集，不读取虚拟队列"""
    env_manager = system_state.environment_manager
    SH, _, _ = env_manager.get_state_components()
    n = len(SH)
    if n == 0:
        return [np.array([], dtype=int)]

    candidates = []
    limit = max(int(getattr(config, "myopic_candidate_limit", 64)), 4)
    active_mask = get_active_ai_server_mask(system_state)
    sh_min = float(np.min(SH))
    sh_span = float(np.max(SH) - sh_min) or 1.0
    normalized_sh = (np.asarray(SH, dtype=float) - sh_min) / sh_span

    # 阈值候选保留旧启发式信息，但最终由queue-unaware目标函数统一评分
    for threshold in [0.25, 0.35, 0.5, 0.65, 0.75]:
        candidates.append(_apply_active_mask((normalized_sh < threshold).astype(int), active_mask))

    candidates.extend(build_guard_candidates(system_state))

    ranked = [
        idx for idx, _ in sorted(
            enumerate(normalized_sh),
            key=lambda item: (-float(item[1]), item[0])
        )
        if idx < len(active_mask) and active_mask[idx]
    ]

    all_cloud = np.ones(n, dtype=int)
    prefix_action = np.ones(n, dtype=int)
    for idx in ranked:
        prefix_action = prefix_action.copy()
        prefix_action[idx] = 0
        candidates.append(prefix_action)

    all_local = _apply_active_mask(np.zeros(n, dtype=int), active_mask)
    for idx in ranked:
        one_local = all_cloud.copy()
        one_local[idx] = 0
        candidates.append(one_local)
        one_cloud = all_local.copy()
        one_cloud[idx] = 1
        candidates.append(_apply_active_mask(one_cloud, active_mask))

    if getattr(config, "myopic_randomness_rate", 0.0) > 0:
        seed_value = int(np.sum(np.asarray(SH, dtype=float) * 1000003)) % (2 ** 32)
        rng = np.random.default_rng(seed_value)
        for _ in range(min(8, limit)):
            random_action = (rng.random(n) < config.myopic_randomness_rate).astype(int)
            candidates.append(_apply_active_mask(random_action, active_mask))

    return _deduplicate_candidates(candidates, n)[:limit]


def select_best_action_from_candidates(candidates: Iterable[np.ndarray], system_state,
                                       config: AblationExperimentConfig,
                                       queue_aware: bool = True) -> Dict:
    """逐候选调用共享evaluator，按DPP或预注册三指标选择动作。"""
    from ResourceAllocation import evaluate_action_dry_run

    start_time = time.perf_counter()
    candidate_list = list(candidates)
    if not candidate_list:
        candidate_list = build_guard_candidates(system_state)
    try:
        action_dim = get_ai_action_dimension(system_state)
    except AttributeError:
        action_dim = infer_candidate_action_dimension(candidate_list)
    if candidate_list and all(
            np.all(_normalize_action(item.get("action") if isinstance(item, dict) else item, action_dim) == 1)
            for item in candidate_list):
        repair_candidates = wrap_candidates_with_pair_projection(
            build_local_repair_candidates(system_state), system_state
        )
        candidate_list.extend(repair_candidates)
    candidate_list = add_energy_claim_resource_variants(
        candidate_list, config, queue_aware=queue_aware, system_state=system_state
    )

    best_eval = None
    best_energy_eval = None
    best_rank = -1
    eval_times = []
    evaluated_actions = []
    evaluated_results = []
    eval_cache = {}
    best_local_score = float("inf")
    best_cloud_score = float("inf")
    local_candidate_feasible_count = 0
    all_cloud_candidate_count = 0
    all_local_candidate_count = 0
    for rank, candidate in enumerate(candidate_list):
        candidate_meta = dict(candidate) if isinstance(candidate, dict) else {}
        candidate_action = candidate_meta.get("action", candidate)
        normalized_action = _normalize_action(candidate_action, action_dim)
        evaluated_actions.append(tuple(int(x) for x in normalized_action))
        eval_start = time.perf_counter()
        pair_key = tuple(int(x) for x in np.asarray(candidate_meta.get("pair_action", []), dtype=int).reshape(-1))
        resource_cache_key = tuple(
            str(candidate_meta.get(key, ""))
            for key in ["resource_hint", "resource_queue_aware", "resource_queue_scale", "resource_mode", "f_gpu", "batch_size", "f_pre", "compression_ratio"]
        )
        candidate_resource_queue_aware = bool(candidate_meta.get("resource_queue_aware", queue_aware))
        # 同一pair动作在不同资源偏好下会触发不同(g,b,f_GPU)/f_pre搜索，不能共用dry-run结果。
        cache_key = (
            tuple(int(x) for x in normalized_action),
            pair_key,
            bool(queue_aware),
            bool(candidate_resource_queue_aware),
            resource_cache_key,
        )
        if cache_key in eval_cache:
            result = dict(eval_cache[cache_key])
        else:
            result = evaluate_action_dry_run(candidate_meta or candidate_action, system_state, config, queue_aware=queue_aware)
            eval_cache[cache_key] = dict(result)
        for key in [
            "action_dim", "pair_action_dim", "pair_universe",
            "routing_policy", "model_mutated", "candidate_source", "action_scope",
            "resource_hint", "resource_queue_aware", "resource_queue_scale", "resource_mode",
        ]:
            if key in candidate_meta:
                result[key] = candidate_meta[key]
        result["eval_rank"] = int(rank)
        result["repaired_pair_action_hash"] = str(
            result.get("repaired_pair_action_hash") or result.get("pair_action_hash") or result.get("action_hash", "")
        )
        evaluated_results.append(dict(result))
        eval_times.append((time.perf_counter() - eval_start) * 1000.0)
        score = float(result.get("paper_dpp_score", float("inf")))
        feasible = bool(result.get("feasible", False))
        local_count = int(result.get("local_count", 0))
        cloud_count = int(result.get("cloud_count", 0))
        if cloud_count > 0 and local_count == 0:
            all_cloud_candidate_count += 1
            if feasible:
                best_cloud_score = min(best_cloud_score, score)
        if local_count > 0 and cloud_count == 0:
            all_local_candidate_count += 1
        if local_count > 0 and feasible:
            local_candidate_feasible_count += 1
            best_local_score = min(best_local_score, score)
        if best_eval is None:
            best_eval = result
            best_rank = rank
            continue
        best_score = float(best_eval.get("paper_dpp_score", float("inf")))
        if feasible and (not best_eval.get("feasible", False) or score < best_score):
            best_eval = result
            best_rank = rank
        elif feasible == best_eval.get("feasible", False) and score < best_score:
            best_eval = result
            best_rank = rank

    deduped_results = dedupe_evaluated_candidates_by_repaired_hash(evaluated_results)
    feasible_results = [row for row in deduped_results if row.get("feasible", False)]
    selectable_results = feasible_results or deduped_results
    if selectable_results:
        best_score = min(float(row.get("paper_dpp_score", float("inf"))) for row in selectable_results)
        dpp_tolerance = max(
            abs(best_score) * float(getattr(config, "energy_hard_dpp_tolerance_ratio", 0.03)),
            1e-9,
        )
        annotate_claim_selector_rows(selectable_results, config, best_score, dpp_tolerance)
        # source摘要使用完整候选列表，需同步写入claim诊断字段。
        annotate_claim_selector_rows(evaluated_results, config, best_score, dpp_tolerance)
        temporal_history = list(getattr(system_state, "_lyham_temporal_metric_history", []) or [])
        # 仅用于诊断：记录候选集中最低system energy的可执行动作，不改变最终选择规则。
        best_energy_eval = min(
            selectable_results,
            key=lambda row: (
                not bool(row.get("feasible", False)),
                float(row.get("energy_j", float("inf"))),
                float(row.get("paper_dpp_score", float("inf"))),
                int(row.get("eval_rank", 0)),
            )
        )
        if (not queue_aware) and getattr(config, "include_energy_claim", False):
            # Myopic不读取虚拟队列；energy-claim场景用即时三指标，避免冷启动时退化为只看cost。
            best_eval = min(
                selectable_results,
                key=lambda row: (
                    not bool(row.get("feasible", False)),
                    float(row.get("claim_score", float("inf"))),
                    float(row.get("paper_dpp_score", float("inf"))),
                    int(row.get("eval_rank", 0)),
                )
            )
            best_eval["selected_by_dpp_or_claim_band"] = "myopic_immediate"
        elif queue_aware and getattr(config, "include_energy_claim", False):
            # energy-hard下先用共享evaluator约束paper DPP，再在带内选三指标非支配候选。
            near_best = [
                row for row in selectable_results
                if bool(row.get("dpp_band_passed", False))
            ]
            if queue_pressure_dpp_required(selectable_results, config):
                queue_dpp_pool = near_best or selectable_results
                temporal_dpp_ratio = max(
                    float(getattr(config, "energy_claim_temporal_guard_dpp_regret_ratio", 0.0)),
                    0.0,
                )
                strict_dpp_best = min(
                    queue_dpp_pool,
                    key=lambda row: (
                        not bool(row.get("feasible", False)),
                        float(row.get("paper_dpp_score", float("inf"))),
                        float(row.get("claim_score", float("inf"))),
                        int(row.get("eval_rank", 0)),
                    )
                )
                temporal_pool = queue_dpp_pool
                if temporal_dpp_ratio > 0.0:
                    strict_dpp_score = _finite_float(strict_dpp_best.get("paper_dpp_score"), float("inf"))
                    if np.isfinite(strict_dpp_score):
                        temporal_dpp_budget = max(
                            abs(strict_dpp_score) * temporal_dpp_ratio,
                            dpp_tolerance,
                        )
                        temporal_pool = [
                            row for row in evaluated_results
                            if bool(row.get("feasible", False)) and
                            _finite_float(row.get("paper_dpp_score"), float("inf"))
                            <= strict_dpp_score + temporal_dpp_budget
                        ] or queue_dpp_pool
                delay_guard = queue_pressure_delay_guard_candidate(
                    temporal_pool, config, strict_dpp_best
                )
                tail_delay_guard = queue_pressure_tail_delay_guard_candidate(
                    evaluated_results, config, strict_dpp_best
                )
                selected_delay_guard = delay_guard
                selection_label = "queue_pressure_dpp_delay_guard"
                if tail_delay_guard:
                    tail_delay = _finite_float(tail_delay_guard.get("delay_ms"), float("inf"))
                    current_guard_delay = _finite_float(
                        delay_guard.get("delay_ms") if delay_guard else strict_dpp_best.get("delay_ms"),
                        float("inf"),
                    )
                    if tail_delay < current_guard_delay:
                        selected_delay_guard = tail_delay_guard
                        selection_label = "queue_pressure_dpp_tail_delay_guard"
                if selected_delay_guard:
                    best_eval = selected_delay_guard
                else:
                    stability_guard = claim_stability_guard_candidate(
                        queue_dpp_pool, config, strict_dpp_best
                    )
                    if stability_guard:
                        best_eval = stability_guard
                        selection_label = "queue_pressure_dpp_stability_guard"
                    else:
                        best_eval = strict_dpp_best
                        selection_label = "queue_pressure_dpp"
                temporal_guard = claim_temporal_guard_candidate(
                    temporal_pool, config, best_eval, temporal_history
                )
                if temporal_guard:
                    apply_temporal_guard = True
                    if "delay_guard" in selection_label:
                        current_delay = _finite_float(best_eval.get("delay_ms"), float("inf"))
                        temporal_delay = _finite_float(temporal_guard.get("delay_ms"), float("inf"))
                        max_delay_regret = max(
                            float(getattr(
                                config,
                                "queue_pressure_delay_guard_temporal_max_delay_regret_ms",
                                20.0,
                            )),
                            0.0,
                        )
                        if (
                            np.isfinite(current_delay) and
                            np.isfinite(temporal_delay) and
                            temporal_delay - current_delay > max_delay_regret
                        ):
                            apply_temporal_guard = False
                    if apply_temporal_guard:
                        best_eval = temporal_guard
                        selection_label = f"{selection_label}_temporal_guard"
                energy_guard_pool = queue_dpp_pool
                energy_guard_dpp_ratio = max(
                    float(getattr(config, "queue_pressure_energy_guard_dpp_slack_ratio", 0.0)),
                    0.0,
                )
                if energy_guard_dpp_ratio > 0.0:
                    strict_dpp_score = _finite_float(strict_dpp_best.get("paper_dpp_score"), float("inf"))
                    if np.isfinite(strict_dpp_score):
                        energy_guard_dpp_budget = max(
                            abs(strict_dpp_score) * energy_guard_dpp_ratio,
                            dpp_tolerance,
                        )
                        energy_guard_pool = [
                            row for row in evaluated_results
                            if bool(row.get("feasible", False)) and
                            _finite_float(row.get("paper_dpp_score"), float("inf"))
                            <= strict_dpp_score + energy_guard_dpp_budget
                        ] or queue_dpp_pool
                energy_guard_current = best_eval
                if "temporal_guard" in selection_label:
                    strict_dpp_score = _finite_float(strict_dpp_best.get("paper_dpp_score"), float("nan"))
                    if np.isfinite(strict_dpp_score):
                        energy_guard_current = dict(best_eval)
                        energy_guard_current["energy_guard_strict_dpp_baseline_score"] = strict_dpp_score
                        energy_guard_current["energy_guard_strict_dpp_baseline_source"] = str(
                            strict_dpp_best.get("candidate_source", "")
                        )
                        energy_guard_current["energy_guard_strict_dpp_baseline_hash"] = str(
                            strict_dpp_best.get("repaired_pair_action_hash") or
                            strict_dpp_best.get("pair_action_hash") or
                            strict_dpp_best.get("action_hash") or
                            ""
                        )
                energy_guard = queue_pressure_energy_guard_candidate(
                    energy_guard_pool, config, energy_guard_current
                )
                if energy_guard:
                    best_eval = energy_guard
                    selection_label = f"{selection_label}_energy_guard"
                best_eval["selected_by_dpp_or_claim_band"] = selection_label
            else:
                pareto_near_best = [
                    row for row in near_best
                    if bool(row.get("is_pareto_candidate", False))
                ]
                claim_pool = pareto_near_best or near_best
                best_claim_score = min(
                    float(row.get("claim_score", float("inf"))) for row in claim_pool
                )
                diversity_tolerance = max(abs(best_claim_score) * 0.08, 0.10)
                near_claim_pool = [
                    row for row in claim_pool
                    if float(row.get("claim_score", float("inf"))) <= best_claim_score + diversity_tolerance
                ]
                noncollapsed_pool = [
                    row for row in near_claim_pool
                    if int(row.get("local_count", 0)) > 0 and int(row.get("cloud_count", 0)) > 0
                ]
                final_claim_pool = near_claim_pool or claim_pool
                if noncollapsed_pool:
                    claim_rank_key = lambda row: (
                        not bool(row.get("feasible", False)),
                        float(row.get("claim_score", float("inf"))),
                        float(row.get("paper_dpp_score", float("inf"))),
                        int(row.get("eval_rank", 0)),
                    )
                    best_overall_claim_row = min(final_claim_pool, key=claim_rank_key)
                    best_noncollapsed_row = min(noncollapsed_pool, key=claim_rank_key)
                    if id(best_overall_claim_row) == id(best_noncollapsed_row):
                        final_claim_pool = noncollapsed_pool
                    else:
                        delay_ref = max(float(getattr(config, "claim_delay_ref_ms", 100.0)), 1e-9)
                        energy_ref = max(float(getattr(config, "claim_energy_ref_j", 2.0)), 1e-9)
                        cost_ref = max(float(getattr(config, "claim_cost_ref", 400.0)), 1e-9)
                        delay_gain = max(
                            float(best_overall_claim_row.get("delay_ms", float("inf"))) -
                            float(best_noncollapsed_row.get("delay_ms", float("inf"))),
                            0.0,
                        ) / delay_ref
                        energy_penalty = max(
                            float(best_noncollapsed_row.get("energy_j", float("inf"))) -
                            float(best_overall_claim_row.get("energy_j", float("inf"))),
                            0.0,
                        ) / energy_ref
                        cost_penalty = max(
                            float(best_noncollapsed_row.get("cost", float("inf"))) -
                            float(best_overall_claim_row.get("cost", float("inf"))),
                            0.0,
                        ) / cost_ref
                        delay_penalty = max(
                            float(best_noncollapsed_row.get("delay_ms", float("inf"))) -
                            float(best_overall_claim_row.get("delay_ms", float("inf"))),
                            0.0,
                        ) / delay_ref
                        energy_gain = max(
                            float(best_overall_claim_row.get("energy_j", float("inf"))) -
                            float(best_noncollapsed_row.get("energy_j", float("inf"))),
                            0.0,
                        ) / energy_ref
                        cost_gain = max(
                            float(best_overall_claim_row.get("cost", float("inf"))) -
                            float(best_noncollapsed_row.get("cost", float("inf"))),
                            0.0,
                        ) / cost_ref
                        claim_gap = max(
                            float(best_noncollapsed_row.get("claim_score", float("inf"))) -
                            float(best_overall_claim_row.get("claim_score", float("inf"))),
                            0.0,
                        )
                        normalized_gain = delay_gain + energy_gain + cost_gain
                        normalized_penalty = delay_penalty + energy_penalty + cost_penalty
                        max_single_penalty = max(delay_penalty, energy_penalty, cost_penalty)
                        diversity_regret_budget = min(float(diversity_tolerance), 0.10)
                        # 已进入near-claim池的hybrid动作可用小幅regret换取动作多样性。
                        prefer_noncollapsed = (
                            normalized_gain > 0.0 and
                            claim_gap <= diversity_regret_budget and
                            max_single_penalty <= diversity_regret_budget and
                            normalized_penalty <= normalized_gain + diversity_regret_budget
                        )
                        if prefer_noncollapsed:
                            final_claim_pool = noncollapsed_pool
                escape_pool = build_claim_escape_pool(
                    selectable_results, final_claim_pool, config, best_score
                )
                selection_label = "claim_escape" if escape_pool else "claim_band"
                if escape_pool:
                    final_claim_pool = escape_pool
                best_eval = min(
                    final_claim_pool,
                    key=lambda row: (
                        not bool(row.get("feasible", False)),
                        float(row.get("claim_score", float("inf"))),
                        float(row.get("paper_dpp_score", float("inf"))),
                        int(row.get("eval_rank", 0)),
                    )
                )
                stability_guard = claim_stability_guard_candidate(
                    selectable_results, config, best_eval
                )
                if stability_guard:
                    best_eval = stability_guard
                    selection_label = f"{selection_label}_stability_guard"
                temporal_guard = claim_temporal_guard_candidate(
                    selectable_results, config, best_eval, temporal_history
                )
                if temporal_guard:
                    best_eval = temporal_guard
                    selection_label = f"{selection_label}_temporal_guard"
                best_eval["selected_by_dpp_or_claim_band"] = selection_label
        else:
            best_eval = min(
                selectable_results,
                key=lambda row: (not bool(row.get("feasible", False)), float(row.get("paper_dpp_score", float("inf"))), int(row.get("eval_rank", 0)))
            )
            best_eval["selected_by_dpp_or_claim_band"] = "dpp"
        best_rank = int(best_eval.get("eval_rank", best_rank))
    if best_eval is None:
        raise ValueError("候选动作为空，无法执行快层控制器")

    decision = dict(best_eval)
    fallback = candidate_list[best_rank].get("action") if isinstance(candidate_list[best_rank], dict) else candidate_list[best_rank]
    decision["action"] = np.array(decision.get("action", fallback), dtype=int)
    decision["candidate_count"] = len(candidate_list)
    decision["selected_candidate_rank"] = int(best_rank)
    decision.setdefault("replay_written", False)
    decision.setdefault("online_update_step", 0)
    decision.setdefault("solver_gap_vs_lycd", 0.0)
    decision.setdefault("action_dim", int(len(decision["action"])))
    decision.setdefault("pair_action_dim", 0)
    decision.setdefault("model_mutated", False)
    decision.setdefault("selected_candidate_source", str(decision.get("candidate_source", "")))
    decision.setdefault("candidate_source", str(decision.get("selected_candidate_source", "")))
    decision.setdefault("claim_score", calculate_claim_score(decision, config))
    decision.setdefault("is_pareto_candidate", False)
    decision.setdefault("dpp_band_passed", False)
    decision.setdefault("selected_by_dpp_or_claim_band", "dpp")
    decision.setdefault("resource_queue_scale", 1.0)
    decision.setdefault("per_pair_delta_delay", "")
    decision.setdefault("per_pair_delta_energy", "")
    decision.setdefault("per_pair_delta_cost", "")
    decision.setdefault("resource_hint", "")
    decision.setdefault("resource_queue_aware", bool(queue_aware))
    decision.setdefault("resource_mode", "")
    decision.setdefault("resource_hint_collapsed", False)
    decision.setdefault("candidate_source_score_summary", "")
    decision.setdefault("candidate_source_family_count", 0.0)
    tail_risk_defaults = {
        "tail_risk_candidate_source": "",
        "tail_risk_candidate_hash": "",
        "tail_risk_candidate_delay_ms": 0.0,
        "tail_risk_candidate_energy_j": 0.0,
        "tail_risk_candidate_cost": 0.0,
        "tail_risk_candidate_claim_score": 0.0,
        "tail_risk_candidate_dpp_score": 0.0,
        "tail_risk_candidate_predicted_avg_y": 0.0,
        "tail_risk_candidate_predicted_avg_z": 0.0,
        "tail_risk_candidate_post_update_queue_drift_term": 0.0,
        "tail_risk_candidate_upper_excess_score": 0.0,
        "tail_risk_selected_upper_excess_score": 0.0,
        "tail_risk_candidate_upper_excess_improvement": 0.0,
        "tail_risk_best_relief_source": "",
        "tail_risk_best_relief_hash": "",
        "tail_risk_best_relief_delay_ms": 0.0,
        "tail_risk_best_relief_energy_j": 0.0,
        "tail_risk_best_relief_cost": 0.0,
        "tail_risk_best_relief_claim_score": 0.0,
        "tail_risk_best_relief_dpp_score": 0.0,
        "tail_risk_best_relief_predicted_avg_y": 0.0,
        "tail_risk_best_relief_predicted_avg_z": 0.0,
        "tail_risk_best_relief_post_update_queue_drift_term": 0.0,
        "tail_risk_best_relief_upper_excess_score": 0.0,
        "tail_risk_best_relief_improvement": 0.0,
        "tail_risk_best_relief_reject_reason": "",
    }
    energy_relief_defaults = {
        "energy_relief_candidate_source": "",
        "energy_relief_candidate_hash": "",
        "energy_relief_candidate_delay_ms": 0.0,
        "energy_relief_candidate_energy_j": 0.0,
        "energy_relief_candidate_cost": 0.0,
        "energy_relief_candidate_claim_score": 0.0,
        "energy_relief_candidate_dpp_score": 0.0,
        "energy_relief_candidate_predicted_avg_y": 0.0,
        "energy_relief_candidate_predicted_avg_z": 0.0,
        "energy_relief_candidate_post_update_queue_drift_term": 0.0,
        "energy_relief_candidate_energy_gain_j": 0.0,
        "energy_relief_candidate_delay_regret_ms": 0.0,
        "energy_relief_candidate_cost_regret": 0.0,
        "energy_relief_candidate_dpp_regret": 0.0,
        "energy_relief_best_lower_source": "",
        "energy_relief_best_lower_hash": "",
        "energy_relief_best_lower_delay_ms": 0.0,
        "energy_relief_best_lower_energy_j": 0.0,
        "energy_relief_best_lower_cost": 0.0,
        "energy_relief_best_lower_claim_score": 0.0,
        "energy_relief_best_lower_dpp_score": 0.0,
        "energy_relief_best_lower_predicted_avg_y": 0.0,
        "energy_relief_best_lower_predicted_avg_z": 0.0,
        "energy_relief_best_lower_post_update_queue_drift_term": 0.0,
        "energy_relief_best_lower_energy_gain_j": 0.0,
        "energy_relief_best_lower_delay_regret_ms": 0.0,
        "energy_relief_best_lower_cost_regret": 0.0,
        "energy_relief_best_lower_dpp_regret": 0.0,
        "energy_relief_best_lower_reject_reason": "",
    }
    decision.update(tail_risk_defaults)
    decision.update(energy_relief_defaults)
    if queue_aware and getattr(config, "include_energy_claim", False):
        decision.update(
            tail_risk_diagnostic_candidate(
                selectable_results, decision, temporal_history, config
            )
        )
        decision.update(
            energy_relief_diagnostic_candidate(
                selectable_results, config, decision
            )
        )
    if best_energy_eval is not None:
        selected_energy = float(decision.get("energy_j", float("inf")))
        selected_dpp = float(decision.get("paper_dpp_score", float("inf")))
        best_energy_value = float(best_energy_eval.get("energy_j", float("inf")))
        best_energy_dpp = float(best_energy_eval.get("paper_dpp_score", float("inf")))
        decision["best_energy_candidate_source"] = str(best_energy_eval.get("candidate_source", ""))
        decision["best_energy_candidate_hash"] = str(
            best_energy_eval.get("repaired_pair_action_hash") or
            best_energy_eval.get("pair_action_hash") or
            best_energy_eval.get("action_hash", "")
        )
        decision["best_energy_candidate_energy_j"] = best_energy_value
        decision["best_energy_candidate_dpp_score"] = best_energy_dpp
        decision["best_energy_candidate_energy_gap"] = float(selected_energy - best_energy_value)
        decision["best_energy_candidate_dpp_gap"] = float(best_energy_dpp - selected_dpp)
    else:
        decision["best_energy_candidate_source"] = ""
        decision["best_energy_candidate_hash"] = ""
        decision["best_energy_candidate_energy_j"] = float("nan")
        decision["best_energy_candidate_dpp_score"] = float("nan")
        decision["best_energy_candidate_energy_gap"] = float("nan")
        decision["best_energy_candidate_dpp_gap"] = float("nan")
    repaired_hashes = [str(row.get("repaired_pair_action_hash", "")) for row in evaluated_results if row.get("repaired_pair_action_hash")]
    unique_actions = len(set(repaired_hashes)) if repaired_hashes else len(set(evaluated_actions))
    decision["repaired_candidate_diversity"] = unique_actions / max(len(evaluated_results), 1)
    decision["repaired_source_distribution"] = ";".join(sorted({str(row.get("candidate_source", "")) for row in deduped_results if row.get("candidate_source", "")}))
    source_score_summary = summarize_candidate_source_scores(evaluated_results)
    decision["candidate_source_score_summary"] = format_candidate_source_score_summary(source_score_summary)
    decision["candidate_source_family_count"] = float(len(source_score_summary))
    decision["best_local_score"] = float(best_local_score)
    decision["best_cloud_score"] = float(best_cloud_score)
    decision["score_gap_local_vs_cloud"] = float(best_local_score - best_cloud_score)
    decision["local_candidate_feasible_count"] = int(local_candidate_feasible_count)
    decision["candidate_diversity"] = unique_actions / max(len(evaluated_actions), 1)
    decision["all_cloud_candidate_count"] = int(all_cloud_candidate_count)
    decision["all_local_candidate_count"] = int(all_local_candidate_count)
    decision["decision_time_ms"] = (time.perf_counter() - start_time) * 1000.0
    decision["p95_decision_time_ms"] = float(np.percentile(eval_times, 95)) if eval_times else 0.0
    return decision













