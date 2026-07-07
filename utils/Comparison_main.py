"""Compatibility entry points for the unified ablation pipeline.

The old standalone comparison implementation was removed from this module to
avoid accidentally reusing legacy result paths. Current experiments must go
through ``run_ablation.py`` and the normal-main algorithm lists in
``ablation_config.py``.
"""

from typing import Dict, List

from run_ablation import run_ablation_experiment


def run_ablation_experiment_from_comparison(
    seeds: List[int] = None,
    time_slots: int = None,
    algorithms: List[str] = None,
    output_dir: str = None,
    silent: bool = True,
) -> Dict:
    """Backward-compatible wrapper around the canonical ablation runner."""
    from ablation_config import AblationExperimentConfig

    config = AblationExperimentConfig()
    if seeds is not None:
        config.seeds = seeds
    if time_slots is not None:
        config.time_slots = time_slots
    if algorithms is not None:
        config.algorithms = algorithms
    if output_dir is not None:
        config.output_dir = output_dir
    return run_ablation_experiment(config=config, silent=silent)


def run_normal_experiment_from_repro_pipeline(
    seeds: List[int] = None,
    time_slots: int = None,
    output_dir: str = None,
    silent: bool = True,
) -> Dict:
    """Run normal-main through the canonical reproducible pipeline."""
    from ablation_config import (
        AblationExperimentConfig,
        NORMAL_MAIN_ALGORITHMS,
        NORMAL_MAIN_BASELINE_ALGORITHMS,
    )

    config = AblationExperimentConfig()
    config.experiment_type = "normal_main"
    config.algorithms = list(NORMAL_MAIN_ALGORITHMS)
    config.claim_baselines = list(NORMAL_MAIN_BASELINE_ALGORITHMS)
    if seeds is not None:
        config.seeds = seeds
    if time_slots is not None:
        config.time_slots = time_slots
    if output_dir is not None:
        config.output_dir = output_dir
    return run_ablation_experiment(config=config, silent=silent)
