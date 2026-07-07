"""
论文图数据实验计划
文件只定义绘图横坐标和统一复现入口的配置骨架，不直接画图。
"""
from copy import deepcopy
from typing import Dict, List

from ablation_config import AblationExperimentConfig, NORMAL_MAIN_ALGORITHMS, NORMAL_MAIN_BASELINE_ALGORITHMS
from run_ablation import apply_energy_claim_profile, apply_heterogeneous_burst_main_profile


def build_figure_sweep_plan() -> Dict[str, List]:
    """返回当前绘图代码使用的横坐标。

    这些值来自 `绘图代码_单位校准/generate_unit_calibrated_static_figures.py`。
    后续正式数据必须按这些横坐标从统一 pipeline 生成，不能继续手填静态数组。
    """
    return {
        "chain_length": [(2, 4), (3, 5), (4, 6), (5, 7), (6, 8)],
        "arrival_rate": [5, 6, 7, 8, 9, 10, 11, 12],
        "edge_nodes": [20, 25, 30, 35, 40, 45],
        "V": [1, 5, 10, 20, 50, 100, 200, 500],
    }


def build_figure_experiment_configs(base_config: AblationExperimentConfig = None) -> List[AblationExperimentConfig]:
    """生成论文图所需的配置矩阵。

    这里只构造配置，不运行实验。每个配置都使用 normal-main 算法集合和 claim baseline，
    便于后续 raw/summary/meta 与正式主实验口径一致。
    """
    base = deepcopy(base_config) if base_config is not None else AblationExperimentConfig()
    base.experiment_type = "normal_main"
    base.algorithms = list(NORMAL_MAIN_ALGORITHMS)
    base.claim_baselines = list(NORMAL_MAIN_BASELINE_ALGORITHMS)
    base.include_energy_claim = True
    base.strict_pair_actor_required = True
    apply_heterogeneous_burst_main_profile(base, preserve_runtime_overrides=True)
    apply_energy_claim_profile(base)

    configs: List[AblationExperimentConfig] = []
    plan = build_figure_sweep_plan()
    for chain_range in plan["chain_length"]:
        cfg = deepcopy(base)
        cfg.chain_length_range = tuple(chain_range)
        cfg.figure_sweep_name = "chain_length"
        cfg.figure_sweep_value = str(tuple(chain_range))
        configs.append(cfg)
    for arrival in plan["arrival_rate"]:
        cfg = deepcopy(base)
        cfg.fixed_arrival_rate = float(arrival)
        cfg.arrival_range_req_s = (float(arrival), float(arrival))
        cfg.figure_sweep_name = "arrival_rate"
        cfg.figure_sweep_value = str(arrival)
        configs.append(cfg)
    for edge_nodes in plan["edge_nodes"]:
        cfg = deepcopy(base)
        cfg.traditional_nodes = int(edge_nodes)
        # 边缘节点数是该组唯一自变量；恢复绘图代码默认负载，避免20节点点位因重载profile失效。
        cfg.request_flow_count = 12
        cfg.chain_length_range = (3, 5)
        cfg.figure_sweep_name = "edge_nodes"
        cfg.figure_sweep_value = str(edge_nodes)
        configs.append(cfg)
    for v_value in plan["V"]:
        cfg = deepcopy(base)
        cfg.V = float(v_value)
        cfg.figure_sweep_name = "V"
        cfg.figure_sweep_value = str(v_value)
        configs.append(cfg)
    return configs
