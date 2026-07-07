"""
C4消融轻量测试
只验证算法边界和导出字段，不跑正式五种子实验。
"""
import copy
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from Constant import create_base_system
from FFD import run_FFD_slow_context
from Deployment import run_PDRS_slow_context, run_LoadAware_slow_context, run_GSLA
from ablation_resource_models import DVFS_RAILS, F_PRE_RAILS
from ablation_config import AblationExperimentConfig, ABLATION_MAIN_ALGORITHMS
from ablation_metrics import AlgorithmSummary, SlotResult, summarize_slot_results
from ai_inference import TrainedAIInference


def make_small_system(seed: int = 38):
    """构造小规模可复现场景"""
    return create_base_system(
        seed=seed,
        chain_length_range=(3, 4),
        fixed_arrival_rate=3.0,
        num_edge_nodes=10,
        ai_node_count=4,
        request_flow_count=4,
        arrival_range_req_s=(2.0, 4.0),
        input_tokens_range=(128, 256),
        output_tokens_range=(32, 64),
        gpu_units_range=(2, 4),
        max_batch_size=16,
    )


class C4AblationBoundaryTest(unittest.TestCase):
    """C4边界测试"""

    def test_ffd_uses_deterministic_routing(self):
        system_state = make_small_system()
        ok = run_FFD_slow_context(system_state, seed=38)
        self.assertTrue(ok)
        self.assertEqual(system_state.ffd_context.get("routing_policy"), "FFD-deterministic")
        self.assertNotIn("softmax_routing_matrix", system_state.ffd_context)

    def test_ffd_is_pure_ffd_not_delay_spare_optimizer(self):
        """FFD正式基线只能是resource-first FFD，不能混入delay/spare/HAPA式评分。"""
        system_state = make_small_system(seed=38)
        ok = run_FFD_slow_context(system_state, seed=38)
        self.assertTrue(ok)
        self.assertEqual(system_state.ffd_context.get("server_order_policy"), "resource_first_ffd")
        self.assertEqual(system_state.ffd_context.get("ai_placement_policy"), "resource_first_ai")
        self.assertFalse(system_state.ffd_context.get("uses_delay_sorted_servers", True))
        self.assertFalse(system_state.ffd_context.get("uses_spare_delay_ai_score", True))

    def test_pdrs_is_independent_engineering_baseline(self):
        system_state = make_small_system()
        ok = run_PDRS_slow_context(system_state)
        self.assertTrue(ok)
        self.assertEqual(system_state.pdrs_context.get("status"), "ok")
        self.assertIn("independent DRS", system_state.pdrs_context.get("boundary", ""))
        self.assertEqual(system_state.pdrs_context.get("routing_policy"), "PDRS-loadaware")

    def test_pdrs_records_traffic_cost_aware_placement_weight(self):
        """PDRS应显式记录traffic-weighted通信成本放置项，区别于普通LoadAware。"""
        system_state = make_small_system(seed=39)
        ok = run_PDRS_slow_context(system_state)
        self.assertTrue(ok)

        weights = system_state.pdrs_context.get("placement_score_weights", {})
        self.assertIn("traffic_cost", weights)
        self.assertGreater(float(weights["traffic_cost"]), 0.0)
        self.assertEqual(system_state.pdrs_context.get("placement_policy"), "loadaware-traffic-cost")

    def test_loadaware_score_can_weight_locality_by_flow_traffic(self):
        """高到达率/大token flow应更强惩罚远距离传统服务放置。"""
        from types import SimpleNamespace
        from Deployment import _loadaware_server_score

        class Topology:
            def get_communication_delay(self, origin, dest):
                return {"near": 2.0, "far": 10.0}.get(dest, 0.0)

        microservices = [SimpleNamespace(ms_id="svc_0"), SimpleNamespace(ms_id="svc_1")]
        service_chain = SimpleNamespace(get_traditional_microservices=lambda: microservices)
        state = SimpleNamespace(
            network_topology=Topology(),
            stream_allocated_resources={"flow": {"svc_0": {"pred": 1.0}}},
        )
        near = SimpleNamespace(
            server_id="near",
            available_cpu=8,
            available_memory=16.0,
            cpu_cores=8,
            memory_capacity=16.0,
        )
        far = SimpleNamespace(
            server_id="far",
            available_cpu=8,
            available_memory=16.0,
            cpu_cores=8,
            memory_capacity=16.0,
        )
        req = {"cores": 1, "memory": 1.2}
        low_traffic_flow = SimpleNamespace(
            flow_id="flow",
            arrival_rate=1.0,
            r_input_data_size=128.0,
            r_output_data_size=32.0,
            service_chain=service_chain,
        )
        high_traffic_flow = SimpleNamespace(
            flow_id="flow",
            arrival_rate=24.0,
            r_input_data_size=2048.0,
            r_output_data_size=512.0,
            service_chain=service_chain,
        )

        low_gap = (
            _loadaware_server_score(far, low_traffic_flow, microservices[1], req, state, traffic_cost_weight=1.0) -
            _loadaware_server_score(near, low_traffic_flow, microservices[1], req, state, traffic_cost_weight=1.0)
        )
        high_gap = (
            _loadaware_server_score(far, high_traffic_flow, microservices[1], req, state, traffic_cost_weight=1.0) -
            _loadaware_server_score(near, high_traffic_flow, microservices[1], req, state, traffic_cost_weight=1.0)
        )

        self.assertGreater(high_gap, low_gap * 2.0)

    def test_loadaware_context_available(self):
        system_state = make_small_system()
        ok = run_LoadAware_slow_context(system_state)
        self.assertTrue(ok)
        self.assertEqual(system_state.loadaware_context.get("routing_policy"), "LoadAware-deterministic")

    def test_routing_dest_placements_prefer_current_flow(self):
        """同名传统服务默认只路由到当前flow实例，避免跨flow通信成本污染。"""
        from Deployment import _get_successor_dest_placements

        system_state = make_small_system()
        flow_ids = sorted(system_state.request_flows.keys())
        traditional_servers = sorted(
            server.server_id for server in system_state.edge_servers.values()
            if server.server_type.value == "traditional"
        )
        self.assertGreaterEqual(len(flow_ids), 2)
        self.assertGreaterEqual(len(traditional_servers), 2)

        service = system_state.request_flows[flow_ids[0]].service_chain.get_traditional_microservices()[0]
        system_state.stream_allocated_resources = {
            flow_ids[0]: {service.ms_id: {traditional_servers[0]: 1.0}},
            flow_ids[1]: {service.ms_id: {traditional_servers[1]: 1.0}},
        }

        placements = _get_successor_dest_placements(system_state, flow_ids[0], service)

        self.assertEqual(placements, {traditional_servers[0]: 1.0})

    def test_no_implicit_nextfit_redeploy_between_slow_epochs(self):
        """正式消融的slot到达更新不能隐式触发Next Fit重部署。"""
        from ablation_state_hash import compute_placement_hash, compute_routing_hash

        system_state = make_small_system(seed=38)
        self.assertTrue(run_GSLA(system_state))
        placement_before = compute_placement_hash(system_state)
        routing_before = compute_routing_hash(system_state)
        new_arrivals = {
            flow_id: flow.arrival_rate * 1.25
            for flow_id, flow in system_state.request_flows.items()
        }

        system_state.environment_manager.update_all_ai_server_states(
            new_arrivals,
            allow_redeployment=False,
        )

        self.assertEqual(placement_before, compute_placement_hash(system_state))
        self.assertEqual(routing_before, compute_routing_hash(system_state))

    def test_algorithm_specific_slow_context_not_overwritten_by_environment_update(self):
        """FFD/GMDA等慢层上下文不能被slot环境更新改成Next Fit或GSLA上下文。"""
        from Deployment import run_GMDA_RMPR_slow_context
        from ablation_state_hash import compute_slow_context_hash

        system_state = make_small_system(seed=39)
        self.assertTrue(run_GMDA_RMPR_slow_context(system_state))
        slow_hash_before = compute_slow_context_hash(system_state, "GMDA-RMPR")
        self.assertEqual(system_state.gmda_rmpr_context.get("routing_policy"), "GMDA-RMPR-probabilistic")
        new_arrivals = {
            flow_id: flow.arrival_rate * 0.85
            for flow_id, flow in system_state.request_flows.items()
        }

        system_state.environment_manager.update_all_ai_server_states(
            new_arrivals,
            allow_redeployment=False,
        )

        self.assertEqual(slow_hash_before, compute_slow_context_hash(system_state, "GMDA-RMPR"))
        self.assertEqual(system_state.gmda_rmpr_context.get("routing_policy"), "GMDA-RMPR-probabilistic")
        self.assertFalse(hasattr(system_state, "nextfit_context"))

    def test_slow_context_hash_stable_between_epoch_slots(self):
        """slow_epoch内部只更新arrival和队列状态，slow context hash必须稳定。"""
        from ablation_state_hash import compute_slow_context_hash

        system_state = make_small_system(seed=40)
        self.assertTrue(run_GSLA(system_state))
        slow_hash_before = compute_slow_context_hash(system_state, "GSLA")
        for scale in (1.05, 0.95, 1.10):
            arrivals = {
                flow_id: flow.arrival_rate * scale
                for flow_id, flow in system_state.request_flows.items()
            }
            system_state.environment_manager.update_all_ai_server_states(
                arrivals,
                allow_redeployment=False,
            )

        self.assertEqual(slow_hash_before, compute_slow_context_hash(system_state, "GSLA"))

    def test_slot_result_exports_slow_context_hashes(self):
        """raw slot result必须导出placement/routing/slow context哈希用于复现实验审计。"""
        from ResourceAllocation import evaluate_ai_action_shared

        system_state = make_small_system(seed=41)
        self.assertTrue(run_GSLA(system_state))
        result = evaluate_ai_action_shared(
            offloading_mode={"action": [0, 0, 0, 0], "candidate_source": "unit_all_local"},
            system_state=system_state,
            V=20.0,
            slot=0,
            seed=41,
            algorithm="GSLA-Myopic",
            slow_policy="GSLA",
            fast_controller="Myopic",
            model_path="",
        )

        self.assertTrue(result.placement_hash)
        self.assertTrue(result.routing_hash)
        self.assertTrue(result.slow_context_hash)

    def test_resource_model_has_dvfs_and_fpre_rails(self):
        self.assertGreaterEqual(len(DVFS_RAILS), 3)
        self.assertIn(1.0, DVFS_RAILS)
        self.assertGreaterEqual(len(F_PRE_RAILS), 3)
        self.assertIn(1.0, F_PRE_RAILS)

    def test_local_latency_service_rate_uses_seconds_unit(self):
        """本地GPU服务率要用req/s口径，不能把毫秒处理时间直接与到达率比较。"""
        from types import SimpleNamespace
        from ResourceAllocation import calculate_latency_with_fixed_gpu_units

        request_flow = SimpleNamespace(
            arrival_rate=10.0,
            r_input_data_size=1000.0,
            r_output_data_size=200.0,
        )
        server = SimpleNamespace(
            prefill_speed_tokens_per_sec=10000.0,
            decode_speed_tokens_per_sec=5000.0,
            max_batch_size=16,
        )

        latency = calculate_latency_with_fixed_gpu_units(
            request_flow, None, server, gpu_units=2, system_state=None
        )

        self.assertIsNotNone(latency)
        total_latency_ms, queue_delay_ms, processing_delay_ms = latency
        self.assertGreater(total_latency_ms, processing_delay_ms)
        self.assertGreater(queue_delay_ms, 0.0)
        self.assertLess(total_latency_ms, 500.0)

    def test_local_ai_energy_scales_with_workload(self):
        """本地AI能耗要随arrival/token负载增长，不能只按静态GPU配置计一次。"""
        from types import SimpleNamespace
        from ablation_resource_models import _scale_local_energy_by_workload

        low_flow = SimpleNamespace(
            arrival_rate=4.0,
            r_input_data_size=512.0,
            r_output_data_size=128.0,
        )
        high_flow = SimpleNamespace(
            arrival_rate=18.0,
            r_input_data_size=2048.0,
            r_output_data_size=512.0,
        )

        low_energy = _scale_local_energy_by_workload(
            base_energy=0.2, request_flow=low_flow, batch_size=2, gpu_frequency_scale=0.7
        )
        high_energy = _scale_local_energy_by_workload(
            base_energy=0.2, request_flow=high_flow, batch_size=2, gpu_frequency_scale=0.7
        )

        self.assertGreater(high_energy, low_energy * 2.0)

    def test_paper_dpp_queue_energy_uses_node_attributable_energy(self):
        """虚拟能耗队列按节点可归属能耗更新，system energy只作为导出指标。"""
        import numpy as np
        from ResourceAllocation import calculate_paper_dpp_components

        components = calculate_paper_dpp_components(
            ai_delays=np.array([10.0]),
            ai_energies=np.array([1.0]),
            cost_value=0.0,
            SQ=np.array([10.0]),
            SZ=np.array([0.0]),
            V=20.0,
            energy_ref_j=1.0,
            delay_ref_ms=50.0,
            system_energy_j=10.0,
        )

        self.assertAlmostEqual(components["energy_queue_term"], 10.0)
        self.assertAlmostEqual(components["scaled_energy_sum"], 10.0)

    def test_cloud_pair_energy_queue_excludes_remote_compute(self):
        """云端远端推理能耗保留在system energy中，但不直接压到边缘节点能耗队列。"""
        from types import SimpleNamespace
        from ResourceAllocation import _edge_queue_energy_for_pair_decision

        instance = SimpleNamespace(
            energy_comm_j=0.12,
            energy_preprocess_j=0.03,
            energy_cloud_compute_j=1.80,
        )

        queue_energy = _edge_queue_energy_for_pair_decision(
            instance, total_energy=1.95, decision_bit=1
        )

        self.assertAlmostEqual(queue_energy, 0.15)

    def test_cloud_f_pre_energy_is_not_counted_twice(self):
        """云端f_pre预处理能耗单独计入，基础云端推理能耗不能再带同一项。"""
        from types import SimpleNamespace
        from ablation_resource_models import solve_cloud_preprocess_config

        flow = SimpleNamespace(
            arrival_rate=8.0,
            r_input_data_size=512.0,
            r_output_data_size=128.0,
        )
        server = SimpleNamespace(energy_threshold=2.0, delay_threshold=80.0)

        with patch("Deployment.evaluate_cloud_deployment") as mock_cloud_eval, \
                patch("EnergyConsumption.calculate_cloud_processing_energy") as mock_cloud_energy, \
                patch("EnergyConsumption.calculate_optimized_communication_energy") as mock_comm_energy:
            mock_cloud_eval.return_value = {
                "total_latency": 120.0,
                "cloud_inference_latency": 50.0,
            }
            mock_cloud_energy.return_value = 1.0
            mock_comm_energy.return_value = 0.2

            config = solve_cloud_preprocess_config(
                flow, None, server, None,
                SQ_value=1.0,
                SZ_value=1.0,
            )

        self.assertIsNotNone(config)
        for call in mock_cloud_energy.call_args_list:
            self.assertEqual(call.kwargs.get("f_pre"), 0.0)

    def test_cloud_remote_inference_energy_has_system_scope_floor(self):
        """system-wide能耗口径下，云端远端推理不能近似成接近零的协调能耗。"""
        from types import SimpleNamespace
        from EnergyConsumption import calculate_cloud_processing_energy

        server = SimpleNamespace(base_power=120.0, cpu_power_coeff=100.0)
        flow = SimpleNamespace(
            arrival_rate=12.0,
            r_input_data_size=2048.0,
            r_output_data_size=512.0,
        )

        energy = calculate_cloud_processing_energy(server, request_flow=flow, f_pre=0.0)

        self.assertGreaterEqual(energy, 1.50)

    def test_cloud_f_pre_search_uses_resource_cache(self):
        """训练候选重复评估时，相同云端f_pre搜索应复用缓存以保持可执行速度。"""
        from types import SimpleNamespace
        from ablation_resource_models import (
            clear_resource_model_cache,
            get_resource_model_cache_stats,
            solve_cloud_preprocess_config,
        )

        flow = SimpleNamespace(
            request_flow_id="flow_cache",
            arrival_rate=8.0,
            r_input_data_size=512.0,
            r_output_data_size=128.0,
        )
        service = SimpleNamespace(service_id="ai_cache")
        server = SimpleNamespace(
            server_id="ai_v_cache",
            energy_threshold=2.0,
            delay_threshold=80.0,
        )
        system_state = SimpleNamespace(time_frame=0)
        clear_resource_model_cache()

        with patch("Deployment.evaluate_cloud_deployment") as mock_cloud_eval, \
                patch("EnergyConsumption.calculate_cloud_processing_energy") as mock_cloud_energy, \
                patch("EnergyConsumption.calculate_optimized_communication_energy") as mock_comm_energy:
            mock_cloud_eval.return_value = {
                "total_latency": 120.0,
                "cloud_inference_latency": 50.0,
            }
            mock_cloud_energy.return_value = 1.0
            mock_comm_energy.return_value = 0.2
            first = solve_cloud_preprocess_config(flow, service, server, system_state, 1.0, 1.0)
            second = solve_cloud_preprocess_config(flow, service, server, system_state, 1.0, 1.0)

        stats = get_resource_model_cache_stats()
        self.assertEqual(first, second)
        self.assertEqual(mock_cloud_eval.call_count, 1)
        self.assertGreaterEqual(stats.get("cloud_hits", 0), 1)

    def test_resource_objective_respects_energy_weight(self):
        """energy-hard配置必须下沉到资源枚举目标，而不是只影响外层候选排序"""
        from ablation_resource_models import _queue_aware_objective

        fast_high_energy = _queue_aware_objective(
            latency_ms=40.0, energy_j=1.5,
            SQ_value=1.0, SZ_value=2.0,
            V=20.0, energy_ref=2.0, delay_ref=50.0,
            omega_energy=1.0, omega_delay=1.0,
        )
        slow_low_energy = _queue_aware_objective(
            latency_ms=90.0, energy_j=0.5,
            SQ_value=1.0, SZ_value=2.0,
            V=20.0, energy_ref=2.0, delay_ref=50.0,
            omega_energy=1.0, omega_delay=1.0,
        )
        self.assertLess(fast_high_energy, slow_low_energy)

        fast_high_energy_weighted = _queue_aware_objective(
            latency_ms=40.0, energy_j=1.5,
            SQ_value=1.0, SZ_value=2.0,
            V=20.0, energy_ref=2.0, delay_ref=50.0,
            omega_energy=8.0, omega_delay=1.0,
        )
        slow_low_energy_weighted = _queue_aware_objective(
            latency_ms=90.0, energy_j=0.5,
            SQ_value=1.0, SZ_value=2.0,
            V=20.0, energy_ref=2.0, delay_ref=50.0,
            omega_energy=8.0, omega_delay=1.0,
        )
        self.assertLess(slow_low_energy_weighted, fast_high_energy_weighted)

    def test_energy_claim_profile_scales_dpp_energy_reference(self):
        """energy-hard运行要同步缩小DPP能耗归一化参考值"""
        from run_ablation import apply_energy_claim_profile

        config = AblationExperimentConfig(
            include_energy_claim=True,
            omega_energy=1.0,
            energy_claim_omega_energy=10.0,
            energy_ref_j=2.0,
            energy_claim_threshold_scale=0.5,
        )
        apply_energy_claim_profile(config)
        self.assertAlmostEqual(config.omega_energy, 10.0)
        self.assertAlmostEqual(config.energy_ref_j, 1.0)
        self.assertTrue(getattr(config, "energy_claim_reference_scaled", False))

    def test_heterogeneous_burst_main_profile_is_frozen_before_formal(self):
        """normal-main正式场景必须在formal前冻结进config hash，不能跑完后改profile。"""
        from run_ablation import apply_heterogeneous_burst_main_profile, config_fingerprint

        config = AblationExperimentConfig()
        config.experiment_type = "normal_main"
        apply_heterogeneous_burst_main_profile(config)
        first_hash = config_fingerprint(config)
        apply_heterogeneous_burst_main_profile(config)

        self.assertEqual(config.scenario_profile, "heterogeneous_burst_main")
        self.assertTrue(config.scenario_profile_frozen)
        self.assertGreaterEqual(config.request_flow_count, 18)
        self.assertGreaterEqual(config.chain_length_range[0], 5)
        self.assertGreaterEqual(config.arrival_range_req_s[0], 8.0)
        self.assertGreaterEqual(config.input_tokens_range[0], 512)
        self.assertGreaterEqual(config.output_tokens_range[0], 128)
        self.assertIn(config.slow_epoch_slots, (10, 20))
        self.assertEqual(first_hash, config_fingerprint(config))

    def test_normal_main_profile_calibrates_edge_ai_inference_scale(self):
        """normal-main必须使用edge-AI吞吐量级，避免HAPA单副本虚假覆盖全部demand。"""
        from run_ablation import apply_heterogeneous_burst_main_profile, create_ablation_system, config_fingerprint

        config = AblationExperimentConfig()
        config.experiment_type = "normal_main"
        apply_heterogeneous_burst_main_profile(config)
        fingerprint = config_fingerprint(config)
        system_state = create_ablation_system(seed=38, config=config)
        ai_servers = [
            server for server in system_state.edge_servers.values()
            if server.server_type.value == "ai_capable"
        ]

        self.assertTrue(ai_servers)
        self.assertLessEqual(max(server.max_batch_size for server in ai_servers), 16)
        self.assertLess(max(server.prefill_speed_tokens_per_sec for server in ai_servers), 500000.0)
        self.assertLess(max(server.decode_speed_tokens_per_sec for server in ai_servers), 30000.0)
        self.assertIn("ai_prefill_speed_range", config.to_dict())
        self.assertIn("ai_decode_speed_range", config.to_dict())
        self.assertEqual(fingerprint, config_fingerprint(config))
    def test_diagnostic_energy_preserves_twenty_slot_runtime(self):
        """diagnostic-energy必须保持20 slot，不能被normal-main profile拉回100 slot。"""
        from argparse import Namespace
        from run_ablation import (
            apply_heterogeneous_burst_main_profile,
            should_preserve_profile_runtime_overrides,
        )

        config = AblationExperimentConfig()
        config.experiment_type = "normal_main"
        config.time_slots = 20
        config.slow_epoch_slots = 5
        args = Namespace(time_slots=None, slow_epoch_slots=None, diagnostic_energy=True)

        apply_heterogeneous_burst_main_profile(
            config,
            preserve_runtime_overrides=should_preserve_profile_runtime_overrides(args),
        )

        self.assertEqual(config.time_slots, 20)
        self.assertEqual(config.slow_epoch_slots, 5)

    def test_all_main_slow_layers_valid_on_validation_profile(self):
        """heterogeneous profile必须先保证主实验慢层可行，否则不能进入性能比较。"""
        from run_ablation import apply_heterogeneous_burst_main_profile, create_ablation_system
        from Deployment import run_GMDA_RMPR_slow_context, run_PDRS_slow_context, run_GSLA

        config = AblationExperimentConfig()
        config.experiment_type = "normal_main"
        apply_heterogeneous_burst_main_profile(config)

        for runner in [run_GSLA, run_GMDA_RMPR_slow_context, run_PDRS_slow_context, run_FFD_slow_context]:
            system_state = create_ablation_system(seed=38, config=config)
            self.assertTrue(runner(system_state), runner.__name__)

    def test_formal_seed42_gsla_valid_after_slot0_workload_update(self):
        """formal seed 42在slot0更新后，GSLA/HAPA首轮慢层必须可行。"""
        from run_ablation import apply_heterogeneous_burst_main_profile, create_ablation_system, build_workload_trace

        config = AblationExperimentConfig()
        config.experiment_type = "normal_main"
        config.include_energy_claim = True
        config.time_slots = 20
        config.slow_epoch_slots = 5
        apply_heterogeneous_burst_main_profile(config, preserve_runtime_overrides=True)

        system_state = create_ablation_system(seed=42, config=config)
        workload_trace = build_workload_trace(system_state, config.time_slots)
        system_state.time_frame = 0
        system_state.environment_manager.update_all_ai_server_states(workload_trace[0], allow_redeployment=False)

        self.assertTrue(run_GSLA(system_state), system_state.gsla_context.get("hapa_failure_reason", ""))

    def test_formal_seed42_ffd_valid_after_slot0_workload_update(self):
        """formal seed 42在slot0更新后，FFD AI放置不能因重复扣模型缓存而失败。"""
        from run_ablation import apply_heterogeneous_burst_main_profile, create_ablation_system, build_workload_trace

        config = AblationExperimentConfig()
        config.experiment_type = "normal_main"
        config.include_energy_claim = True
        config.time_slots = 20
        config.slow_epoch_slots = 5
        apply_heterogeneous_burst_main_profile(config, preserve_runtime_overrides=True)

        system_state = create_ablation_system(seed=42, config=config)
        workload_trace = build_workload_trace(system_state, config.time_slots)
        system_state.time_frame = 0
        system_state.environment_manager.update_all_ai_server_states(workload_trace[0], allow_redeployment=False)

        self.assertTrue(run_FFD_slow_context(system_state), getattr(system_state, "ffd_context", {}).get("failure_reason", ""))
    def test_claim_aware_selector_uses_pareto_band_not_algorithm_name(self):
        """DPP容忍带内按三指标claim score选非支配候选，不能按source/算法名偏置。"""
        import numpy as np
        from ablation_algorithms import select_best_action_from_candidates

        system_state = make_small_system(seed=42)
        config = AblationExperimentConfig(include_energy_claim=True)
        config.claim_delay_ref_ms = 100.0
        config.claim_energy_ref_j = 2.0
        config.claim_cost_ref = 200.0
        config.energy_hard_dpp_tolerance_ratio = 0.05
        candidates = [
            {"action": np.array([0, 1, 1, 1]), "candidate_source": "lyham_named_best_dpp"},
            {"action": np.array([0, 0, 1, 1]), "candidate_source": "neutral_claim_better"},
            {"action": np.array([1, 1, 1, 1]), "candidate_source": "neutral_dominated"},
        ]
        rows = {
            "lyham_named_best_dpp": {
                "paper_dpp_score": 1000.0, "delay_ms": 100.0,
                "energy_j": 2.0, "cost": 180.0, "feasible": True,
                "local_count": 1, "cloud_count": 3,
            },
            "neutral_claim_better": {
                "paper_dpp_score": 1020.0, "delay_ms": 70.0,
                "energy_j": 1.2, "cost": 120.0, "feasible": True,
                "local_count": 2, "cloud_count": 2,
            },
            "neutral_dominated": {
                "paper_dpp_score": 1010.0, "delay_ms": 130.0,
                "energy_j": 2.4, "cost": 220.0, "feasible": True,
                "local_count": 0, "cloud_count": 4,
            },
        }

        def fake_eval(candidate, *_args, **_kwargs):
            source = candidate["candidate_source"]
            row = dict(rows[source])
            row.update({
                "action": candidate["action"],
                "candidate_source": source,
                "selected_candidate_source": source,
                "repaired_pair_action_hash": source,
                "pair_action_hash": source,
            })
            return row

        with patch("ResourceAllocation.evaluate_action_dry_run", side_effect=fake_eval):
            decision = select_best_action_from_candidates(candidates, system_state, config, queue_aware=True)

        self.assertEqual(decision["selected_candidate_source"], "neutral_claim_better")
        self.assertEqual(decision["selected_by_dpp_or_claim_band"], "claim_band")
        self.assertTrue(decision["dpp_band_passed"])
        self.assertTrue(decision["is_pareto_candidate"])
        self.assertLess(decision["claim_score"], rows["lyham_named_best_dpp"]["delay_ms"])

    def test_claim_selector_prefers_near_tied_noncollapsed_action(self):
        """claim-band内非坍缩hybrid动作近似打平时，应优先保留UAC动作多样性。"""
        import numpy as np
        from ablation_algorithms import select_best_action_from_candidates

        system_state = make_small_system()
        config = AblationExperimentConfig(include_energy_claim=True)
        config.claim_delay_ref_ms = 100.0
        config.claim_energy_ref_j = 12.0
        config.claim_cost_ref = 400.0
        config.energy_hard_dpp_tolerance_ratio = 0.05
        all_local = {"action": np.array([0, 0]), "candidate_source": "actor_all_local"}
        hybrid = {"action": np.array([0, 1]), "candidate_source": "uac_energy_cloud_relief"}

        def fake_eval(candidate, *_args, **_kwargs):
            source = candidate["candidate_source"]
            if source == "actor_all_local":
                return {
                    "action": candidate["action"],
                    "pair_action": np.array([0, 0]),
                    "paper_dpp_score": 100.0,
                    "feasible": True,
                    "failure_reason": "",
                    "delay_ms": 160.0,
                    "energy_j": 12.0,
                    "cost": 820.0,
                    "local_count": 2,
                    "cloud_count": 0,
                    "repaired_pair_action_hash": "all-local",
                    "candidate_source": source,
                }
            return {
                "action": candidate["action"],
                "pair_action": np.array([0, 1]),
                "paper_dpp_score": 102.0,
                "feasible": True,
                "failure_reason": "",
                "delay_ms": 168.0,
                "energy_j": 11.3,
                "cost": 828.0,
                "local_count": 1,
                "cloud_count": 1,
                "repaired_pair_action_hash": "hybrid",
                "candidate_source": source,
            }

        with patch("ResourceAllocation.evaluate_action_dry_run", side_effect=fake_eval):
            decision = select_best_action_from_candidates(
                [all_local, hybrid], system_state, config, queue_aware=True
            )

        self.assertEqual(decision["selected_candidate_source"], "uac_energy_cloud_relief")
        self.assertEqual(decision["repaired_pair_action_hash"], "hybrid")

    def test_slot_result_exports_claim_selector_fields(self):
        """raw必须导出claim selector诊断字段，便于审计DPP与三指标选择是否错位。"""
        from ablation_export import RAW_FIELDS

        for field in [
            "claim_score", "is_pareto_candidate", "dpp_band_passed",
            "selected_by_dpp_or_claim_band",
            "per_pair_delta_delay", "per_pair_delta_energy", "per_pair_delta_cost",
            "repaired_hamming_vs_reference", "reference_pair_action_bits",
            "reference_pair_action_hash", "reference_pair_action_source",
        ]:
            self.assertIn(field, RAW_FIELDS)

        result = SlotResult(
            slot=0, seed=1, algorithm="LyHAM-CO", slow_policy="GSLA",
            fast_controller="UAC-DO", status="ok", failure_reason="",
            delay_ms=1.0, energy_j=1.0, cost=1.0, avg_y=0.0, avg_z=0.0,
            dpp_score=1.0, legacy_reward=0.0, feasible=True,
            local_count=1, cloud_count=1, forced_cloud_count=0,
            decision_time_ms=0.0, slow_context_reused=False, model_path="",
            claim_score=0.5, is_pareto_candidate=True, dpp_band_passed=True,
            selected_by_dpp_or_claim_band="claim_band",
            per_pair_delta_delay="0.1", per_pair_delta_energy="-0.2",
            per_pair_delta_cost="-1.0",
        )
        row = result.to_dict()
        self.assertEqual(row["claim_score"], 0.5)
        self.assertEqual(row["selected_by_dpp_or_claim_band"], "claim_band")

    def test_idle_replica_energy_is_active_pair_aware(self):
        """active AI chain energy只应高额计入本时隙被pair动作消费的副本。"""
        from Deployment import run_GSLA
        from ResourceAllocation import calculate_ai_replica_idle_energy

        system_state = make_small_system(seed=38)
        self.assertTrue(run_GSLA(system_state))
        ai_instances = [
            instance for instance in system_state.microservice_instances.values()
            if instance.microservice.service_type == "ai"
        ]
        self.assertGreaterEqual(len(ai_instances), 1)
        for instance in ai_instances:
            instance.gpu_units_reserved = 4.0
            instance.gpu_memory_reserved = 8.0
            instance.model_storage_reserved = 8.0
            instance.processing_mode = "local_processing"
            instance.active_pair_count = 0
            instance.active_local_pair_count = 0
            instance.active_cloud_pair_count = 0

        standby_idle = calculate_ai_replica_idle_energy(system_state)
        for instance in ai_instances:
            instance.active_pair_count = 1
            instance.active_local_pair_count = 1
        active_idle = calculate_ai_replica_idle_energy(system_state)

        self.assertGreater(active_idle, standby_idle * 2.0)
    def test_energy_budget_frontier_has_medium_local_budgets(self):
        """energy-budget候选要覆盖中等本地pair数量，避免只生成过度云端动作"""
        from ablation_algorithms import _energy_budget_frontier_sizes

        sizes = _energy_budget_frontier_sizes(12)
        self.assertIn(6, sizes)
        self.assertIn(8, sizes)
        self.assertEqual(sizes, sorted(set(sizes)))
        self.assertLessEqual(max(sizes), 12)


    def test_energy_hard_selector_uses_energy_inside_dpp_guard(self):
        """DPP仍可接受时，energy-hard选择低system energy的repaired动作"""
        import numpy as np
        from ablation_algorithms import select_best_action_from_candidates

        system_state = make_small_system()
        config = AblationExperimentConfig(include_energy_claim=True)
        high_energy = {"action": np.array([0, 1, 1, 1]), "candidate_source": "unit_high_energy"}
        low_energy = {"action": np.array([1, 0, 1, 1]), "candidate_source": "unit_low_energy"}

        def fake_eval(candidate, state, cfg, queue_aware=True):
            source = candidate["candidate_source"]
            if source == "unit_high_energy":
                return {
                    "action": candidate["action"],
                    "pair_action": candidate["action"],
                    "feasible": True,
                    "paper_dpp_score": 1000.0,
                    "energy_j": 2.0,
                    "cost": 50.0,
                    "delay_ms": 40.0,
                    "local_count": 1,
                    "cloud_count": 3,
                    "repaired_pair_action_hash": "high",
                    "candidate_source": source,
                }
            return {
                "action": candidate["action"],
                "pair_action": candidate["action"],
                "feasible": True,
                "paper_dpp_score": 1020.0,
                "energy_j": 1.0,
                "cost": 51.0,
                "delay_ms": 42.0,
                "local_count": 1,
                "cloud_count": 3,
                "repaired_pair_action_hash": "low",
                "candidate_source": source,
            }

        with patch("ResourceAllocation.evaluate_action_dry_run", side_effect=fake_eval):
            decision = select_best_action_from_candidates(
                [high_energy, low_energy], system_state, config, queue_aware=True
            )

        self.assertEqual(decision["selected_candidate_source"], "unit_low_energy")
        self.assertEqual(decision["repaired_pair_action_hash"], "low")


    def test_selector_exports_best_energy_candidate_diagnostics(self):
        """energy gate失败时，要能看出候选集中最低能耗动作与最终动作的差距。"""
        import numpy as np
        from ablation_algorithms import select_best_action_from_candidates

        system_state = make_small_system()
        config = AblationExperimentConfig(include_energy_claim=True)
        best_dpp = {"action": np.array([0, 1, 1, 1]), "candidate_source": "unit_best_dpp"}
        best_energy = {"action": np.array([1, 0, 1, 1]), "candidate_source": "unit_best_energy"}

        def fake_eval(candidate, *_args, **_kwargs):
            source = candidate["candidate_source"]
            if source == "unit_best_dpp":
                return {
                    "action": candidate["action"], "pair_action": candidate["action"],
                    "feasible": True, "paper_dpp_score": 100.0, "energy_j": 10.0,
                    "local_count": 1, "cloud_count": 3,
                    "repaired_pair_action_hash": "best-dpp", "candidate_source": source,
                }
            return {
                "action": candidate["action"], "pair_action": candidate["action"],
                "feasible": True, "paper_dpp_score": 140.0, "energy_j": 1.0,
                "local_count": 1, "cloud_count": 3,
                "repaired_pair_action_hash": "best-energy", "candidate_source": source,
            }

        with patch("ResourceAllocation.evaluate_action_dry_run", side_effect=fake_eval):
            decision = select_best_action_from_candidates(
                [best_dpp, best_energy], system_state, config, queue_aware=True
            )

        self.assertEqual(decision["selected_candidate_source"], "unit_best_dpp")
        self.assertEqual(decision["best_energy_candidate_source"], "unit_best_energy")
        self.assertAlmostEqual(decision["best_energy_candidate_dpp_gap"], 40.0)
        self.assertAlmostEqual(decision["best_energy_candidate_energy_gap"], 9.0)
    def test_energy_claim_profile_tightens_virtual_energy_threshold(self):
        """energy-hard运行要让能耗队列约束进入系统状态和config hash"""
        from run_ablation import create_ablation_system

        base_config = AblationExperimentConfig(
            traditional_nodes=10,
            ai_nodes=4,
            request_flow_count=4,
            chain_length_range=(3, 4),
            arrival_range_req_s=(2.0, 4.0),
            input_tokens_range=(128, 256),
            output_tokens_range=(32, 64),
            include_energy_claim=False,
        )
        hard_config = AblationExperimentConfig(
            traditional_nodes=10,
            ai_nodes=4,
            request_flow_count=4,
            chain_length_range=(3, 4),
            arrival_range_req_s=(2.0, 4.0),
            input_tokens_range=(128, 256),
            output_tokens_range=(32, 64),
            include_energy_claim=True,
            energy_claim_threshold_scale=0.5,
        )
        base_system = create_ablation_system(38, base_config)
        hard_system = create_ablation_system(38, hard_config)
        for server_id, hard_queue in hard_system.virtual_energy_queues.items():
            base_threshold = base_system.virtual_energy_queues[server_id].energy_threshold
            self.assertAlmostEqual(hard_queue.energy_threshold, base_threshold * 0.5)
            self.assertAlmostEqual(
                hard_system.edge_servers[server_id].energy_threshold,
                hard_queue.energy_threshold,
            )

    def test_slow_context_does_not_pollute_copy(self):
        base = make_small_system()
        left = copy.deepcopy(base)
        right = copy.deepcopy(base)
        self.assertTrue(run_PDRS_slow_context(left))
        self.assertTrue(run_LoadAware_slow_context(right))
        self.assertNotEqual(
            getattr(left, "pdrs_context", {}).get("policy"),
            getattr(right, "loadaware_context", {}).get("policy"),
        )

    def test_gsla_hapa_exports_demand_covering_quality(self):
        system_state = make_small_system(seed=39)
        first_flow = next(iter(system_state.request_flows.values()))
        first_flow.arrival_rate = 8.0
        first_flow.r_input_data_size = 512
        first_flow.r_output_data_size = 128

        self.assertTrue(run_GSLA(system_state))
        covering = system_state.gsla_context.get("hapa_demand_covering", {})
        self.assertTrue(covering)
        for flow_id, item in covering.items():
            self.assertIn("coverage_ratio", item, flow_id)
            self.assertIn("replica_count", item, flow_id)
            self.assertIn("uncovered_demand_req_s", item, flow_id)
            self.assertGreaterEqual(item["coverage_ratio"], 0.95, flow_id)

    def test_gsla_default_seed_39_has_feasible_hapa_covering(self):
        """默认诊断场景下HAPA不能因单server单副本限制导致首slot invalid"""
        from run_ablation import create_ablation_system

        config = AblationExperimentConfig()
        system_state = create_ablation_system(39, config)
        self.assertTrue(run_GSLA(system_state), system_state.gsla_context.get("hapa_failure_reason", ""))
        covering = system_state.gsla_context.get("hapa_demand_covering", {})
        self.assertTrue(covering)
        self.assertGreaterEqual(
            min(item["coverage_ratio"] for item in covering.values()),
            0.95,
        )

    def test_normal_main_hapa_adds_diversity_replicas_for_uac_endpoint_choice(self):
        """normal-main中HAPA应为高负载AI流提供少量可行副本，给UAC endpoint选择空间。"""
        from run_ablation import apply_heterogeneous_burst_main_profile, create_ablation_system

        config = AblationExperimentConfig()
        config.experiment_type = "normal_main"
        apply_heterogeneous_burst_main_profile(config)
        system_state = create_ablation_system(38, config)

        self.assertTrue(run_GSLA(system_state), system_state.gsla_context.get("hapa_failure_reason", ""))
        ai_instances = [
            instance for instance in system_state.microservice_instances.values()
            if instance.microservice.service_type == "ai"
        ]
        ai_flow_count = sum(1 for flow in system_state.request_flows.values() if flow.service_chain.ai_microservice)
        self.assertGreater(len(ai_instances), ai_flow_count)
        self.assertGreater(system_state.gsla_context.get("hapa_latency_diversity_replicas", 0), 0)
    def test_slow_failure_reason_propagates_hapa_reason(self):
        """慢层失败原因要写入SlotResult/CSV，不能只留下空字符串"""
        from ablation_algorithms import run_slow_context_for_algorithm

        system_state = make_small_system(seed=39)

        def fake_gsla(state):
            state.gsla_context = {"hapa_failure_reason": "flow_x AI需求无法覆盖"}
            return False

        with patch("Deployment.run_GSLA", side_effect=fake_gsla):
            ok, reason = run_slow_context_for_algorithm(
                "LyHAM-CO", system_state, AblationExperimentConfig(), slot=0, seed=39
            )
        self.assertFalse(ok)
        self.assertIn("flow_x", reason)

    def test_hapa_records_server_spread_counts(self):
        """HAPA需要记录实例分散度，避免慢层覆盖后快层全云端坍缩"""
        system_state = make_small_system(seed=41)
        self.assertTrue(run_GSLA(system_state))
        counts = system_state.gsla_context.get("hapa_server_instance_counts", {})
        self.assertTrue(counts)
        used_servers = [server_id for server_id, count in counts.items() if count > 0]
        self.assertGreaterEqual(len(used_servers), 2)

    def test_gsla_routing_records_cost_delay_aware_components(self):
        system_state = make_small_system(seed=39)
        self.assertTrue(run_GSLA(system_state))
        context = system_state.gsla_context
        self.assertEqual(context.get("routing_policy"), "GSLA-cost-delay-softmax")
        components = context.get("routing_score_components", {})
        self.assertTrue(components)
        first_row = next(iter(components.values()))
        first_dest = next(iter(first_row.values()))
        for key in ["delay_score", "spare_score", "cost_score", "queue_score", "final_score"]:
            self.assertIn(key, first_dest)

    def test_gsla_softmax_prefers_high_score_routes(self):
        """GSLA Softmax routing不能接近均匀分流，否则delay优势会被概率平均抵消"""
        from Deployment import build_softmax_routing_context

        system_state = make_small_system(seed=39)
        self.assertTrue(run_GSLA(system_state))
        flow_id, request_flow = next(iter(system_state.request_flows.items()))
        chain = request_flow.service_chain.microservices
        origin_ms = chain[0]
        dest_ms = chain[1]
        traditional_servers = sorted(
            server.server_id for server in system_state.edge_servers.values()
            if server.server_type.value == "traditional"
        )
        system_state.stream_allocated_resources.setdefault(flow_id, {})
        system_state.stream_allocated_resources[flow_id][origin_ms.ms_id] = {traditional_servers[0]: 1}
        system_state.stream_allocated_resources[flow_id][dest_ms.ms_id] = {
            traditional_servers[1]: 1,
            traditional_servers[-1]: 1,
        }
        build_softmax_routing_context(system_state, system_state.gsla_context)
        rows = [
            row for row in system_state.gsla_context.get("softmax_routing_matrix", {}).values()
            if len(row) > 1
        ]
        self.assertTrue(rows)
        average_best_prob = sum(max(float(v) for v in row.values()) for row in rows) / len(rows)
        self.assertGreaterEqual(average_best_prob, 0.75)

    def test_uac_state_vector_pads_to_checkpoint_dimension(self):
        short_state = [0.1] * 12
        padded = TrainedAIInference.fit_state_vector_to_checkpoint(short_state, expected_dim=30)
        self.assertEqual(len(padded), 30)
        self.assertAlmostEqual(float(padded[0]), 0.1)
        self.assertAlmostEqual(float(padded[-1]), 0.0)

    def test_uac_candidate_pool_uses_myopic_seed_only_for_repair(self):
        """UAC可使用Myopic种子构造邻域，但不能把Myopic原动作作为最终候选"""
        import numpy as np
        from ablation_algorithms import get_ai_action_dimension, run_UAC_DO

        system_state = make_small_system(seed=38)
        self.assertTrue(run_GSLA(system_state))
        config = AblationExperimentConfig(
            time_slots=3,
            traditional_nodes=10,
            ai_nodes=4,
            request_flow_count=4,
            chain_length_range=(3, 4),
            arrival_range_req_s=(2.0, 4.0),
            input_tokens_range=(128, 256),
            output_tokens_range=(32, 64),
        )
        n = get_ai_action_dimension(system_state)

        class FakeInference:
            def make_candidates(self, state, cfg):
                return [np.ones(n, dtype=int)]

        captured_sources = []

        def fake_select(candidates, state, cfg, queue_aware=True):
            captured_sources.extend([
                str(item.get("candidate_source", "")) for item in candidates
                if isinstance(item, dict)
            ])
            return {
                "action": np.ones(n, dtype=int),
                "paper_dpp_score": 1.0,
                "candidate_count": len(candidates),
                "selected_candidate_rank": 0,
            }

        with patch("ablation_algorithms.get_cached_uac_inference", return_value=FakeInference()), \
             patch("ablation_algorithms.select_best_action_from_candidates", side_effect=fake_select):
            run_UAC_DO(system_state, config)

        self.assertNotIn("uac_myopic_seed", captured_sources)
        self.assertTrue(any(source.startswith("uac_") for source in captured_sources))

    def test_pair_actor_checkpoint_generates_pair_candidates(self):
        """pair-level checkpoint应严格加载并输出pair动作候选。"""
        import tempfile
        import torch
        from ai_inference import create_trained_ai_inference

        system_state = make_small_system(seed=38)
        self.assertTrue(run_GSLA(system_state))
        model = torch.nn.Sequential(
            torch.nn.Linear(15, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 1),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pair_actor_test.pth"
            torch.save({
                "state_dict": model.state_dict(),
                "feature_dim": 15,
                "hidden_dim": 8,
                "training_dataset_hash": "unit-test",
            }, path)
            inference = create_trained_ai_inference(str(path), strict=True)
            candidates = inference.make_candidates(system_state, AblationExperimentConfig(ai_nodes=4))

        self.assertTrue(candidates)
        self.assertTrue(any(item.get("action_scope") == "pair" for item in candidates))
        self.assertTrue(any("pair_actor" in item.get("candidate_source", "") for item in candidates))

    def test_local_resource_check_uses_dvfs_enumeration(self):
        """本地可行性检查应优先用C4的(g,b,f_GPU)枚举，而不是旧GPU需求硬判定"""
        from ResourceAllocation import check_local_resource_sufficiency, find_ai_instances_and_flows

        system_state = make_small_system(seed=42)
        self.assertTrue(run_GSLA(system_state))
        server_id = next(
            sid for sid in sorted(system_state.edge_servers)
            if find_ai_instances_and_flows(sid, system_state)
        )
        server = system_state.edge_servers[server_id]
        server.available_gpu_units = max(server.available_gpu_units, 8)
        server.available_gpu_memory = max(server.available_gpu_memory, 128.0)
        server.available_model_storage = max(server.available_model_storage, 512.0)
        feasible_config = {
            "gpu_units": 1,
            "gpu_memory": 1.0,
            "model_storage": 1.0,
            "latency_ms": 10.0,
        }

        with patch("Deployment.calculate_local_ai_inference_latency",
                   return_value=(10.0, 1.0, 9.0, 64)), \
             patch("ablation_resource_models.select_local_ai_config",
                   return_value=feasible_config) as selector:
            self.assertTrue(check_local_resource_sufficiency(server_id, system_state))
            self.assertTrue(selector.called)

    def test_gsla_local_candidate_can_beat_all_cloud(self):
        """GSLA/HAPA存在本地可行副本时，evaluator应让低延迟本地候选有胜出机会"""
        import numpy as np
        from ResourceAllocation import evaluate_action_dry_run

        system_state = make_small_system(seed=38)
        self.assertTrue(run_GSLA(system_state))
        config = AblationExperimentConfig(
            time_slots=3,
            traditional_nodes=10,
            ai_nodes=4,
            request_flow_count=4,
            chain_length_range=(3, 4),
            arrival_range_req_s=(2.0, 4.0),
            input_tokens_range=(128, 256),
            output_tokens_range=(32, 64),
        )
        n = len([
            server for server in system_state.edge_servers.values()
            if server.server_type.value == "ai_capable"
        ])
        all_cloud = np.ones(n, dtype=int)
        cloud_score = evaluate_action_dry_run(all_cloud, system_state, config, queue_aware=True)["paper_dpp_score"]
        best_local_score = float("inf")
        for idx in range(n):
            candidate = np.ones(n, dtype=int)
            candidate[idx] = 0
            result = evaluate_action_dry_run(candidate, system_state, config, queue_aware=True)
            if result.get("feasible") and result.get("local_count", 0) > 0:
                best_local_score = min(best_local_score, float(result["paper_dpp_score"]))
        self.assertLess(best_local_score, cloud_score)

    def test_cost_delay_fallback_is_deterministic(self):
        """Cost fallback不能使用Python内置hash，否则跨进程复现实验成本会漂移"""
        from cost import CostCalculator

        class EmptyState:
            pass

        calculator = CostCalculator()
        delay = calculator._get_server_communication_delay("ai_v1", "tra_v7", EmptyState())

        def stable_id_value(server_id: str) -> int:
            return sum((idx + 1) * ord(ch) for idx, ch in enumerate(str(server_id))) % 100

        expected = abs(stable_id_value("ai_v1") - stable_id_value("tra_v7")) * 0.1 + 0.5
        self.assertAlmostEqual(delay, expected)

        system_state = make_small_system(seed=40)
        topo_delay = system_state.network_topology.get_communication_delay("tra_v1", "ai_v1")
        self.assertAlmostEqual(
            calculator._get_server_communication_delay("tra_v1", "ai_v1", system_state),
            topo_delay,
        )
    def test_routing_probabilities_affect_cost_and_delay(self):
        """同一部署下改变routing概率，通信成本和端到端延迟必须改变"""
        from cost import CostCalculator
        from Waitingtime import calculate_flow_end_to_end_delay_global

        system_state = make_small_system(seed=40)
        self.assertTrue(run_GSLA(system_state))
        flow_id, origin_server, origin_ms, dest_ms, near_server, far_server = self._build_near_far_route(
            system_state
        )

        near_key = (origin_server, origin_ms, near_server, dest_ms)
        far_key = (origin_server, origin_ms, far_server, dest_ms)
        original_routes = copy.deepcopy(system_state.stream_transfer_probabilities)
        try:
            system_state.stream_transfer_probabilities = {flow_id: {near_key: Fraction(1, 1)}}
            near_cost = CostCalculator().calculate_total_system_cost(system_state).total_cost
            near_delay = calculate_flow_end_to_end_delay_global(flow_id, system_state)

            system_state.stream_transfer_probabilities = {flow_id: {far_key: Fraction(1, 1)}}
            far_cost = CostCalculator().calculate_total_system_cost(system_state).total_cost
            far_delay = calculate_flow_end_to_end_delay_global(flow_id, system_state)
        finally:
            system_state.stream_transfer_probabilities = original_routes

        self.assertNotAlmostEqual(near_cost, far_cost, places=6)
        self.assertNotAlmostEqual(near_delay, far_delay, places=6)

    def test_pair_repair_consumes_hapa_reserved_envelope(self):
        """pair本地执行应优先消费HAPA实例预留包络，而不是只看server剩余量"""
        import numpy as np
        from ablation_pair_actions import build_active_pair_universe, project_pair_action_to_server_action
        from ResourceAllocation import evaluate_action_dry_run

        system_state = make_small_system(seed=38)
        self.assertTrue(run_GSLA(system_state))
        pair_universe = build_active_pair_universe(system_state)
        self.assertTrue(pair_universe)
        target_pos = None
        for pos, item in enumerate(pair_universe):
            instance = system_state.microservice_instances.get(item.get("instance_id", ""))
            if instance is None:
                continue
            if (float(getattr(instance, "gpu_units_reserved", 0.0)) > 0 and
                    float(getattr(instance, "model_storage_reserved", 0.0)) > 0):
                target_pos = pos
                break
        self.assertIsNotNone(target_pos)
        target_item = pair_universe[target_pos]
        server = system_state.edge_servers[target_item["server_id"]]
        server.available_gpu_units = 0
        server.available_gpu_memory = 0.0
        server.available_model_storage = 0.0

        pair_bits = np.ones(len(pair_universe), dtype=int)
        pair_bits[target_pos] = 0
        candidate = {
            "pair_action": pair_bits,
            "action": project_pair_action_to_server_action(pair_bits, pair_universe, 4),
            "pair_universe": pair_universe,
            "action_scope": "pair",
            "candidate_source": "unit_reserved_local",
        }
        result = evaluate_action_dry_run(candidate, system_state, AblationExperimentConfig(), queue_aware=True)
        self.assertTrue(result.get("feasible"), result.get("failure_reason", ""))
        self.assertEqual(result.get("repair_changed_pair_count", 0), 0)
        self.assertEqual(result.get("local_pair_count", 0), 1)
    def test_pair_dry_run_reports_capacity_repair_collapse(self):
        """pair候选被资源修复改写时，raw诊断必须能追踪修复前后差异"""
        import numpy as np
        from ablation_pair_actions import build_active_pair_universe, project_pair_action_to_server_action
        from ResourceAllocation import evaluate_action_dry_run

        system_state = make_small_system(seed=38)
        self.assertTrue(run_GSLA(system_state))
        pair_universe = build_active_pair_universe(system_state)
        self.assertTrue(pair_universe)
        for server in system_state.edge_servers.values():
            if server.server_type.value == "ai_capable":
                server.available_gpu_units = 0
                server.available_gpu_memory = 0.0
                server.available_model_storage = 0.0
                server.available_model_storage = 0.0
        for instance in system_state.microservice_instances.values():
            if instance.microservice.service_type == "ai":
                instance.gpu_units_reserved = 0.0
                instance.gpu_memory_reserved = 0.0
                instance.model_storage_reserved = 0.0
        pair_bits = np.zeros(len(pair_universe), dtype=int)
        candidate = {
            "pair_action": pair_bits,
            "action": project_pair_action_to_server_action(pair_bits, pair_universe, 4),
            "pair_universe": pair_universe,
            "action_scope": "pair",
            "candidate_source": "unit_all_local",
        }
        result = evaluate_action_dry_run(candidate, system_state, AblationExperimentConfig(), queue_aware=True)
        self.assertTrue(result.get("feasible"), result.get("failure_reason", ""))
        self.assertNotEqual(result.get("original_pair_action_hash"), result.get("pair_action_hash"))
        self.assertGreater(result.get("repair_changed_pair_count", 0), 0)
        self.assertGreater(result.get("repair_changed_ratio", 0.0), 0.0)

    def test_pair_endpoint_repair_keeps_one_local_endpoint_per_flow_service(self):
        """同一flow/service的多副本只能选择一个本地endpoint，不能重复执行同一AI服务。"""
        import numpy as np
        from ablation_pair_actions import build_active_pair_universe, project_pair_action_to_server_action
        from ResourceAllocation import evaluate_action_dry_run

        system_state = make_small_system(seed=38)
        self.assertTrue(run_GSLA(system_state))
        base_universe = build_active_pair_universe(system_state)
        self.assertTrue(base_universe)
        base_item = base_universe[0]
        base_instance = system_state.microservice_instances[base_item["instance_id"]]
        ai_servers = sorted([
            server.server_id for server in system_state.edge_servers.values()
            if server.server_type.value == "ai_capable" and server.server_id != base_item["server_id"]
        ])
        self.assertTrue(ai_servers)

        # 构造同一flow/service的第二个可行本地endpoint，模拟HAPA多实例覆盖。
        replica = copy.deepcopy(base_instance)
        replica.server_id = ai_servers[0]
        replica.instance_id = f"{base_item['flow_id']}_{base_item['microservice_id']}_{ai_servers[0]}_unitrep"
        system_state.microservice_instances[replica.instance_id] = replica

        pair_universe = build_active_pair_universe(system_state)
        group_positions = [
            idx for idx, item in enumerate(pair_universe)
            if item["flow_id"] == base_item["flow_id"] and
            item["microservice_id"] == base_item["microservice_id"]
        ]
        self.assertGreaterEqual(len(group_positions), 2)

        pair_bits = np.ones(len(pair_universe), dtype=int)
        for pos in group_positions[:2]:
            pair_bits[pos] = 0
        candidate = {
            "pair_action": pair_bits,
            "action": project_pair_action_to_server_action(pair_bits, pair_universe, 4),
            "pair_universe": pair_universe,
            "action_scope": "pair",
            "candidate_source": "unit_duplicate_local_endpoint",
        }
        result = evaluate_action_dry_run(candidate, system_state, AblationExperimentConfig(), queue_aware=True)
        self.assertTrue(result.get("feasible"), result.get("failure_reason", ""))
        repaired_bits = str(result.get("pair_action_bits", ""))
        local_in_group = sum(
            1 for pos in group_positions
            if pos < len(repaired_bits) and repaired_bits[pos] == "0"
        )
        self.assertEqual(local_in_group, 1)
        self.assertEqual(result.get("local_pair_count", 0), 1)
    def test_paper_dpp_components_are_traceable(self):
        """论文DPP必须导出V*C、能耗队列项和延迟队列项"""
        from ResourceAllocation import calculate_paper_dpp_components
        import numpy as np

        components = calculate_paper_dpp_components(
            ai_delays=np.array([10.0, 20.0]),
            ai_energies=np.array([0.2, 0.4]),
            cost_value=3.0,
            SQ=np.array([1.0, 2.0]),
            SZ=np.array([3.0, 4.0]),
            V=5.0,
            energy_ref_j=1.0,
            delay_ref_ms=10.0,
        )
        expected = (
            components["v_cost_term"] +
            components["energy_queue_term"] +
            components["delay_queue_term"]
        )
        self.assertAlmostEqual(components["paper_dpp_score"], expected)

    def test_paper_dpp_score_wrapper_returns_finite_value(self):
        """兼容评分函数不能因旧参数残留返回异常。"""
        from ResourceAllocation import calculate_paper_dpp_score
        import numpy as np

        score, scaled_energy, scaled_delay = calculate_paper_dpp_score(
            ai_delays=np.array([10.0]),
            ai_energies=np.array([0.5]),
            cost_value=2.0,
            SQ=np.array([1.0]),
            SZ=np.array([1.0]),
            V=3.0,
            energy_ref_j=1.0,
            delay_ref_ms=10.0,
        )
        self.assertTrue(np.isfinite(score))
        self.assertGreater(score, 0.0)
        self.assertGreaterEqual(scaled_energy, 0.0)
        self.assertGreaterEqual(scaled_delay, 0.0)
    def test_paper_dpp_uses_pair_delay_burden_when_given(self):
        """pair级UAC评分应使用active-pair延迟负载，不把多pair压成server均值"""
        from ResourceAllocation import calculate_paper_dpp_components
        import numpy as np

        mean_delay_components = calculate_paper_dpp_components(
            ai_delays=np.array([10.0, 10.0]),
            ai_energies=np.array([1.0, 1.0]),
            cost_value=1.0,
            SQ=np.array([0.0, 0.0]),
            SZ=np.array([5.0, 5.0]),
            V=1.0,
            delay_ref_ms=10.0,
        )
        pair_burden_components = calculate_paper_dpp_components(
            ai_delays=np.array([10.0, 10.0]),
            ai_energies=np.array([1.0, 1.0]),
            cost_value=1.0,
            SQ=np.array([0.0, 0.0]),
            SZ=np.array([5.0, 5.0]),
            V=1.0,
            delay_ref_ms=10.0,
            delay_burden_vec=np.array([30.0, 10.0]),
        )
        self.assertGreater(
            pair_burden_components["delay_queue_term"],
            mean_delay_components["delay_queue_term"],
        )

    def test_delay_burden_ignores_cloud_repaired_pair_on_local_server(self):
        """延迟队列负担要消费repaired pair action，云端pair不能继续压到本地AI服务器"""
        from ablation_pair_actions import build_active_pair_universe
        from ResourceAllocation import calculate_request_level_delay_burden_by_server

        system_state = make_small_system(seed=38)
        self.assertTrue(run_GSLA(system_state))
        pair_universe = build_active_pair_universe(system_state)
        self.assertTrue(pair_universe)
        target = pair_universe[0]
        target_instance = system_state.microservice_instances[target["instance_id"]]
        target_flow = system_state.request_flows[target["flow_id"]]
        ai_server_ids = sorted([
            server.server_id for server in system_state.edge_servers.values()
            if server.server_type.value == "ai_capable"
        ])
        self.assertIn(target_instance.server_id, ai_server_ids)

        for flow in system_state.request_flows.values():
            flow.ca_latency = 1.0
        target_flow.ca_latency = 100.0
        target_instance.pair_action_bit = 0
        target_instance.processing_mode = "local_processing"
        before = calculate_request_level_delay_burden_by_server(system_state, ai_server_ids, 1.0)
        target_idx = ai_server_ids.index(target_instance.server_id)

        target_instance.pair_action_bit = 1
        target_instance.processing_mode = "cloud_offloaded"
        after = calculate_request_level_delay_burden_by_server(system_state, ai_server_ids, 1.0)
        self.assertLess(after[target_idx], before[target_idx])
        self.assertGreater(after[target_idx], 0.0)
    def test_main_gate_rejects_all_cloud_degeneracy(self):
        """正式门禁必须拒绝全云端动作坍缩"""
        from run_ablation import evaluate_main_c4_gate

        config = AblationExperimentConfig(time_slots=100)
        rows = [
            self._valid_aggregate_summary("LyHAM-CO", all_cloud_ratio=1.0, delay_mean=80.0, cost_mean=20.0),
            self._valid_aggregate_summary("GSLA-Myopic", all_cloud_ratio=1.0, delay_mean=60.0, cost_mean=15.0),
            self._valid_aggregate_summary("FFD-UAC", all_cloud_ratio=1.0, delay_mean=70.0, cost_mean=18.0),
        ]
        gate_passed, notes, mechanism_gate_passed, claim_supported = evaluate_main_c4_gate(rows, config)
        self.assertFalse(gate_passed)
        self.assertFalse(mechanism_gate_passed)
        self.assertFalse(claim_supported)
        self.assertTrue(any("all_cloud_ratio" in note for note in notes))

    def test_formal_table_requires_claim_gate(self):
        """正式表必须同时满足formal gate和claim gate，不能只看seed/slot有效"""
        from run_ablation import evaluate_main_c4_gate

        config = AblationExperimentConfig(time_slots=100)
        rows = [
            self._valid_aggregate_summary("LyHAM-CO", all_cloud_ratio=0.0, delay_mean=50.0, cost_mean=10.0,
                                          paper_dpp_score_mean=100.0),
            self._valid_aggregate_summary("GSLA-Myopic", all_cloud_ratio=0.0, delay_mean=40.0, cost_mean=11.0,
                                          paper_dpp_score_mean=110.0),
            self._valid_aggregate_summary("FFD-UAC", all_cloud_ratio=0.0, delay_mean=60.0, cost_mean=12.0,
                                          paper_dpp_score_mean=120.0),
            self._valid_aggregate_summary("PDRS-Myopic", all_cloud_ratio=0.0, delay_mean=70.0, cost_mean=15.0,
                                          paper_dpp_score_mean=150.0),
            self._valid_aggregate_summary("LoadAware-Myopic", all_cloud_ratio=0.0, delay_mean=40.0, cost_mean=16.0,
                                          paper_dpp_score_mean=160.0),
            self._valid_aggregate_summary("FFD-Myopic", all_cloud_ratio=0.0, delay_mean=80.0, cost_mean=17.0,
                                          paper_dpp_score_mean=170.0),
        ]
        gate_passed, notes, mechanism_gate_passed, claim_supported = evaluate_main_c4_gate(rows, config)
        self.assertFalse(gate_passed)
        self.assertTrue(mechanism_gate_passed)
        self.assertFalse(claim_supported)
        self.assertTrue(any("delay" in note for note in notes))

    def test_energy_scope_gate(self):
        """当用户要求energy作为主张指标时，能耗口径无变化必须拒绝formal gate"""
        from run_ablation import evaluate_main_c4_gate

        config = AblationExperimentConfig(time_slots=100)
        config.include_energy_claim = True
        rows = [
            self._valid_aggregate_summary("LyHAM-CO", all_cloud_ratio=0.0, delay_mean=30.0, cost_mean=10.0,
                                          paper_dpp_score_mean=100.0, energy_std=0.0,
                                          energy_scope_gate_passed=False),
            self._valid_aggregate_summary("GSLA-Myopic", all_cloud_ratio=0.0, delay_mean=55.0, cost_mean=11.0,
                                          paper_dpp_score_mean=110.0),
            self._valid_aggregate_summary("FFD-UAC", all_cloud_ratio=0.0, delay_mean=60.0, cost_mean=12.0,
                                          paper_dpp_score_mean=120.0),
            self._valid_aggregate_summary("PDRS-Myopic", all_cloud_ratio=0.0, delay_mean=70.0, cost_mean=15.0,
                                          paper_dpp_score_mean=150.0),
            self._valid_aggregate_summary("LoadAware-Myopic", all_cloud_ratio=0.0, delay_mean=80.0, cost_mean=16.0,
                                          paper_dpp_score_mean=160.0),
            self._valid_aggregate_summary("FFD-Myopic", all_cloud_ratio=0.0, delay_mean=90.0, cost_mean=17.0,
                                          paper_dpp_score_mean=170.0),
        ]
        gate_passed, notes, _, _ = evaluate_main_c4_gate(rows, config)
        self.assertFalse(gate_passed)
        self.assertTrue(any("energy_scope_gate" in note for note in notes))

    def test_energy_hard_claim_rejects_energy_loss(self):
        """energy纳入claim后，LyHAM-CO能耗不优于强baseline时只能出draft"""
        from run_ablation import evaluate_main_c4_gate

        config = AblationExperimentConfig(time_slots=100)
        config.include_energy_claim = True
        rows = [
            self._valid_aggregate_summary("LyHAM-CO", all_cloud_ratio=0.0, delay_mean=30.0,
                                          energy_mean=2.0, cost_mean=10.0,
                                          paper_dpp_score_mean=100.0),
            self._valid_aggregate_summary("GSLA-Myopic", all_cloud_ratio=0.0, delay_mean=35.0,
                                          energy_mean=1.0, cost_mean=11.0,
                                          paper_dpp_score_mean=110.0),
            self._valid_aggregate_summary("FFD-UAC", all_cloud_ratio=0.0, delay_mean=40.0,
                                          energy_mean=2.2, cost_mean=12.0,
                                          paper_dpp_score_mean=120.0),
            self._valid_aggregate_summary("PDRS-Myopic", all_cloud_ratio=0.0, delay_mean=70.0,
                                          energy_mean=1.0, cost_mean=15.0,
                                          paper_dpp_score_mean=150.0),
            self._valid_aggregate_summary("LoadAware-Myopic", all_cloud_ratio=0.0, delay_mean=80.0,
                                          energy_mean=1.1, cost_mean=16.0,
                                          paper_dpp_score_mean=160.0),
            self._valid_aggregate_summary("FFD-Myopic", all_cloud_ratio=0.0, delay_mean=90.0,
                                          energy_mean=1.2, cost_mean=17.0,
                                          paper_dpp_score_mean=170.0),
        ]
        gate_passed, notes, _, claim_supported = evaluate_main_c4_gate(rows, config)
        self.assertFalse(gate_passed)
        self.assertFalse(claim_supported)
        self.assertTrue(any("energy" in note for note in notes))

    def test_canonical_export_requires_energy_claim(self):
        """canonical产物必须来自energy-hard配置"""
        from run_ablation import resolve_canonical_export_gate

        allowed, reason = resolve_canonical_export_gate(True, False)
        self.assertFalse(allowed)
        self.assertIn("include_energy_claim=false", reason)

        allowed, reason = resolve_canonical_export_gate(False, True)
        self.assertFalse(allowed)
        self.assertIn("gate", reason)

        allowed, reason = resolve_canonical_export_gate(True, True)
        self.assertTrue(allowed)
        self.assertEqual(reason, "")

    def test_latex_table_requires_energy_claim_to_be_citable(self):
        """LaTeX正式表必须显式通过energy claim gate"""
        from ablation_export import export_latex_table

        rows = [
            self._valid_aggregate_summary("LyHAM-CO", all_cloud_ratio=0.0),
            self._valid_aggregate_summary("GSLA-Myopic", all_cloud_ratio=0.0),
            self._valid_aggregate_summary("FFD-UAC", all_cloud_ratio=0.0),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            draft_path = export_latex_table(
                base_dir, rows, formal_gate_passed=True,
                claim_supported=True, include_energy_claim=False
            )
            self.assertEqual(draft_path.name, "ablation_table_draft.tex")
            self.assertIn("Draft ablation", draft_path.read_text(encoding="utf-8"))

            formal_path = export_latex_table(
                base_dir, rows, formal_gate_passed=True,
                claim_supported=True, include_energy_claim=True
            )
            self.assertEqual(formal_path.name, "ablation_table.tex")
            self.assertIn("Energy-hard ablation", formal_path.read_text(encoding="utf-8"))

    def test_energy_scope_gate_requires_component_traceability(self):
        """能耗主张不能只看energy_j波动，必须有分量或动作/路由解释"""

        def row(slot, energy, local_component, action_hash):
            return SlotResult(
                slot=slot, seed=38, algorithm="LyHAM-CO", slow_policy="GSLA",
                fast_controller="UAC-DO", status="ok", failure_reason="",
                delay_ms=1.0, energy_j=energy, cost=1.0, avg_y=1.0, avg_z=1.0,
                dpp_score=1.0, legacy_reward=0.0, feasible=True,
                local_count=1, cloud_count=1, forced_cloud_count=0,
                decision_time_ms=1.0, slow_context_reused=False, model_path="",
                paper_dpp_score=1.0, active_ai_energy_j=energy,
                system_active_ai_chain_energy_j=energy,
                energy_local_gpu_j=local_component,
                action_hash=action_hash,
            )

        no_component = summarize_slot_results([
            row(0, 1.0, 0.0, "a"),
            row(1, 1.2, 0.0, "b"),
        ])
        self.assertFalse(no_component.energy_scope_gate_passed)

        traceable = summarize_slot_results([
            row(0, 1.0, 0.4, "a"),
            row(1, 1.2, 0.5, "b"),
        ])
        self.assertTrue(traceable.energy_scope_gate_passed)

    def test_pair_level_action_can_split_instances_on_same_server(self):
        """同一AI服务器上的不同pair应能执行不同本地/云端动作"""
        import numpy as np
        from ablation_algorithms import build_active_pair_universe, project_pair_action_to_server_action, get_ai_action_dimension
        from ResourceAllocation import evaluate_action_dry_run

        system_state = make_small_system(seed=38)
        self.assertTrue(run_GSLA(system_state))
        ai_instances = [
            (instance_id, instance) for instance_id, instance in system_state.microservice_instances.items()
            if instance.microservice.service_type == "ai"
        ]
        self.assertGreaterEqual(len(ai_instances), 2)
        anchor_server = ai_instances[0][1].server_id
        # 测试内构造同server多AI实例，验证pair动作不是server级全同处理
        ai_instances[1][1].server_id = anchor_server
        universe = build_active_pair_universe(system_state)
        same_server_positions = [idx for idx, item in enumerate(universe) if item["server_id"] == anchor_server]
        self.assertGreaterEqual(len(same_server_positions), 2)
        pair_action = np.ones(len(universe), dtype=int)
        pair_action[same_server_positions[0]] = 0
        pair_action[same_server_positions[1]] = 1
        action = project_pair_action_to_server_action(pair_action, universe, get_ai_action_dimension(system_state))
        result = evaluate_action_dry_run({
            "action": action,
            "pair_action": pair_action,
            "pair_universe": universe,
            "action_scope": "pair",
            "candidate_source": "unit_pair_split",
        }, system_state, AblationExperimentConfig(), queue_aware=True)
        self.assertTrue(result["feasible"])
        self.assertGreaterEqual(result["local_pair_count"], 1)
        self.assertGreaterEqual(result["cloud_pair_count"], 1)
        bits = result["pair_action_bits"]
        self.assertEqual(bits[same_server_positions[0]], "0")
        self.assertEqual(bits[same_server_positions[1]], "1")

    def test_pair_dry_run_exports_executed_cloud_pair_signature(self):
        """dry-run必须导出实际执行的云端pair签名，不能只依赖修复后的bit串。"""
        import numpy as np
        from ablation_algorithms import build_active_pair_universe, project_pair_action_to_server_action, get_ai_action_dimension
        from ResourceAllocation import evaluate_action_dry_run

        system_state = make_small_system(seed=38)
        self.assertTrue(run_GSLA(system_state))
        universe = build_active_pair_universe(system_state)
        self.assertTrue(universe)
        cloud_pos = min(1, len(universe) - 1)
        pair_action = np.zeros(len(universe), dtype=int)
        pair_action[cloud_pos] = 1
        action = project_pair_action_to_server_action(
            pair_action, universe, get_ai_action_dimension(system_state)
        )

        result = evaluate_action_dry_run({
            "action": action,
            "pair_action": pair_action,
            "pair_universe": universe,
            "action_scope": "pair",
            "candidate_source": "unit_cloud_signature",
        }, system_state, AblationExperimentConfig(), queue_aware=True)

        self.assertTrue(result["feasible"], result.get("failure_reason", ""))
        self.assertEqual(result["active_cloud_pair_indices"], [cloud_pos])
        self.assertEqual(result["active_cloud_pair_ids"], [universe[cloud_pos]["pair_id"]])
        self.assertEqual(
            result["active_cloud_pair_signature"],
            f"{cloud_pos}:{universe[cloud_pos]['pair_id']}",
        )

    def test_uac_pair_candidates_include_energy_and_hybrid_sources(self):
        """UAC候选集必须包含energy repair、hybrid和pair flip来源"""
        import numpy as np
        from ablation_algorithms import build_pair_repair_candidates, build_active_pair_universe

        system_state = make_small_system(seed=39)
        self.assertTrue(run_GSLA(system_state))
        universe = build_active_pair_universe(system_state)
        seed_candidate = {
            "pair_action": np.ones(len(universe), dtype=int),
            "pair_universe": universe,
            "candidate_source": "actor_topk",
        }
        candidates = build_pair_repair_candidates(
            system_state, AblationExperimentConfig(), queue_aware=True,
            seed_candidates=[seed_candidate], source_prefix="uac"
        )
        sources = {item["candidate_source"] for item in candidates}
        energy_sources = (
            "energy_low_dvfs_local",
            "energy_cloud_relief",
            "energy_queue_relief",
        )
        self.assertTrue(any(any(key in source for key in energy_sources) for source in sources))
        self.assertTrue(any("hybrid" in source for source in sources))
        self.assertTrue(any("pair_flip" in source for source in sources))

    def test_uac_pair_candidates_include_normal_main_frontier_sources(self):
        """normal-main失败后补充的frontier候选必须进Omega，但最终仍由evaluator选择。"""
        import numpy as np
        from ablation_algorithms import build_pair_repair_candidates, build_active_pair_universe

        system_state = make_small_system(seed=39)
        self.assertTrue(run_GSLA(system_state))
        universe = build_active_pair_universe(system_state)
        seed_candidate = {
            "pair_action": np.ones(len(universe), dtype=int),
            "pair_universe": universe,
            "candidate_source": "actor_topk",
        }
        candidates = build_pair_repair_candidates(
            system_state, AblationExperimentConfig(), queue_aware=True,
            seed_candidates=[seed_candidate], source_prefix="uac"
        )
        sources = {item["candidate_source"] for item in candidates}
        required_fragments = [
            "low_energy_delay_feasible",
            "low_cost_ffd_like_local",
            "gmda_delay_frontier",
            "pdrs_energy_frontier",
        ]
        for fragment in required_fragments:
            self.assertTrue(
                any(fragment in source for source in sources),
                f"缺少UAC frontier候选来源: {fragment}",
            )

    def test_paper_compact_candidates_exclude_baseline_named_and_reference_sources(self):
        """正式compact候选族只保留论文可描述的actor/repair/frontier，不把baseline名和Myopic邻居放进Omega。"""
        import numpy as np
        from ablation_algorithms import build_pair_repair_candidates, build_active_pair_universe

        system_state = make_small_system(seed=39)
        self.assertTrue(run_GSLA(system_state))
        universe = build_active_pair_universe(system_state)
        seed_candidate = {
            "pair_action": np.ones(len(universe), dtype=int),
            "pair_universe": universe,
            "candidate_source": "myopic_reference_seed_0",
        }
        config = AblationExperimentConfig(
            uac_candidate_mechanism="paper_compact",
            uac_compact_pair_repair_limit=20,
            uac_compact_frontier_width=4,
        )
        candidates = build_pair_repair_candidates(
            system_state, config, queue_aware=True,
            seed_candidates=[seed_candidate], source_prefix="uac"
        )
        sources = {item["candidate_source"] for item in candidates}

        forbidden_fragments = [
            "low_cost_ffd_like_local",
            "gmda_delay_frontier",
            "pdrs_energy_frontier",
            "reference_low_impact",
            "balanced_tail_relief",
            "pair_flip",
            "lycd_neighborhood",
            "latency_recovery_frontier",
        ]
        for fragment in forbidden_fragments:
            self.assertFalse(
                any(fragment in source for source in sources),
                f"compact候选不应包含来源: {fragment}",
            )
        required_fragments = [
            "low_energy_delay_feasible",
            "cost_aware_hybrid",
            "delay_energy_frontier",
            "local_density_frontier",
            "queue_balanced_frontier",
            "queue_pressure",
            "energy_queue_relief",
        ]
        for fragment in required_fragments:
            self.assertTrue(
                any(fragment in source for source in sources),
                f"compact候选缺少必要frontier: {fragment}",
            )

    def test_paper_compact_anchor_candidates_include_source_clean_resource_refinement(self):
        """paper_compact需要actor/anchor种子上的资源细化候选，而不是回退到reference邻居。"""
        import numpy as np
        from ablation_algorithms import build_pair_repair_candidates, build_active_pair_universe

        system_state = make_small_system(seed=39)
        self.assertTrue(run_GSLA(system_state))
        universe = build_active_pair_universe(system_state)
        seed_candidate = {
            "pair_action": np.ones(len(universe), dtype=int),
            "pair_universe": universe,
            "candidate_source": "uac_lyapunov_anchor_seed_0",
        }
        config = AblationExperimentConfig(
            uac_candidate_mechanism="paper_compact",
            uac_compact_pair_repair_limit=20,
            uac_compact_frontier_width=4,
        )

        candidates = build_pair_repair_candidates(
            system_state, config, queue_aware=True,
            seed_candidates=[seed_candidate], source_prefix="uac_anchor"
        )
        refinement_candidates = [
            item for item in candidates
            if "actor_delay_resource_refinement" in item.get("candidate_source", "")
        ]

        self.assertTrue(refinement_candidates, "compact anchor候选缺少actor/anchor资源细化来源")
        forbidden_fragments = [
            "myopic",
            "reference",
            "gmda",
            "pdrs",
            "ffd",
            "latency_recovery_frontier",
            "pair_flip",
            "lycd_neighborhood",
        ]
        for candidate in refinement_candidates:
            source = candidate.get("candidate_source", "")
            for fragment in forbidden_fragments:
                self.assertNotIn(fragment, source)
            self.assertEqual(candidate.get("resource_hint"), "latency_saver_hybrid")
            self.assertEqual(candidate.get("action_scope"), "pair")
            self.assertEqual(len(candidate.get("pair_action", [])), len(universe))

    def test_paper_compact_delay_resource_refinement_does_not_cloudify_local_seed(self):
        """delay资源细化候选不能在能耗压力下把actor/anchor本地种子改成云端候选。"""
        import numpy as np
        from ablation_algorithms import build_pair_repair_candidates, build_active_pair_universe

        system_state = make_small_system(seed=39)
        self.assertTrue(run_GSLA(system_state))
        for queue in system_state.virtual_energy_queues.values():
            queue.queue_state = 100.0
        for queue in system_state.virtual_delay_queues.values():
            queue.queue_state = 1.0
        universe = build_active_pair_universe(system_state)
        seed_bits = np.zeros(len(universe), dtype=int)
        seed_candidate = {
            "pair_action": seed_bits,
            "pair_universe": universe,
            "candidate_source": "uac_lyapunov_anchor_seed_0",
        }
        config = AblationExperimentConfig(
            uac_candidate_mechanism="paper_compact",
            uac_compact_pair_repair_limit=20,
            uac_compact_frontier_width=4,
        )

        def fake_local_config(*args, **kwargs):
            return {"latency_ms": 8.0, "energy_j": 10.0}

        def fake_cloud_config(*args, **kwargs):
            return {"latency_ms": 24.0, "energy_j": 1.0}

        with patch("ablation_resource_models.select_local_ai_config", side_effect=fake_local_config), \
                patch("ablation_resource_models.solve_cloud_preprocess_config", side_effect=fake_cloud_config):
            candidates = build_pair_repair_candidates(
                system_state, config, queue_aware=True,
                seed_candidates=[seed_candidate], source_prefix="uac_anchor"
            )

        seed_cloud_count = int(np.sum(seed_bits == 1))
        for candidate in candidates:
            if "actor_delay_resource_refinement" not in candidate.get("candidate_source", ""):
                continue
            pair_bits = np.asarray(candidate.get("pair_action", []), dtype=int)
            self.assertLessEqual(
                int(np.sum(pair_bits == 1)),
                seed_cloud_count,
                "delay资源细化候选不能比actor/anchor种子更cloud-heavy",
            )

    def test_hapa_feedback_consumed_by_pair_resource_model(self):
        """HAPA的覆盖率、readiness和D_loc要进入pair级资源评分"""
        from ablation_algorithms import build_active_pair_universe
        from ResourceAllocation import _pair_hapa_features, _apply_hapa_feedback

        system_state = make_small_system(seed=39)
        self.assertTrue(run_GSLA(system_state))
        universe = build_active_pair_universe(system_state)
        self.assertTrue(universe)

        features = None
        for item in universe:
            current = _pair_hapa_features(item, system_state)
            if current.get("has_hapa_feedback", 0.0) > 0.0:
                features = current
                break
        self.assertIsNotNone(features)
        for key in ["coverage_ratio", "replica_readiness", "hapa_psi", "hapa_d_loc"]:
            self.assertIn(key, features)

        stressed = dict(features)
        stressed["coverage_ratio"] = 0.5
        stressed["replica_readiness"] = 0.2
        stressed["hapa_psi"] = 0.3
        stressed["hapa_d_loc"] = 0.2
        stressed_delay, stressed_energy = _apply_hapa_feedback(10.0, 1.0, stressed)

        ready = dict(features)
        ready["coverage_ratio"] = 1.2
        ready["replica_readiness"] = 1.0
        ready["hapa_psi"] = 1.0
        ready["hapa_d_loc"] = 1.0
        ready_delay, ready_energy = _apply_hapa_feedback(10.0, 1.0, ready)

        self.assertGreater(stressed_delay, ready_delay)
        self.assertGreater(stressed_energy, ready_energy)
        # ready副本具备模型热缓存和包络匹配优势，能耗反馈需要足够进入energy claim。
        self.assertLessEqual(ready_energy, 0.80)

    def test_uac_pair_candidates_include_queue_pressure_source(self):
        """UAC应有基于队列影子价格的候选来源，不能只靠actor/server投影"""
        import numpy as np
        from ablation_algorithms import build_pair_repair_candidates, build_active_pair_universe

        system_state = make_small_system(seed=39)
        self.assertTrue(run_GSLA(system_state))
        for idx, server_id in enumerate(sorted(system_state.virtual_delay_queues.keys())):
            system_state.virtual_delay_queues[server_id].queue_state = 500.0 if idx % 2 else 0.0
        universe = build_active_pair_universe(system_state)
        seed_candidate = {
            "pair_action": np.ones(len(universe), dtype=int),
            "pair_universe": universe,
            "candidate_source": "actor_topk",
        }
        candidates = build_pair_repair_candidates(
            system_state, AblationExperimentConfig(), queue_aware=True,
            seed_candidates=[seed_candidate], source_prefix="uac"
        )
        sources = {item["candidate_source"] for item in candidates}
        self.assertTrue(any("queue_pressure" in source for source in sources))

    def test_uac_pair_candidates_include_energy_queue_relief_source(self):
        """UAC应生成能耗队列驱动的本地转云端修复候选"""
        import numpy as np
        from ablation_algorithms import build_pair_repair_candidates, build_active_pair_universe

        system_state = make_small_system(seed=39)
        self.assertTrue(run_GSLA(system_state))
        for idx, server_id in enumerate(sorted(system_state.virtual_energy_queues.keys())):
            system_state.virtual_energy_queues[server_id].queue_state = 500.0 if idx % 2 == 0 else 0.0
        universe = build_active_pair_universe(system_state)
        seed_candidate = {
            "pair_action": np.zeros(len(universe), dtype=int),
            "pair_universe": universe,
            "candidate_source": "actor_all_local_seed",
        }
        candidates = build_pair_repair_candidates(
            system_state, AblationExperimentConfig(), queue_aware=True,
            seed_candidates=[seed_candidate], source_prefix="uac"
        )
        sources = {item["candidate_source"] for item in candidates}
        self.assertTrue(any("energy_queue_relief" in source for source in sources))

    def test_uac_pair_candidates_include_queue_balanced_frontier(self):
        """UAC应生成队列平衡前沿候选，拉开与queue-unaware Myopic的动作空间"""
        import numpy as np
        from ablation_algorithms import build_pair_repair_candidates, build_active_pair_universe

        system_state = make_small_system(seed=39)
        self.assertTrue(run_GSLA(system_state))
        for idx, server_id in enumerate(sorted(system_state.virtual_energy_queues.keys())):
            system_state.virtual_energy_queues[server_id].queue_state = 650.0 if idx % 2 == 0 else 10.0
        for idx, server_id in enumerate(sorted(system_state.virtual_delay_queues.keys())):
            system_state.virtual_delay_queues[server_id].queue_state = 650.0 if idx % 2 == 1 else 10.0
        universe = build_active_pair_universe(system_state)
        seed_candidate = {
            "pair_action": np.ones(len(universe), dtype=int),
            "pair_universe": universe,
            "candidate_source": "actor_topk",
        }
        candidates = build_pair_repair_candidates(
            system_state, AblationExperimentConfig(uac_pair_repair_limit=48), queue_aware=True,
            seed_candidates=[seed_candidate], source_prefix="uac"
        )
        sources = {item["candidate_source"] for item in candidates}
        self.assertTrue(any("queue_balanced_frontier" in source for source in sources))

    def test_myopic_pair_candidates_ignore_virtual_queues(self):
        """Myopic候选生成不能读取Y/Z队列，队列变化只允许影响UAC评分"""
        from ablation_algorithms import build_pair_repair_candidates

        system_state = make_small_system(seed=39)
        self.assertTrue(run_GSLA(system_state))
        config = AblationExperimentConfig(uac_pair_repair_limit=24)

        base_candidates = build_pair_repair_candidates(
            system_state, config, queue_aware=False, source_prefix="myopic"
        )
        base_bits = [tuple(int(v) for v in item["pair_action"]) for item in base_candidates]

        for idx, server_id in enumerate(sorted(system_state.virtual_energy_queues.keys())):
            system_state.virtual_energy_queues[server_id].queue_state = 0.0 if idx % 2 else 500.0
        for idx, server_id in enumerate(sorted(system_state.virtual_delay_queues.keys())):
            system_state.virtual_delay_queues[server_id].queue_state = 500.0 if idx % 2 else 0.0

        changed_candidates = build_pair_repair_candidates(
            system_state, config, queue_aware=False, source_prefix="myopic"
        )
        changed_bits = [tuple(int(v) for v in item["pair_action"]) for item in changed_candidates]
        self.assertEqual(base_bits, changed_bits)

    def test_myopic_execution_resource_search_ignores_virtual_queues(self):
        """Myopic正式执行的资源枚举不读Y/Z，但队列仍可作为结果指标更新"""
        import numpy as np
        from ablation_algorithms import build_active_pair_universe, get_ai_action_dimension
        from ResourceAllocation import evaluate_ai_action_shared

        base = make_small_system(seed=39)
        self.assertTrue(run_GSLA(base))
        universe = build_active_pair_universe(base)
        action_dim = get_ai_action_dimension(base)
        candidate = {
            "pair_action": np.zeros(len(universe), dtype=int),
            "action": np.zeros(action_dim, dtype=int),
            "pair_universe": universe,
            "action_scope": "pair",
            "candidate_source": "test_myopic_boundary",
            "action_dim": action_dim,
            "pair_action_dim": len(universe),
        }
        low_queue = copy.deepcopy(base)
        high_queue = copy.deepcopy(base)
        for queue in high_queue.virtual_energy_queues.values():
            queue.queue_state = 500.0
        for queue in high_queue.virtual_delay_queues.values():
            queue.queue_state = 500.0

        low_result = evaluate_ai_action_shared(
            candidate, low_queue, V=20.0, slot=0, seed=39,
            algorithm="GSLA-Myopic", slow_policy="GSLA", fast_controller="Myopic",
            resource_queue_aware=False
        )
        high_result = evaluate_ai_action_shared(
            candidate, high_queue, V=20.0, slot=0, seed=39,
            algorithm="GSLA-Myopic", slow_policy="GSLA", fast_controller="Myopic",
            resource_queue_aware=False
        )
        self.assertAlmostEqual(low_result.energy_j, high_result.energy_j, places=6)
        self.assertAlmostEqual(low_result.delay_ms, high_result.delay_ms, places=6)

    def test_uac_mechanism_diagnostics_from_pair_bits(self):
        """UAC-vs-Myopic pair hamming和UAC来源占比应能从SlotResult复算"""
        from run_ablation import calculate_uac_mechanism_diagnostics
        from ablation_metrics import SlotResult

        def row(algorithm, bits, source=False):
            return SlotResult(
                slot=0, seed=38, algorithm=algorithm, slow_policy="GSLA",
                fast_controller="UAC-DO" if algorithm == "LyHAM-CO" else "Myopic",
                status="ok", failure_reason="", delay_ms=1.0, energy_j=1.0,
                cost=1.0, avg_y=0.0, avg_z=0.0, dpp_score=1.0,
                legacy_reward=0.0, feasible=True, local_count=1, cloud_count=1,
                forced_cloud_count=0, decision_time_ms=1.0,
                slow_context_reused=False, model_path="", paper_dpp_score=1.0,
                pair_action_bits=bits, uac_selected_source=source,
            )

        diagnostics = calculate_uac_mechanism_diagnostics({
            ("LyHAM-CO", 38): [row("LyHAM-CO", "0101", True)],
            ("GSLA-Myopic", 38): [row("GSLA-Myopic", "0001", False)],
        })
        self.assertAlmostEqual(diagnostics["lyham_vs_gsla_myopic_pair_hamming_mean"], 0.25)
        self.assertAlmostEqual(diagnostics["uac_selected_source_ratio"], 1.0)

    def test_normal_main_mechanism_uses_same_context_reference_bits(self):
        """normal-main不能跨GMDA/PDRS/FFD慢层比较pair bits，必须用LyHAM内部同上下文参考动作。"""
        from run_ablation import calculate_uac_mechanism_diagnostics
        from ablation_metrics import SlotResult

        def row(algorithm, bits):
            return SlotResult(
                slot=0, seed=38, algorithm=algorithm,
                slow_policy="GSLA" if algorithm == "LyHAM-CO" else "GMDA-RMPR",
                fast_controller="UAC-DO" if algorithm == "LyHAM-CO" else "Myopic",
                status="ok", failure_reason="", delay_ms=1.0, energy_j=1.0,
                cost=1.0, avg_y=0.0, avg_z=0.0, dpp_score=1.0,
                legacy_reward=0.0, feasible=True, local_count=1, cloud_count=1,
                forced_cloud_count=0, decision_time_ms=1.0,
                slow_context_reused=False, model_path="", paper_dpp_score=1.0,
                pair_action_bits=bits, uac_selected_source=True,
            )

        lyham = row("LyHAM-CO", "0101")
        # 该参考动作来自同一GSLA慢层、同一pair universe下的Myopic dry-run。
        lyham.reference_pair_action_bits = "0001"
        gmda = row("GMDA-RMPR-Myopic", "000100")

        diagnostics = calculate_uac_mechanism_diagnostics({
            ("LyHAM-CO", 38): [lyham],
            ("GMDA-RMPR-Myopic", 38): [gmda],
        })

        self.assertEqual(diagnostics["hamming_sample_count"], 1)
        self.assertAlmostEqual(diagnostics["lyham_vs_reference_myopic_repaired_hamming_mean"], 0.25)

    def test_artifact_guard_requires_canonical_allowed(self):
        """旧formal但没有canonical gate的run不能作为正文可引用产物"""
        from ablation_artifact_guard import classify_run

        old_formal = {
            "run_id": "20260619_old",
            "meta": {
                "main_c4_gate_passed": True,
                "claim_supported": True,
                "config": {"include_energy_claim": False},
            },
            "raw_dir": "raw/20260619_old",
            "summary_path": "summary/ablation_summary_20260619_old.csv",
            "meta_path": "summary/ablation_run_meta_20260619_old.json",
        }
        self.assertEqual(classify_run(old_formal), "draft")

        current_formal = copy.deepcopy(old_formal)
        current_formal["meta"]["canonical_export_allowed"] = True
        self.assertEqual(classify_run(current_formal), "formal_valid")
    def test_cleanup_manifest_marks_deleted_runs(self):
        """不可引用run必须先写manifest，再删除raw/summary/meta/draft表"""
        from ablation_artifact_guard import delete_non_citable_runs, scan_runs

        with tempfile.TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir)
            run_id = "20260619_bad"
            raw_dir = base_dir / "raw" / run_id / "LyHAM-CO"
            raw_dir.mkdir(parents=True)
            (raw_dir / "seed_38_per_slot.csv").write_text("slot,status\n0,ok\n", encoding="utf-8")
            summary_dir = base_dir / "summary"
            summary_dir.mkdir(parents=True)
            (summary_dir / f"ablation_run_meta_{run_id}.json").write_text(
                '{"run_id":"20260619_bad","main_c4_gate_passed":false}', encoding="utf-8"
            )
            (summary_dir / f"ablation_summary_{run_id}.csv").write_text("algorithm\nLyHAM-CO\n", encoding="utf-8")
            table_dir = base_dir / "tables"
            table_dir.mkdir(parents=True)
            (table_dir / "ablation_table_draft.tex").write_text("draft", encoding="utf-8")

            manifest = delete_non_citable_runs(base_dir, dry_run=False)
            self.assertTrue(manifest.exists())
            self.assertFalse((base_dir / "raw" / run_id).exists())
            self.assertFalse((summary_dir / f"ablation_run_meta_{run_id}.json").exists())
            self.assertFalse((summary_dir / f"ablation_summary_{run_id}.csv").exists())
            remaining = scan_runs(base_dir)
            self.assertNotIn(run_id, remaining)

    def test_normal_experiment_uses_repro_pipeline(self):
        """正常实验入口必须调用统一复现管线，并写入normal_main实验类型"""
        from Comparison_main import run_normal_experiment_from_repro_pipeline

        with patch("Comparison_main.run_ablation_experiment") as runner:
            runner.return_value = {"run_id": "normal_test"}
            result = run_normal_experiment_from_repro_pipeline(seeds=[38], time_slots=3)
            self.assertEqual(result["run_id"], "normal_test")
            config = runner.call_args.kwargs["config"]
            self.assertEqual(config.experiment_type, "normal_main")
            self.assertIn("LyHAM-CO", config.algorithms)
            self.assertNotIn("LyEU", config.algorithms)

    def test_hapa_fields_consumed_by_routing(self):
        """HAPA replica readiness应进入routing score，改变readiness会改变最终路由分"""
        from Deployment import build_softmax_routing_context

        system_state = make_small_system(seed=39)
        self.assertTrue(run_GSLA(system_state))
        flow_id, request_flow = next(iter(system_state.request_flows.items()))
        chain = request_flow.service_chain.microservices
        origin_ms = chain[0]
        dest_ms = chain[-1]
        ai_servers = sorted(
            server.server_id for server in system_state.edge_servers.values()
            if server.server_type.value == "ai_capable"
        )
        system_state.stream_allocated_resources.setdefault(flow_id, {})
        system_state.stream_allocated_resources[flow_id][origin_ms.ms_id] = {ai_servers[0]: 1}
        system_state.stream_allocated_resources[flow_id][dest_ms.ms_id] = {
            ai_servers[0]: 1,
            ai_servers[-1]: 1,
        }
        system_state.gsla_context["hapa_replica_readiness"] = {
            ai_servers[0]: 1.0,
            ai_servers[-1]: 0.0,
        }
        build_softmax_routing_context(system_state, system_state.gsla_context)
        high_ready = copy.deepcopy(system_state.gsla_context.get("routing_score_components", {}))
        system_state.gsla_context["hapa_replica_readiness"] = {
            ai_servers[0]: 0.0,
            ai_servers[-1]: 1.0,
        }
        build_softmax_routing_context(system_state, system_state.gsla_context)
        low_ready = system_state.gsla_context.get("routing_score_components", {})
        self.assertNotEqual(high_ready, low_ready)

    def test_gsla_refine_records_cost_delay_proxy(self):
        """GSLA局部调整需要记录delay/cost/DPP代理权重，避免只是名称包装"""
        system_state = make_small_system(seed=38)
        self.assertTrue(run_GSLA(system_state))
        refine = system_state.gsla_context.get("local_refine", {})
        weights = refine.get("weights", {})
        self.assertEqual(weights.get("delay"), 0.45)
        self.assertGreaterEqual(weights.get("cost"), 0.30)
        self.assertIn("accepted_moves", refine)
        self.assertIn("accepted_swaps", refine)

    def test_gsla_softmax_routing_has_cost_weight_for_normal_main(self):
        """GSLA routing的cost分量要进入预注册权重，支撑cost-aware routing。"""
        system_state = make_small_system(seed=39)
        self.assertTrue(run_GSLA(system_state))
        weights = system_state.gsla_context.get("softmax_routing", {}).get("weights", {})

        self.assertIn("cost", weights)
        self.assertGreaterEqual(float(weights["cost"]), 0.35)
        self.assertGreater(float(weights.get("link_latency", 0.0)), 0.0)
    def test_gsla_post_routing_refine_consumes_routing_paths(self):
        """GSLA二次局部调整必须在routing生成后消费概率路径，不能只停留在预路由proxy。"""
        system_state = make_small_system(seed=39)
        self.assertTrue(run_GSLA(system_state))
        refine = system_state.gsla_context.get("post_routing_refine", {})

        self.assertTrue(refine.get("routing_aware"), refine)
        self.assertGreater(refine.get("routing_paths", 0), 0)
        self.assertIn("routing-weighted", refine.get("objective", ""))

    def test_pair_universe_stable_hash_and_distance_report(self):
        """pair universe和动作差异报告必须可复现，供UAC机制门禁复算。"""
        from ablation_pair_actions import (
            build_active_pair_universe,
            pair_action_distance_report,
            pair_universe_hash,
        )

        left_state = make_small_system(seed=39)
        right_state = make_small_system(seed=39)
        self.assertTrue(run_GSLA(left_state))
        self.assertTrue(run_GSLA(right_state))
        left_universe = build_active_pair_universe(left_state)
        right_universe = build_active_pair_universe(right_state)
        self.assertEqual(pair_universe_hash(left_universe), pair_universe_hash(right_universe))

        left_bits = [0, 1, 0, 1][:len(left_universe)]
        right_bits = [1, 1, 0, 0][:len(left_universe)]
        report = pair_action_distance_report(left_bits, right_bits, left_universe)
        self.assertIn("hamming", report)
        self.assertIn("local_to_cloud_count", report)
        self.assertIn("changed_service_count", report)
        self.assertGreaterEqual(report["hamming"], 0.0)

    def test_uac_candidate_source_quota_keeps_diverse_sources(self):
        """UAC候选去重后仍需保留各类工程候选，避免单一threshold来源占主导。"""
        from ablation_algorithms import build_pair_repair_candidates

        system_state = make_small_system(seed=39)
        self.assertTrue(run_GSLA(system_state))
        config = AblationExperimentConfig(uac_pair_repair_limit=12)
        candidates = build_pair_repair_candidates(
            system_state, config, queue_aware=True,
            seed_candidates=None, source_prefix="uac"
        )
        sources = {candidate.get("candidate_source", "") for candidate in candidates}
        required_sources = {
            "uac_energy_low_dvfs_local",
            "uac_energy_cloud_relief",
            "uac_energy_budget_frontier",
            "uac_cost_aware_hybrid",
            "uac_delay_energy_frontier",
            "uac_replica_ready_local",
            "uac_queue_pressure_flip",
            "uac_lycd_neighborhood",
        }
        self.assertTrue(required_sources.issubset(sources), sources)
    def test_uac_energy_cost_pareto_relief_has_middle_actions(self):
        """UAC需要生成中等云端释放强度的能耗-成本折中候选，避免只在全云/高本地两端选择。"""
        from ablation_algorithms import build_pair_repair_candidates

        system_state = make_small_system(seed=39)
        self.assertTrue(run_GSLA(system_state))
        config = AblationExperimentConfig(uac_pair_repair_limit=32)
        candidates = build_pair_repair_candidates(
            system_state, config, queue_aware=True,
            seed_candidates=None, source_prefix="uac"
        )
        pareto = [c for c in candidates if "energy_cost_pareto_relief" in str(c.get("candidate_source", ""))]
        pair_dim = len(pareto[0]["pair_action"]) if pareto else 0
        local_counts = sorted({int((1 - c["pair_action"]).sum()) for c in pareto})

        self.assertGreaterEqual(len(pareto), 2)
        self.assertTrue(any(0 < count < pair_dim for count in local_counts), local_counts)
        self.assertTrue(any(count <= max(pair_dim // 2, 1) for count in local_counts), local_counts)

    def test_uac_cloud_relief_keeps_incremental_cloud_counts(self):
        """全本地种子需要保留多档cloud relief候选，避免只暴露单pair释放动作。"""
        import numpy as np
        from ablation_algorithms import build_pair_repair_candidates, build_active_pair_universe

        system_state = make_small_system(seed=39)
        self.assertTrue(run_GSLA(system_state))
        universe = build_active_pair_universe(system_state)
        seed_candidate = {
            "pair_action": np.zeros(len(universe), dtype=int),
            "pair_universe": universe,
            "candidate_source": "actor_all_local_seed",
        }

        candidates = build_pair_repair_candidates(
            system_state, AblationExperimentConfig(uac_pair_repair_limit=48),
            queue_aware=True, seed_candidates=[seed_candidate], source_prefix="uac"
        )
        cloud_relief = [
            candidate for candidate in candidates
            if candidate.get("candidate_source") == "uac_energy_cloud_relief"
        ]
        cloud_counts = sorted({int(candidate["pair_action"].sum()) for candidate in cloud_relief})

        self.assertGreaterEqual(len(cloud_counts), 3, cloud_counts)
        self.assertIn(1, cloud_counts)

    def test_pair_actor_v4_config_excludes_formal_seeds_and_uses_large_scale(self):
        """pair actor v4默认按大规模质量训练，formal seed不得进入训练/验证。"""
        from train_pair_uac_actor import PairActorTrainingConfig, assert_no_formal_seed_leakage

        config = PairActorTrainingConfig(version="v4")
        assert_no_formal_seed_leakage(config)
        self.assertEqual(config.version, "v4")
        self.assertGreaterEqual(config.time_slots, 100)
        self.assertGreaterEqual(config.max_candidates_per_seed, 128)
        self.assertEqual(config.max_evaluated_candidates_per_slot, 16)
        self.assertLessEqual(config.max_evaluated_candidates_per_slot, config.max_candidates_per_seed)
        self.assertTrue(set(range(20, 38)).issubset(set(config.train_seeds)))
        self.assertTrue(set(range(43, 53)).issubset(set(config.train_seeds)))
        self.assertEqual(config.val_seeds, [53, 54, 55, 56, 57])
        self.assertTrue(set(range(43, 58)).issubset(set(config.train_seeds) | set(config.val_seeds)))
        self.assertFalse(set(config.train_seeds) & set(config.val_seeds))
        self.assertFalse(set(config.formal_seed_blacklist) & set(config.train_seeds))
        self.assertFalse(set(config.formal_seed_blacklist) & set(config.val_seeds))

    def test_pair_actor_v4_training_uses_normal_main_profile(self):
        """v4训练必须对齐normal-main异构profile，避免学到default小场景模板。"""
        from train_pair_uac_actor import PairActorTrainingConfig, build_training_ablation_config

        config = PairActorTrainingConfig(version="v4")
        ablation_config = build_training_ablation_config(config)

        self.assertEqual(ablation_config.experiment_type, "normal_main")
        self.assertEqual(ablation_config.scenario_profile, "heterogeneous_burst_main")
        self.assertTrue(ablation_config.scenario_profile_frozen)
        self.assertGreaterEqual(ablation_config.request_flow_count, 18)
        self.assertGreaterEqual(ablation_config.traditional_nodes, 40)
        self.assertGreaterEqual(ablation_config.ai_nodes, 16)
        self.assertGreaterEqual(ablation_config.uac_pair_repair_limit, config.max_candidates_per_seed)

    def test_pair_actor_v4_training_system_uses_repro_profile(self):
        """v4训练系统必须复用复现实验profile，并禁止slot更新隐式重部署。"""
        from train_pair_uac_actor import PairActorTrainingConfig, build_training_ablation_config, make_training_system

        train_config = PairActorTrainingConfig(version="v4", train_slots=1)
        ablation_config = build_training_ablation_config(train_config)
        system_state = make_training_system(20, ablation_config)

        self.assertEqual(getattr(system_state, "scenario_profile", ""), "heterogeneous_burst_main")
        self.assertTrue(getattr(system_state, "workload_burst_enabled", False))
        self.assertFalse(getattr(system_state, "allow_environment_redeployment", True))
        self.assertGreaterEqual(len(system_state.request_flows), 18)
        ai_count = sum(
            1 for server in system_state.edge_servers.values()
            if getattr(server.server_type, "value", server.server_type) == "ai_capable"
        )
        self.assertGreaterEqual(ai_count, 16)

    def test_pair_actor_training_queue_profile_is_server_heterogeneous(self):
        """v4训练队列扰动必须按server/slot异构，避免queue-aware标签坍缩成单一动作。"""
        from train_pair_uac_actor import (
            PairActorTrainingConfig,
            apply_training_virtual_queue_profile,
            build_training_ablation_config,
            make_training_system,
        )

        train_config = PairActorTrainingConfig(version="v4", train_slots=1)
        ablation_config = build_training_ablation_config(train_config)
        system_a = make_training_system(20, ablation_config)
        system_b = make_training_system(20, ablation_config)

        profile_a = apply_training_virtual_queue_profile(system_a, seed=20, slot=3)
        profile_b = apply_training_virtual_queue_profile(system_b, seed=20, slot=3)

        energy_a = [
            float(queue.queue_state)
            for _, queue in sorted(system_a.virtual_energy_queues.items())
        ]
        energy_b = [
            float(queue.queue_state)
            for _, queue in sorted(system_b.virtual_energy_queues.items())
        ]
        delay_a = [
            float(queue.queue_state)
            for _, queue in sorted(system_a.virtual_delay_queues.items())
        ]

        self.assertEqual(energy_a, energy_b)
        self.assertGreater(profile_a["energy_queue_unique"], 1)
        self.assertGreater(profile_a["delay_queue_unique"], 1)
        self.assertGreater(len(set(energy_a)), 1)
        self.assertGreater(len(set(delay_a)), 1)
        self.assertEqual(profile_a, profile_b)

    def test_training_label_quality_gate_rejects_collapsed_labels(self):
        """训练标签多样性过低时必须在meta中阻断formal使用。"""
        from train_pair_uac_actor import summarize_training_label_quality

        payload = {
            "dataset_meta": [
                {"status": "ok", "repaired_label_diversity": 0.01, "repaired_label_hash_count": 1},
                {"status": "ok", "repaired_label_diversity": 0.02, "repaired_label_hash_count": 2},
            ],
            "validation_summary": [
                {"status": "ok", "repaired_label_diversity": 0.01, "repaired_label_hash_count": 1},
            ],
        }
        quality = summarize_training_label_quality(payload, min_diversity=0.05)

        self.assertFalse(quality["label_quality_gate_passed"])
        self.assertLess(quality["validation_label_diversity_mean"], 0.05)
        self.assertIn("label diversity", quality["label_quality_failure_reason"])

    def test_pair_actor_training_uses_frontier_labels_not_single_best(self):
        """v4训练应从repaired frontier选多个标签，避免每个slot只学一个模板动作。"""
        import numpy as np
        from train_pair_uac_actor import select_frontier_training_rows

        evaluated_rows = [
            (10.0, {"candidate_source": "best"}, np.array([0, 0, 0, 0]), {"delay_ms": 20.0, "energy_j": 2.0, "cost": 8.0}),
            (10.2, {"candidate_source": "low_energy"}, np.array([1, 0, 0, 0]), {"delay_ms": 23.0, "energy_j": 1.0, "cost": 8.5}),
            (10.3, {"candidate_source": "low_delay"}, np.array([0, 1, 0, 0]), {"delay_ms": 15.0, "energy_j": 2.5, "cost": 8.4}),
            (10.4, {"candidate_source": "low_cost"}, np.array([0, 0, 1, 0]), {"delay_ms": 21.0, "energy_j": 2.4, "cost": 6.0}),
            (15.0, {"candidate_source": "dominated"}, np.array([1, 1, 1, 1]), {"delay_ms": 30.0, "energy_j": 4.0, "cost": 12.0}),
        ]
        rows = select_frontier_training_rows(evaluated_rows, max_positive=4)
        hashes = {"".join(str(int(x)) for x in row[2]) for row in rows}

        self.assertGreaterEqual(len(rows), 3)
        self.assertIn("0000", hashes)
        self.assertTrue({"1000", "0100", "0010"} & hashes)
        self.assertNotIn("1111", hashes)

    def test_pair_actor_cli_accepts_v4_choice(self):
        """训练入口必须允许v4，避免配置类和CLI参数漂移。"""
        script = Path(__file__).with_name("train_pair_uac_actor.py")
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("v4", result.stdout)
        self.assertIn("--max-evaluated-candidates-per-slot", result.stdout)
    def test_summary_exports_repaired_hamming_and_action_diversity(self):
        """summary CSV需要导出UAC机制门禁指标，避免只在meta里可见。"""
        from ablation_export import SUMMARY_FIELDS
        from ablation_metrics import AlgorithmSummary

        required = {
            "repaired_hamming_vs_reference_mean",
            "selected_action_diversity_mean",
            "uac_source_family_count_mean",
        }
        self.assertTrue(required.issubset(set(SUMMARY_FIELDS)))
        summary = AlgorithmSummary(
            algorithm="LyHAM-CO",
            seed=-1,
            n_valid_slots=1,
            n_failed_slots=0,
            delay_mean=1.0,
            delay_std=0.0,
            energy_mean=1.0,
            energy_std=0.0,
            cost_mean=1.0,
            cost_std=0.0,
            avg_y_mean=0.0,
            avg_y_std=0.0,
            avg_z_mean=0.0,
            avg_z_std=0.0,
            dpp_score_mean=1.0,
            dpp_score_std=0.0,
            decision_time_mean_ms=1.0,
            decision_time_p95_ms=1.0,
            feasible_ratio=1.0,
            valid=True,
            repaired_hamming_vs_reference_mean=0.12,
            selected_action_diversity_mean=0.34,
            uac_source_family_count_mean=5.0,
        )
        row = summary.to_dict()
        self.assertAlmostEqual(row["repaired_hamming_vs_reference_mean"], 0.12)
        self.assertAlmostEqual(row["selected_action_diversity_mean"], 0.34)
        self.assertAlmostEqual(row["uac_source_family_count_mean"], 5.0)

    def test_cleanup_keeps_latest_diagnostic_when_requested(self):
        """清理旧结果时保留最近一轮diagnostic证据，避免根因链路丢失。"""
        from ablation_artifact_guard import delete_non_citable_runs

        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            raw_dir = base_dir / "raw"
            summary_dir = base_dir / "summary"
            raw_dir.mkdir(parents=True)
            summary_dir.mkdir(parents=True)
            for run_id in ["20260623_old", "20260623_latest"]:
                (raw_dir / run_id).mkdir()
                (raw_dir / run_id / "marker.txt").write_text(run_id, encoding="utf-8")
                (summary_dir / f"ablation_run_meta_{run_id}.json").write_text(
                    '{"run_id":"' + run_id + '","canonical_export_allowed":false,"experiment_type":"normal_main"}',
                    encoding="utf-8"
                )
                (summary_dir / f"ablation_summary_{run_id}.csv").write_text(
                    "algorithm\nLyHAM-CO\n", encoding="utf-8"
                )

            delete_non_citable_runs(base_dir, dry_run=False, keep_latest_diagnostic=True)
            self.assertFalse((raw_dir / "20260623_old").exists())
            self.assertFalse((summary_dir / "ablation_run_meta_20260623_old.json").exists())
            self.assertTrue((raw_dir / "20260623_latest").exists())
            self.assertTrue((summary_dir / "ablation_run_meta_20260623_latest.json").exists())
    def test_pair_actor_v4_hard_negative_quality_metadata(self):
        """v4训练元信息要记录hard negative类型和repaired label多样性。"""
        import numpy as np
        from train_pair_uac_actor import summarize_hard_negative_quality

        best_bits = np.array([0, 1, 0, 1], dtype=np.float32)
        evaluated_rows = [
            (10.0, {"candidate_source": "pair_train_best"}, best_bits, {"energy_j": 1.0, "delay_ms": 10.0}),
            (12.0, {"candidate_source": "pair_train_myopic_seed"}, np.array([0, 1, 1, 1]), {"energy_j": 1.1, "delay_ms": 11.0}),
            (13.0, {"candidate_source": "pair_train_all_local"}, np.array([0, 0, 0, 0]), {"energy_j": 1.2, "delay_ms": 12.0}),
            (14.0, {"candidate_source": "pair_train_all_cloud"}, np.array([1, 1, 1, 1]), {"energy_j": 0.8, "delay_ms": 20.0}),
            (11.0, {"candidate_source": "pair_train_near"}, np.array([0, 1, 0, 0]), {"energy_j": 0.7, "delay_ms": 25.0}),
        ]
        quality = summarize_hard_negative_quality(evaluated_rows, best_score=10.0, best_bits=best_bits)
        self.assertGreaterEqual(quality["hard_negative_type_counts"]["myopic_repaired_action"], 1)
        self.assertGreaterEqual(quality["hard_negative_type_counts"]["all_local_collapse"], 1)
        self.assertGreaterEqual(quality["hard_negative_type_counts"]["all_cloud_collapse"], 1)
        self.assertGreaterEqual(quality["hard_negative_type_counts"]["low_energy_high_delay"], 1)
        self.assertGreater(quality["repaired_label_diversity"], 0.5)

    def test_hard_negatives_enter_loss_not_only_meta(self):
        """hard negative必须提高冲突pair位的训练权重，不能只写在meta里。"""
        import numpy as np
        from train_pair_uac_actor import PairActorTrainingConfig, make_hard_negative_bit_weights

        config = PairActorTrainingConfig(version="v4")
        best_bits = np.array([0, 1, 0, 1], dtype=np.float32)
        evaluated_rows = [
            (10.0, {"candidate_source": "pair_train_best"}, best_bits, {"energy_j": 1.0, "delay_ms": 10.0}),
            (14.0, {"candidate_source": "pair_train_myopic_seed"}, np.array([1, 1, 0, 0]), {"energy_j": 1.2, "delay_ms": 11.0}),
            (15.0, {"candidate_source": "pair_train_all_cloud"}, np.array([1, 1, 1, 1]), {"energy_j": 0.8, "delay_ms": 20.0}),
        ]
        weights = make_hard_negative_bit_weights(
            evaluated_rows, best_score=10.0, best_bits=best_bits,
            base_weight=1.0, hard_negative_weight=config.hard_negative_loss_weight,
        )

        self.assertEqual(config.loss_mode, "pairwise_hard_negative_weighted_bce")
        self.assertEqual(len(weights), len(best_bits))
        self.assertGreater(weights[0], 1.0)
        self.assertGreater(weights[2], 1.0)
        self.assertGreater(weights[3], 1.0)
    def test_claim_gate_checks_all_declared_baselines(self):
        """energy-hard正文claim必须优于全部声明baseline，而不是只优于三个强baseline。"""
        from run_ablation import evaluate_claim_support

        rows = {
            "LyHAM-CO": self._valid_aggregate_summary(
                "LyHAM-CO", all_cloud_ratio=0.2,
                delay_mean=10.0, energy_mean=1.0, cost_mean=10.0,
                paper_dpp_score_mean=100.0
            ),
            "PDRS-Myopic": self._valid_aggregate_summary("PDRS-Myopic", 0.2, 20.0, 2.0, 20.0, 200.0),
            "LoadAware-Myopic": self._valid_aggregate_summary("LoadAware-Myopic", 0.2, 20.0, 2.0, 20.0, 200.0),
            "FFD-Myopic": self._valid_aggregate_summary("FFD-Myopic", 0.2, 20.0, 2.0, 20.0, 200.0),
            "Random-Myopic": self._valid_aggregate_summary("Random-Myopic", 0.2, 20.0, 2.0, 20.0, 200.0),
            "GSLA-Myopic": self._valid_aggregate_summary("GSLA-Myopic", 0.2, 9.0, 0.9, 9.0, 90.0),
            "FFD-UAC": self._valid_aggregate_summary("FFD-UAC", 0.2, 20.0, 2.0, 20.0, 200.0),
        }
        supported, notes = evaluate_claim_support(rows, include_energy=True)
        self.assertFalse(supported)
        self.assertTrue(any("GSLA-Myopic" in note or "6" in note for note in notes), notes)

    def test_normal_main_claim_requires_registered_margin(self):
        """normal-main claim gate必须支持预注册margin，不能只要求略优。"""
        from run_ablation import evaluate_claim_support
        from ablation_config import NORMAL_MAIN_BASELINE_ALGORITHMS

        rows = {
            "LyHAM-CO": self._valid_aggregate_summary(
                "LyHAM-CO", all_cloud_ratio=0.1,
                delay_mean=91.0, energy_mean=1.81, cost_mean=181.0,
                paper_dpp_score_mean=181.0,
            ),
            "GMDA-RMPR-Myopic": self._valid_aggregate_summary("GMDA-RMPR-Myopic", 0.1, 100.0, 2.0, 200.0, 200.0),
            "PDRS-Myopic": self._valid_aggregate_summary("PDRS-Myopic", 0.1, 100.0, 2.0, 200.0, 200.0),
            "FFD-Myopic": self._valid_aggregate_summary("FFD-Myopic", 0.1, 100.0, 2.0, 200.0, 200.0),
        }
        supported, notes = evaluate_claim_support(
            rows, include_energy=True, include_paper_dpp=True,
            claim_baselines=NORMAL_MAIN_BASELINE_ALGORITHMS,
            claim_margin=0.10,
        )
        self.assertFalse(supported)
        self.assertTrue(any("10%" in note for note in notes), notes)

        rows["LyHAM-CO"] = self._valid_aggregate_summary(
            "LyHAM-CO", all_cloud_ratio=0.1,
            delay_mean=89.0, energy_mean=1.79, cost_mean=179.0,
            paper_dpp_score_mean=179.0,
        )
        supported, notes = evaluate_claim_support(
            rows, include_energy=True, include_paper_dpp=True,
            claim_baselines=NORMAL_MAIN_BASELINE_ALGORITHMS,
            claim_margin=0.10,
        )
        self.assertTrue(supported, notes)
    def test_normal_main_uses_gmda_pdrs_ffd_only(self):
        """主实验声明baseline只保留GMDA/PDRS/FFD，旧诊断算法不得进入normal-main claim。"""
        from ablation_config import (
            ABLATION_MAIN_ALGORITHMS,
            ABLATION_SOLVER_ALGORITHMS,
            DIAGNOSTIC_ALGORITHMS,
            NORMAL_MAIN_ALGORITHMS,
            NORMAL_MAIN_BASELINE_ALGORITHMS,
        )

        self.assertEqual(ABLATION_MAIN_ALGORITHMS, ["LyHAM-CO", "GSLA-Myopic", "FFD-UAC"])
        self.assertEqual(ABLATION_SOLVER_ALGORITHMS, ["LyHAM-CO", "GSLA-LyCD", "GSLA-Myopic"])
        self.assertEqual(
            NORMAL_MAIN_ALGORITHMS,
            ["LyHAM-CO", "GMDA-RMPR-Myopic", "PDRS-Myopic", "FFD-Myopic"],
        )
        self.assertEqual(
            NORMAL_MAIN_BASELINE_ALGORITHMS,
            ["GMDA-RMPR-Myopic", "PDRS-Myopic", "FFD-Myopic"],
        )
        self.assertIn("LoadAware-Myopic", DIAGNOSTIC_ALGORITHMS)
        self.assertIn("Random-Myopic", DIAGNOSTIC_ALGORITHMS)
        self.assertNotIn("LoadAware-Myopic", NORMAL_MAIN_ALGORITHMS)
        self.assertNotIn("Random-Myopic", NORMAL_MAIN_BASELINE_ALGORITHMS)

    def test_gmda_rmpr_exports_probabilistic_routing_policy(self):
        """GMDA-RMPR必须有独立慢层上下文，不能只是LoadAware或Random改名。"""
        from Deployment import run_GMDA_RMPR_slow_context

        system_state = make_small_system(seed=40)
        ok = run_GMDA_RMPR_slow_context(system_state)
        self.assertTrue(ok)
        context = getattr(system_state, "gmda_rmpr_context", {})
        self.assertEqual(context.get("routing_policy"), "GMDA-RMPR-probabilistic")
        self.assertEqual(context.get("implementation_boundary"), "GMDA/RMPR engineering adaptation")
        self.assertTrue(context.get("resource_splitting_groups"))
        self.assertTrue(context.get("reservation_envelope"))
        self.assertTrue(context.get("rmpr_routing_matrix"))
        self.assertFalse(hasattr(system_state, "loadaware_context"))

    def test_formal_energy_claim_rejects_server_level_checkpoint(self):
        """energy/formal运行必须使用pair actor checkpoint，禁止旧server-level模型fallback。"""
        from ai_inference import validate_pair_actor_checkpoint_metadata

        with tempfile.TemporaryDirectory() as tmpdir:
            bad_path = Path(tmpdir) / "server_actor_like.pth"
            bad_path.write_bytes(b"not a pair actor checkpoint")
            ok, reason, meta = validate_pair_actor_checkpoint_metadata(str(bad_path), require_pair_actor=True)
            self.assertFalse(ok)
            self.assertIn("pair", reason.lower())
            self.assertFalse(meta.get("pair_actor", False))

    def test_pair_actor_training_labels_use_repaired_action(self):
        """训练标签必须来自evaluator修复后的可执行pair动作，而不是原始候选。"""
        from train_pair_uac_actor import extract_repaired_label_from_eval_result

        candidate = {"pair_action": [0, 0, 0, 0]}
        eval_result = {"pair_action": [1, 0, 1, 0], "pair_action_bits": "1010"}
        label = extract_repaired_label_from_eval_result(candidate, eval_result)
        self.assertEqual(label.tolist(), [1, 0, 1, 0])

    def test_candidate_dedupe_uses_repaired_pair_hash(self):
        """候选去重和多样性统计必须基于dry-run后的repaired pair hash。"""
        from ablation_algorithms import dedupe_evaluated_candidates_by_repaired_hash

        evaluated = [
            {"candidate_source": "uac_actor", "pair_action_hash": "raw-a", "repaired_pair_action_hash": "rep-1",
             "paper_dpp_score": 9.0},
            {"candidate_source": "uac_energy", "pair_action_hash": "raw-b", "repaired_pair_action_hash": "rep-1",
             "paper_dpp_score": 7.0},
            {"candidate_source": "uac_hybrid", "pair_action_hash": "raw-b", "repaired_pair_action_hash": "rep-2",
             "paper_dpp_score": 8.0},
        ]
        kept = dedupe_evaluated_candidates_by_repaired_hash(evaluated)
        self.assertEqual([row["repaired_pair_action_hash"] for row in kept], ["rep-1", "rep-2"])
        self.assertEqual(kept[0]["candidate_source"], "uac_energy")

    def test_candidate_dedupe_preserves_distinct_resource_hints(self):
        """同一repaired动作下，不同资源偏好代表不同资源搜索结果，不能被误删。"""
        from ablation_algorithms import dedupe_evaluated_candidates_by_repaired_hash

        evaluated = [
            {
                "candidate_source": "pair_actor_threshold_0.35",
                "repaired_pair_action_hash": "same-action",
                "paper_dpp_score": 100.0,
                "energy_j": 9.0,
                "resource_hint": "",
                "f_gpu": 0.90,
                "batch_size": 16,
                "f_pre": 0.0,
            },
            {
                "candidate_source": "uac_energy_low_dvfs_local",
                "repaired_pair_action_hash": "same-action",
                "paper_dpp_score": 101.0,
                "energy_j": 6.0,
                "resource_hint": "energy_saver_local",
                "f_gpu": 0.45,
                "batch_size": 8,
                "f_pre": 0.0,
            },
        ]

        kept = dedupe_evaluated_candidates_by_repaired_hash(evaluated)

        self.assertEqual(len(kept), 2)
        self.assertEqual(
            {row["candidate_source"] for row in kept},
            {"pair_actor_threshold_0.35", "uac_energy_low_dvfs_local"},
        )

    def test_select_cache_key_includes_resource_hint(self):
        """dry-run memo必须区分resource_hint，否则UAC资源偏好不会真正进入评分。"""
        import numpy as np
        from unittest.mock import patch

        from ablation_algorithms import select_best_action_from_candidates

        system_state = make_small_system()
        config = AblationExperimentConfig(include_energy_claim=True)
        config.energy_hard_dpp_tolerance_ratio = 0.05
        action = np.array([0, 0, 0, 0])
        candidates = [
            {
                "action": action,
                "pair_action": action,
                "candidate_source": "pair_actor_threshold_0.35",
                "resource_hint": "",
            },
            {
                "action": action,
                "pair_action": action,
                "candidate_source": "uac_energy_low_dvfs_local",
                "resource_hint": "energy_saver_local",
            },
        ]
        calls = []

        def fake_eval(candidate, *_args, **_kwargs):
            calls.append(str(candidate.get("resource_hint", "")))
            hint = str(candidate.get("resource_hint", ""))
            hinted = hint == "energy_saver_local"
            claim_hinted = hint == "claim_energy_saver_local"
            return {
                "action": candidate["action"],
                "pair_action": candidate["pair_action"],
                "feasible": True,
                "paper_dpp_score": 102.0 if claim_hinted else (101.0 if hinted else 100.0),
                "energy_j": 3.0 if claim_hinted else (4.0 if hinted else 9.0),
                "delay_ms": 40.0,
                "cost": 100.0,
                "local_count": 4,
                "cloud_count": 0,
                "repaired_pair_action_hash": "same-action",
                "candidate_source": candidate["candidate_source"],
                "resource_hint": candidate.get("resource_hint", ""),
                "f_gpu": 0.35 if claim_hinted else (0.45 if hinted else 0.90),
                "batch_size": 4 if claim_hinted else (8 if hinted else 16),
            }

        with patch("ResourceAllocation.evaluate_action_dry_run", side_effect=fake_eval):
            decision = select_best_action_from_candidates(candidates, system_state, config, queue_aware=True)

        self.assertEqual(calls, ["", "energy_saver_local", "claim_energy_saver_local"])
        self.assertEqual(decision["resource_hint"], "claim_energy_saver_local")
        self.assertTrue(decision["selected_candidate_source"].endswith("_claim_energy_saver"))
        self.assertEqual(float(decision["energy_j"]), 3.0)

    def test_claim_gate_uses_normal_main_baselines_only(self):
        """normal-main claim只比较GMDA/PDRS/FFD，不被C4诊断算法污染。"""
        from ablation_config import NORMAL_MAIN_BASELINE_ALGORITHMS
        from run_ablation import evaluate_claim_support

        rows = {
            "LyHAM-CO": self._valid_aggregate_summary(
                "LyHAM-CO", all_cloud_ratio=0.1, delay_mean=10.0, energy_mean=1.0,
                cost_mean=10.0, paper_dpp_score_mean=100.0
            ),
            "GMDA-RMPR-Myopic": self._valid_aggregate_summary(
                "GMDA-RMPR-Myopic", 0.1, 20.0, 2.0, 20.0, 200.0
            ),
            "PDRS-Myopic": self._valid_aggregate_summary("PDRS-Myopic", 0.1, 21.0, 2.1, 21.0, 210.0),
            "FFD-Myopic": self._valid_aggregate_summary("FFD-Myopic", 0.1, 22.0, 2.2, 22.0, 220.0),
            "GSLA-Myopic": self._valid_aggregate_summary("GSLA-Myopic", 0.1, 5.0, 0.5, 5.0, 50.0),
            "FFD-UAC": self._valid_aggregate_summary("FFD-UAC", 0.1, 6.0, 0.6, 6.0, 60.0),
        }
        supported, notes = evaluate_claim_support(
            rows, include_energy=True,
            claim_baselines=NORMAL_MAIN_BASELINE_ALGORITHMS,
            include_paper_dpp=True,
        )
        self.assertTrue(supported, notes)

    def test_normal_main_latex_requires_four_metric_claim(self):
        """normal-main正式表只导出主实验四行，且必须通过四指标claim。"""
        from ablation_export import export_latex_table

        rows = [
            self._valid_aggregate_summary("LyHAM-CO", all_cloud_ratio=0.0),
            self._valid_aggregate_summary("GMDA-RMPR-Myopic", all_cloud_ratio=0.0),
            self._valid_aggregate_summary("PDRS-Myopic", all_cloud_ratio=0.0),
            self._valid_aggregate_summary("FFD-Myopic", all_cloud_ratio=0.0),
            self._valid_aggregate_summary("GSLA-Myopic", all_cloud_ratio=0.0),
            self._valid_aggregate_summary("FFD-UAC", all_cloud_ratio=0.0),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = export_latex_table(
                Path(tmpdir), rows, formal_gate_passed=True, claim_supported=True,
                include_energy_claim=True, experiment_type="normal_main",
            )
            text = path.read_text(encoding="utf-8")
            self.assertEqual(path.name, "normal_main_table.tex")
            self.assertIn("GMDA-RMPR-Myopic", text)
            self.assertNotIn("GSLA-Myopic", text)

    def test_figure_sweep_grid_matches_plot_code(self):
        """统一实验入口必须覆盖当前绘图代码里的横坐标。"""
        from ablation_figure_experiments import build_figure_sweep_plan

        plan = build_figure_sweep_plan()
        self.assertEqual(plan["chain_length"], [(2, 4), (3, 5), (4, 6), (5, 7), (6, 8)])
        self.assertEqual(plan["arrival_rate"], [5, 6, 7, 8, 9, 10, 11, 12])
        self.assertEqual(plan["edge_nodes"], [20, 25, 30, 35, 40, 45])
        self.assertEqual(plan["V"], [1, 5, 10, 20, 50, 100, 200, 500])

    def test_uac_candidate_diagnostics_export_best_source_scores(self):
        """UAC候选诊断需要记录每类候选的最优评分，避免只看最终动作。"""
        from ablation_algorithms import summarize_candidate_source_scores

        rows = [
            {"candidate_source": "pair_actor_threshold_0.35", "paper_dpp_score": 10.0, "energy_j": 2.0, "cost": 5.0, "delay_ms": 20.0, "claim_score": 0.8, "local_count": 3, "cloud_count": 1},
            {"candidate_source": "uac_energy_cloud_relief", "paper_dpp_score": 12.0, "energy_j": 1.0, "cost": 6.0, "delay_ms": 24.0, "claim_score": 0.7, "local_count": 1, "cloud_count": 3},
            {"candidate_source": "uac_energy_cloud_relief", "paper_dpp_score": 9.0, "energy_j": 1.5, "cost": 5.5, "delay_ms": 22.0, "claim_score": 0.9, "local_count": 2, "cloud_count": 2},
        ]
        summary = summarize_candidate_source_scores(rows)
        self.assertIn("actor", summary)
        self.assertIn("energy_cloud_relief", summary)
        self.assertEqual(summary["energy_cloud_relief"]["best_dpp"], 9.0)
        self.assertEqual(summary["energy_cloud_relief"]["best_energy"], 1.0)
        self.assertEqual(summary["energy_cloud_relief"]["best_claim_score"], 0.7)
        self.assertEqual(summary["energy_cloud_relief"]["best_claim_local_count"], 1)
        self.assertEqual(summary["energy_cloud_relief"]["best_claim_cloud_count"], 3)

    def test_selector_source_summary_uses_annotated_claim_rows(self):
        """候选家族摘要必须使用已注释的claim score，不能导出inf。"""
        import numpy as np
        from unittest.mock import patch
        from ablation_algorithms import select_best_action_from_candidates

        system_state = make_small_system()
        config = AblationExperimentConfig(include_energy_claim=True)
        config.energy_hard_dpp_tolerance_ratio = 0.10
        candidates = [
            {"action": np.array([0, 0, 0, 0]), "candidate_source": "pair_actor_threshold_0.35"},
            {"action": np.array([1, 0, 1, 0]), "candidate_source": "uac_energy_cloud_relief"},
        ]

        def fake_eval(candidate, *_args, **_kwargs):
            source = candidate.get("candidate_source", "")
            if "energy_cloud_relief" in source:
                return {
                    "action": candidate["action"], "feasible": True,
                    "paper_dpp_score": 102.0, "energy_j": 3.0,
                    "cost": 52.0, "delay_ms": 120.0,
                    "local_count": 2, "cloud_count": 2,
                    "pair_action_hash": "relief",
                }
            return {
                "action": candidate["action"], "feasible": True,
                "paper_dpp_score": 100.0, "energy_j": 5.0,
                "cost": 50.0, "delay_ms": 110.0,
                "local_count": 4, "cloud_count": 0,
                "pair_action_hash": "actor",
            }

        with patch("ResourceAllocation.evaluate_action_dry_run", side_effect=fake_eval):
            decision = select_best_action_from_candidates(candidates, system_state, config, queue_aware=True)

        summary = decision["candidate_source_score_summary"]
        self.assertIn("actor:n=1", summary)
        self.assertIn("energy_cloud_relief:n=1", summary)
        self.assertNotIn("cs=inf", summary)
        self.assertIn("lc=4,cc=0", summary)
        self.assertIn("lc=2,cc=2", summary)

    def test_resource_hint_changes_result_or_marks_collapse(self):
        """资源偏好若没有改变可执行资源配置，evaluator必须显式标记collapse。"""
        from ResourceAllocation import mark_resource_hint_collapse

        base = {"f_gpu": 0.55, "batch_size": 8, "f_pre": 0.0}
        same = {"f_gpu": 0.55, "batch_size": 8, "f_pre": 0.0}
        changed = {"f_gpu": 0.70, "batch_size": 8, "f_pre": 0.0}
        self.assertTrue(mark_resource_hint_collapse("energy_saver_local", base, same))
        self.assertFalse(mark_resource_hint_collapse("energy_saver_local", base, changed))

    def test_resource_hint_selects_energy_saver_within_dpp_band(self):
        """energy hint应在DPP带内选择更低能耗配置，而不是永远选最低objective。"""
        from ablation_resource_models import select_config_by_resource_hint

        configs = [
            {"objective": 100.0, "energy_j": 5.0, "latency_ms": 20.0, "cost_proxy": 10.0},
            {"objective": 102.0, "energy_j": 3.0, "latency_ms": 21.0, "cost_proxy": 10.5},
            {"objective": 150.0, "energy_j": 1.0, "latency_ms": 80.0, "cost_proxy": 30.0},
        ]
        plain = select_config_by_resource_hint(configs, "")
        hinted = select_config_by_resource_hint(configs, "energy_saver_local", dpp_band_ratio=0.05)
        self.assertEqual(plain["energy_j"], 5.0)
        self.assertEqual(hinted["energy_j"], 3.0)

    def test_resource_hint_selects_cloud_relief_f_pre_within_dpp_band(self):
        """cloud relief hint应偏向低能耗f_pre配置，但仍受DPP带约束。"""
        from ablation_resource_models import select_config_by_resource_hint

        configs = [
            {"objective": 200.0, "energy_j": 8.0, "latency_ms": 60.0, "f_pre": 0.25},
            {"objective": 206.0, "energy_j": 6.0, "latency_ms": 63.0, "f_pre": 0.55},
            {"objective": 280.0, "energy_j": 3.0, "latency_ms": 120.0, "f_pre": 1.0},
        ]
        hinted = select_config_by_resource_hint(configs, "cloud_relief_f_pre", dpp_band_ratio=0.05)
        self.assertEqual(hinted["energy_j"], 6.0)

    def test_silent_run_suppresses_legacy_logs(self):
        """silent模式下底层部署日志应被重定向，保证复现实验输出可读。"""
        from run_ablation import suppress_output_if_needed
        import io
        import contextlib

        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            with suppress_output_if_needed(True):
                print("legacy verbose line")
        self.assertEqual(stream.getvalue(), "")
    def _build_near_far_route(self, system_state):
        """构造同一服务转移的近/远两个目的节点"""
        flow_id, request_flow = next(iter(system_state.request_flows.items()))
        chain = request_flow.service_chain.microservices
        self.assertGreaterEqual(len(chain), 2)
        origin_ms = chain[0].ms_id
        dest_ms = chain[1].ms_id
        origin_server = sorted(system_state.edge_servers.keys())[0]
        candidates = []
        for server_id in sorted(system_state.edge_servers.keys()):
            delay = system_state.network_topology.get_communication_delay(origin_server, server_id)
            candidates.append((float(delay), server_id))
        candidates.sort()
        near_server = candidates[0][1]
        far_server = candidates[-1][1]
        self.assertNotEqual(
            system_state.network_topology.get_communication_delay(origin_server, near_server),
            system_state.network_topology.get_communication_delay(origin_server, far_server),
        )
        return flow_id, origin_server, origin_ms, dest_ms, near_server, far_server

    def _valid_aggregate_summary(self, algorithm: str, all_cloud_ratio: float,
                                 delay_mean: float = 100.0,
                                 energy_mean: float = 1.0,
                                 cost_mean: float = 10.0,
                                 paper_dpp_score_mean: float = 100.0,
                                 energy_std: float = 0.1,
                                 energy_scope_gate_passed: bool = True) -> AlgorithmSummary:
        """构造正式门禁测试用聚合行"""
        return AlgorithmSummary(
            algorithm=algorithm,
            seed=-1,
            n_valid_slots=500,
            n_failed_slots=0,
            delay_mean=delay_mean,
            delay_std=1.0,
            energy_mean=energy_mean,
            energy_std=energy_std,
            cost_mean=cost_mean,
            cost_std=0.5,
            avg_y_mean=0.1,
            avg_y_std=0.0,
            avg_z_mean=0.2,
            avg_z_std=0.0,
            dpp_score_mean=100.0,
            dpp_score_std=1.0,
            decision_time_mean_ms=10.0,
            decision_time_p95_ms=20.0,
            feasible_ratio=1.0,
            valid=True,
            valid_seed_count=5,
            all_cloud_ratio=all_cloud_ratio,
            mechanism_gate_passed=True,
            paper_dpp_score_mean=paper_dpp_score_mean,
            routing_metric_consumed_ratio=1.0,
            routing_delay_consumed_ratio=1.0,
            energy_scope_gate_passed=energy_scope_gate_passed,
        )


if __name__ == "__main__":
    unittest.main()






























