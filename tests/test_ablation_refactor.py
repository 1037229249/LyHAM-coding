import tempfile
import shutil
import unittest
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np


UTILS_DIR = Path(__file__).resolve().parents[1] / "utils"


def _workspace_test_tmp(name: str) -> Path:
    path = Path(__file__).resolve().parents[1] / "_test_artifacts" / name
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path



class AblationRefactorTest(unittest.TestCase):
    def test_candidate_economics_pair_action_detail_lists_local_and_cloud_indices(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from p5_candidate_economics_audit import _pair_action_detail

        pair_universe = [
            {"pair_id": "flow_0:ms@server_a", "flow_id": "flow_0", "microservice_id": "ms", "server_id": "server_a", "instance_id": "flow_0_ms_a"},
            {"pair_id": "flow_1:ms@server_b", "flow_id": "flow_1", "microservice_id": "ms", "server_id": "server_b", "instance_id": "flow_1_ms_b"},
            {"pair_id": "flow_2:ms@server_c", "flow_id": "flow_2", "microservice_id": "ms", "server_id": "server_c", "instance_id": "flow_2_ms_c"},
        ]

        detail = _pair_action_detail("010", pair_universe)

        self.assertEqual(detail["pair_count"], 3)
        self.assertEqual(detail["local_pair_indices"], [0, 2])
        self.assertEqual(detail["cloud_pair_indices"], [1])
        self.assertEqual(detail["cloud_pairs"][0]["pair_index"], 1)
        self.assertEqual(detail["cloud_pairs"][0]["pair_id"], "flow_1:ms@server_b")
        self.assertEqual(detail["local_pairs"][1]["server_id"], "server_c")

    def test_candidate_economics_instance_rows_include_executed_pair_bit(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from types import SimpleNamespace
        from p5_candidate_economics_audit import _instance_rows

        ai_instance = SimpleNamespace(
            microservice=SimpleNamespace(service_type="ai"),
            server_id="server_a",
            processing_mode="cloud_offloaded",
            pair_action_bit=1,
            active_pair_count=1,
            active_local_pair_count=0,
            active_cloud_pair_count=1,
            gpu_units_allocated=1,
            batch_size_allocated=16,
            gpu_frequency_scale=0.85,
            preprocess_frequency_scale=1.0,
            compression_ratio=0.62,
            resource_config_source="cloud_relief",
            resource_hint="cloud_relief_f_pre",
            inference_latency=0.0,
            cloud_latency=438.0,
            energy_local_gpu_j=0.0,
            energy_cloud_compute_j=0.7,
            energy_comm_j=0.01,
            energy_preprocess_j=0.02,
        )
        inactive_ai = SimpleNamespace(
            microservice=SimpleNamespace(service_type="ai"),
            server_id="server_b",
            processing_mode="",
            active_pair_count=0,
        )
        system_state = SimpleNamespace(microservice_instances={
            "flow_1_ms_a": ai_instance,
            "flow_2_ms_b": inactive_ai,
        })

        rows = _instance_rows(system_state)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["instance_id"], "flow_1_ms_a")
        self.assertEqual(rows[0]["pair_action_bit"], 1)
        self.assertEqual(rows[0]["active_cloud_pair_count"], 1)

    def test_pair_actor_training_cli_accepts_lr_epochs_and_output_dir(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from train_pair_uac_actor import build_arg_parser, PairActorTrainingConfig

        parser = build_arg_parser()
        args = parser.parse_args([
            "--version", "v4",
            "--lr", "0.0003",
            "--epochs", "7",
            "--output-dir", "custom_pair_actor_dir",
            "--checkpoint-every-epochs", "5",
            "--resume",
        ])
        config = PairActorTrainingConfig(
            version=args.version,
            lr=args.lr,
            epochs=args.epochs,
            output_dir=args.output_dir,
            checkpoint_every_epochs=args.checkpoint_every_epochs,
            resume=args.resume,
        )

        self.assertEqual(config.version, "v4")
        self.assertAlmostEqual(config.lr, 0.0003)
        self.assertEqual(config.epochs, 7)
        self.assertEqual(config.output_dir, "custom_pair_actor_dir")
        self.assertEqual(config.checkpoint_every_epochs, 5)
        self.assertTrue(config.resume)

    def test_pair_actor_training_loop_records_epoch_loss_history(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        import numpy as np

        from train_pair_uac_actor import PairActorTrainingConfig, fit_pair_actor_model

        config = PairActorTrainingConfig(version="v1", epochs=3, lr=1e-3, hidden_dim=8)
        x_train = np.asarray([
            [0.0, 0.0, 0.2],
            [1.0, 0.0, 0.8],
            [0.0, 1.0, 0.4],
            [1.0, 1.0, 0.9],
        ], dtype=np.float32)
        y_train = np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
        sample_weights = np.ones_like(y_train, dtype=np.float32)

        _model, history = fit_pair_actor_model(x_train, y_train, sample_weights, config)

        self.assertEqual(len(history), 3)
        self.assertTrue(all(np.isfinite(item["loss"]) for item in history))
        self.assertEqual([item["epoch"] for item in history], [1, 2, 3])
        required_diagnostics = {
            "gradient_l2_norm",
            "gradient_nonfinite_count",
            "parameter_nonfinite_count",
            "logit_nonfinite_count",
            "probability_mean",
            "predicted_positive_ratio",
            "target_positive_ratio",
            "actor_collapse_detected",
            "instability_detected",
        }
        for item in history:
            self.assertTrue(required_diagnostics.issubset(item.keys()))
            self.assertGreaterEqual(item["gradient_l2_norm"], 0.0)
            self.assertEqual(item["gradient_nonfinite_count"], 0)
            self.assertEqual(item["parameter_nonfinite_count"], 0)
            self.assertEqual(item["logit_nonfinite_count"], 0)
            self.assertFalse(item["actor_collapse_detected"])
            self.assertFalse(item["instability_detected"])

    def test_pair_actor_training_flags_threshold_level_actor_collapse(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        import numpy as np

        from train_pair_uac_actor import PairActorTrainingConfig, fit_pair_actor_model

        config = PairActorTrainingConfig(version="v1", epochs=1, lr=1e-3, hidden_dim=8)
        x_train = np.zeros((4, 3), dtype=np.float32)
        y_train = np.ones((4,), dtype=np.float32)
        sample_weights = np.ones_like(y_train, dtype=np.float32)

        _model, history = fit_pair_actor_model(x_train, y_train, sample_weights, config)

        self.assertIn(history[-1]["predicted_positive_ratio"], (0.0, 1.0))
        self.assertTrue(history[-1]["actor_collapse_detected"])

    def test_pair_actor_training_records_validation_loss_with_validation_samples(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        import numpy as np

        from train_pair_uac_actor import PairActorTrainingConfig, train_pair_actor

        class FakeAblationConfig:
            def to_dict(self):
                return {"fake": True}

        x_train = np.asarray([
            [0.0, 0.0, 0.2],
            [1.0, 0.0, 0.8],
            [0.0, 1.0, 0.4],
            [1.0, 1.0, 0.9],
        ], dtype=np.float32)
        y_train = np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
        w_train = np.ones_like(y_train, dtype=np.float32)
        x_val = np.asarray([
            [0.0, 0.5, 0.1],
            [1.0, 0.5, 0.9],
        ], dtype=np.float32)
        y_val = np.asarray([0.0, 1.0], dtype=np.float32)
        w_val = np.ones_like(y_val, dtype=np.float32)
        ok_meta = {
            "status": "ok",
            "repaired_label_diversity": 0.5,
            "repaired_label_hash_count": 2,
        }

        with patch("train_pair_uac_actor.build_training_ablation_config", return_value=FakeAblationConfig()), \
             patch("train_pair_uac_actor.build_dataset", return_value=(x_train, y_train, w_train, [ok_meta])), \
             patch("train_pair_uac_actor.collect_seed_examples", return_value=(x_val, y_val, w_val, ok_meta)):
            output_dir = _workspace_test_tmp("pair_actor_validation_loss_unit")
            result = train_pair_actor(PairActorTrainingConfig(
                version="v1",
                val_seeds=[53],
                epochs=2,
                lr=1e-3,
                hidden_dim=8,
                output_dir=str(output_dir),
            ))

        self.assertEqual(len(result["loss_history"]), 2)
        self.assertIn("validation_loss", result["loss_history"][0])
        self.assertTrue(np.isfinite(result["loss_history"][0]["validation_loss"]))
        self.assertEqual(result["validation_sample_count"], 2)

    def test_pair_actor_training_resume_extends_checkpoint_history(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        import numpy as np

        from train_pair_uac_actor import PairActorTrainingConfig, train_pair_actor

        class FakeAblationConfig:
            def to_dict(self):
                return {"fake": True}

        x_train = np.asarray([
            [0.0, 0.0, 0.2],
            [1.0, 0.0, 0.8],
            [0.0, 1.0, 0.4],
            [1.0, 1.0, 0.9],
        ], dtype=np.float32)
        y_train = np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
        w_train = np.ones_like(y_train, dtype=np.float32)
        ok_meta = {
            "status": "ok",
            "repaired_label_diversity": 0.5,
            "repaired_label_hash_count": 2,
        }
        output_dir = _workspace_test_tmp("pair_actor_resume_unit")

        with patch("train_pair_uac_actor.build_training_ablation_config", return_value=FakeAblationConfig()), \
             patch("train_pair_uac_actor.build_dataset", return_value=(x_train, y_train, w_train, [ok_meta])), \
             patch("train_pair_uac_actor.collect_seed_examples", return_value=(np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32), ok_meta)):
            first = train_pair_actor(PairActorTrainingConfig(
                version="v1",
                val_seeds=[53],
                epochs=2,
                lr=1e-3,
                hidden_dim=8,
                output_dir=str(output_dir),
                checkpoint_every_epochs=1,
            ))
            resumed = train_pair_actor(PairActorTrainingConfig(
                version="v1",
                val_seeds=[53],
                epochs=4,
                lr=1e-3,
                hidden_dim=8,
                output_dir=str(output_dir),
                checkpoint_every_epochs=1,
                resume=True,
            ))

        self.assertEqual([item["epoch"] for item in first["loss_history"]], [1, 2])
        self.assertEqual([item["epoch"] for item in resumed["loss_history"]], [1, 2, 3, 4])
        self.assertEqual(resumed["resumed_from_epoch"], 2)
        self.assertTrue(resumed["resume_loaded"])
        self.assertTrue(Path(resumed["resume_checkpoint_path"]).exists())

    def test_pair_actor_training_progress_records_checkpoint_gate_metadata(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        import numpy as np

        from train_pair_uac_actor import PairActorTrainingConfig, train_pair_actor

        class FakeAblationConfig:
            def to_dict(self):
                return {"fake": True}

        x_train = np.asarray([
            [0.0, 0.0, 0.2],
            [1.0, 0.0, 0.8],
            [0.0, 1.0, 0.4],
            [1.0, 1.0, 0.9],
        ], dtype=np.float32)
        y_train = np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
        w_train = np.ones_like(y_train, dtype=np.float32)
        ok_meta = {
            "status": "ok",
            "repaired_label_diversity": 0.5,
            "repaired_label_hash_count": 2,
        }

        with patch("train_pair_uac_actor.build_training_ablation_config", return_value=FakeAblationConfig()), \
             patch("train_pair_uac_actor.build_dataset", return_value=(x_train, y_train, w_train, [ok_meta])), \
             patch("train_pair_uac_actor.collect_seed_examples", return_value=(np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32), ok_meta)):
            output_dir = _workspace_test_tmp("pair_actor_progress_unit")
            result = train_pair_actor(PairActorTrainingConfig(
                version="v1",
                val_seeds=[53],
                epochs=2,
                lr=1e-3,
                hidden_dim=8,
                output_dir=str(output_dir),
                checkpoint_every_epochs=1,
            ))

        progress = json.loads(Path(result["progress_path"]).read_text(encoding="utf-8"))
        self.assertEqual(progress["current_epoch"], 2)
        self.assertEqual(progress["target_epochs"], 2)
        self.assertEqual(progress["train_config_hash"], result["train_config_hash"])
        self.assertEqual(progress["training_dataset_hash"], result["training_dataset_hash"])
        self.assertEqual(len(progress["loss_history"]), 2)
        self.assertEqual(progress["last_epoch"]["gradient_nonfinite_count"], 0)
        self.assertEqual(progress["last_epoch"]["parameter_nonfinite_count"], 0)
        self.assertEqual(progress["last_epoch"]["logit_nonfinite_count"], 0)
        self.assertTrue(Path(progress["last_checkpoint_path"]).exists())

    def test_config_defaults_use_paper_named_algorithms(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_config import AblationExperimentConfig, ABLATION_MAIN_ALGORITHMS

        config = AblationExperimentConfig()

        self.assertEqual(config.seeds, [38, 39, 40, 41, 42])
        self.assertEqual(config.time_slots, 100)
        self.assertEqual(config.slow_epoch_slots, 20)
        self.assertEqual(config.traditional_nodes, 20)
        self.assertEqual(config.ai_nodes, 10)
        self.assertEqual(config.request_flow_count, 12)
        self.assertEqual(ABLATION_MAIN_ALGORITHMS, ["LyHAM-CO", "GSLA-Myopic", "FFD-UAC"])
        self.assertNotIn("LyEU", ABLATION_MAIN_ALGORITHMS)

    def test_export_writes_raw_summary_and_latex_table(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_export import export_latex_table, export_raw_slot_results, export_summary_csv
        from ablation_metrics import SlotResult, summarize_slot_results

        rows = [
            SlotResult(
                slot=0,
                seed=38,
                algorithm="LyHAM-CO",
                slow_policy="GSLA",
                fast_controller="UAC-DO",
                status="ok",
                failure_reason="",
                delay_ms=10.0,
                energy_j=2.0,
                cost=1.0,
                avg_y=0.2,
                avg_z=0.3,
                dpp_score=1.5,
                legacy_reward=8.0,
                feasible=True,
                local_count=8,
                cloud_count=2,
                forced_cloud_count=0,
                decision_time_ms=1.2,
                slow_context_reused=False,
                model_path="model.pth",
            ),
            SlotResult(
                slot=1,
                seed=38,
                algorithm="LyHAM-CO",
                slow_policy="GSLA",
                fast_controller="UAC-DO",
                status="invalid",
                failure_reason="模型文件不存在",
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
                slow_context_reused=False,
                model_path="missing.pth",
            )
        ]

        base = _workspace_test_tmp("export_roundtrip")
        try:
            raw_path = export_raw_slot_results(base, "run1", "LyHAM-CO", 38, rows)
            summary = summarize_slot_results(rows)
            summary_path = export_summary_csv(base, [summary])
            table_path = export_latex_table(base, [summary])

            self.assertTrue(raw_path.exists())
            self.assertTrue(summary_path.exists())
            self.assertTrue(table_path.exists())
            self.assertNotIn("nan", raw_path.read_text(encoding="utf-8").lower())
            self.assertNotIn("nan", summary_path.read_text(encoding="utf-8").lower())
            self.assertIn("LyHAM-CO", table_path.read_text(encoding="utf-8"))
            self.assertIn("Average Response Delay", table_path.read_text(encoding="utf-8"))

        finally:
            shutil.rmtree(base, ignore_errors=True)
    def test_missing_model_marks_uac_algorithm_invalid(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_config import AblationExperimentConfig
        from ablation_algorithms import validate_algorithm_prerequisites

        config = AblationExperimentConfig(model_path=str(Path("missing-model-file.pth").resolve()))

        valid, reason = validate_algorithm_prerequisites("LyHAM-CO", config)

        self.assertFalse(valid)
        self.assertIn("模型文件不存在", reason)

    def test_default_model_path_is_project_local(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_config import get_default_model_path

        model_path = get_default_model_path()

        self.assertEqual(model_path.name, "trained_ai_offloading_model_1.2.pth")
        self.assertIn("Lyapunov_Edge_Unloading10duibi", str(model_path))
        self.assertTrue(model_path.exists())

    def test_resume_meta_reader_preserves_same_hash_online_diagnostics(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from run_ablation import _read_existing_run_meta, aggregate_online_final_model_hash

        base = _workspace_test_tmp("resume_meta_reader")
        try:
            meta_path = base / "summary" / "ablation_run_meta_run1.json"
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            meta_path.write_text(json.dumps({
                "config_hash": "same-hash",
                "uac_online_model_diagnostics": {
                    "entries": [
                        {
                            "algorithm": "LyHAM-CO",
                            "seed": 38,
                            "replay_size": 100,
                            "model_mutated": True,
                            "final_model_hash": "model-b",
                        },
                        {
                            "algorithm": "FFD-UAC",
                            "seed": 38,
                            "replay_size": 100,
                            "model_mutated": True,
                            "final_model_hash": "model-a",
                        },
                    ]
                },
            }), encoding="utf-8")

            loaded = _read_existing_run_meta(meta_path, "same-hash")
            rejected = _read_existing_run_meta(meta_path, "other-hash")

            self.assertEqual(len(loaded["uac_online_model_diagnostics"]["entries"]), 2)
            self.assertEqual(rejected, {})
            self.assertNotEqual(
                aggregate_online_final_model_hash(
                    "initial", loaded["uac_online_model_diagnostics"]
                ),
                "initial",
            )

        finally:
            shutil.rmtree(base, ignore_errors=True)

    def test_base_system_accepts_ablation_request_flow_count(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from Constant import create_base_system

        system = create_base_system(
            seed=38,
            chain_length_range=(4, 4),
            num_edge_nodes=4,
            ai_node_count=2,
            request_flow_count=3,
            input_tokens_range=(128, 128),
            output_tokens_range=(32, 32),
        )

        self.assertEqual(len(system.request_flows), 3)
        self.assertEqual(len([s for s in system.edge_servers.values() if s.server_type.value == "ai_capable"]), 2)

    def test_normal_main_energy_claim_profile_calibrates_virtual_queue_thresholds(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_config import AblationExperimentConfig
        from run_ablation import (
            apply_energy_claim_profile,
            apply_heterogeneous_burst_main_profile,
            create_ablation_system,
        )

        config = AblationExperimentConfig()
        config.experiment_type = "normal_main"
        config.include_energy_claim = True
        apply_heterogeneous_burst_main_profile(config, preserve_runtime_overrides=True)
        apply_energy_claim_profile(config)

        expected_energy_threshold = max(
            config.energy_ref_j,
            config.claim_energy_ref_j * config.energy_claim_threshold_scale * 0.40,
        )
        expected_delay_threshold = config.claim_delay_ref_ms * 3.25
        self.assertAlmostEqual(config.queue_energy_threshold_j, expected_energy_threshold)
        self.assertAlmostEqual(config.queue_delay_threshold_ms, expected_delay_threshold)

        system = create_ablation_system(seed=38, config=config)
        energy_thresholds = {
            queue.energy_threshold
            for queue in system.virtual_energy_queues.values()
        }
        delay_thresholds = {
            queue.delay_threshold
            for queue in system.virtual_delay_queues.values()
        }

        self.assertEqual(energy_thresholds, {config.queue_energy_threshold_j})
        self.assertEqual(delay_thresholds, {config.queue_delay_threshold_ms})

    def test_lyham_temporal_history_is_past_only_per_seed(self):
        import sys
        from types import SimpleNamespace
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_config import AblationExperimentConfig
        from ablation_metrics import SlotResult
        from run_ablation import run_algorithm_for_seed

        class EnvironmentManager:
            def update_all_ai_server_states(self, arrivals, allow_redeployment=False):
                self.last_arrivals = arrivals

        config = AblationExperimentConfig(time_slots=2, slow_epoch_slots=10)
        base_system = SimpleNamespace(environment_manager=EnvironmentManager())
        seen_histories = []

        def slot_result(slot, slow_context_reused=False):
            return SlotResult(
                slot=slot,
                seed=38,
                algorithm="LyHAM-CO",
                slow_policy="GSLA",
                fast_controller="UAC-DO",
                status="ok",
                failure_reason="",
                delay_ms=100.0 + slot,
                energy_j=10.0 + slot,
                cost=500.0 + slot,
                avg_y=1.0,
                avg_z=1.0,
                dpp_score=1.0,
                legacy_reward=0.0,
                feasible=True,
                local_count=2 + slot,
                cloud_count=3 + slot,
                forced_cloud_count=0,
                decision_time_ms=0.1,
                slow_context_reused=slow_context_reused,
                model_path="model.pth",
                selected_candidate_source=f"uac_reference_low_impact_neighbor_{slot}",
                local_pair_count=4 + slot,
                cloud_pair_count=5 + slot,
                predicted_avg_y=10.0 + slot,
                predicted_avg_z=20.0 + slot,
                post_update_queue_drift_term=30.0 + slot,
                post_update_queue_delta_term=40.0 + slot,
                post_update_energy_queue_delta_term=50.0 + slot,
                post_update_delay_queue_delta_term=60.0 + slot,
            )

        def fake_run_named_algorithm(algorithm, system_state, config, slot, seed, slow_context_reused=False):
            seen_histories.append(list(getattr(system_state, "_lyham_temporal_metric_history", []) or []))
            return slot_result(slot, slow_context_reused=slow_context_reused)

        with patch("run_ablation.run_slow_context_for_algorithm", return_value=(True, "")), \
                patch("run_ablation._run_named_algorithm", side_effect=fake_run_named_algorithm):
            results = run_algorithm_for_seed(
                algorithm="LyHAM-CO",
                seed=38,
                base_system=base_system,
                workload_trace=[{"arrival": 1.0}, {"arrival": 2.0}],
                config=config,
                silent=True,
            )

        self.assertEqual([row.slot for row in results], [0, 1])
        self.assertEqual(seen_histories[0], [])
        self.assertEqual(seen_histories[1], [
            {
                "delay_ms": 100.0,
                "energy_j": 10.0,
                "cost": 500.0,
                "selected_candidate_source": "uac_reference_low_impact_neighbor_0",
                "local_count": 2,
                "cloud_count": 3,
                "local_pair_count": 4,
                "cloud_pair_count": 5,
                "predicted_avg_y": 10.0,
                "predicted_avg_z": 20.0,
                "post_update_queue_drift_term": 30.0,
                "post_update_queue_delta_term": 40.0,
                "post_update_energy_queue_delta_term": 50.0,
                "post_update_delay_queue_delta_term": 60.0,
            }
        ])

    def test_energy_claim_profile_raises_delay_queue_weight(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_config import AblationExperimentConfig
        from run_ablation import apply_energy_claim_profile

        config = AblationExperimentConfig(include_energy_claim=True, omega_delay=1.0)
        config.energy_claim_omega_delay = 4.0

        apply_energy_claim_profile(config)

        self.assertEqual(config.omega_delay, 4.0)
        self.assertEqual(config.to_dict()["energy_claim_omega_delay"], 4.0)
        self.assertIn("queue_pressure_energy_guard_dpp_improvement_delay_slack_ratio", config.to_dict())
        self.assertIn("queue_pressure_energy_guard_dpp_improvement_queue_drift_offset_ratio", config.to_dict())
        self.assertIn("queue_pressure_energy_guard_strict_dpp_regret_ratio", config.to_dict())

    def test_arrival_generation_uses_stable_hash(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from Constant import create_base_system

        system1 = create_base_system(seed=38, num_edge_nodes=4, ai_node_count=2, request_flow_count=3)
        system2 = create_base_system(seed=38, num_edge_nodes=4, ai_node_count=2, request_flow_count=3)

        arrivals1 = system1.environment_manager.generate_time_varying_arrivals(7)
        arrivals2 = system2.environment_manager.generate_time_varying_arrivals(7)

        self.assertEqual(arrivals1, arrivals2)

    def test_slot_result_exports_paper_dpp_fields(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_export import RAW_FIELDS, SUMMARY_FIELDS
        from ablation_metrics import SlotResult

        row = SlotResult(
            slot=0,
            seed=38,
            algorithm="GSLA-Myopic",
            slow_policy="GSLA",
            fast_controller="Myopic",
            status="ok",
            failure_reason="",
            delay_ms=10.0,
            energy_j=2.0,
            cost=3.0,
            avg_y=0.1,
            avg_z=0.2,
            dpp_score=64.0,
            legacy_reward=0.0,
            feasible=True,
            local_count=1,
            cloud_count=1,
            forced_cloud_count=0,
            decision_time_ms=1.0,
            slow_context_reused=False,
            model_path="",
            paper_dpp_score=64.0,
            scaled_energy_sum=1.0,
            scaled_delay_burden_sum=2.0,
            candidate_count=4,
            selected_candidate_rank=2,
            replay_written=False,
            online_update_step=0,
            solver_gap_vs_lycd=0.0,
            p95_decision_time_ms=1.0,
            routing_entropy=0.5,
            slow_profile_reused=False,
            retry_count=0,
            resource_hint="cloud_relief_f_pre",
            resource_queue_aware=False,
            resource_queue_scale=0.0,
            resource_mode="queue_relaxed_cloud_relief",
            predicted_avg_y=7.5,
            predicted_avg_z=2.5,
            post_update_queue_drift_term=0.9,
            post_update_queue_pressure_term=49.0,
            post_update_queue_delta_term=3.6,
            post_update_energy_queue_delta_term=2.0,
            post_update_delay_queue_delta_term=1.6,
            post_update_queue_drift_enabled=True,
            tail_risk_candidate_source="reference_low_impact_neighbor",
            tail_risk_candidate_hash="abc123",
            tail_risk_candidate_delay_ms=12.0,
            tail_risk_candidate_energy_j=2.5,
            tail_risk_candidate_cost=4.0,
            tail_risk_candidate_claim_score=5.0,
            tail_risk_candidate_dpp_score=66.0,
            tail_risk_candidate_predicted_avg_y=7.0,
            tail_risk_candidate_predicted_avg_z=2.0,
            tail_risk_candidate_post_update_queue_drift_term=0.4,
            tail_risk_candidate_upper_excess_score=0.2,
            tail_risk_selected_upper_excess_score=0.8,
            tail_risk_candidate_upper_excess_improvement=0.6,
            tail_risk_best_relief_source="unsafe_queue_relief",
            tail_risk_best_relief_hash="unsafe123",
            tail_risk_best_relief_delay_ms=10.0,
            tail_risk_best_relief_energy_j=2.0,
            tail_risk_best_relief_cost=3.0,
            tail_risk_best_relief_claim_score=5.1,
            tail_risk_best_relief_dpp_score=65.0,
            tail_risk_best_relief_predicted_avg_y=12.0,
            tail_risk_best_relief_predicted_avg_z=2.1,
            tail_risk_best_relief_post_update_queue_drift_term=1.2,
            tail_risk_best_relief_upper_excess_score=0.1,
            tail_risk_best_relief_improvement=0.7,
            tail_risk_best_relief_reject_reason="queue_predicted_avg_y",
            energy_relief_candidate_source="eligible_energy_relief",
            energy_relief_candidate_hash="energy123",
            energy_relief_candidate_delay_ms=260.0,
            energy_relief_candidate_energy_j=18.0,
            energy_relief_candidate_cost=1010.0,
            energy_relief_candidate_claim_score=7.8,
            energy_relief_candidate_dpp_score=90.0,
            energy_relief_candidate_predicted_avg_y=6.0,
            energy_relief_candidate_predicted_avg_z=1.8,
            energy_relief_candidate_post_update_queue_drift_term=0.3,
            energy_relief_candidate_energy_gain_j=2.0,
            energy_relief_candidate_delay_regret_ms=10.0,
            energy_relief_candidate_cost_regret=20.0,
            energy_relief_candidate_dpp_regret=-1.0,
            energy_relief_best_lower_source="delay_blocked_energy_relief",
            energy_relief_best_lower_hash="energy-low123",
            energy_relief_best_lower_delay_ms=285.0,
            energy_relief_best_lower_energy_j=16.0,
            energy_relief_best_lower_cost=1020.0,
            energy_relief_best_lower_claim_score=8.0,
            energy_relief_best_lower_dpp_score=89.0,
            energy_relief_best_lower_predicted_avg_y=6.5,
            energy_relief_best_lower_predicted_avg_z=2.0,
            energy_relief_best_lower_post_update_queue_drift_term=0.8,
            energy_relief_best_lower_energy_gain_j=4.0,
            energy_relief_best_lower_delay_regret_ms=35.0,
            energy_relief_best_lower_cost_regret=30.0,
            energy_relief_best_lower_dpp_regret=-2.0,
            energy_relief_best_lower_reject_reason="delay_regret",
        )

        exported = row.to_dict()

        self.assertEqual(exported["paper_dpp_score"], 64.0)
        self.assertIn("paper_dpp_score", RAW_FIELDS)
        self.assertIn("paper_dpp_score_mean", SUMMARY_FIELDS)
        self.assertIn("candidate_count", RAW_FIELDS)
        self.assertFalse(exported["resource_queue_aware"])
        self.assertEqual(exported["resource_queue_scale"], 0.0)
        self.assertEqual(exported["resource_mode"], "queue_relaxed_cloud_relief")
        self.assertIn("resource_queue_aware", RAW_FIELDS)
        self.assertIn("resource_queue_scale", RAW_FIELDS)
        self.assertIn("resource_mode", RAW_FIELDS)
        self.assertEqual(exported["predicted_avg_y"], 7.5)
        self.assertIn("predicted_avg_y", RAW_FIELDS)
        self.assertIn("predicted_avg_z", RAW_FIELDS)
        self.assertIn("post_update_queue_drift_term", RAW_FIELDS)
        self.assertIn("post_update_queue_pressure_term", RAW_FIELDS)
        self.assertIn("post_update_queue_delta_term", RAW_FIELDS)
        self.assertIn("post_update_energy_queue_delta_term", RAW_FIELDS)
        self.assertIn("post_update_delay_queue_delta_term", RAW_FIELDS)
        self.assertIn("post_update_queue_drift_enabled", RAW_FIELDS)
        self.assertEqual(exported["tail_risk_candidate_source"], "reference_low_impact_neighbor")
        self.assertAlmostEqual(exported["tail_risk_candidate_upper_excess_improvement"], 0.6)
        self.assertIn("tail_risk_candidate_source", RAW_FIELDS)
        self.assertIn("tail_risk_candidate_hash", RAW_FIELDS)
        self.assertIn("tail_risk_candidate_delay_ms", RAW_FIELDS)
        self.assertIn("tail_risk_candidate_energy_j", RAW_FIELDS)
        self.assertIn("tail_risk_candidate_cost", RAW_FIELDS)
        self.assertIn("tail_risk_candidate_claim_score", RAW_FIELDS)
        self.assertIn("tail_risk_candidate_dpp_score", RAW_FIELDS)
        self.assertIn("tail_risk_candidate_predicted_avg_y", RAW_FIELDS)
        self.assertIn("tail_risk_candidate_predicted_avg_z", RAW_FIELDS)
        self.assertIn("tail_risk_candidate_post_update_queue_drift_term", RAW_FIELDS)
        self.assertIn("tail_risk_candidate_upper_excess_score", RAW_FIELDS)
        self.assertIn("tail_risk_selected_upper_excess_score", RAW_FIELDS)
        self.assertIn("tail_risk_candidate_upper_excess_improvement", RAW_FIELDS)
        self.assertEqual(exported["tail_risk_best_relief_source"], "unsafe_queue_relief")
        self.assertEqual(exported["tail_risk_best_relief_reject_reason"], "queue_predicted_avg_y")
        self.assertIn("tail_risk_best_relief_source", RAW_FIELDS)
        self.assertIn("tail_risk_best_relief_hash", RAW_FIELDS)
        self.assertIn("tail_risk_best_relief_delay_ms", RAW_FIELDS)
        self.assertIn("tail_risk_best_relief_energy_j", RAW_FIELDS)
        self.assertIn("tail_risk_best_relief_cost", RAW_FIELDS)
        self.assertIn("tail_risk_best_relief_claim_score", RAW_FIELDS)
        self.assertIn("tail_risk_best_relief_dpp_score", RAW_FIELDS)
        self.assertIn("tail_risk_best_relief_predicted_avg_y", RAW_FIELDS)
        self.assertIn("tail_risk_best_relief_predicted_avg_z", RAW_FIELDS)
        self.assertIn("tail_risk_best_relief_post_update_queue_drift_term", RAW_FIELDS)
        self.assertIn("tail_risk_best_relief_upper_excess_score", RAW_FIELDS)
        self.assertIn("tail_risk_best_relief_improvement", RAW_FIELDS)
        self.assertIn("tail_risk_best_relief_reject_reason", RAW_FIELDS)
        self.assertEqual(exported["energy_relief_candidate_source"], "eligible_energy_relief")
        self.assertAlmostEqual(exported["energy_relief_candidate_energy_gain_j"], 2.0)
        self.assertEqual(exported["energy_relief_best_lower_source"], "delay_blocked_energy_relief")
        self.assertEqual(exported["energy_relief_best_lower_reject_reason"], "delay_regret")
        self.assertIn("energy_relief_candidate_source", RAW_FIELDS)
        self.assertIn("energy_relief_candidate_hash", RAW_FIELDS)
        self.assertIn("energy_relief_candidate_delay_ms", RAW_FIELDS)
        self.assertIn("energy_relief_candidate_energy_j", RAW_FIELDS)
        self.assertIn("energy_relief_candidate_cost", RAW_FIELDS)
        self.assertIn("energy_relief_candidate_claim_score", RAW_FIELDS)
        self.assertIn("energy_relief_candidate_dpp_score", RAW_FIELDS)
        self.assertIn("energy_relief_candidate_predicted_avg_y", RAW_FIELDS)
        self.assertIn("energy_relief_candidate_predicted_avg_z", RAW_FIELDS)
        self.assertIn("energy_relief_candidate_post_update_queue_drift_term", RAW_FIELDS)
        self.assertIn("energy_relief_candidate_energy_gain_j", RAW_FIELDS)
        self.assertIn("energy_relief_candidate_delay_regret_ms", RAW_FIELDS)
        self.assertIn("energy_relief_candidate_cost_regret", RAW_FIELDS)
        self.assertIn("energy_relief_candidate_dpp_regret", RAW_FIELDS)
        self.assertIn("energy_relief_best_lower_source", RAW_FIELDS)
        self.assertIn("energy_relief_best_lower_hash", RAW_FIELDS)
        self.assertIn("energy_relief_best_lower_delay_ms", RAW_FIELDS)
        self.assertIn("energy_relief_best_lower_energy_j", RAW_FIELDS)
        self.assertIn("energy_relief_best_lower_cost", RAW_FIELDS)
        self.assertIn("energy_relief_best_lower_claim_score", RAW_FIELDS)
        self.assertIn("energy_relief_best_lower_dpp_score", RAW_FIELDS)
        self.assertIn("energy_relief_best_lower_predicted_avg_y", RAW_FIELDS)
        self.assertIn("energy_relief_best_lower_predicted_avg_z", RAW_FIELDS)
        self.assertIn("energy_relief_best_lower_post_update_queue_drift_term", RAW_FIELDS)
        self.assertIn("energy_relief_best_lower_energy_gain_j", RAW_FIELDS)
        self.assertIn("energy_relief_best_lower_delay_regret_ms", RAW_FIELDS)
        self.assertIn("energy_relief_best_lower_cost_regret", RAW_FIELDS)
        self.assertIn("energy_relief_best_lower_dpp_regret", RAW_FIELDS)
        self.assertIn("energy_relief_best_lower_reject_reason", RAW_FIELDS)

    def test_candidate_source_summary_reports_best_claim_row_metrics(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_algorithms import format_candidate_source_score_summary, summarize_candidate_source_scores

        summary = summarize_candidate_source_scores([
            {
                "candidate_source": "actor_fast",
                "paper_dpp_score": 10.0,
                "delay_ms": 50.0,
                "energy_j": 30.0,
                "cost": 900.0,
                "claim_score": 9.0,
                "local_count": 1,
                "cloud_count": 1,
            },
            {
                "candidate_source": "actor_balanced",
                "paper_dpp_score": 20.0,
                "delay_ms": 70.0,
                "energy_j": 10.0,
                "cost": 100.0,
                "claim_score": 2.0,
                "local_count": 2,
                "cloud_count": 0,
            },
        ])
        text = format_candidate_source_score_summary(summary)

        self.assertIn("bc_d=70", text)
        self.assertIn("bc_e=10", text)
        self.assertIn("bc_c=100", text)
        self.assertIn("bc_dpp=20", text)

    def test_energy_relief_diagnostic_reports_guard_eligible_and_best_lower_reject(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_algorithms import energy_relief_diagnostic_candidate
        from ablation_config import AblationExperimentConfig

        selected = {
            "candidate_source": "selected_high_energy",
            "repaired_pair_action_hash": "selected-hash",
            "feasible": True,
            "paper_dpp_score": 100.0,
            "claim_score": 8.0,
            "delay_ms": 250.0,
            "energy_j": 30.0,
            "cost": 1000.0,
            "predicted_avg_y": 4.0,
            "predicted_avg_z": 3.0,
            "post_update_queue_drift_term": 4.0,
            "eval_rank": 0,
        }
        eligible = {
            "candidate_source": "eligible_energy_relief",
            "repaired_pair_action_hash": "eligible-hash",
            "feasible": True,
            "paper_dpp_score": 100.0,
            "claim_score": 8.1,
            "delay_ms": 260.0,
            "energy_j": 28.0,
            "cost": 1030.0,
            "predicted_avg_y": 4.2,
            "predicted_avg_z": 3.1,
            "post_update_queue_drift_term": 3.5,
            "eval_rank": 1,
        }
        delay_blocked = dict(eligible)
        delay_blocked.update({
            "candidate_source": "delay_blocked_energy_relief",
            "repaired_pair_action_hash": "delay-blocked-hash",
            "paper_dpp_score": 99.0,
            "delay_ms": 285.0,
            "energy_j": 25.0,
            "cost": 1040.0,
            "post_update_queue_drift_term": 3.0,
            "eval_rank": 2,
        })

        config = AblationExperimentConfig(
            queue_pressure_energy_guard_min_energy_gain_j=1.0,
            queue_pressure_energy_guard_max_delay_regret_ms=20.0,
            queue_pressure_energy_guard_max_cost_regret=80.0,
            queue_pressure_energy_guard_dpp_slack_ratio=0.0,
        )
        diagnostic = energy_relief_diagnostic_candidate(
            [selected, delay_blocked, eligible],
            config,
            selected,
        )

        self.assertEqual(diagnostic["energy_relief_candidate_source"], "eligible_energy_relief")
        self.assertEqual(diagnostic["energy_relief_candidate_hash"], "eligible-hash")
        self.assertAlmostEqual(diagnostic["energy_relief_candidate_energy_gain_j"], 2.0)
        self.assertAlmostEqual(diagnostic["energy_relief_candidate_delay_regret_ms"], 10.0)
        self.assertEqual(diagnostic["energy_relief_best_lower_source"], "delay_blocked_energy_relief")
        self.assertEqual(diagnostic["energy_relief_best_lower_hash"], "delay-blocked-hash")
        self.assertAlmostEqual(diagnostic["energy_relief_best_lower_energy_gain_j"], 5.0)
        self.assertAlmostEqual(diagnostic["energy_relief_best_lower_delay_regret_ms"], 35.0)
        self.assertEqual(diagnostic["energy_relief_best_lower_reject_reason"], "delay_regret")

    def test_energy_relief_diagnostic_reports_no_lower_energy_candidate(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_algorithms import energy_relief_diagnostic_candidate
        from ablation_config import AblationExperimentConfig

        selected = {
            "candidate_source": "selected_low_energy",
            "feasible": True,
            "paper_dpp_score": 100.0,
            "claim_score": 8.0,
            "delay_ms": 250.0,
            "energy_j": 20.0,
            "cost": 1000.0,
        }
        higher_energy = dict(selected)
        higher_energy.update({"candidate_source": "higher_energy", "energy_j": 20.5})

        diagnostic = energy_relief_diagnostic_candidate(
            [selected, higher_energy],
            AblationExperimentConfig(),
            selected,
        )

        self.assertEqual(
            diagnostic["energy_relief_best_lower_reject_reason"],
            "no_lower_energy_candidate",
        )
        self.assertEqual(diagnostic["energy_relief_candidate_source"], "")

    def test_tail_risk_diagnostic_prefers_queue_safe_upper_tail_reduction(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_algorithms import tail_risk_diagnostic_candidate
        from ablation_config import AblationExperimentConfig

        history = [
            {"delay_ms": 100.0, "energy_j": 10.0, "cost": 1000.0},
            {"delay_ms": 102.0, "energy_j": 10.5, "cost": 1010.0},
            {"delay_ms": 98.0, "energy_j": 9.5, "cost": 990.0},
            {"delay_ms": 101.0, "energy_j": 10.1, "cost": 1005.0},
            {"delay_ms": 99.0, "energy_j": 9.8, "cost": 995.0},
        ]
        selected = {
            "candidate_source": "selected_spike",
            "paper_dpp_score": 100.0,
            "claim_score": 10.0,
            "delay_ms": 165.0,
            "energy_j": 18.0,
            "cost": 1380.0,
            "predicted_avg_y": 8.0,
            "predicted_avg_z": 6.0,
            "post_update_energy_queue_delta_term": 2.0,
            "post_update_delay_queue_delta_term": 2.0,
            "post_update_queue_drift_term": 0.5,
            "eval_rank": 0,
        }
        safe_candidate = {
            "candidate_source": "safe_tail_relief",
            "repaired_pair_action_hash": "safe-hash",
            "feasible": True,
            "paper_dpp_score": 101.0,
            "claim_score": 10.4,
            "delay_ms": 120.0,
            "energy_j": 12.0,
            "cost": 1090.0,
            "predicted_avg_y": 7.5,
            "predicted_avg_z": 5.5,
            "post_update_energy_queue_delta_term": 1.0,
            "post_update_delay_queue_delta_term": 1.0,
            "post_update_queue_drift_term": 0.2,
            "local_count": 7,
            "cloud_count": 11,
            "eval_rank": 1,
        }
        unsafe_candidate = dict(safe_candidate)
        unsafe_candidate.update({
            "candidate_source": "unsafe_queue_relief",
            "repaired_pair_action_hash": "unsafe-hash",
            "paper_dpp_score": 100.5,
            "claim_score": 10.1,
            "delay_ms": 110.0,
            "energy_j": 11.0,
            "cost": 1060.0,
            "predicted_avg_y": 25.0,
            "post_update_energy_queue_delta_term": 15.0,
            "eval_rank": 2,
        })

        diagnostic = tail_risk_diagnostic_candidate(
            [selected, unsafe_candidate, safe_candidate],
            selected,
            history,
            AblationExperimentConfig(),
        )

        self.assertEqual(diagnostic["tail_risk_candidate_source"], "safe_tail_relief")
        self.assertEqual(diagnostic["tail_risk_candidate_hash"], "safe-hash")
        self.assertGreater(diagnostic["tail_risk_candidate_upper_excess_improvement"], 0.0)
        self.assertLess(
            diagnostic["tail_risk_candidate_upper_excess_score"],
            diagnostic["tail_risk_selected_upper_excess_score"],
        )
        self.assertEqual(diagnostic["tail_risk_best_relief_source"], "unsafe_queue_relief")
        self.assertEqual(diagnostic["tail_risk_best_relief_hash"], "unsafe-hash")
        self.assertEqual(diagnostic["tail_risk_best_relief_reject_reason"], "queue_predicted_avg_y")
        self.assertGreater(
            diagnostic["tail_risk_best_relief_improvement"],
            diagnostic["tail_risk_candidate_upper_excess_improvement"],
        )

    def test_tail_risk_diagnostic_returns_empty_when_selected_has_no_upper_tail_excess(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_algorithms import tail_risk_diagnostic_candidate
        from ablation_config import AblationExperimentConfig

        history = [
            {"delay_ms": 100.0, "energy_j": 10.0, "cost": 1000.0},
            {"delay_ms": 101.0, "energy_j": 10.2, "cost": 1005.0},
            {"delay_ms": 99.0, "energy_j": 9.8, "cost": 995.0},
            {"delay_ms": 102.0, "energy_j": 10.1, "cost": 1010.0},
            {"delay_ms": 98.0, "energy_j": 9.9, "cost": 990.0},
        ]
        selected = {
            "candidate_source": "selected_good_low",
            "feasible": True,
            "paper_dpp_score": 100.0,
            "claim_score": 10.0,
            "delay_ms": 99.0,
            "energy_j": 9.8,
            "cost": 995.0,
        }
        candidate = dict(selected)
        candidate.update({"candidate_source": "higher_but_smoother", "delay_ms": 105.0})

        self.assertEqual(
            tail_risk_diagnostic_candidate(
                [selected, candidate], selected, history, AblationExperimentConfig()
            ),
            {},
        )

    def test_tail_risk_diagnostic_records_selected_risk_without_safe_alternative(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_algorithms import tail_risk_diagnostic_candidate
        from ablation_config import AblationExperimentConfig

        history = [
            {"delay_ms": 100.0, "energy_j": 10.0, "cost": 1000.0},
            {"delay_ms": 102.0, "energy_j": 10.1, "cost": 1010.0},
            {"delay_ms": 98.0, "energy_j": 9.9, "cost": 990.0},
            {"delay_ms": 101.0, "energy_j": 10.2, "cost": 1005.0},
            {"delay_ms": 99.0, "energy_j": 9.8, "cost": 995.0},
        ]
        selected = {
            "candidate_source": "selected_spike",
            "feasible": True,
            "paper_dpp_score": 100.0,
            "claim_score": 10.0,
            "delay_ms": 170.0,
            "energy_j": 19.0,
            "cost": 1400.0,
            "predicted_avg_y": 5.0,
            "predicted_avg_z": 5.0,
        }
        unsafe_candidate = dict(selected)
        unsafe_candidate.update({
            "candidate_source": "unsafe_queue_relief",
            "delay_ms": 120.0,
            "energy_j": 12.0,
            "cost": 1100.0,
            "predicted_avg_y": 30.0,
        })

        diagnostic = tail_risk_diagnostic_candidate(
            [selected, unsafe_candidate], selected, history, AblationExperimentConfig()
        )

        self.assertEqual(diagnostic["tail_risk_candidate_source"], "")
        self.assertEqual(diagnostic["tail_risk_candidate_upper_excess_improvement"], 0.0)
        self.assertGreater(diagnostic["tail_risk_selected_upper_excess_score"], 0.0)
        self.assertEqual(diagnostic["tail_risk_best_relief_source"], "unsafe_queue_relief")
        self.assertEqual(diagnostic["tail_risk_best_relief_reject_reason"], "queue_predicted_avg_y")
        self.assertGreater(diagnostic["tail_risk_best_relief_improvement"], 0.0)

    def test_select_best_action_uses_paper_dpp(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_algorithms import select_best_action_from_candidates
        from ablation_config import AblationExperimentConfig

        candidates = [
            np.array([1, 1], dtype=int),
            np.array([0, 0], dtype=int),
            np.array([1, 0], dtype=int),
        ]

        def fake_eval(action, system_state, config, queue_aware=True):
            return {
                "action": action,
                "paper_dpp_score": float(np.sum(action) * 10 + action[0]),
                "feasible": True,
                "failure_reason": "",
            }

        with patch("ResourceAllocation.evaluate_action_dry_run", side_effect=fake_eval):
            decision = select_best_action_from_candidates(
                candidates=candidates,
                system_state=object(),
                config=AblationExperimentConfig(),
                queue_aware=True,
            )

        self.assertEqual(decision["selected_candidate_rank"], 1)
        np.testing.assert_array_equal(decision["action"], np.array([0, 0], dtype=int))
        self.assertEqual(decision["candidate_count"], 3)

    def test_random_slow_context_is_integrated(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_algorithms import run_slow_context_for_algorithm
        from ablation_config import AblationExperimentConfig
        from Constant import create_base_system

        config = AblationExperimentConfig()
        system = create_base_system(
            seed=38,
            chain_length_range=(4, 4),
            num_edge_nodes=6,
            ai_node_count=2,
            request_flow_count=3,
            input_tokens_range=(128, 128),
            output_tokens_range=(32, 32),
        )

        ok, reason = run_slow_context_for_algorithm("Random-Myopic", system, config, slot=0, seed=38)

        self.assertTrue(ok, reason)
        self.assertNotIn("尚未纳入", reason)
        self.assertTrue(hasattr(system, "random_context"))

    def test_gsla_builds_algorithm_context(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from Deployment import run_GSLA
        from Constant import create_base_system

        system = create_base_system(
            seed=38,
            chain_length_range=(4, 4),
            num_edge_nodes=6,
            ai_node_count=2,
            request_flow_count=3,
            input_tokens_range=(128, 128),
            output_tokens_range=(32, 32),
        )

        ok = run_GSLA(system)

        self.assertTrue(ok)
        self.assertTrue(hasattr(system, "gsla_context"))
        self.assertIn("rappa_clusters", system.gsla_context)
        self.assertIn("hapa_profile", system.gsla_context)
        self.assertGreaterEqual(system.gsla_context.get("routing_entropy", 0.0), 0.0)

    def test_gsla_post_routing_refine_moves_ai_endpoint_for_comm_cost_gain(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from types import SimpleNamespace
        from Deployment import post_refine_gsla_after_routing

        class Network:
            def get_communication_delay(self, source, dest):
                if source == "trad_src" and dest == "ai_slow":
                    return 120.0
                if source == "trad_src" and dest == "ai_near":
                    return 8.0
                return 1.0

        server_type = lambda value: SimpleNamespace(value=value)
        trad_src = SimpleNamespace(
            server_id="trad_src",
            server_type=server_type("traditional"),
            cpu_cores=8,
            memory_capacity=32.0,
            available_cpu=7,
            available_memory=30.0,
        )
        ai_slow = SimpleNamespace(
            server_id="ai_slow",
            server_type=server_type("ai_capable"),
            cpu_cores=8,
            memory_capacity=32.0,
            available_cpu=7,
            available_memory=32.0,
            gpu_units=4,
            gpu_memory=32.0,
            model_storage=64.0,
            available_gpu_units=4,
            available_gpu_memory=32.0,
            available_model_storage=64.0,
        )
        ai_near = SimpleNamespace(
            server_id="ai_near",
            server_type=server_type("ai_capable"),
            cpu_cores=8,
            memory_capacity=32.0,
            available_cpu=8,
            available_memory=32.0,
            gpu_units=4,
            gpu_memory=32.0,
            model_storage=64.0,
            available_gpu_units=4,
            available_gpu_memory=32.0,
            available_model_storage=64.0,
        )
        ai_ms = SimpleNamespace(ms_id="ai", service_type="ai")
        instance = SimpleNamespace(
            instance_id="flow_0_ai_ai_slow",
            microservice=ai_ms,
            server_id="ai_slow",
            allocated_cores=1,
            gpu_units_reserved=1.0,
            gpu_memory_reserved=4.0,
            model_storage_reserved=4.0,
            context_units_reserved=1.0,
        )
        request_flow = SimpleNamespace(
            flow_id="flow_0",
            arrival_rate=12.0,
            r_input_data_size=1024.0,
            r_output_data_size=256.0,
            routing_probabilities={},
        )
        system = SimpleNamespace(
            edge_servers={
                "trad_src": trad_src,
                "ai_slow": ai_slow,
                "ai_near": ai_near,
            },
            request_flows={"flow_0": request_flow},
            microservice_instances={"flow_0_ai_ai_slow": instance},
            stream_allocated_resources={
                "flow_0": {
                    "trad": {"trad_src": 1},
                    "ai": {"ai_slow": 1},
                }
            },
            stream_transfer_probabilities={
                "flow_0": {
                    ("trad_src", "trad", "ai_slow", "ai"): 1.0,
                }
            },
            network_topology=Network(),
            virtual_energy_queues={},
            virtual_delay_queues={},
        )
        context = {
            "hapa_profile": {
                "flow_0_ai_ai_slow": {
                    "flow_id": "flow_0",
                    "server_id": "ai_slow",
                    "microservice": "ai",
                    "capacity_req_s": 1.0,
                }
            },
        }

        moved = post_refine_gsla_after_routing(system, context, max_moves=1)

        self.assertTrue(moved)
        self.assertEqual(instance.server_id, "ai_near")
        self.assertNotIn("ai_slow", system.stream_allocated_resources["flow_0"]["ai"])
        self.assertEqual(system.stream_allocated_resources["flow_0"]["ai"]["ai_near"], 1)
        self.assertEqual(context["post_routing_refine"]["ai_accepted_moves"], 1)

    def test_solver_benchmark_table_export(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_export import export_solver_benchmark_table
        from ablation_metrics import AlgorithmSummary

        summaries = [
            AlgorithmSummary("LyHAM-CO", -1, 3, 0, 10, 1, 2, 0.1, 3, 0.2, 0.1, 0, 0.2, 0, 50, 1, 4, 5, 1.0, True,
                             paper_dpp_score_mean=50, paper_dpp_score_std=1,
                             solver_gap_vs_lycd_mean=0.2, candidate_count_mean=8),
            AlgorithmSummary("GSLA-LyCD", -1, 3, 0, 9, 1, 2, 0.1, 3, 0.2, 0.1, 0, 0.2, 0, 48, 1, 12, 15, 1.0, True,
                             paper_dpp_score_mean=48, paper_dpp_score_std=1,
                             solver_gap_vs_lycd_mean=0.0, candidate_count_mean=4),
        ]

        base = _workspace_test_tmp("solver_benchmark")
        try:
            path = export_solver_benchmark_table(base, summaries)

            text = path.read_text(encoding="utf-8")
            self.assertIn("gap vs. GSLA-LyCD", text)
            self.assertIn("P95 decision time", text)

        finally:
            shutil.rmtree(base, ignore_errors=True)
    def test_c4_cloud_f_pre_search_respects_extended_rails(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from types import SimpleNamespace
        from unittest.mock import patch

        from ablation_resource_models import solve_cloud_preprocess_config

        flow = SimpleNamespace(
            request_flow_id="flow_c4",
            arrival_rate=8.0,
            r_input_data_size=2048.0,
            r_output_data_size=512.0,
        )
        service = SimpleNamespace(service_id="ai_c4")
        server = SimpleNamespace(
            server_id="ai_v1",
            energy_threshold=2.0,
            delay_threshold=80.0,
        )
        system_state = SimpleNamespace(time_frame=0, cloud_f_pre_rails=(0.25, 1.0, 1.25))

        with patch("Deployment.evaluate_cloud_deployment") as mock_cloud_eval, \
                patch("EnergyConsumption.calculate_cloud_processing_energy") as mock_cloud_energy, \
                patch("EnergyConsumption.calculate_optimized_communication_energy") as mock_comm_energy:
            mock_cloud_eval.return_value = {
                "total_latency": 120.0,
                "cloud_inference_latency": 50.0,
            }
            mock_cloud_energy.return_value = 1.0
            mock_comm_energy.return_value = 0.2
            config = solve_cloud_preprocess_config(flow, service, server, system_state, 1.0, 1.0, resource_hint="cloud_relief_f_pre")

        self.assertEqual(config["f_pre"], 1.25)

    def test_default_cloud_f_pre_search_uses_canonical_rails_without_hint(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from types import SimpleNamespace
        from unittest.mock import patch

        from ablation_resource_models import solve_cloud_preprocess_config

        flow = SimpleNamespace(
            request_flow_id="flow_c4_default",
            arrival_rate=8.0,
            r_input_data_size=2048.0,
            r_output_data_size=512.0,
        )
        service = SimpleNamespace(service_id="ai_c4")
        server = SimpleNamespace(
            server_id="ai_v1",
            energy_threshold=2.0,
            delay_threshold=80.0,
        )
        system_state = SimpleNamespace(time_frame=0, cloud_f_pre_rails=(0.25, 1.0, 1.25))

        with patch("Deployment.evaluate_cloud_deployment") as mock_cloud_eval, \
                patch("EnergyConsumption.calculate_cloud_processing_energy") as mock_cloud_energy, \
                patch("EnergyConsumption.calculate_optimized_communication_energy") as mock_comm_energy:
            mock_cloud_eval.return_value = {
                "total_latency": 120.0,
                "cloud_inference_latency": 50.0,
            }
            mock_cloud_energy.return_value = 1.0
            mock_comm_energy.return_value = 0.2
            config = solve_cloud_preprocess_config(flow, service, server, system_state, 1.0, 1.0)

        self.assertEqual(config["f_pre"], 1.0)

    def test_cloud_remote_energy_factor_applies_before_gsla_uac_relief(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from types import SimpleNamespace
        from unittest.mock import patch

        from ablation_resource_models import clear_resource_model_cache, solve_cloud_preprocess_config

        clear_resource_model_cache()
        flow = SimpleNamespace(
            request_flow_id="flow_cloud_energy_factor",
            arrival_rate=8.0,
            r_input_data_size=384.0,
            r_output_data_size=128.0,
        )
        service = SimpleNamespace(service_id="ai_cloud_energy_factor")
        server = SimpleNamespace(
            server_id="ai_v1",
            energy_threshold=2.0,
            delay_threshold=80.0,
        )
        system_state = SimpleNamespace(
            time_frame=0,
            cloud_f_pre_rails=(1.0,),
            cloud_remote_energy_factor=1.50,
            scenario_profile="heterogeneous_burst_main",
            current_slow_policy="GSLA",
            current_fast_controller="UAC-DO",
            gsla_uac_cloud_latency_factor=1.0,
            gsla_uac_cloud_energy_factor=0.80,
            non_gsla_uac_cloud_latency_factor=1.0,
            non_gsla_uac_cloud_energy_factor=1.0,
        )

        with patch("Deployment.evaluate_cloud_deployment") as mock_cloud_eval, \
                patch("EnergyConsumption.calculate_cloud_processing_energy") as mock_cloud_energy, \
                patch("EnergyConsumption.calculate_optimized_communication_energy") as mock_comm_energy:
            mock_cloud_eval.return_value = {
                "total_latency": 120.0,
                "cloud_inference_latency": 50.0,
            }
            mock_cloud_energy.return_value = 10.0
            mock_comm_energy.return_value = 2.0
            config = solve_cloud_preprocess_config(flow, service, server, system_state, 1.0, 1.0)

        self.assertAlmostEqual(config["cloud_compute_energy_j"], 10.0 * 1.50 * 0.80)
        self.assertAlmostEqual(config["preprocess_energy_j"], 0.006 * 1.50 * 0.80)
        self.assertAlmostEqual(config["communication_energy_j"], 2.0)

    def test_energy_claim_profile_uses_pair_actor_and_online_update(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_config import AblationExperimentConfig, get_default_model_path
        from ai_inference import validate_pair_actor_checkpoint_metadata
        from run_ablation import apply_energy_claim_profile

        config = AblationExperimentConfig(include_energy_claim=True)
        self.assertEqual(config.resolve_model_path(), get_default_model_path().resolve())

        apply_energy_claim_profile(config)

        self.assertTrue(config.strict_pair_actor_required)
        self.assertTrue(config.enable_online_update)
        self.assertIn("PairUAC", str(config.resolve_model_path()))
        ok, reason, meta = validate_pair_actor_checkpoint_metadata(
            str(config.resolve_model_path()),
            require_pair_actor=True,
        )
        self.assertTrue(ok, reason)
        self.assertTrue(meta.get("formal_seed_excluded", False))
        self.assertFalse(meta.get("model_mutated", True))

    def test_pair_actor_online_update_mutates_model_from_repaired_label(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_config import AblationExperimentConfig
        from ai_inference import PairUACActorNetwork, TrainedAIInference

        inference = TrainedAIInference("in-memory-pair-actor")
        inference.model = PairUACActorNetwork(feature_dim=3, hidden_dim=4)
        inference.model_kind = "pair_actor"
        inference.is_loaded = True
        config = AblationExperimentConfig(
            enable_online_update=True,
            online_update_interval=1,
            online_update_batch_size=1,
            online_learning_rate=0.05,
        )
        features = np.asarray([[0.1, 0.2, 0.3], [0.4, 0.1, 0.9]], dtype=np.float32)
        repaired_label = np.asarray([0, 1], dtype=np.float32)

        initial_hash = inference.model_state_hash()
        update_meta = inference.online_update_from_feature_batch(
            features=features,
            repaired_pair_action=repaired_label,
            config=config,
            seed=38,
            slot=0,
        )
        final_hash = inference.model_state_hash()

        self.assertTrue(update_meta["replay_written"])
        self.assertTrue(update_meta["model_mutated"])
        self.assertEqual(update_meta["online_update_step"], 1)
        self.assertNotEqual(initial_hash, final_hash)
        self.assertEqual(inference.online_replay_records[-1]["seed"], 38)
        self.assertEqual(inference.online_replay_records[-1]["slot"], 0)

    def test_frozen_lyham_maps_to_gsla_uac_and_requires_model(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        import ablation_algorithms
        from ablation_config import AblationExperimentConfig, UAC_DO_ALGORITHMS

        config = AblationExperimentConfig(enable_online_update=True, include_energy_claim=True)

        self.assertEqual(ablation_algorithms.get_algorithm_policies("LyHAM-CO-Frozen"), ("GSLA", "UAC-DO"))
        self.assertIn("LyHAM-CO-Frozen", UAC_DO_ALGORITHMS)
        self.assertFalse(ablation_algorithms.is_online_update_enabled_for_algorithm(config, "LyHAM-CO-Frozen"))
        self.assertTrue(ablation_algorithms.is_online_update_enabled_for_algorithm(config, "LyHAM-CO"))

    def test_frozen_lyham_disables_online_update_per_algorithm(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        import ablation_algorithms
        from ablation_config import AblationExperimentConfig

        class FakeInference:
            def __init__(self):
                self.updated = False

            def make_candidates(self, system_state, config):
                return [{"action": np.array([0]), "candidate_source": "actor"}]

            def online_update_from_decision(self, system_state, config, decision, seed, slot):
                self.updated = True
                return {
                    "replay_written": True,
                    "online_update_step": 1,
                    "model_mutated": True,
                    "online_loss": 0.5,
                    "final_model_hash": "mutated",
                }

        inference = FakeInference()

        def fake_select(candidates, system_state, config, queue_aware=True):
            return {
                "action": np.array([0]),
                "pair_action_bits": "0",
                "paper_dpp_score": 1.0,
                "selected_candidate_source": "actor",
            }

        config = AblationExperimentConfig(enable_online_update=True, include_energy_claim=True)

        with patch.object(ablation_algorithms, "get_cached_uac_inference", return_value=inference), \
             patch.object(ablation_algorithms, "wrap_candidates_with_pair_projection", side_effect=lambda c, s: list(c)), \
             patch.object(ablation_algorithms, "build_myopic_candidates", return_value=[{"action": np.array([1])}]), \
             patch.object(ablation_algorithms, "build_pair_repair_candidates", return_value=[]), \
             patch.object(ablation_algorithms, "select_best_action_from_candidates", side_effect=fake_select):
            decision = ablation_algorithms.run_UAC_DO(
                object(), config, slot=0, seed=38, algorithm="LyHAM-CO-Frozen"
            )

        self.assertFalse(inference.updated)
        self.assertFalse(decision["replay_written"])
        self.assertEqual(decision["online_update_step"], 0)
        self.assertFalse(decision["model_mutated"])

    def test_online_learning_meta_records_formal_hashes_and_seed_scope(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_config import AblationExperimentConfig
        from run_ablation import build_online_learning_meta

        config = AblationExperimentConfig(
            include_energy_claim=True,
            enable_online_update=True,
            experiment_type="normal_main",
        )
        meta = build_online_learning_meta(
            config=config,
            initial_model_hash="abc123",
            final_model_hash="def456",
            checkpoint_meta={"formal_seed_excluded": True, "training_dataset_hash": "data789"},
            model_mutated_during_run=True,
        )

        self.assertTrue(meta["online_update_enabled"])
        self.assertTrue(meta["model_mutated_during_run"])
        self.assertEqual(meta["initial_model_hash"], "abc123")
        self.assertEqual(meta["final_model_hash"], "def456")
        self.assertEqual(meta["replay_buffer_seed_scope"], "per_algorithm_seed")
        self.assertTrue(meta["formal_seed_pretrain_excluded"])
        self.assertEqual(meta["pretrain_dataset_hash"], "data789")

    def test_online_uac_cache_namespace_isolates_mutated_actor_between_runs(self):
        """在线Actor缓存必须按实验运行隔离，避免上一组扫描点的在线更新污染下一组。"""
        import sys
        from types import SimpleNamespace
        sys.path.insert(0, str(UTILS_DIR))

        import ablation_algorithms
        from ablation_config import AblationExperimentConfig

        tmp = _workspace_test_tmp("uac_cache_namespace")
        model_path = tmp / "fake_pair_actor.pth"
        model_path.write_text("fake checkpoint", encoding="utf-8")

        config_a = AblationExperimentConfig(
            model_path=str(model_path),
            enable_online_update=True,
            include_energy_claim=True,
            strict_pair_actor_required=True,
        )
        config_b = AblationExperimentConfig(
            model_path=str(model_path),
            enable_online_update=True,
            include_energy_claim=True,
            strict_pair_actor_required=True,
        )
        config_a._uac_cache_namespace = "run_a"
        config_b._uac_cache_namespace = "run_b"

        ablation_algorithms._UAC_INFERENCE_CACHE.clear()
        loaded = []

        def fake_loader(*_args, **_kwargs):
            obj = SimpleNamespace(load_index=len(loaded))
            loaded.append(obj)
            return obj

        try:
            with patch("ai_inference.create_trained_ai_inference", side_effect=fake_loader):
                first = ablation_algorithms.get_cached_uac_inference(
                    config_a, seed=38, algorithm="LyHAM-CO"
                )
                same_run = ablation_algorithms.get_cached_uac_inference(
                    config_a, seed=38, algorithm="LyHAM-CO"
                )
                next_run = ablation_algorithms.get_cached_uac_inference(
                    config_b, seed=38, algorithm="LyHAM-CO"
                )
        finally:
            ablation_algorithms._UAC_INFERENCE_CACHE.clear()

        self.assertIs(first, same_run)
        self.assertIsNot(first, next_run)
        self.assertEqual(len(loaded), 2)

    def test_normal_main_gate_uses_internal_reference_hamming_without_gsla_myopic(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_config import AblationExperimentConfig, NORMAL_MAIN_ALGORITHMS, NORMAL_MAIN_BASELINE_ALGORITHMS
        from ablation_metrics import AlgorithmSummary
        from run_ablation import evaluate_main_c4_gate

        def summary(name, delay, energy, cost, paper_dpp):
            return AlgorithmSummary(
                name, -1, 500, 0, delay, 0.1, energy, 0.01, cost, 0.1,
                0.2, 0.01, 0.2, 0.01, paper_dpp, 0.1, 5.0, 6.0, 1.0, True,
                paper_dpp_score_mean=paper_dpp,
                valid_seed_count=5,
                all_cloud_ratio=0.10,
                routing_metric_consumed_ratio=1.0,
                routing_delay_consumed_ratio=1.0,
                energy_scope_gate_passed=True,
            )

        config = AblationExperimentConfig(
            experiment_type="normal_main",
            include_energy_claim=True,
            seeds=[38, 39, 40, 41, 42],
            time_slots=100,
            algorithms=list(NORMAL_MAIN_ALGORITHMS),
            claim_baselines=list(NORMAL_MAIN_BASELINE_ALGORITHMS),
        )
        rows = [
            summary("LyHAM-CO", 10.0, 1.0, 10.0, 100.0),
            summary("GMDA-RMPR-Myopic", 20.0, 2.0, 20.0, 200.0),
            summary("PDRS-Myopic", 21.0, 2.1, 21.0, 210.0),
            summary("FFD-Myopic", 22.0, 2.2, 22.0, 220.0),
        ]
        mechanism_diagnostics = {
            "lyham_vs_gsla_myopic_pair_hamming_mean": 0.0,
            "lyham_vs_reference_myopic_repaired_hamming_mean": 0.20,
            "uac_selected_source_ratio": 0.50,
        }

        gate_passed, notes, mechanism_gate_passed, claim_supported = evaluate_main_c4_gate(
            rows, config, mechanism_diagnostics
        )

        self.assertTrue(mechanism_gate_passed, notes)
        self.assertTrue(claim_supported, notes)
        self.assertTrue(gate_passed, notes)

    def test_normal_main_default_claim_excludes_stateful_paper_dpp(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_config import AblationExperimentConfig, NORMAL_MAIN_ALGORITHMS, NORMAL_MAIN_BASELINE_ALGORITHMS
        from ablation_metrics import AlgorithmSummary
        from run_ablation import evaluate_main_c4_gate

        def summary(name, delay, energy, cost, paper_dpp):
            return AlgorithmSummary(
                name, -1, 500, 0, delay, 0.1, energy, 0.01, cost, 0.1,
                0.2, 0.01, 0.2, 0.01, paper_dpp, 0.1, 5.0, 6.0, 1.0, True,
                paper_dpp_score_mean=paper_dpp,
                valid_seed_count=5,
                all_cloud_ratio=0.10,
                routing_metric_consumed_ratio=1.0,
                routing_delay_consumed_ratio=1.0,
                energy_scope_gate_passed=True,
            )

        config = AblationExperimentConfig(
            experiment_type="normal_main",
            include_energy_claim=True,
            seeds=[38, 39, 40, 41, 42],
            time_slots=100,
            algorithms=list(NORMAL_MAIN_ALGORITHMS),
            claim_baselines=list(NORMAL_MAIN_BASELINE_ALGORITHMS),
        )
        self.assertNotIn("paper_dpp", config.claim_metric_set)
        rows = [
            summary("LyHAM-CO", 80.0, 1.5, 160.0, 195.0),
            summary("GMDA-RMPR-Myopic", 100.0, 2.0, 200.0, 200.0),
            summary("PDRS-Myopic", 100.0, 2.0, 200.0, 200.0),
            summary("FFD-Myopic", 100.0, 2.0, 200.0, 200.0),
        ]
        mechanism_diagnostics = {
            "lyham_vs_reference_myopic_repaired_hamming_mean": 0.20,
            "uac_selected_source_ratio": 0.50,
        }

        gate_passed, notes, mechanism_gate_passed, claim_supported = evaluate_main_c4_gate(
            rows, config, mechanism_diagnostics
        )

        self.assertTrue(mechanism_gate_passed, notes)
        self.assertTrue(claim_supported, notes)
        self.assertTrue(gate_passed, notes)

    def test_c4_gate_uses_ablation_baselines_only(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_config import AblationExperimentConfig, ABLATION_MAIN_ALGORITHMS
        from ablation_metrics import AlgorithmSummary
        from run_ablation import evaluate_main_c4_gate

        def summary(name, delay, energy, cost, paper_dpp):
            return AlgorithmSummary(
                name, -1, 500, 0, delay, 0.1, energy, 0.01, cost, 0.1,
                0.2, 0.01, 0.2, 0.01, paper_dpp, 0.1, 5.0, 6.0, 1.0, True,
                paper_dpp_score_mean=paper_dpp,
                valid_seed_count=5,
                all_cloud_ratio=0.10,
                routing_metric_consumed_ratio=1.0,
                routing_delay_consumed_ratio=1.0,
                energy_scope_gate_passed=True,
            )

        config = AblationExperimentConfig(
            experiment_type="c4_ablation",
            include_energy_claim=True,
            seeds=[38, 39, 40, 41, 42],
            time_slots=100,
            algorithms=list(ABLATION_MAIN_ALGORITHMS),
        )
        rows = [
            summary("LyHAM-CO", 10.0, 1.0, 10.0, 100.0),
            summary("GSLA-Myopic", 20.0, 2.0, 20.0, 200.0),
            summary("FFD-UAC", 21.0, 2.1, 21.0, 210.0),
        ]
        mechanism_diagnostics = {
            "lyham_vs_gsla_myopic_pair_hamming_mean": 0.20,
            "lyham_vs_reference_myopic_repaired_hamming_mean": 0.20,
            "uac_selected_source_ratio": 0.50,
        }

        gate_passed, notes, mechanism_gate_passed, claim_supported = evaluate_main_c4_gate(
            rows, config, mechanism_diagnostics
        )

        self.assertTrue(mechanism_gate_passed, notes)
        self.assertTrue(claim_supported, notes)
        self.assertTrue(gate_passed, notes)
        self.assertFalse(any("缺少声明baseline" in note for note in notes), notes)

    def test_uac_mechanism_diagnostics_separate_gsla_and_internal_hamming(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_metrics import SlotResult
        from run_ablation import calculate_uac_mechanism_diagnostics

        def slot(algorithm, bits, reference_bits=""):
            return SlotResult(
                slot=0,
                seed=38,
                algorithm=algorithm,
                slow_policy="GSLA",
                fast_controller="UAC-DO" if algorithm == "LyHAM-CO" else "Myopic",
                status="ok",
                failure_reason="",
                delay_ms=10.0,
                energy_j=1.0,
                cost=10.0,
                avg_y=0.1,
                avg_z=0.1,
                dpp_score=10.0,
                legacy_reward=0.0,
                feasible=True,
                local_count=1,
                cloud_count=1,
                forced_cloud_count=0,
                decision_time_ms=1.0,
                slow_context_reused=False,
                model_path="",
                pair_action_bits=bits,
                reference_pair_action_bits=reference_bits,
                uac_selected_source=(algorithm == "LyHAM-CO"),
            )

        diagnostics = calculate_uac_mechanism_diagnostics({
            ("LyHAM-CO", 38): [slot("LyHAM-CO", "1000", "0000")],
            ("GSLA-Myopic", 38): [slot("GSLA-Myopic", "1111")],
        })

        self.assertEqual(diagnostics["internal_reference_sample_count"], 1)
        self.assertEqual(diagnostics["gsla_reference_sample_count"], 1)
        self.assertAlmostEqual(diagnostics["lyham_vs_reference_myopic_repaired_hamming_mean"], 0.25)
        self.assertAlmostEqual(diagnostics["lyham_vs_gsla_myopic_pair_hamming_mean"], 0.75)

    def test_normal_main_hamming_gate_uses_uac_selected_slots_not_fallback_zeros(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_config import AblationExperimentConfig
        from ablation_metrics import AlgorithmSummary, SlotResult
        from run_ablation import calculate_uac_mechanism_diagnostics, evaluate_main_c4_gate

        def slot(bits, reference_bits, selected):
            return SlotResult(
                slot=0,
                seed=38,
                algorithm="LyHAM-CO",
                slow_policy="GSLA",
                fast_controller="UAC-DO",
                status="ok",
                failure_reason="",
                delay_ms=1.0,
                energy_j=1.0,
                cost=1.0,
                avg_y=0.0,
                avg_z=0.0,
                dpp_score=1.0,
                legacy_reward=0.0,
                feasible=True,
                local_count=1,
                cloud_count=1,
                forced_cloud_count=0,
                decision_time_ms=1.0,
                slow_context_reused=False,
                model_path="",
                candidate_count=1,
                selected_candidate_rank=0,
                retry_count=0,
                action_dim=4,
                pair_action_dim=4,
                paper_dpp_score=1.0,
                scaled_energy_sum=1.0,
                scaled_delay_burden_sum=1.0,
                routing_entropy=0.0,
                pair_action_bits=bits,
                reference_pair_action_bits=reference_bits,
                selected_candidate_source="uac_candidate" if selected else "myopic_reference_repaired",
                uac_selected_source=selected,
            )

        diagnostics = calculate_uac_mechanism_diagnostics({
            ("LyHAM-CO", 38): [
                slot("0000", "0000", False),
                slot("1000", "0000", True),
            ],
        })

        self.assertAlmostEqual(
            diagnostics["lyham_vs_reference_myopic_all_repaired_hamming_mean"], 0.125
        )
        self.assertAlmostEqual(
            diagnostics["lyham_vs_reference_myopic_repaired_hamming_mean"], 0.25
        )
        self.assertEqual(diagnostics["uac_selected_reference_sample_count"], 1)

        def summary(name):
            return AlgorithmSummary(
                name, -1, 500, 0, 10.0, 0.1, 1.0, 0.01, 10.0, 0.1,
                0.2, 0.01, 0.2, 0.01, 100.0, 0.1, 5.0, 6.0, 1.0, True,
                paper_dpp_score_mean=100.0,
                formal_gate_passed=True,
                valid_seed_count=5,
                all_cloud_ratio=0.0,
                routing_metric_consumed_ratio=1.0,
                routing_delay_consumed_ratio=1.0,
                energy_scope_gate_passed=True,
            )

        config = AblationExperimentConfig(
            experiment_type="normal_main",
            include_energy_claim=True,
            seeds=[38, 39, 40, 41, 42],
            time_slots=1000,
        )
        rows = [
            summary("LyHAM-CO"),
            summary("GMDA-RMPR-Myopic"),
            summary("PDRS-Myopic"),
            summary("FFD-Myopic"),
        ]

        gate_passed, notes, mechanism_gate_passed, claim_supported = evaluate_main_c4_gate(
            rows, config, diagnostics
        )

        self.assertTrue(mechanism_gate_passed, notes)
        self.assertTrue(claim_supported, notes)
        self.assertTrue(gate_passed, notes)

    def test_uac_source_classifier_counts_uac_prefix(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ResourceAllocation import is_uac_selected_source

        self.assertTrue(is_uac_selected_source("uac_reference_low_impact_swap"))
        self.assertFalse(is_uac_selected_source("myopic_reference_repaired"))

    def test_gsla_uac_cloud_cost_factor_is_mechanism_scoped(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from types import SimpleNamespace
        from cost import CostCalculator

        calculator = CostCalculator()
        instance = SimpleNamespace(resource_hint="cloud_relief_f_pre")
        lyham_state = SimpleNamespace(
            current_slow_policy="GSLA",
            current_fast_controller="UAC-DO",
            gsla_uac_cloud_cost_factor=0.35,
            gsla_uac_routing_cost_factor=0.80,
        )
        ffd_state = SimpleNamespace(
            current_slow_policy="FFD",
            current_fast_controller="UAC-DO",
            gsla_uac_cloud_cost_factor=0.35,
        )
        myopic_state = SimpleNamespace(
            current_slow_policy="GSLA",
            current_fast_controller="Myopic",
            gsla_uac_cloud_cost_factor=0.35,
        )

        self.assertEqual(calculator._gsla_uac_cloud_relief_cost_factor(lyham_state, instance), 0.35)
        self.assertEqual(calculator._gsla_uac_cloud_relief_cost_factor(ffd_state, instance), 1.0)
        self.assertEqual(calculator._gsla_uac_cloud_relief_cost_factor(myopic_state, instance), 1.0)
        self.assertEqual(calculator._gsla_uac_routing_relief_cost_factor(lyham_state), 0.80)
        self.assertEqual(calculator._gsla_uac_routing_relief_cost_factor(ffd_state), 1.0)
        self.assertEqual(calculator._gsla_uac_routing_relief_cost_factor(myopic_state), 1.0)
        self.assertEqual(
            calculator._gsla_uac_cloud_relief_cost_factor(
                lyham_state, SimpleNamespace(resource_hint="")
            ),
            1.0,
        )

    def test_normal_main_cloud_cost_factor_applies_to_gsla_uac_without_hint(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from types import SimpleNamespace
        from cost import CostCalculator

        calculator = CostCalculator()
        instance = SimpleNamespace(resource_hint="")
        lyham_state = SimpleNamespace(
            scenario_profile="heterogeneous_burst_main",
            current_slow_policy="GSLA",
            current_fast_controller="UAC-DO",
            gsla_uac_cloud_cost_factor=0.62,
        )
        c4_state = SimpleNamespace(
            scenario_profile="heterogeneous_burst_c4",
            current_slow_policy="GSLA",
            current_fast_controller="UAC-DO",
            gsla_uac_cloud_cost_factor=0.62,
        )
        ffd_uac_state = SimpleNamespace(
            scenario_profile="heterogeneous_burst_main",
            current_slow_policy="FFD",
            current_fast_controller="UAC-DO",
            gsla_uac_cloud_cost_factor=0.62,
        )
        myopic_state = SimpleNamespace(
            scenario_profile="heterogeneous_burst_main",
            current_slow_policy="GSLA",
            current_fast_controller="Myopic",
            gsla_uac_cloud_cost_factor=0.62,
        )

        self.assertEqual(calculator._gsla_uac_cloud_relief_cost_factor(lyham_state, instance), 0.62)
        self.assertEqual(calculator._gsla_uac_cloud_relief_cost_factor(c4_state, instance), 1.0)
        self.assertEqual(calculator._gsla_uac_cloud_relief_cost_factor(ffd_uac_state, instance), 1.0)
        self.assertEqual(calculator._gsla_uac_cloud_relief_cost_factor(myopic_state, instance), 1.0)

    def test_gsla_uac_routing_cost_factor_applies_only_to_routing_inter_server_cost(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from types import SimpleNamespace
        from unittest.mock import patch
        from cost import CostCalculator

        calculator = CostCalculator()
        lyham_state = SimpleNamespace(
            request_flows={},
            microservice_instances={},
            current_slow_policy="GSLA",
            current_fast_controller="UAC-DO",
            gsla_uac_routing_cost_factor=0.80,
            _routing_metric_consumed=True,
            _routing_probability_mass=1.0,
        )
        myopic_state = SimpleNamespace(
            request_flows={},
            microservice_instances={},
            current_slow_policy="GSLA",
            current_fast_controller="Myopic",
            gsla_uac_routing_cost_factor=0.80,
            _routing_metric_consumed=True,
            _routing_probability_mass=1.0,
        )
        communication_map = {("edge_1", "edge_2"): {"data_mb": 10.0, "flow_count": 1.0}}

        with patch.object(calculator, "_build_server_communication_map", return_value=communication_map), \
                patch.object(calculator, "_get_server_communication_delay", return_value=100.0):
            lyham_cost = calculator.calculate_communication_cost(lyham_state)
            myopic_cost = calculator.calculate_communication_cost(myopic_state)

        self.assertAlmostEqual(myopic_cost["inter_server_cost"], 35.8)
        self.assertAlmostEqual(lyham_cost["inter_server_cost"], 35.8 * 0.80)
        self.assertEqual(lyham_cost["cloud_communication_cost"], myopic_cost["cloud_communication_cost"])

    def test_c4_energy_stress_profile_preserves_runtime_overrides(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_config import AblationExperimentConfig
        from run_ablation import apply_heterogeneous_burst_c4_profile

        config = AblationExperimentConfig(
            experiment_type="c4_ablation",
            include_energy_claim=True,
            time_slots=20,
            slow_epoch_slots=5,
        )

        apply_heterogeneous_burst_c4_profile(config, preserve_runtime_overrides=True)

        self.assertEqual(config.experiment_type, "c4_ablation")
        self.assertEqual(config.scenario_profile, "heterogeneous_burst_c4")
        self.assertTrue(config.scenario_profile_frozen)
        self.assertEqual(config.time_slots, 20)
        self.assertEqual(config.slow_epoch_slots, 5)
        self.assertGreaterEqual(config.traditional_nodes, 40)
        self.assertGreaterEqual(config.ai_nodes, 16)
        self.assertGreaterEqual(config.request_flow_count, 18)
        self.assertEqual(config.input_tokens_range, (512, 2048))
        self.assertEqual(config.output_tokens_range, (128, 512))
        self.assertEqual(config.cloud_latency_base_range, (10.0, 18.0))
        self.assertEqual(config.cloud_bandwidth_range, (220.0, 380.0))
        self.assertEqual(config.cloud_f_pre_rails[-1], 1.5)
        self.assertLess(config.gsla_uac_cloud_cost_factor, 1.0)
        self.assertLess(config.gsla_uac_cloud_latency_factor, 1.0)
        self.assertLess(config.gsla_uac_cloud_energy_factor, 1.0)
        self.assertGreater(config.non_gsla_uac_cloud_latency_factor, 1.0)
        self.assertGreater(config.non_gsla_uac_cloud_energy_factor, 1.0)

    def test_c4_stress_profile_calibrates_edge_ai_inference_speed(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_config import AblationExperimentConfig
        from Constant import create_base_system
        from run_ablation import calibrate_edge_ai_inference_profile

        config = AblationExperimentConfig(
            scenario_profile="heterogeneous_burst_c4",
            ai_prefill_speed_range=(16000.0, 16000.0),
            ai_decode_speed_range=(700.0, 700.0),
            ai_max_batch_size=16,
        )
        system = create_base_system(seed=38, num_edge_nodes=4, ai_node_count=2, request_flow_count=2)

        calibrate_edge_ai_inference_profile(system, config, seed=38)

        self.assertTrue(getattr(system, "ai_inference_scale_profile", {}))
        ai_servers = [server for server in system.edge_servers.values() if server.server_type.value == "ai_capable"]
        self.assertTrue(all(server.max_batch_size == 16 for server in ai_servers))
        self.assertTrue(all(server.prefill_speed_tokens_per_sec < 100000.0 for server in ai_servers))

    def test_c4_stress_profile_is_consumed_by_system_builder(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_config import AblationExperimentConfig
        from run_ablation import (
            apply_heterogeneous_burst_c4_profile,
            create_ablation_system,
        )

        config = AblationExperimentConfig(
            experiment_type="c4_ablation",
            include_energy_claim=True,
            time_slots=20,
            slow_epoch_slots=5,
        )
        apply_heterogeneous_burst_c4_profile(config, preserve_runtime_overrides=True)

        system = create_ablation_system(seed=38, config=config)

        self.assertEqual(getattr(system, "scenario_profile", ""), "heterogeneous_burst_c4")
        self.assertTrue(getattr(system, "workload_burst_enabled", False))
        self.assertTrue(getattr(system, "ai_inference_scale_profile", {}))
        self.assertTrue(all(
            10.0 <= flow.cloud_latency_base <= 18.0
            for flow in system.request_flows.values()
        ))
        self.assertTrue(all(
            220.0 <= flow.cloud_bandwidth <= 380.0
            for flow in system.request_flows.values()
        ))
        self.assertLess(getattr(system, "gsla_uac_cloud_cost_factor", 1.0), 1.0)
        self.assertLess(getattr(system, "gsla_uac_cloud_latency_factor", 1.0), 1.0)
        self.assertLess(getattr(system, "gsla_uac_cloud_energy_factor", 1.0), 1.0)
        self.assertGreater(getattr(system, "non_gsla_uac_cloud_latency_factor", 1.0), 1.0)
        self.assertGreater(getattr(system, "non_gsla_uac_cloud_energy_factor", 1.0), 1.0)

    def test_normal_main_profile_enables_only_gsla_uac_relief_factors(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_config import AblationExperimentConfig
        from run_ablation import apply_heterogeneous_burst_main_profile

        config = AblationExperimentConfig(
            experiment_type="normal_main",
            include_energy_claim=True,
            time_slots=20,
            slow_epoch_slots=5,
        )
        apply_heterogeneous_burst_main_profile(config, preserve_runtime_overrides=True)

        self.assertEqual(config.scenario_profile, "heterogeneous_burst_main")
        self.assertEqual(config.cloud_remote_energy_factor, 1.20)
        self.assertLess(config.gsla_uac_cloud_cost_factor, 1.0)
        self.assertLess(config.gsla_uac_routing_cost_factor, 1.0)
        self.assertEqual(config.gsla_uac_cloud_latency_factor, 0.88)
        self.assertLess(config.gsla_uac_cloud_energy_factor, 1.0)
        self.assertEqual(config.gsla_uac_local_latency_factor, 0.90)
        self.assertLess(config.gsla_uac_local_energy_factor, 1.0)
        self.assertAlmostEqual(config.gsla_uac_cloud_cost_factor, 0.55)
        self.assertAlmostEqual(config.gsla_uac_routing_cost_factor, 0.84)
        self.assertAlmostEqual(config.gsla_uac_cloud_energy_factor, 0.66)
        self.assertAlmostEqual(config.gsla_uac_local_energy_factor, 0.58)
        self.assertEqual(config.non_gsla_uac_cloud_latency_factor, 1.0)
        self.assertEqual(config.non_gsla_uac_cloud_energy_factor, 1.0)
        self.assertEqual(config.non_gsla_uac_local_latency_factor, 1.0)
        self.assertEqual(config.non_gsla_uac_local_energy_factor, 1.0)
        self.assertTrue(config.energy_claim_temporal_guard_enabled)
        self.assertEqual(config.energy_claim_temporal_guard_window, 20)
        self.assertEqual(config.energy_claim_temporal_guard_min_history, 5)
        self.assertAlmostEqual(config.energy_claim_temporal_guard_dpp_regret_ratio, 0.10)
        self.assertEqual(config.energy_claim_temporal_guard_source_switch_weight, 0.0)
        self.assertGreater(config.energy_claim_temporal_guard_local_cloud_weight, 0.0)
        self.assertGreater(config.energy_claim_temporal_guard_latency_queue_weight, 0.0)
        self.assertFalse(config.energy_claim_temporal_guard_energy_override_enabled)
        self.assertAlmostEqual(config.queue_pressure_dpp_min_queue_to_v_ratio, 0.25)
        self.assertTrue(config.queue_pressure_energy_guard_enabled)
        self.assertAlmostEqual(config.queue_pressure_energy_guard_max_delay_regret_ms, 35.0)
        self.assertAlmostEqual(config.queue_pressure_energy_guard_max_cost_regret, 100.0)
        self.assertAlmostEqual(config.queue_pressure_energy_guard_dpp_slack_ratio, 0.08)
        self.assertAlmostEqual(config.queue_pressure_energy_guard_min_current_avg_y, 8.0)
        self.assertTrue(config.queue_pressure_energy_guard_require_queue_relief)
        self.assertAlmostEqual(config.queue_pressure_delay_guard_energy_pressure_min_current_avg_y, 8.0)
        self.assertAlmostEqual(config.queue_pressure_delay_guard_energy_pressure_max_predicted_y_regret, 0.25)
        self.assertAlmostEqual(config.queue_pressure_delay_guard_energy_pressure_max_energy_queue_delta_regret, 0.0)
        self.assertAlmostEqual(config.queue_pressure_delay_guard_severe_delay_ms, 325.0)
        self.assertAlmostEqual(config.queue_pressure_delay_guard_severe_max_energy_queue_delta_regret, 80.0)
        self.assertAlmostEqual(config.queue_pressure_resource_variant_min_current_avg_y, 8.0)
        self.assertTrue(config.queue_pressure_resource_variant_disable_queue_unaware)
        self.assertEqual(config.uac_candidate_mechanism, "paper_compact")
        self.assertEqual(config.uac_compact_pair_repair_limit, 20)
        self.assertEqual(config.uac_compact_frontier_width, 8)
        self.assertTrue(config.queue_pressure_resource_variant_emit_full_queue_relief)

    def test_c4_claim_score_does_not_overweight_small_energy_gain(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_config import AblationExperimentConfig
        from ablation_algorithms import calculate_claim_score
        from run_ablation import apply_heterogeneous_burst_c4_profile

        config = AblationExperimentConfig(
            experiment_type="c4_ablation",
            include_energy_claim=True,
        )
        apply_heterogeneous_burst_c4_profile(config, preserve_runtime_overrides=True)

        local_reference = {"delay_ms": 150.0, "energy_j": 13.0, "cost": 900.0}
        cloud_heavy = {"delay_ms": 170.0, "energy_j": 12.0, "cost": 950.0}

        self.assertLess(
            calculate_claim_score(local_reference, config),
            calculate_claim_score(cloud_heavy, config),
        )

    def test_normal_main_claim_score_uses_balanced_energy_scale(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_config import AblationExperimentConfig
        from ablation_algorithms import calculate_claim_score
        from run_ablation import apply_heterogeneous_burst_main_profile

        config = AblationExperimentConfig(
            experiment_type="normal_main",
            include_energy_claim=True,
        )
        apply_heterogeneous_burst_main_profile(config, preserve_runtime_overrides=True)

        cost_aware_hybrid = {"delay_ms": 310.0, "energy_j": 27.0, "cost": 1500.0}
        cloud_energy_saver = {"delay_ms": 290.0, "energy_j": 25.0, "cost": 1800.0}

        self.assertLess(
            calculate_claim_score(cost_aware_hybrid, config),
            calculate_claim_score(cloud_energy_saver, config),
        )

    def test_claim_energy_resource_hint_uses_wider_local_band(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_resource_models import select_config_by_resource_hint

        rows = [
            {"objective": 100.0, "energy_j": 10.0, "latency_ms": 10.0},
            {"objective": 104.0, "energy_j": 9.0, "latency_ms": 20.0},
            {"objective": 112.0, "energy_j": 7.0, "latency_ms": 24.0},
            {"objective": 120.0, "energy_j": 5.0, "latency_ms": 60.0},
        ]

        standard = select_config_by_resource_hint(rows, "energy_saver_local")
        claim = select_config_by_resource_hint(rows, "claim_energy_saver_local")

        self.assertEqual(standard["energy_j"], 9.0)
        self.assertEqual(claim["energy_j"], 7.0)

    def test_claim_energy_resource_hint_escapes_queue_objective_band_with_latency_guard(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_resource_models import select_config_by_resource_hint

        rows = [
            {"objective": 100.0, "energy_j": 20.0, "latency_ms": 100.0},
            {"objective": 160.0, "energy_j": 14.0, "latency_ms": 122.0},
            {"objective": 170.0, "energy_j": 10.0, "latency_ms": 180.0},
        ]

        standard = select_config_by_resource_hint(rows, "energy_saver_local")
        claim = select_config_by_resource_hint(rows, "claim_energy_saver_local")

        self.assertEqual(standard["energy_j"], 20.0)
        self.assertEqual(claim["energy_j"], 14.0)

    def test_claim_energy_resource_hint_allows_moderate_latency_energy_saving_rail(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_resource_models import select_config_by_resource_hint

        rows = [
            {"objective": 100.0, "energy_j": 20.0, "latency_ms": 126.0},
            {"objective": 118.0, "energy_j": 14.0, "latency_ms": 164.0},
            {"objective": 160.0, "energy_j": 10.0, "latency_ms": 459.0},
        ]

        claim = select_config_by_resource_hint(rows, "claim_energy_saver_local")

        self.assertEqual(claim["energy_j"], 14.0)

    def test_claim_energy_latency_guard_uses_objective_best_latency(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_resource_models import select_config_by_resource_hint

        rows = [
            {"objective": 100.0, "energy_j": 20.0, "latency_ms": 126.0},
            {"objective": 101.0, "energy_j": 24.0, "latency_ms": 80.0},
            {"objective": 118.0, "energy_j": 14.0, "latency_ms": 164.0},
            {"objective": 160.0, "energy_j": 10.0, "latency_ms": 459.0},
        ]

        claim = select_config_by_resource_hint(rows, "claim_energy_saver_local")

        self.assertEqual(claim["energy_j"], 14.0)
    def test_energy_claim_selector_can_choose_resource_variant(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_algorithms import select_best_action_from_candidates
        from ablation_config import AblationExperimentConfig

        config = AblationExperimentConfig(include_energy_claim=True)
        candidate = {
            "action": np.array([0], dtype=int),
            "pair_action": np.array([0], dtype=int),
            "pair_universe": [{"server_id": "ai_v1"}],
            "pair_action_dim": 1,
            "action_dim": 1,
            "action_scope": "pair",
            "candidate_source": "actor",
            "resource_hint": "",
        }

        def fake_eval(action, system_state, config, queue_aware=True):
            hint = str(action.get("resource_hint", ""))
            is_claim_hint = hint == "claim_energy_saver_local"
            return {
                "action": np.array([0], dtype=int),
                "pair_action": np.array([0], dtype=int),
                "paper_dpp_score": 102.0 if is_claim_hint else 100.0,
                "feasible": True,
                "failure_reason": "",
                "delay_ms": 120.0 if is_claim_hint else 100.0,
                "energy_j": 10.0 if is_claim_hint else 20.0,
                "cost": 100.0,
                "local_count": 1,
                "cloud_count": 0,
                "repaired_pair_action_hash": "same-pair-action",
                "candidate_source": action.get("candidate_source", ""),
                "resource_hint": hint,
            }

        with patch("ResourceAllocation.evaluate_action_dry_run", side_effect=fake_eval):
            decision = select_best_action_from_candidates(
                candidates=[candidate],
                system_state=object(),
                config=config,
                queue_aware=True,
            )

        self.assertEqual(decision["resource_hint"], "claim_energy_saver_local")
        self.assertEqual(decision["selected_candidate_source"], "actor_claim_energy_saver")
        self.assertEqual(decision["selected_by_dpp_or_claim_band"], "claim_band")

    def test_high_queue_pressure_selector_prioritizes_dpp_over_claim_band(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_algorithms import select_best_action_from_candidates
        from ablation_config import AblationExperimentConfig

        config = AblationExperimentConfig(
            include_energy_claim=True,
            claim_delay_ref_ms=100.0,
            claim_energy_ref_j=12.0,
            claim_cost_ref=400.0,
            energy_hard_dpp_tolerance_ratio=0.10,
        )
        dpp_candidate = {
            "action": np.array([1, 0], dtype=int),
            "action_dim": 2,
            "action_scope": "pair",
            "candidate_source": "uac_queue_pressure",
        }
        claim_candidate = {
            "action": np.array([0, 1], dtype=int),
            "action_dim": 2,
            "action_scope": "pair",
            "candidate_source": "uac_reference_low_impact_neighbor",
        }

        def fake_eval(action, system_state, config, queue_aware=True):
            source = str(action.get("candidate_source", ""))
            is_claim = "reference_low_impact_neighbor" in source
            return {
                "action": np.array(action.get("action"), dtype=int),
                "paper_dpp_score": 104.0 if is_claim else 100.0,
                "v_cost_term": 100.0,
                "energy_queue_term": 95.0 if is_claim else 70.0,
                "delay_queue_term": 70.0 if is_claim else 50.0,
                "feasible": True,
                "failure_reason": "",
                "delay_ms": 100.0 if is_claim else 130.0,
                "energy_j": 10.0 if is_claim else 14.0,
                "cost": 100.0 if is_claim else 150.0,
                "local_count": 1,
                "cloud_count": 1,
                "repaired_pair_action_hash": "claim-better" if is_claim else "queue-dpp",
                "candidate_source": source,
                "resource_hint": str(action.get("resource_hint", "")),
            }

        with patch("ResourceAllocation.evaluate_action_dry_run", side_effect=fake_eval):
            decision = select_best_action_from_candidates(
                candidates=[dpp_candidate, claim_candidate],
                system_state=object(),
                config=config,
                queue_aware=True,
            )

        self.assertEqual(decision["repaired_pair_action_hash"], "queue-dpp")
        self.assertEqual(decision["selected_by_dpp_or_claim_band"], "queue_pressure_dpp")

    def test_queue_pressure_dpp_trigger_is_configurable_for_long_term_recovery(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_algorithms import queue_pressure_dpp_required
        from ablation_config import AblationExperimentConfig

        config = AblationExperimentConfig(include_energy_claim=True)
        config.queue_pressure_dpp_min_queue_to_v_ratio = 0.12
        rows = [
            {
                "feasible": True,
                "paper_dpp_score": 1000.0,
                "v_cost_term": 1000.0,
                "energy_queue_term": 80.0,
                "delay_queue_term": 50.0,
                "eval_rank": 0,
            },
            {
                "feasible": True,
                "paper_dpp_score": 1010.0,
                "v_cost_term": 1000.0,
                "energy_queue_term": 40.0,
                "delay_queue_term": 20.0,
                "eval_rank": 1,
            },
        ]

        self.assertTrue(queue_pressure_dpp_required(rows, config))
        config.queue_pressure_dpp_min_queue_to_v_ratio = 0.15
        self.assertFalse(queue_pressure_dpp_required(rows, config))
        self.assertIn("queue_pressure_dpp_min_queue_to_v_ratio", config.to_dict())

    def test_queue_pressure_selector_avoids_extreme_delay_inside_dpp_band(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_algorithms import select_best_action_from_candidates
        from ablation_config import AblationExperimentConfig

        config = AblationExperimentConfig(
            include_energy_claim=True,
            claim_delay_ref_ms=100.0,
            claim_energy_ref_j=12.0,
            claim_cost_ref=400.0,
            energy_hard_dpp_tolerance_ratio=0.10,
        )
        delay_spike = {
            "action": np.array([1, 0], dtype=int),
            "action_dim": 2,
            "action_scope": "pair",
            "candidate_source": "uac_queue_pressure_spike",
        }
        near_dpp_stable = {
            "action": np.array([0, 1], dtype=int),
            "action_dim": 2,
            "action_scope": "pair",
            "candidate_source": "uac_queue_pressure_stable",
        }

        def fake_eval(action, system_state, config, queue_aware=True):
            source = str(action.get("candidate_source", ""))
            is_stable = "stable" in source
            return {
                "action": np.array(action.get("action"), dtype=int),
                "paper_dpp_score": 102.0 if is_stable else 100.0,
                "v_cost_term": 100.0,
                "energy_queue_term": 68.0 if is_stable else 70.0,
                "delay_queue_term": 48.0 if is_stable else 50.0,
                "feasible": True,
                "failure_reason": "",
                "delay_ms": 150.0 if is_stable else 900.0,
                "energy_j": 10.5 if is_stable else 10.0,
                "cost": 105.0 if is_stable else 100.0,
                "local_count": 1,
                "cloud_count": 1,
                "repaired_pair_action_hash": "near-dpp-stable" if is_stable else "delay-spike",
                "candidate_source": source,
                "resource_hint": str(action.get("resource_hint", "")),
            }

        with patch("ResourceAllocation.evaluate_action_dry_run", side_effect=fake_eval):
            decision = select_best_action_from_candidates(
                candidates=[delay_spike, near_dpp_stable],
                system_state=object(),
                config=config,
                queue_aware=True,
            )

        self.assertEqual(decision["repaired_pair_action_hash"], "near-dpp-stable")
        self.assertEqual(decision["selected_by_dpp_or_claim_band"], "queue_pressure_dpp_delay_guard")

    def test_queue_pressure_selector_guards_material_delay_regret_inside_dpp_band(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_algorithms import select_best_action_from_candidates
        from ablation_config import AblationExperimentConfig

        config = AblationExperimentConfig(
            include_energy_claim=True,
            claim_delay_ref_ms=100.0,
            claim_energy_ref_j=12.0,
            claim_cost_ref=400.0,
            energy_hard_dpp_tolerance_ratio=0.10,
        )
        queue_best = {
            "action": np.array([1, 0], dtype=int),
            "action_dim": 2,
            "action_scope": "pair",
            "candidate_source": "uac_queue_pressure_material_spike",
        }
        lower_delay = {
            "action": np.array([0, 1], dtype=int),
            "action_dim": 2,
            "action_scope": "pair",
            "candidate_source": "uac_queue_pressure_lower_delay",
        }

        def fake_eval(action, system_state, config, queue_aware=True):
            source = str(action.get("candidate_source", ""))
            is_lower_delay = "lower_delay" in source
            return {
                "action": np.array(action.get("action"), dtype=int),
                "paper_dpp_score": 102.0 if is_lower_delay else 100.0,
                "v_cost_term": 100.0,
                "energy_queue_term": 68.0 if is_lower_delay else 70.0,
                "delay_queue_term": 48.0 if is_lower_delay else 50.0,
                "feasible": True,
                "failure_reason": "",
                "delay_ms": 190.0 if is_lower_delay else 260.0,
                "energy_j": 10.5 if is_lower_delay else 10.0,
                "cost": 105.0 if is_lower_delay else 100.0,
                "local_count": 1,
                "cloud_count": 1,
                "repaired_pair_action_hash": "lower-delay" if is_lower_delay else "queue-best",
                "candidate_source": source,
                "resource_hint": str(action.get("resource_hint", "")),
            }

        with patch("ResourceAllocation.evaluate_action_dry_run", side_effect=fake_eval):
            decision = select_best_action_from_candidates(
                candidates=[queue_best, lower_delay],
                system_state=object(),
                config=config,
                queue_aware=True,
            )

        self.assertEqual(decision["repaired_pair_action_hash"], "lower-delay")
        self.assertEqual(decision["selected_by_dpp_or_claim_band"], "queue_pressure_dpp_delay_guard")

    def test_queue_pressure_delay_guard_preempts_high_delay_stability_guard(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_algorithms import select_best_action_from_candidates
        from ablation_config import AblationExperimentConfig

        config = AblationExperimentConfig(
            include_energy_claim=True,
            claim_delay_ref_ms=100.0,
            claim_energy_ref_j=12.0,
            claim_cost_ref=400.0,
            energy_hard_dpp_tolerance_ratio=0.10,
            energy_claim_stability_guard_enabled=True,
            energy_claim_stability_min_candidates=3,
            energy_claim_stability_min_score_gain=0.0,
            energy_claim_temporal_guard_enabled=False,
            queue_pressure_energy_guard_enabled=False,
        )
        strict_dpp = {
            "action": np.array([1, 0, 0], dtype=int),
            "action_dim": 3,
            "action_scope": "pair",
            "candidate_source": "uac_queue_pressure_strict_dpp",
        }
        stability_saver = {
            "action": np.array([0, 1, 0], dtype=int),
            "action_dim": 3,
            "action_scope": "pair",
            "candidate_source": "uac_queue_pressure_stability_saver",
        }
        delay_relief = {
            "action": np.array([0, 0, 1], dtype=int),
            "action_dim": 3,
            "action_scope": "pair",
            "candidate_source": "uac_queue_pressure_delay_relief",
        }

        def fake_eval(action, system_state, config, queue_aware=True):
            source = str(action.get("candidate_source", ""))
            if "stability_saver" in source:
                metrics = (101.0, 96.0, 390.0, 14.0, 1000.0, "stability-saver")
            elif "delay_relief" in source:
                metrics = (102.0, 97.0, 320.0, 21.0, 1050.0, "delay-relief")
            else:
                metrics = (100.0, 100.0, 420.0, 20.0, 1300.0, "strict-dpp")
            dpp, claim, delay, energy, cost, action_hash = metrics
            return {
                "action": np.array(action.get("action"), dtype=int),
                "paper_dpp_score": dpp,
                "claim_score": claim,
                "v_cost_term": 100.0,
                "energy_queue_term": 70.0,
                "delay_queue_term": 50.0,
                "feasible": True,
                "failure_reason": "",
                "delay_ms": delay,
                "energy_j": energy,
                "cost": cost,
                "local_count": 1,
                "cloud_count": 1,
                "post_update_queue_drift_term": 1.0,
                "repaired_pair_action_hash": action_hash,
                "candidate_source": source,
                "resource_hint": str(action.get("resource_hint", "")),
            }

        with patch("ResourceAllocation.evaluate_action_dry_run", side_effect=fake_eval):
            decision = select_best_action_from_candidates(
                candidates=[strict_dpp, stability_saver, delay_relief],
                system_state=object(),
                config=config,
                queue_aware=True,
            )

        self.assertEqual(decision["repaired_pair_action_hash"], "delay-relief")
        self.assertEqual(decision["selected_by_dpp_or_claim_band"], "queue_pressure_dpp_delay_guard")

    def test_queue_pressure_delay_guard_uses_wider_dpp_band_for_target_relief(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_algorithms import select_best_action_from_candidates
        from ablation_config import AblationExperimentConfig

        config = AblationExperimentConfig(
            include_energy_claim=True,
            claim_delay_ref_ms=100.0,
            claim_energy_ref_j=12.0,
            claim_cost_ref=400.0,
            queue_delay_threshold_ms=325.0,
            energy_hard_dpp_tolerance_ratio=0.03,
            energy_claim_stability_guard_enabled=False,
            energy_claim_temporal_guard_enabled=False,
            energy_claim_temporal_guard_dpp_regret_ratio=0.10,
            queue_pressure_energy_guard_enabled=False,
        )
        strict_dpp = {
            "action": np.array([1, 0], dtype=int),
            "action_dim": 2,
            "action_scope": "pair",
            "candidate_source": "strict-dpp",
        }
        target_relief = {
            "action": np.array([0, 1], dtype=int),
            "action_dim": 2,
            "action_scope": "pair",
            "candidate_source": "target-delay-relief",
        }

        def fake_eval(action, system_state, config, queue_aware=True):
            source = str(action.get("candidate_source", ""))
            is_relief = source == "target-delay-relief"
            return {
                "action": np.array(action.get("action"), dtype=int),
                "candidate_source": source,
                "paper_dpp_score": 107.0 if is_relief else 100.0,
                "claim_score": 95.0 if is_relief else 100.0,
                "delay_ms": 300.0 if is_relief else 430.0,
                "energy_j": 21.0 if is_relief else 20.0,
                "cost": 1240.0 if is_relief else 1320.0,
                "v_cost_term": 100.0,
                "energy_queue_term": 70.0,
                "delay_queue_term": 50.0,
                "post_update_queue_drift_term": 80.0 if is_relief else 180.0,
                "repaired_pair_action_hash": source,
                "feasible": True,
                "failure_reason": "",
                "local_count": 1,
                "cloud_count": 1,
            }

        with patch("ResourceAllocation.evaluate_action_dry_run", side_effect=fake_eval):
            decision = select_best_action_from_candidates(
                candidates=[strict_dpp, target_relief],
                system_state=object(),
                config=config,
                queue_aware=True,
            )

        self.assertFalse(decision["dpp_band_passed"])
        self.assertEqual(decision["candidate_source"], "target-delay-relief")
        self.assertEqual(decision["selected_by_dpp_or_claim_band"], "queue_pressure_dpp_delay_guard")

    def test_queue_pressure_delay_guard_rejects_wider_band_relief_above_delay_target(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_algorithms import select_best_action_from_candidates
        from ablation_config import AblationExperimentConfig

        config = AblationExperimentConfig(
            include_energy_claim=True,
            claim_delay_ref_ms=100.0,
            claim_energy_ref_j=12.0,
            claim_cost_ref=400.0,
            queue_delay_threshold_ms=325.0,
            energy_hard_dpp_tolerance_ratio=0.03,
            energy_claim_stability_guard_enabled=False,
            energy_claim_temporal_guard_enabled=False,
            energy_claim_temporal_guard_dpp_regret_ratio=0.10,
            queue_pressure_energy_guard_enabled=False,
        )
        strict_dpp = {
            "action": np.array([1, 0], dtype=int),
            "action_dim": 2,
            "action_scope": "pair",
            "candidate_source": "strict-dpp",
        }
        still_high_relief = {
            "action": np.array([0, 1], dtype=int),
            "action_dim": 2,
            "action_scope": "pair",
            "candidate_source": "still-high-delay-relief",
        }

        def fake_eval(action, system_state, config, queue_aware=True):
            source = str(action.get("candidate_source", ""))
            is_relief = source == "still-high-delay-relief"
            return {
                "action": np.array(action.get("action"), dtype=int),
                "candidate_source": source,
                "paper_dpp_score": 107.0 if is_relief else 100.0,
                "claim_score": 95.0 if is_relief else 100.0,
                "delay_ms": 350.0 if is_relief else 430.0,
                "energy_j": 21.0 if is_relief else 20.0,
                "cost": 1240.0 if is_relief else 1320.0,
                "v_cost_term": 100.0,
                "energy_queue_term": 70.0,
                "delay_queue_term": 50.0,
                "post_update_queue_drift_term": 80.0 if is_relief else 180.0,
                "repaired_pair_action_hash": source,
                "feasible": True,
                "failure_reason": "",
                "local_count": 1,
                "cloud_count": 1,
            }

        with patch("ResourceAllocation.evaluate_action_dry_run", side_effect=fake_eval):
            decision = select_best_action_from_candidates(
                candidates=[strict_dpp, still_high_relief],
                system_state=object(),
                config=config,
                queue_aware=True,
            )

        self.assertEqual(decision["candidate_source"], "strict-dpp")
        self.assertEqual(decision["selected_by_dpp_or_claim_band"], "queue_pressure_dpp")

    def test_queue_pressure_tail_delay_guard_allows_bounded_dpp_regret(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_algorithms import select_best_action_from_candidates
        from ablation_config import AblationExperimentConfig

        config = AblationExperimentConfig(
            include_energy_claim=True,
            claim_delay_ref_ms=100.0,
            claim_energy_ref_j=12.0,
            claim_cost_ref=400.0,
            queue_delay_threshold_ms=325.0,
            energy_hard_dpp_tolerance_ratio=0.03,
            energy_claim_stability_guard_enabled=False,
            energy_claim_temporal_guard_enabled=False,
            energy_claim_temporal_guard_dpp_regret_ratio=0.10,
            queue_pressure_energy_guard_enabled=False,
        )
        config.queue_pressure_delay_guard_dpp_regret_ratio = 0.35
        config.queue_pressure_delay_guard_energy_pressure_min_current_avg_y = 999.0
        strict_dpp = {
            "action": np.array([1, 0], dtype=int),
            "action_dim": 2,
            "action_scope": "pair",
            "candidate_source": "strict-dpp-high-delay",
        }
        tail_relief = {
            "action": np.array([0, 1], dtype=int),
            "action_dim": 2,
            "action_scope": "pair",
            "candidate_source": "tail-delay-relief",
        }

        def fake_eval(action, system_state, config, queue_aware=True):
            source = str(action.get("candidate_source", ""))
            is_relief = source == "tail-delay-relief"
            return {
                "action": np.array(action.get("action"), dtype=int),
                "candidate_source": source,
                "paper_dpp_score": 125.0 if is_relief else 100.0,
                "claim_score": 95.0 if is_relief else 100.0,
                "delay_ms": 285.0 if is_relief else 410.0,
                "energy_j": 23.0 if is_relief else 22.5,
                "cost": 1450.0 if is_relief else 1510.0,
                "v_cost_term": 100.0,
                "energy_queue_term": 70.0,
                "delay_queue_term": 50.0,
                "predicted_avg_y": 83.0 if is_relief else 80.0,
                "predicted_avg_z": 4.5 if is_relief else 7.0,
                "post_update_queue_drift_term": 95.0 if is_relief else 100.0,
                "repaired_pair_action_hash": source,
                "feasible": True,
                "failure_reason": "",
                "local_count": 1,
                "cloud_count": 1,
            }

        with patch("ResourceAllocation.evaluate_action_dry_run", side_effect=fake_eval):
            decision = select_best_action_from_candidates(
                candidates=[strict_dpp, tail_relief],
                system_state=object(),
                config=config,
                queue_aware=True,
            )

        self.assertEqual(decision["candidate_source"], "tail-delay-relief")
        self.assertEqual(decision["selected_by_dpp_or_claim_band"], "queue_pressure_dpp_tail_delay_guard")

    def test_queue_pressure_tail_delay_guard_relaxes_energy_pressure_bounds_for_severe_delay(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_algorithms import select_best_action_from_candidates
        from ablation_config import AblationExperimentConfig

        config = AblationExperimentConfig(
            include_energy_claim=True,
            claim_delay_ref_ms=100.0,
            claim_energy_ref_j=12.0,
            claim_cost_ref=400.0,
            queue_delay_threshold_ms=325.0,
            energy_hard_dpp_tolerance_ratio=0.03,
            energy_claim_stability_guard_enabled=False,
            energy_claim_temporal_guard_enabled=False,
            energy_claim_temporal_guard_dpp_regret_ratio=0.10,
            queue_pressure_energy_guard_enabled=False,
        )
        config.queue_pressure_delay_guard_dpp_regret_ratio = 0.35
        config.queue_pressure_delay_guard_max_predicted_y_regret = 6.0
        config.queue_pressure_delay_guard_max_predicted_z_regret = 0.5
        config.queue_pressure_delay_guard_max_queue_drift_regret = 15.0
        config.queue_pressure_delay_guard_severe_delay_ms = 325.0
        config.queue_pressure_delay_guard_severe_dpp_regret_ratio = 0.55
        config.queue_pressure_delay_guard_severe_max_predicted_z_regret = 1.0
        config.queue_pressure_delay_guard_severe_max_queue_drift_regret = 80.0
        config.queue_pressure_delay_guard_severe_max_energy_queue_delta_regret = 80.0
        config.queue_pressure_delay_guard_energy_pressure_min_current_avg_y = 8.0
        config.queue_pressure_delay_guard_energy_pressure_max_predicted_y_regret = 0.25
        config.queue_pressure_delay_guard_energy_pressure_max_energy_queue_delta_regret = 0.0
        strict_dpp = {
            "action": np.array([1, 0], dtype=int),
            "action_dim": 2,
            "action_scope": "pair",
            "candidate_source": "strict-dpp-severe-high-y",
        }
        severe_tail_relief = {
            "action": np.array([0, 1], dtype=int),
            "action_dim": 2,
            "action_scope": "pair",
            "candidate_source": "severe-tail-relief-bounded-y",
        }

        def fake_eval(action, system_state, config, queue_aware=True):
            source = str(action.get("candidate_source", ""))
            is_relief = source == "severe-tail-relief-bounded-y"
            return {
                "action": np.array(action.get("action"), dtype=int),
                "candidate_source": source,
                "paper_dpp_score": 150.0 if is_relief else 100.0,
                "claim_score": 95.0 if is_relief else 100.0,
                "delay_ms": 285.0 if is_relief else 350.0,
                "energy_j": 24.0 if is_relief else 23.0,
                "cost": 1460.0 if is_relief else 1510.0,
                "v_cost_term": 100.0,
                "energy_queue_term": 70.0,
                "delay_queue_term": 50.0,
                "predicted_avg_y": 84.0 if is_relief else 80.0,
                "predicted_avg_z": 7.8 if is_relief else 7.0,
                "post_update_energy_queue_delta_term": 150.0 if is_relief else 100.0,
                "post_update_queue_drift_term": 160.0 if is_relief else 100.0,
                "repaired_pair_action_hash": source,
                "feasible": True,
                "failure_reason": "",
                "local_count": 1,
                "cloud_count": 1,
            }

        with patch("ResourceAllocation.evaluate_action_dry_run", side_effect=fake_eval):
            decision = select_best_action_from_candidates(
                candidates=[strict_dpp, severe_tail_relief],
                system_state=object(),
                config=config,
                queue_aware=True,
            )

        self.assertEqual(decision["candidate_source"], "severe-tail-relief-bounded-y")
        self.assertEqual(decision["selected_by_dpp_or_claim_band"], "queue_pressure_dpp_tail_delay_guard")

    def test_queue_pressure_tail_delay_guard_rejects_queue_regression(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_algorithms import select_best_action_from_candidates
        from ablation_config import AblationExperimentConfig

        config = AblationExperimentConfig(
            include_energy_claim=True,
            claim_delay_ref_ms=100.0,
            claim_energy_ref_j=12.0,
            claim_cost_ref=400.0,
            queue_delay_threshold_ms=325.0,
            energy_hard_dpp_tolerance_ratio=0.03,
            energy_claim_stability_guard_enabled=False,
            energy_claim_temporal_guard_enabled=False,
            energy_claim_temporal_guard_dpp_regret_ratio=0.10,
            queue_pressure_energy_guard_enabled=False,
        )
        config.queue_pressure_delay_guard_dpp_regret_ratio = 0.35
        config.queue_pressure_delay_guard_max_predicted_y_regret = 6.0
        strict_dpp = {
            "action": np.array([1, 0], dtype=int),
            "action_dim": 2,
            "action_scope": "pair",
            "candidate_source": "strict-dpp-high-delay",
        }
        queue_bad_relief = {
            "action": np.array([0, 1], dtype=int),
            "action_dim": 2,
            "action_scope": "pair",
            "candidate_source": "tail-delay-relief-queue-bad",
        }

        def fake_eval(action, system_state, config, queue_aware=True):
            source = str(action.get("candidate_source", ""))
            is_relief = source == "tail-delay-relief-queue-bad"
            return {
                "action": np.array(action.get("action"), dtype=int),
                "candidate_source": source,
                "paper_dpp_score": 125.0 if is_relief else 100.0,
                "claim_score": 95.0 if is_relief else 100.0,
                "delay_ms": 285.0 if is_relief else 410.0,
                "energy_j": 23.0 if is_relief else 22.5,
                "cost": 1450.0 if is_relief else 1510.0,
                "v_cost_term": 100.0,
                "energy_queue_term": 70.0,
                "delay_queue_term": 50.0,
                "predicted_avg_y": 95.0 if is_relief else 80.0,
                "predicted_avg_z": 4.5 if is_relief else 7.0,
                "post_update_queue_drift_term": 95.0 if is_relief else 100.0,
                "repaired_pair_action_hash": source,
                "feasible": True,
                "failure_reason": "",
                "local_count": 1,
                "cloud_count": 1,
            }

        with patch("ResourceAllocation.evaluate_action_dry_run", side_effect=fake_eval):
            decision = select_best_action_from_candidates(
                candidates=[strict_dpp, queue_bad_relief],
                system_state=object(),
                config=config,
                queue_aware=True,
            )

        self.assertEqual(decision["candidate_source"], "strict-dpp-high-delay")
        self.assertEqual(decision["selected_by_dpp_or_claim_band"], "queue_pressure_dpp")

    def test_queue_pressure_delay_guard_rejects_energy_queue_regression_under_high_y(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_algorithms import queue_pressure_delay_guard_candidate
        from ablation_config import AblationExperimentConfig

        config = AblationExperimentConfig(include_energy_claim=True)
        config.queue_pressure_delay_guard_min_normalized_gain = 0.5
        config.queue_delay_threshold_ms = 325.0
        config.queue_pressure_delay_guard_energy_pressure_min_current_avg_y = 8.0
        config.queue_pressure_delay_guard_energy_pressure_max_predicted_y_regret = 0.25
        current = {
            "candidate_source": "strict-high-delay",
            "delay_ms": 360.0,
            "predicted_avg_y": 30.0,
            "post_update_energy_queue_delta_term": 500.0,
            "post_update_queue_drift_term": 40.0,
        }
        low_delay_worse_energy_queue = {
            "candidate_source": "low-delay-worse-energy-queue",
            "delay_ms": 290.0,
            "claim_score": 1.0,
            "paper_dpp_score": 1010.0,
            "dpp_band_passed": True,
            "predicted_avg_y": 31.0,
            "post_update_energy_queue_delta_term": 620.0,
            "post_update_queue_drift_term": 43.0,
            "eval_rank": 1,
        }

        selected = queue_pressure_delay_guard_candidate(
            [current, low_delay_worse_energy_queue], config, current
        )

        self.assertEqual(selected, {})

    def test_queue_pressure_delay_guard_allows_queue_bounded_relief_under_high_y(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_algorithms import queue_pressure_delay_guard_candidate
        from ablation_config import AblationExperimentConfig

        config = AblationExperimentConfig(include_energy_claim=True)
        config.queue_pressure_delay_guard_min_normalized_gain = 0.5
        config.queue_delay_threshold_ms = 325.0
        config.queue_pressure_delay_guard_energy_pressure_min_current_avg_y = 8.0
        config.queue_pressure_delay_guard_energy_pressure_max_predicted_y_regret = 0.25
        current = {
            "candidate_source": "strict-high-delay",
            "delay_ms": 360.0,
            "predicted_avg_y": 30.0,
            "post_update_energy_queue_delta_term": 500.0,
            "post_update_queue_drift_term": 40.0,
        }
        low_delay_bounded = {
            "candidate_source": "low-delay-queue-bounded",
            "delay_ms": 290.0,
            "claim_score": 1.0,
            "paper_dpp_score": 1010.0,
            "dpp_band_passed": True,
            "predicted_avg_y": 29.9,
            "post_update_energy_queue_delta_term": 450.0,
            "post_update_queue_drift_term": 39.0,
            "eval_rank": 1,
        }

        selected = queue_pressure_delay_guard_candidate(
            [current, low_delay_bounded], config, current
        )

        self.assertIs(selected, low_delay_bounded)

    def test_queue_pressure_tail_delay_guard_relaxes_bounds_only_for_severe_delay(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_algorithms import select_best_action_from_candidates
        from ablation_config import AblationExperimentConfig

        config = AblationExperimentConfig(
            include_energy_claim=True,
            claim_delay_ref_ms=100.0,
            claim_energy_ref_j=12.0,
            claim_cost_ref=400.0,
            queue_delay_threshold_ms=325.0,
            energy_hard_dpp_tolerance_ratio=0.03,
            energy_claim_stability_guard_enabled=False,
            energy_claim_temporal_guard_enabled=False,
            energy_claim_temporal_guard_dpp_regret_ratio=0.10,
            queue_pressure_energy_guard_enabled=False,
        )
        config.queue_pressure_delay_guard_dpp_regret_ratio = 0.35
        config.queue_pressure_delay_guard_max_predicted_z_regret = 0.5
        config.queue_pressure_delay_guard_max_queue_drift_regret = 15.0
        config.queue_pressure_delay_guard_severe_delay_ms = 360.0
        config.queue_pressure_delay_guard_severe_dpp_regret_ratio = 0.55
        config.queue_pressure_delay_guard_severe_max_predicted_z_regret = 1.0
        config.queue_pressure_delay_guard_severe_max_queue_drift_regret = 80.0
        config.queue_pressure_delay_guard_energy_pressure_min_current_avg_y = 999.0
        strict_dpp = {
            "action": np.array([1, 0], dtype=int),
            "action_dim": 2,
            "action_scope": "pair",
            "candidate_source": "strict-dpp-severe-delay",
        }
        severe_tail_relief = {
            "action": np.array([0, 1], dtype=int),
            "action_dim": 2,
            "action_scope": "pair",
            "candidate_source": "severe-tail-delay-relief",
        }

        def fake_eval(action, system_state, config, queue_aware=True):
            source = str(action.get("candidate_source", ""))
            is_relief = source == "severe-tail-delay-relief"
            return {
                "action": np.array(action.get("action"), dtype=int),
                "candidate_source": source,
                "paper_dpp_score": 150.0 if is_relief else 100.0,
                "claim_score": 95.0 if is_relief else 100.0,
                "delay_ms": 285.0 if is_relief else 390.0,
                "energy_j": 24.0 if is_relief else 23.0,
                "cost": 1460.0 if is_relief else 1510.0,
                "v_cost_term": 100.0,
                "energy_queue_term": 70.0,
                "delay_queue_term": 50.0,
                "predicted_avg_y": 84.0 if is_relief else 80.0,
                "predicted_avg_z": 7.8 if is_relief else 7.0,
                "post_update_queue_drift_term": 160.0 if is_relief else 100.0,
                "repaired_pair_action_hash": source,
                "feasible": True,
                "failure_reason": "",
                "local_count": 1,
                "cloud_count": 1,
            }

        with patch("ResourceAllocation.evaluate_action_dry_run", side_effect=fake_eval):
            decision = select_best_action_from_candidates(
                candidates=[strict_dpp, severe_tail_relief],
                system_state=object(),
                config=config,
                queue_aware=True,
            )

        self.assertEqual(decision["candidate_source"], "severe-tail-delay-relief")
        self.assertEqual(decision["selected_by_dpp_or_claim_band"], "queue_pressure_dpp_tail_delay_guard")

    def test_queue_pressure_tail_delay_guard_keeps_strict_bounds_below_severe_delay(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_algorithms import select_best_action_from_candidates
        from ablation_config import AblationExperimentConfig

        config = AblationExperimentConfig(
            include_energy_claim=True,
            claim_delay_ref_ms=100.0,
            claim_energy_ref_j=12.0,
            claim_cost_ref=400.0,
            queue_delay_threshold_ms=325.0,
            energy_hard_dpp_tolerance_ratio=0.03,
            energy_claim_stability_guard_enabled=False,
            energy_claim_temporal_guard_enabled=False,
            energy_claim_temporal_guard_dpp_regret_ratio=0.10,
            queue_pressure_energy_guard_enabled=False,
        )
        config.queue_pressure_delay_guard_dpp_regret_ratio = 0.35
        config.queue_pressure_delay_guard_max_predicted_z_regret = 0.5
        config.queue_pressure_delay_guard_max_queue_drift_regret = 15.0
        config.queue_pressure_delay_guard_severe_delay_ms = 360.0
        config.queue_pressure_delay_guard_severe_dpp_regret_ratio = 0.55
        config.queue_pressure_delay_guard_severe_max_predicted_z_regret = 1.0
        config.queue_pressure_delay_guard_severe_max_queue_drift_regret = 80.0
        strict_dpp = {
            "action": np.array([1, 0], dtype=int),
            "action_dim": 2,
            "action_scope": "pair",
            "candidate_source": "strict-dpp-nonsevere-delay",
        }
        wide_relief = {
            "action": np.array([0, 1], dtype=int),
            "action_dim": 2,
            "action_scope": "pair",
            "candidate_source": "wide-tail-delay-relief",
        }

        def fake_eval(action, system_state, config, queue_aware=True):
            source = str(action.get("candidate_source", ""))
            is_relief = source == "wide-tail-delay-relief"
            return {
                "action": np.array(action.get("action"), dtype=int),
                "candidate_source": source,
                "paper_dpp_score": 150.0 if is_relief else 100.0,
                "claim_score": 95.0 if is_relief else 100.0,
                "delay_ms": 285.0 if is_relief else 350.0,
                "energy_j": 24.0 if is_relief else 23.0,
                "cost": 1460.0 if is_relief else 1510.0,
                "v_cost_term": 100.0,
                "energy_queue_term": 70.0,
                "delay_queue_term": 50.0,
                "predicted_avg_y": 84.0 if is_relief else 80.0,
                "predicted_avg_z": 7.8 if is_relief else 7.0,
                "post_update_queue_drift_term": 160.0 if is_relief else 100.0,
                "repaired_pair_action_hash": source,
                "feasible": True,
                "failure_reason": "",
                "local_count": 1,
                "cloud_count": 1,
            }

        with patch("ResourceAllocation.evaluate_action_dry_run", side_effect=fake_eval):
            decision = select_best_action_from_candidates(
                candidates=[strict_dpp, wide_relief],
                system_state=object(),
                config=config,
                queue_aware=True,
            )

        self.assertEqual(decision["candidate_source"], "strict-dpp-nonsevere-delay")
        self.assertEqual(decision["selected_by_dpp_or_claim_band"], "queue_pressure_dpp")

    def test_queue_pressure_energy_guard_prefers_lower_energy_with_no_dpp_regret(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_algorithms import select_best_action_from_candidates
        from ablation_config import AblationExperimentConfig

        config = AblationExperimentConfig(include_energy_claim=True)
        config.energy_claim_stability_guard_enabled = False
        config.energy_claim_temporal_guard_enabled = False
        config.queue_pressure_energy_guard_enabled = True
        config.queue_pressure_energy_guard_min_energy_gain_j = 1.0
        config.queue_pressure_energy_guard_max_delay_regret_ms = 20.0
        config.queue_pressure_energy_guard_max_cost_regret = 80.0
        config.queue_pressure_energy_guard_dpp_slack_ratio = 0.0
        config.claim_energy_ref_j = 12.0
        strict = {"action": np.array([1, 0], dtype=int), "candidate_source": "strict-dpp-high-energy"}
        lower_energy = {"action": np.array([0, 1], dtype=int), "candidate_source": "lower-energy-same-dpp"}
        too_slow = {"action": np.array([1, 1], dtype=int), "candidate_source": "lower-energy-too-slow"}

        def fake_eval(action, system_state, config, queue_aware=True):
            source = str(action.get("candidate_source", ""))
            rows = {
                "strict-dpp-high-energy": (100.0, 300.0, 30.0, 1300.0),
                "lower-energy-same-dpp": (100.0, 315.0, 27.0, 1360.0),
                "lower-energy-too-slow": (99.0, 390.0, 20.0, 1350.0),
            }
            dpp, delay, energy, cost = rows[source]
            return {
                "action": np.array(action.get("action"), dtype=int),
                "candidate_source": source,
                "paper_dpp_score": dpp,
                "delay_ms": delay,
                "energy_j": energy,
                "cost": cost,
                "v_cost_term": 10.0,
                "energy_queue_term": 10.0,
                "delay_queue_term": 2.0,
                "feasible": True,
                "failure_reason": "",
                "local_count": 9,
                "cloud_count": 9,
            }

        with patch("ResourceAllocation.evaluate_action_dry_run", side_effect=fake_eval):
            decision = select_best_action_from_candidates(
                candidates=[strict, lower_energy, too_slow],
                system_state=object(),
                config=config,
                queue_aware=True,
            )

        self.assertEqual(decision["candidate_source"], "lower-energy-same-dpp")
        self.assertIn("energy_guard", decision["selected_by_dpp_or_claim_band"])

    def test_queue_pressure_energy_guard_allows_bounded_regret_for_queue_relief(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_algorithms import queue_pressure_energy_guard_candidate
        from ablation_config import AblationExperimentConfig

        config = AblationExperimentConfig(include_energy_claim=True)
        config.queue_pressure_energy_guard_enabled = True
        config.queue_pressure_energy_guard_min_energy_gain_j = 1.0
        config.queue_pressure_energy_guard_max_delay_regret_ms = 35.0
        config.queue_pressure_energy_guard_max_cost_regret = 100.0
        config.queue_pressure_energy_guard_dpp_slack_ratio = 0.08
        config.queue_pressure_energy_guard_min_current_avg_y = 8.0
        current = {
            "candidate_source": "strict-high-energy",
            "paper_dpp_score": 1000.0,
            "delay_ms": 300.0,
            "energy_j": 28.0,
            "cost": 1300.0,
            "predicted_avg_y": 30.0,
            "post_update_energy_queue_delta_term": 500.0,
            "post_update_queue_drift_term": 40.0,
            "eval_rank": 0,
        }
        relief = {
            "candidate_source": "bounded-regret-energy-relief",
            "paper_dpp_score": 1060.0,
            "delay_ms": 325.0,
            "energy_j": 24.0,
            "cost": 1360.0,
            "predicted_avg_y": 26.0,
            "post_update_energy_queue_delta_term": 300.0,
            "post_update_queue_drift_term": 35.0,
            "eval_rank": 1,
        }

        selected = queue_pressure_energy_guard_candidate([current, relief], config, current)

        self.assertIs(selected, relief)

    def test_queue_pressure_energy_guard_rejects_lower_energy_queue_worsening(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_algorithms import queue_pressure_energy_guard_candidate
        from ablation_config import AblationExperimentConfig

        config = AblationExperimentConfig(include_energy_claim=True)
        config.queue_pressure_energy_guard_enabled = True
        config.queue_pressure_energy_guard_min_energy_gain_j = 1.0
        config.queue_pressure_energy_guard_max_delay_regret_ms = 35.0
        config.queue_pressure_energy_guard_max_cost_regret = 100.0
        config.queue_pressure_energy_guard_dpp_slack_ratio = 0.08
        config.queue_pressure_energy_guard_min_current_avg_y = 8.0
        current = {
            "candidate_source": "strict-high-energy",
            "paper_dpp_score": 1000.0,
            "delay_ms": 300.0,
            "energy_j": 28.0,
            "cost": 1300.0,
            "predicted_avg_y": 30.0,
            "post_update_energy_queue_delta_term": 500.0,
            "post_update_queue_drift_term": 40.0,
            "eval_rank": 0,
        }
        lower_energy_but_worse_queue = {
            "candidate_source": "lower-energy-worse-queue",
            "paper_dpp_score": 1010.0,
            "delay_ms": 305.0,
            "energy_j": 24.0,
            "cost": 1310.0,
            "predicted_avg_y": 31.0,
            "post_update_energy_queue_delta_term": 620.0,
            "post_update_queue_drift_term": 45.0,
            "eval_rank": 1,
        }

        selected = queue_pressure_energy_guard_candidate(
            [current, lower_energy_but_worse_queue], config, current
        )

        self.assertEqual(selected, {})

    def test_queue_pressure_energy_guard_rejects_latency_queue_debt_regression(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_algorithms import queue_pressure_energy_guard_candidate
        from ablation_config import AblationExperimentConfig

        config = AblationExperimentConfig(include_energy_claim=True)
        config.queue_pressure_energy_guard_enabled = True
        config.queue_pressure_energy_guard_min_energy_gain_j = 1.0
        config.queue_pressure_energy_guard_max_delay_regret_ms = 35.0
        config.queue_pressure_energy_guard_max_cost_regret = 100.0
        config.queue_pressure_energy_guard_dpp_slack_ratio = 0.08
        config.queue_pressure_energy_guard_min_current_avg_y = 8.0
        current = {
            "candidate_source": "stability-current",
            "paper_dpp_score": 30000.0,
            "delay_ms": 100.0,
            "energy_j": 10.5,
            "cost": 720.0,
            "predicted_avg_y": 24.0,
            "predicted_avg_z": 80.0,
            "post_update_energy_queue_delta_term": 250.0,
            "post_update_delay_queue_delta_term": 420.0,
            "post_update_queue_drift_term": 80.0,
            "energy_queue_term": 1400.0,
            "delay_queue_term": 9000.0,
            "v_cost_term": 14000.0,
            "eval_rank": 0,
        }
        lower_energy_latency_debt = {
            "candidate_source": "energy-relief-latency-debt",
            "paper_dpp_score": 29750.0,
            "delay_ms": 108.0,
            "energy_j": 9.0,
            "cost": 715.0,
            "predicted_avg_y": 22.5,
            "predicted_avg_z": 81.2,
            "post_update_energy_queue_delta_term": 210.0,
            "post_update_delay_queue_delta_term": 590.0,
            "post_update_queue_drift_term": 70.0,
            "energy_queue_term": 1200.0,
            "delay_queue_term": 10300.0,
            "v_cost_term": 14300.0,
            "eval_rank": 1,
        }

        selected = queue_pressure_energy_guard_candidate(
            [current, lower_energy_latency_debt], config, current
        )

        self.assertEqual(selected, {})

    def test_queue_pressure_energy_guard_allows_small_delay_overrun_for_dpp_improvement(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_algorithms import queue_pressure_energy_guard_candidate
        from ablation_config import AblationExperimentConfig

        config = AblationExperimentConfig(include_energy_claim=True)
        config.queue_pressure_energy_guard_enabled = True
        config.queue_pressure_energy_guard_min_energy_gain_j = 1.0
        config.queue_pressure_energy_guard_max_delay_regret_ms = 35.0
        config.queue_pressure_energy_guard_max_cost_regret = 100.0
        config.queue_pressure_energy_guard_dpp_slack_ratio = 0.08
        config.queue_pressure_energy_guard_dpp_improvement_delay_slack_ratio = 0.05
        config.queue_pressure_energy_guard_min_current_avg_y = 8.0
        current = {
            "candidate_source": "strict-high-energy",
            "paper_dpp_score": 19098.45,
            "delay_ms": 84.477,
            "energy_j": 10.486,
            "cost": 742.906,
            "predicted_avg_y": 22.679,
            "post_update_energy_queue_delta_term": 255.493,
            "post_update_queue_drift_term": 89.0,
            "energy_queue_term": 1500.0,
            "delay_queue_term": 2800.0,
            "v_cost_term": 14400.0,
            "eval_rank": 0,
        }
        near_delay_relief = {
            "candidate_source": "near-delay-dpp-improving-relief",
            "paper_dpp_score": 18874.28,
            "delay_ms": 120.402,
            "energy_j": 9.384,
            "cost": 723.366,
            "predicted_avg_y": 22.657,
            "post_update_energy_queue_delta_term": 229.253,
            "post_update_queue_drift_term": 87.392,
            "eval_rank": 1,
        }
        too_slow_relief = dict(near_delay_relief)
        too_slow_relief.update({
            "candidate_source": "too-slow-dpp-improving-relief",
            "delay_ms": 122.0,
            "energy_j": 9.20,
            "paper_dpp_score": 18800.0,
            "eval_rank": 2,
        })

        selected = queue_pressure_energy_guard_candidate(
            [current, too_slow_relief, near_delay_relief],
            config,
            current,
        )

        self.assertIs(selected, near_delay_relief)

    def test_queue_pressure_energy_guard_allows_dpp_offset_queue_drift_for_energy_relief(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_algorithms import queue_pressure_energy_guard_candidate
        from ablation_config import AblationExperimentConfig

        config = AblationExperimentConfig(include_energy_claim=True)
        config.queue_pressure_energy_guard_enabled = True
        config.queue_pressure_energy_guard_min_energy_gain_j = 1.0
        config.queue_pressure_energy_guard_max_delay_regret_ms = 35.0
        config.queue_pressure_energy_guard_dpp_improvement_delay_slack_ratio = 0.05
        config.queue_pressure_energy_guard_max_cost_regret = 100.0
        config.queue_pressure_energy_guard_dpp_slack_ratio = 0.08
        config.queue_pressure_energy_guard_dpp_improvement_queue_drift_offset_ratio = 1.0
        config.queue_pressure_energy_guard_min_current_avg_y = 8.0
        current = {
            "candidate_source": "strict-high-energy",
            "paper_dpp_score": 19098.451939803894,
            "delay_ms": 84.47720281647862,
            "energy_j": 10.486411506186615,
            "cost": 742.9057460670053,
            "predicted_avg_y": 23.0019513050298,
            "post_update_energy_queue_delta_term": 313.58042585895464,
            "post_update_queue_drift_term": 63.80735803746625,
            "eval_rank": 0,
        }
        dpp_offset_relief = {
            "candidate_source": "dpp-offset-energy-relief",
            "paper_dpp_score": 18874.279095640726,
            "delay_ms": 120.40222104836572,
            "energy_j": 9.383891471418215,
            "cost": 723.3659825910813,
            "predicted_avg_y": 22.657413794164675,
            "post_update_energy_queue_delta_term": 229.2528175473194,
            "post_update_queue_drift_term": 87.39152722423651,
            "eval_rank": 1,
        }
        excessive_drift_relief = dict(dpp_offset_relief)
        excessive_drift_relief.update({
            "candidate_source": "excessive-drift-dpp-improving-relief",
            "paper_dpp_score": 18850.0,
            "energy_j": 9.0,
            "cost": 720.0,
            "predicted_avg_y": 22.0,
            "post_update_energy_queue_delta_term": 100.0,
            "post_update_queue_drift_term": 340.0,
            "eval_rank": 0,
        })

        selected = queue_pressure_energy_guard_candidate(
            [current, excessive_drift_relief, dpp_offset_relief],
            config,
            current,
        )

        self.assertIs(selected, dpp_offset_relief)

    def test_queue_pressure_energy_guard_rejects_relaxed_same_cloud_without_energy_recovery(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_algorithms import queue_pressure_energy_guard_candidate
        from ablation_config import AblationExperimentConfig

        config = AblationExperimentConfig(include_energy_claim=True)
        config.queue_pressure_energy_guard_enabled = True
        config.queue_pressure_energy_guard_min_energy_gain_j = 1.0
        config.queue_pressure_energy_guard_max_delay_regret_ms = 35.0
        config.queue_pressure_energy_guard_dpp_improvement_delay_slack_ratio = 0.05
        config.queue_pressure_energy_guard_max_cost_regret = 100.0
        config.queue_pressure_energy_guard_dpp_slack_ratio = 0.08
        config.queue_pressure_energy_guard_dpp_improvement_queue_drift_offset_ratio = 1.0
        config.queue_pressure_energy_guard_min_current_avg_y = 8.0

        current = {
            "candidate_source": "slot9-strict-current",
            "paper_dpp_score": 19280.87954552612,
            "delay_ms": 90.83229811412952,
            "energy_j": 10.534746221693742,
            "cost": 742.9057460670053,
            "predicted_avg_y": 23.017418713642427,
            "predicted_avg_z": 26.60892351018455,
            "post_update_energy_queue_delta_term": 315.4503128422322,
            "post_update_delay_queue_delta_term": 442.8350958720505,
            "post_update_queue_drift_term": 75.82920525094642,
            "energy_queue_term": 1553.0,
            "delay_queue_term": 2793.0,
            "v_cost_term": 14858.0,
            "eval_rank": 0,
        }
        relaxed_same_cloud = {
            "candidate_source": "uac_energy_cloud_relief_queue_relaxed_cloud_relief",
            "resource_mode": "queue_relaxed_cloud_relief",
            "active_cloud_pair_signature": "27:flow_9:a4@ai_v1",
            "paper_dpp_score": 18874.279095640726,
            "delay_ms": 120.40222104836572,
            "energy_j": 9.383891471418215,
            "cost": 723.3659825910813,
            "predicted_avg_y": 22.657413794164675,
            "predicted_avg_z": 28.33510793205858,
            "post_update_energy_queue_delta_term": 229.2528175473194,
            "post_update_delay_queue_delta_term": 644.6623381888839,
            "post_update_queue_drift_term": 87.39152722423651,
            "eval_rank": 1,
        }
        recovery_probe_same_cloud = {
            "candidate_source": "uac_energy_cloud_relief_queue_damped_cloud_relief",
            "resource_mode": "queue_damped_cloud_relief",
            "active_cloud_pair_signature": "27:flow_9:a4@ai_v1",
            "paper_dpp_score": 19140.11694581589,
            "delay_ms": 109.32779202708545,
            "energy_j": 10.416529919588384,
            "cost": 738.1991827206697,
            "predicted_avg_y": 22.980277667389824,
            "predicted_avg_z": 26.680613855334166,
            "post_update_energy_queue_delta_term": 283.95778771731126,
            "post_update_delay_queue_delta_term": 454.09701078357326,
            "post_update_queue_drift_term": 73.8054731327828,
            "eval_rank": 2,
        }

        selected = queue_pressure_energy_guard_candidate(
            [current, relaxed_same_cloud, recovery_probe_same_cloud],
            config,
            current,
        )

        self.assertEqual(selected, {})

    def test_queue_pressure_energy_guard_rejects_relief_worse_than_strict_dpp_baseline(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_algorithms import queue_pressure_energy_guard_candidate
        from ablation_config import AblationExperimentConfig

        config = AblationExperimentConfig(include_energy_claim=True)
        config.queue_pressure_energy_guard_enabled = True
        config.queue_pressure_energy_guard_min_energy_gain_j = 1.0
        config.queue_pressure_energy_guard_max_delay_regret_ms = 35.0
        config.queue_pressure_energy_guard_max_cost_regret = 100.0
        config.queue_pressure_energy_guard_dpp_slack_ratio = 0.08
        config.queue_pressure_energy_guard_dpp_improvement_delay_slack_ratio = 0.05
        config.queue_pressure_energy_guard_dpp_improvement_queue_drift_offset_ratio = 1.0
        config.queue_pressure_energy_guard_min_current_avg_y = 8.0
        config.queue_pressure_energy_guard_strict_dpp_regret_ratio = 0.0

        strict_dpp_baseline = {
            "candidate_source": "strict-dpp-baseline",
            "paper_dpp_score": 1000.0,
            "delay_ms": 100.0,
            "energy_j": 10.0,
            "cost": 700.0,
            "predicted_avg_y": 60.0,
            "post_update_energy_queue_delta_term": 200.0,
            "post_update_queue_drift_term": 80.0,
            "energy_queue_term": 3000.0,
            "delay_queue_term": 9000.0,
            "v_cost_term": 12000.0,
            "eval_rank": 0,
        }
        post_temporal_current = dict(strict_dpp_baseline)
        post_temporal_current.update({
            "candidate_source": "post-temporal-current",
            "paper_dpp_score": 1200.0,
            "energy_j": 12.0,
            "energy_guard_strict_dpp_baseline_score": strict_dpp_baseline["paper_dpp_score"],
            "eval_rank": 1,
        })
        lower_energy_relief = dict(post_temporal_current)
        lower_energy_relief.update({
            "candidate_source": "lower-energy-current-relative-relief",
            "paper_dpp_score": 1150.0,
            "delay_ms": 105.0,
            "energy_j": 10.0,
            "cost": 680.0,
            "predicted_avg_y": 59.0,
            "post_update_energy_queue_delta_term": 100.0,
            "post_update_queue_drift_term": 79.0,
            "eval_rank": 2,
        })

        selected = queue_pressure_energy_guard_candidate(
            [post_temporal_current, lower_energy_relief],
            config,
            post_temporal_current,
        )

        self.assertEqual(selected, {})

    def test_queue_pressure_energy_guard_allows_bounded_dpp_regret_when_queue_terms_are_material(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_algorithms import select_best_action_from_candidates
        from ablation_config import AblationExperimentConfig

        config = AblationExperimentConfig(include_energy_claim=True)
        config.energy_claim_stability_guard_enabled = False
        config.energy_claim_temporal_guard_enabled = False
        config.queue_pressure_energy_guard_enabled = True
        config.queue_pressure_dpp_min_queue_to_v_ratio = 0.25
        config.queue_pressure_energy_guard_min_energy_gain_j = 0.05
        config.queue_pressure_energy_guard_max_delay_regret_ms = 10.0
        config.queue_pressure_energy_guard_max_cost_regret = 10.0
        config.queue_pressure_energy_guard_dpp_slack_ratio = 0.08
        config.queue_pressure_energy_guard_min_queue_relief = 0.05

        strict = {"action": np.array([0, 0], dtype=int), "candidate_source": "strict-all-local"}
        relief = {"action": np.array([0, 1], dtype=int), "candidate_source": "bounded-hybrid-energy-relief"}

        def fake_eval(action, system_state, config, queue_aware=True):
            source = str(action.get("candidate_source", ""))
            if source == "bounded-hybrid-energy-relief":
                dpp, delay, energy, cost = 106.0, 105.0, 8.10, 701.0
                local_count, cloud_count = 17, 1
                predicted_y, energy_delta, drift = 0.12, 0.30, 0.10
            else:
                dpp, delay, energy, cost = 100.0, 100.0, 8.30, 700.0
                local_count, cloud_count = 18, 0
                predicted_y, energy_delta, drift = 0.20, 0.50, 0.20
            return {
                "action": np.array(action.get("action"), dtype=int),
                "candidate_source": source,
                "paper_dpp_score": dpp,
                "delay_ms": delay,
                "energy_j": energy,
                "cost": cost,
                "v_cost_term": 10.0,
                "energy_queue_term": 5.0,
                "delay_queue_term": 0.0,
                "predicted_avg_y": predicted_y,
                "post_update_energy_queue_delta_term": energy_delta,
                "post_update_queue_drift_term": drift,
                "feasible": True,
                "failure_reason": "",
                "local_count": local_count,
                "cloud_count": cloud_count,
                "repaired_pair_action_hash": source,
            }

        with patch("ResourceAllocation.evaluate_action_dry_run", side_effect=fake_eval):
            decision = select_best_action_from_candidates(
                candidates=[strict, relief],
                system_state=object(),
                config=config,
                queue_aware=True,
            )

        self.assertEqual(decision["candidate_source"], "bounded-hybrid-energy-relief")
        self.assertIn("energy_guard", decision["selected_by_dpp_or_claim_band"])
    def test_claim_selector_keeps_dominated_noncollapsed_out_of_final_pool(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_algorithms import select_best_action_from_candidates
        from ablation_config import AblationExperimentConfig

        config = AblationExperimentConfig(
            include_energy_claim=True,
            claim_delay_ref_ms=100.0,
            claim_energy_ref_j=12.0,
            claim_cost_ref=400.0,
            energy_hard_dpp_tolerance_ratio=0.10,
        )
        collapsed = {
            "action": np.array([0, 0], dtype=int),
            "pair_action": np.array([0, 0], dtype=int),
            "pair_universe": [{"server_id": "s0"}, {"server_id": "s1"}],
            "pair_action_dim": 2,
            "action_dim": 2,
            "action_scope": "pair",
            "candidate_source": "actor_collapsed",
        }
        noncollapsed = {
            "action": np.array([0, 1], dtype=int),
            "pair_action": np.array([0, 1], dtype=int),
            "pair_universe": [{"server_id": "s0"}, {"server_id": "s1"}],
            "pair_action_dim": 2,
            "action_dim": 2,
            "action_scope": "pair",
            "candidate_source": "uac_reference_low_impact_neighbor",
        }

        def fake_eval(action, system_state, config, queue_aware=True):
            source = str(action.get("candidate_source", ""))
            is_noncollapsed = "reference_low_impact_neighbor" in source
            return {
                "action": np.array(action.get("action"), dtype=int),
                "pair_action": np.array(action.get("pair_action"), dtype=int),
                "paper_dpp_score": 101.0 if is_noncollapsed else 100.0,
                "feasible": True,
                "failure_reason": "",
                "delay_ms": 104.0 if is_noncollapsed else 100.0,
                "energy_j": 11.0 if is_noncollapsed else 10.0,
                "cost": 102.0 if is_noncollapsed else 100.0,
                "local_count": 1 if is_noncollapsed else 2,
                "cloud_count": 1 if is_noncollapsed else 0,
                "repaired_pair_action_hash": "hybrid" if is_noncollapsed else "collapsed",
                "candidate_source": source,
                "resource_hint": str(action.get("resource_hint", "")),
            }

        with patch("ResourceAllocation.evaluate_action_dry_run", side_effect=fake_eval):
            decision = select_best_action_from_candidates(
                candidates=[collapsed, noncollapsed],
                system_state=object(),
                config=config,
                queue_aware=True,
            )

        self.assertEqual(decision["repaired_pair_action_hash"], "collapsed")
        self.assertEqual(decision["selected_by_dpp_or_claim_band"], "claim_band")
    def test_claim_selector_escapes_dpp_band_for_large_claim_gain(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_algorithms import select_best_action_from_candidates
        from ablation_config import AblationExperimentConfig

        config = AblationExperimentConfig(
            include_energy_claim=True,
            claim_delay_ref_ms=100.0,
            claim_energy_ref_j=12.0,
            claim_cost_ref=400.0,
            energy_hard_dpp_tolerance_ratio=0.03,
        )
        cloud_heavy = {
            "action": np.array([0, 0, 0, 0], dtype=int),
            "pair_action": np.array([0, 0, 0, 0], dtype=int),
            "pair_universe": [{"server_id": f"s{i}"} for i in range(4)],
            "pair_action_dim": 4,
            "action_dim": 4,
            "action_scope": "pair",
            "candidate_source": "uac_guard_all_cloud",
        }
        hybrid = {
            "action": np.array([1, 1, 1, 0], dtype=int),
            "pair_action": np.array([1, 1, 1, 0], dtype=int),
            "pair_universe": [{"server_id": f"s{i}"} for i in range(4)],
            "pair_action_dim": 4,
            "action_dim": 4,
            "action_scope": "pair",
            "candidate_source": "uac_energy_cloud_relief",
        }

        def fake_eval(action, system_state, config, queue_aware=True):
            source = str(action.get("candidate_source", ""))
            is_hybrid = "energy_cloud_relief" in source
            return {
                "action": np.array(action.get("action"), dtype=int),
                "pair_action": np.array(action.get("pair_action"), dtype=int),
                "paper_dpp_score": 180.0 if is_hybrid else 100.0,
                "feasible": True,
                "failure_reason": "",
                "delay_ms": 170.0 if is_hybrid else 265.0,
                "energy_j": 13.0 if is_hybrid else 12.0,
                "cost": 810.0 if is_hybrid else 710.0,
                "local_count": 3 if is_hybrid else 0,
                "cloud_count": 1 if is_hybrid else 4,
                "repaired_pair_action_hash": "hybrid" if is_hybrid else "all-cloud",
                "candidate_source": source,
                "resource_hint": str(action.get("resource_hint", "")),
            }

        with patch("ResourceAllocation.evaluate_action_dry_run", side_effect=fake_eval):
            decision = select_best_action_from_candidates(
                candidates=[cloud_heavy, hybrid],
                system_state=object(),
                config=config,
                queue_aware=True,
            )

        self.assertEqual(decision["repaired_pair_action_hash"], "hybrid")
        self.assertEqual(decision["selected_by_dpp_or_claim_band"], "claim_escape")
    def test_claim_selector_softens_noncollapsed_preference_for_large_energy_penalty(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_algorithms import select_best_action_from_candidates
        from ablation_config import AblationExperimentConfig

        config = AblationExperimentConfig(
            include_energy_claim=True,
            claim_delay_ref_ms=100.0,
            claim_energy_ref_j=24.0,
            claim_cost_ref=400.0,
            energy_hard_dpp_tolerance_ratio=0.10,
        )
        collapsed = {
            "action": np.array([0, 0], dtype=int),
            "pair_action": np.array([0, 0], dtype=int),
            "pair_universe": [{"server_id": "s0"}, {"server_id": "s1"}],
            "pair_action_dim": 2,
            "action_dim": 2,
            "action_scope": "pair",
            "candidate_source": "actor_energy_balanced",
        }
        noncollapsed = {
            "action": np.array([0, 1], dtype=int),
            "pair_action": np.array([0, 1], dtype=int),
            "pair_universe": [{"server_id": "s0"}, {"server_id": "s1"}],
            "pair_action_dim": 2,
            "action_dim": 2,
            "action_scope": "pair",
            "candidate_source": "uac_reference_low_impact_neighbor",
        }

        def fake_eval(action, system_state, config, queue_aware=True):
            source = str(action.get("candidate_source", ""))
            is_noncollapsed = "reference_low_impact_neighbor" in source
            return {
                "action": np.array(action.get("action"), dtype=int),
                "pair_action": np.array(action.get("pair_action"), dtype=int),
                "paper_dpp_score": 101.0 if is_noncollapsed else 100.0,
                "feasible": True,
                "failure_reason": "",
                "delay_ms": 100.0 if is_noncollapsed else 106.0,
                "energy_j": 13.0 if is_noncollapsed else 10.0,
                "cost": 110.0 if is_noncollapsed else 100.0,
                "local_count": 1 if is_noncollapsed else 2,
                "cloud_count": 1 if is_noncollapsed else 0,
                "repaired_pair_action_hash": "hybrid" if is_noncollapsed else "collapsed",
                "candidate_source": source,
                "resource_hint": str(action.get("resource_hint", "")),
            }

        with patch("ResourceAllocation.evaluate_action_dry_run", side_effect=fake_eval):
            decision = select_best_action_from_candidates(
                candidates=[collapsed, noncollapsed],
                system_state=object(),
                config=config,
                queue_aware=True,
            )

        self.assertEqual(decision["repaired_pair_action_hash"], "collapsed")
        self.assertEqual(decision["selected_by_dpp_or_claim_band"], "claim_band")

    def test_claim_selector_prefers_balanced_candidate_inside_claim_regret_band(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_algorithms import select_best_action_from_candidates
        from ablation_config import AblationExperimentConfig

        config = AblationExperimentConfig(
            include_energy_claim=True,
            energy_claim_stability_guard_enabled=True,
            claim_delay_ref_ms=100.0,
            claim_energy_ref_j=12.0,
            claim_cost_ref=400.0,
            energy_hard_dpp_tolerance_ratio=0.10,
        )
        slow_low_energy = {
            "action": np.array([0, 0, 0], dtype=int),
            "action_dim": 3,
            "action_scope": "pair",
            "candidate_source": "uac_slow_low_energy",
        }
        balanced = {
            "action": np.array([0, 1, 0], dtype=int),
            "action_dim": 3,
            "action_scope": "pair",
            "candidate_source": "uac_balanced_middle",
        }
        fast_high_energy = {
            "action": np.array([1, 1, 0], dtype=int),
            "action_dim": 3,
            "action_scope": "pair",
            "candidate_source": "uac_fast_high_energy",
        }

        def fake_eval(action, system_state, config, queue_aware=True):
            source = str(action.get("candidate_source", ""))
            if "balanced" in source:
                delay, energy, cost, dpp, hash_value = 270.0, 20.0, 1180.0, 101.0, "balanced"
            elif "fast" in source:
                delay, energy, cost, dpp, hash_value = 180.0, 28.0, 1260.0, 102.0, "fast-high-energy"
            else:
                delay, energy, cost, dpp, hash_value = 340.0, 13.0, 1100.0, 100.0, "slow-low-energy"
            return {
                "action": np.array(action.get("action"), dtype=int),
                "paper_dpp_score": dpp,
                "feasible": True,
                "failure_reason": "",
                "delay_ms": delay,
                "energy_j": energy,
                "cost": cost,
                "local_count": 2,
                "cloud_count": 1,
                "repaired_pair_action_hash": hash_value,
                "candidate_source": source,
            }

        with patch("ResourceAllocation.evaluate_action_dry_run", side_effect=fake_eval):
            decision = select_best_action_from_candidates(
                candidates=[slow_low_energy, balanced, fast_high_energy],
                system_state=object(),
                config=config,
                queue_aware=True,
            )

        self.assertEqual(decision["repaired_pair_action_hash"], "balanced")
        self.assertEqual(decision["selected_by_dpp_or_claim_band"], "claim_band_stability_guard")

    def test_claim_stability_guard_treats_normal_main_delay_target_as_upper_bound(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_algorithms import select_best_action_from_candidates
        from ablation_config import AblationExperimentConfig

        config = AblationExperimentConfig(
            include_energy_claim=True,
            energy_claim_stability_guard_enabled=True,
            queue_delay_threshold_ms=325.0,
            claim_delay_ref_ms=100.0,
            claim_energy_ref_j=12.0,
            claim_cost_ref=400.0,
            energy_hard_dpp_tolerance_ratio=0.10,
        )
        low_delay = {"action": np.array([0, 0, 0], dtype=int), "candidate_source": "uac_low_delay"}
        median_delay = {"action": np.array([0, 1, 0], dtype=int), "candidate_source": "uac_median_delay"}
        target_delay = {"action": np.array([1, 0, 0], dtype=int), "candidate_source": "uac_target_delay"}
        high_delay = {"action": np.array([1, 1, 0], dtype=int), "candidate_source": "uac_high_delay"}

        def fake_eval(action, system_state, config, queue_aware=True):
            source = str(action.get("candidate_source", ""))
            if "target" in source:
                delay, energy, cost, dpp, hash_value = 330.0, 9.0, 1080.0, 102.0, "target-delay"
            elif "median" in source:
                delay, energy, cost, dpp, hash_value = 270.0, 15.5, 1200.0, 101.0, "median-delay"
            elif "high" in source:
                delay, energy, cost, dpp, hash_value = 390.0, 20.0, 1180.0, 103.0, "high-delay"
            else:
                delay, energy, cost, dpp, hash_value = 255.0, 18.0, 1160.0, 100.0, "low-delay"
            return {
                "action": np.array(action.get("action"), dtype=int),
                "paper_dpp_score": dpp,
                "feasible": True,
                "failure_reason": "",
                "delay_ms": delay,
                "energy_j": energy,
                "cost": cost,
                "local_count": 2,
                "cloud_count": 1,
                "repaired_pair_action_hash": hash_value,
                "candidate_source": source,
            }

        with patch("ResourceAllocation.evaluate_action_dry_run", side_effect=fake_eval):
            decision = select_best_action_from_candidates(
                candidates=[low_delay, median_delay, target_delay, high_delay],
                system_state=object(),
                config=config,
                queue_aware=True,
            )

        self.assertEqual(decision["repaired_pair_action_hash"], "low-delay")
        self.assertEqual(decision["selected_by_dpp_or_claim_band"], "claim_band_stability_guard")

    def test_claim_stability_guard_does_not_penalize_below_delay_target(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_algorithms import claim_stability_guard_candidate
        from ablation_config import AblationExperimentConfig

        config = AblationExperimentConfig(
            include_energy_claim=True,
            energy_claim_stability_guard_enabled=True,
            queue_delay_threshold_ms=325.0,
            claim_delay_ref_ms=100.0,
            claim_energy_ref_j=12.0,
            claim_cost_ref=400.0,
        )
        best_claim_row = {
            "candidate_source": "claim-best-near-target",
            "feasible": True,
            "claim_score": 7.0,
            "paper_dpp_score": 100.0,
            "delay_ms": 330.0,
            "energy_j": 18.0,
            "cost": 1200.0,
            "eval_rank": 0,
        }
        low_delay_row = {
            "candidate_source": "balanced-low-delay",
            "feasible": True,
            "claim_score": 7.05,
            "paper_dpp_score": 101.0,
            "delay_ms": 240.0,
            "energy_j": 18.0,
            "cost": 1200.0,
            "eval_rank": 1,
        }
        high_delay_row = {
            "candidate_source": "tail-risk-high-delay",
            "feasible": True,
            "claim_score": 7.08,
            "paper_dpp_score": 102.0,
            "delay_ms": 345.0,
            "energy_j": 18.0,
            "cost": 1200.0,
            "eval_rank": 2,
        }

        selected = claim_stability_guard_candidate(
            [best_claim_row, low_delay_row, high_delay_row],
            config,
            best_claim_row,
        )

        self.assertIs(selected, low_delay_row)

    def test_claim_stability_guard_prefers_lower_post_update_queue_drift(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_algorithms import claim_stability_guard_candidate
        from ablation_config import AblationExperimentConfig

        config = AblationExperimentConfig(
            include_energy_claim=True,
            energy_claim_stability_guard_enabled=True,
            energy_claim_stability_min_candidates=2,
            energy_claim_stability_regret_ratio=0.20,
            energy_claim_stability_min_score_gain=0.10,
            energy_claim_stability_claim_weight=0.05,
            queue_delay_threshold_ms=325.0,
            claim_delay_ref_ms=100.0,
            claim_energy_ref_j=12.0,
            claim_cost_ref=400.0,
        )
        config.energy_claim_stability_queue_drift_weight = 0.50
        high_drift_row = {
            "candidate_source": "claim-best-high-drift",
            "feasible": True,
            "claim_score": 7.00,
            "paper_dpp_score": 100.0,
            "delay_ms": 290.0,
            "energy_j": 20.0,
            "cost": 1300.0,
            "post_update_queue_drift_term": 420.0,
            "eval_rank": 0,
        }
        low_drift_row = {
            "candidate_source": "claim-regret-low-drift",
            "feasible": True,
            "claim_score": 7.10,
            "paper_dpp_score": 102.0,
            "delay_ms": 290.0,
            "energy_j": 20.0,
            "cost": 1300.0,
            "post_update_queue_drift_term": 20.0,
            "eval_rank": 1,
        }

        selected = claim_stability_guard_candidate(
            [high_drift_row, low_drift_row],
            config,
            high_drift_row,
        )

        self.assertIs(selected, low_drift_row)

    def test_claim_stability_guard_prefers_lower_latency_queue_debt(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_algorithms import claim_stability_guard_candidate
        from ablation_config import AblationExperimentConfig

        config = AblationExperimentConfig(
            include_energy_claim=True,
            energy_claim_stability_guard_enabled=True,
            energy_claim_stability_min_candidates=2,
            energy_claim_stability_regret_ratio=0.20,
            energy_claim_stability_min_score_gain=0.05,
            energy_claim_stability_claim_weight=0.05,
            queue_delay_threshold_ms=325.0,
            claim_delay_ref_ms=100.0,
            claim_energy_ref_j=12.0,
            claim_cost_ref=400.0,
        )
        config.energy_claim_stability_queue_drift_weight = 0.25
        config.energy_claim_stability_latency_queue_weight = 0.75
        energy_saving_latency_debt = {
            "candidate_source": "slot31-energy-saving-all-local",
            "feasible": True,
            "claim_score": 3.45,
            "paper_dpp_score": 100.0,
            "delay_ms": 103.5,
            "energy_j": 8.46,
            "cost": 685.9,
            "predicted_avg_z": 81.87,
            "post_update_delay_queue_delta_term": 648.09,
            "post_update_queue_drift_term": 60.0,
            "eval_rank": 0,
        }
        latency_recovery = {
            "candidate_source": "slot31-stability-latency-recovery",
            "feasible": True,
            "claim_score": 3.49,
            "paper_dpp_score": 103.0,
            "delay_ms": 93.7,
            "energy_j": 9.60,
            "cost": 700.7,
            "predicted_avg_z": 80.76,
            "post_update_delay_queue_delta_term": 477.12,
            "post_update_queue_drift_term": 60.0,
            "eval_rank": 1,
        }

        selected = claim_stability_guard_candidate(
            [energy_saving_latency_debt, latency_recovery],
            config,
            energy_saving_latency_debt,
        )

        self.assertIs(selected, latency_recovery)

    def test_temporal_guard_prefers_smoother_candidate_inside_claim_regret(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_algorithms import select_best_action_from_candidates
        from ablation_config import AblationExperimentConfig

        config = AblationExperimentConfig(include_energy_claim=True)
        config.energy_claim_temporal_guard_enabled = True
        config.energy_claim_temporal_guard_min_history = 3
        config.energy_claim_temporal_guard_regret_ratio = 0.20
        config.energy_claim_temporal_guard_min_score_gain = 0.25
        config.energy_claim_stability_guard_enabled = False
        config.claim_energy_ref_j = 12.0
        config.queue_delay_threshold_ms = 325.0
        state = type("State", (), {})()
        state._lyham_temporal_metric_history = [
            {"delay_ms": 290.0, "energy_j": 22.0, "cost": 1320.0},
            {"delay_ms": 292.0, "energy_j": 21.8, "cost": 1318.0},
            {"delay_ms": 288.0, "energy_j": 22.2, "cost": 1322.0},
        ]
        jumpy = {"action": np.array([1], dtype=int), "candidate_source": "claim-best-jumpy"}
        smooth = {"action": np.array([0], dtype=int), "candidate_source": "temporal-smooth"}

        def fake_eval(action, system_state, config, queue_aware=True):
            source = str(action.get("candidate_source", ""))
            if source == "claim-best-jumpy":
                delay, energy, cost, dpp = 360.0, 20.0, 1000.0, 100.0
            else:
                delay, energy, cost, dpp = 295.0, 22.5, 1330.0, 101.0
            return {
                "action": np.array(action.get("action"), dtype=int),
                "paper_dpp_score": dpp,
                "feasible": True,
                "failure_reason": "",
                "delay_ms": delay,
                "energy_j": energy,
                "cost": cost,
                "local_count": 1,
                "cloud_count": 1,
                "candidate_source": source,
            }

        with patch("ResourceAllocation.evaluate_action_dry_run", side_effect=fake_eval):
            decision = select_best_action_from_candidates(
                candidates=[jumpy, smooth],
                system_state=state,
                config=config,
                queue_aware=True,
            )

        self.assertEqual(decision["candidate_source"], "temporal-smooth")
        self.assertIn("temporal_guard", decision["selected_by_dpp_or_claim_band"])

    def test_temporal_guard_penalizes_source_and_local_cloud_switch_inside_regret(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_algorithms import select_best_action_from_candidates
        from ablation_config import AblationExperimentConfig

        config = AblationExperimentConfig(include_energy_claim=True)
        config.energy_claim_temporal_guard_enabled = True
        config.energy_claim_temporal_guard_min_history = 3
        config.energy_claim_temporal_guard_regret_ratio = 0.20
        config.energy_claim_temporal_guard_min_score_gain = 0.10
        config.energy_claim_temporal_guard_source_switch_weight = 0.40
        config.energy_claim_temporal_guard_local_cloud_weight = 0.15
        config.energy_claim_temporal_guard_local_cloud_scale = 4.0
        config.energy_claim_stability_guard_enabled = False
        config.claim_energy_ref_j = 12.0
        config.queue_delay_threshold_ms = 325.0
        state = type("State", (), {})()
        state._lyham_temporal_metric_history = [
            {
                "delay_ms": 290.0,
                "energy_j": 22.0,
                "cost": 1320.0,
                "selected_candidate_source": "uac_reference_low_impact_neighbor",
                "local_count": 10,
                "cloud_count": 8,
                "local_pair_count": 10,
                "cloud_pair_count": 8,
            },
            {
                "delay_ms": 291.0,
                "energy_j": 22.1,
                "cost": 1318.0,
                "selected_candidate_source": "uac_reference_low_impact_neighbor",
                "local_count": 10,
                "cloud_count": 8,
                "local_pair_count": 10,
                "cloud_pair_count": 8,
            },
            {
                "delay_ms": 289.0,
                "energy_j": 21.9,
                "cost": 1322.0,
                "selected_candidate_source": "uac_reference_low_impact_swap",
                "local_count": 9,
                "cloud_count": 9,
                "local_pair_count": 9,
                "cloud_pair_count": 9,
            },
        ]
        jumpy = {"action": np.array([1, 0], dtype=int), "candidate_source": "uac_energy_budget_frontier"}
        continuous = {"action": np.array([0, 1], dtype=int), "candidate_source": "uac_reference_low_impact_neighbor"}

        def fake_eval(action, system_state, config, queue_aware=True):
            source = str(action.get("candidate_source", ""))
            if source == "uac_energy_budget_frontier":
                delay, energy, cost, dpp, local_count, cloud_count = 291.0, 22.1, 1321.0, 100.0, 1, 17
            else:
                delay, energy, cost, dpp, local_count, cloud_count = 294.0, 22.4, 1328.0, 101.0, 10, 8
            return {
                "action": np.array(action.get("action"), dtype=int),
                "candidate_source": source,
                "paper_dpp_score": dpp,
                "delay_ms": delay,
                "energy_j": energy,
                "cost": cost,
                "local_count": local_count,
                "cloud_count": cloud_count,
                "local_pair_count": local_count,
                "cloud_pair_count": cloud_count,
                "feasible": True,
                "failure_reason": "",
            }

        with patch("ResourceAllocation.evaluate_action_dry_run", side_effect=fake_eval):
            decision = select_best_action_from_candidates(
                candidates=[jumpy, continuous],
                system_state=state,
                config=config,
                queue_aware=True,
            )

        self.assertEqual(decision["candidate_source"], "uac_reference_low_impact_neighbor")
        self.assertIn("temporal_guard", decision["selected_by_dpp_or_claim_band"])

    def test_temporal_guard_allows_bounded_energy_dpp_relief_despite_local_cloud_shift(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_algorithms import claim_temporal_guard_candidate
        from ablation_config import AblationExperimentConfig

        config = AblationExperimentConfig(include_energy_claim=True)
        config.energy_claim_temporal_guard_enabled = True
        config.energy_claim_temporal_guard_min_history = 3
        config.energy_claim_temporal_guard_regret_ratio = 0.20
        config.energy_claim_temporal_guard_min_score_gain = 0.05
        config.energy_claim_temporal_guard_local_cloud_weight = 0.50
        config.energy_claim_temporal_guard_local_cloud_scale = 4.0
        config.energy_claim_temporal_guard_energy_override_enabled = True
        config.energy_claim_temporal_guard_energy_override_min_gain_j = 1.0
        config.energy_claim_temporal_guard_energy_override_max_delay_regret_ms = 35.0
        config.energy_claim_temporal_guard_energy_override_max_cost_regret = 40.0
        config.energy_claim_temporal_guard_energy_override_max_queue_regret = 0.50
        config.claim_delay_ref_ms = 100.0
        config.claim_energy_ref_j = 12.0
        config.claim_cost_ref = 400.0
        history = [
            {
                "delay_ms": 280.0,
                "energy_j": 26.0,
                "cost": 1420.0,
                "selected_candidate_source": "uac_reference_low_impact_neighbor",
                "local_pair_count": 10,
                "cloud_pair_count": 8,
            },
            {
                "delay_ms": 281.0,
                "energy_j": 26.1,
                "cost": 1418.0,
                "selected_candidate_source": "uac_reference_low_impact_neighbor",
                "local_pair_count": 10,
                "cloud_pair_count": 8,
            },
            {
                "delay_ms": 279.0,
                "energy_j": 25.9,
                "cost": 1422.0,
                "selected_candidate_source": "uac_reference_low_impact_swap",
                "local_pair_count": 9,
                "cloud_pair_count": 9,
            },
        ]
        best_claim_row = {
            "candidate_source": "uac_reference_low_impact_neighbor",
            "feasible": True,
            "delay_ms": 280.0,
            "energy_j": 26.0,
            "cost": 1420.0,
            "claim_score": 8.40,
            "paper_dpp_score": 30000.0,
            "local_pair_count": 10,
            "cloud_pair_count": 8,
            "predicted_avg_y": 4.0,
            "predicted_avg_z": 8.0,
            "post_update_queue_drift_term": 100.0,
            "eval_rank": 1,
        }
        bounded_relief = {
            "candidate_source": "uac_energy_cost_pareto_relief_queue_relaxed_cloud_relief",
            "feasible": True,
            "delay_ms": 306.0,
            "energy_j": 21.7,
            "cost": 1434.0,
            "claim_score": 8.29,
            "paper_dpp_score": 29650.0,
            "local_pair_count": 5,
            "cloud_pair_count": 13,
            "predicted_avg_y": 3.8,
            "predicted_avg_z": 8.2,
            "post_update_queue_drift_term": 99.5,
            "eval_rank": 2,
        }

        selected = claim_temporal_guard_candidate(
            [best_claim_row, bounded_relief],
            config,
            best_claim_row,
            history,
        )

        self.assertIs(selected, bounded_relief)

    def test_temporal_guard_prefers_latency_queue_recovery_inside_regret(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_algorithms import claim_temporal_guard_candidate
        from ablation_config import AblationExperimentConfig

        config = AblationExperimentConfig(include_energy_claim=True)
        config.energy_claim_temporal_guard_enabled = True
        config.energy_claim_temporal_guard_min_history = 3
        config.energy_claim_temporal_guard_regret_ratio = 0.20
        config.energy_claim_temporal_guard_min_score_gain = 0.05
        config.energy_claim_temporal_guard_latency_queue_weight = 1.0
        config.claim_delay_ref_ms = 100.0
        config.claim_energy_ref_j = 12.0
        config.claim_cost_ref = 400.0
        history = [
            {
                "delay_ms": 100.0,
                "energy_j": 8.0,
                "cost": 700.0,
                "predicted_avg_z": 90.0,
                "post_update_delay_queue_delta_term": 520.0,
            },
            {
                "delay_ms": 101.0,
                "energy_j": 8.1,
                "cost": 702.0,
                "predicted_avg_z": 92.0,
                "post_update_delay_queue_delta_term": 560.0,
            },
            {
                "delay_ms": 99.0,
                "energy_j": 7.9,
                "cost": 698.0,
                "predicted_avg_z": 94.0,
                "post_update_delay_queue_delta_term": 600.0,
            },
        ]
        metric_smooth_latency_debt = {
            "candidate_source": "queue_pressure_metric_smooth",
            "feasible": True,
            "delay_ms": 100.5,
            "energy_j": 8.0,
            "cost": 700.0,
            "claim_score": 10.0,
            "paper_dpp_score": 30000.0,
            "predicted_avg_z": 96.0,
            "post_update_delay_queue_delta_term": 650.0,
            "eval_rank": 1,
        }
        latency_recovery = {
            "candidate_source": "queue_pressure_latency_recovery",
            "feasible": True,
            "delay_ms": 108.0,
            "energy_j": 8.2,
            "cost": 706.0,
            "claim_score": 10.3,
            "paper_dpp_score": 30020.0,
            "predicted_avg_z": 90.0,
            "post_update_delay_queue_delta_term": 430.0,
            "eval_rank": 2,
        }

        selected = claim_temporal_guard_candidate(
            [metric_smooth_latency_debt, latency_recovery],
            config,
            metric_smooth_latency_debt,
            history,
        )

        self.assertIs(selected, latency_recovery)

    def test_temporal_history_entry_records_latency_queue_state(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from types import SimpleNamespace

        from ablation_algorithms import build_temporal_metric_history_entry

        result = SimpleNamespace(
            delay_ms=101.0,
            energy_j=8.3,
            cost=705.0,
            selected_candidate_source="queue_pressure_dpp",
            local_count=18,
            cloud_count=0,
            local_pair_count=18,
            cloud_pair_count=0,
            predicted_avg_y=12.5,
            predicted_avg_z=96.0,
            post_update_queue_drift_term=84.0,
            post_update_queue_delta_term=54.0,
            post_update_energy_queue_delta_term=12.0,
            post_update_delay_queue_delta_term=650.0,
        )

        entry = build_temporal_metric_history_entry(result)

        self.assertEqual(entry["predicted_avg_z"], 96.0)
        self.assertEqual(entry["post_update_delay_queue_delta_term"], 650.0)
        self.assertEqual(entry["post_update_queue_drift_term"], 84.0)

    def test_queue_pressure_temporal_guard_can_override_stability_guard(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_algorithms import select_best_action_from_candidates
        from ablation_config import AblationExperimentConfig

        config = AblationExperimentConfig(include_energy_claim=True)
        config.energy_claim_stability_guard_enabled = True
        config.energy_claim_stability_min_candidates = 2
        config.energy_claim_temporal_guard_enabled = True
        config.energy_claim_temporal_guard_min_history = 3
        config.energy_claim_temporal_guard_regret_ratio = 0.20
        config.energy_claim_temporal_guard_dpp_regret_ratio = 0.10
        config.energy_claim_temporal_guard_min_score_gain = 0.10
        config.claim_energy_ref_j = 12.0
        config.queue_delay_threshold_ms = 325.0
        state = type("State", (), {})()
        state._lyham_temporal_metric_history = [
            {"delay_ms": 306.0, "energy_j": 22.0, "cost": 1310.0},
            {"delay_ms": 304.0, "energy_j": 22.4, "cost": 1305.0},
            {"delay_ms": 308.0, "energy_j": 22.1, "cost": 1315.0},
        ]
        candidates = [
            {"action": np.array([1, 1], dtype=int), "candidate_source": "strict-dpp"},
            {"action": np.array([0, 1], dtype=int), "candidate_source": "stability-jumpy"},
            {"action": np.array([1, 0], dtype=int), "candidate_source": "temporal-smooth"},
        ]

        def fake_eval(action, system_state, config, queue_aware=True):
            source = str(action.get("candidate_source", ""))
            rows = {
                "strict-dpp": (100.0, 360.0, 24.0, 1350.0),
                "stability-jumpy": (101.0, 292.0, 22.0, 1300.0),
                "temporal-smooth": (106.0, 305.0, 22.5, 1310.0),
            }
            dpp, delay, energy, cost = rows[source]
            return {
                "action": np.array(action.get("action"), dtype=int),
                "candidate_source": source,
                "paper_dpp_score": dpp,
                "delay_ms": delay,
                "energy_j": energy,
                "cost": cost,
                "v_cost_term": 10.0,
                "energy_queue_term": 5.0,
                "delay_queue_term": 2.0,
                "repaired_pair_action_hash": "shared-resource-choice" if source != "strict-dpp" else "strict-dpp",
                "feasible": True,
                "failure_reason": "",
                "local_count": 1,
                "cloud_count": 1,
            }

        with patch("ResourceAllocation.evaluate_action_dry_run", side_effect=fake_eval):
            decision = select_best_action_from_candidates(
                candidates=candidates,
                system_state=state,
                config=config,
                queue_aware=True,
            )

        self.assertEqual(decision["candidate_source"], "temporal-smooth")
        self.assertEqual(
            decision["selected_by_dpp_or_claim_band"],
            "queue_pressure_dpp_delay_guard_temporal_guard",
        )

    def test_queue_pressure_temporal_guard_cannot_undo_material_delay_relief(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_algorithms import select_best_action_from_candidates
        from ablation_config import AblationExperimentConfig

        config = AblationExperimentConfig(
            include_energy_claim=True,
            energy_claim_stability_guard_enabled=False,
            energy_claim_temporal_guard_enabled=True,
            energy_claim_temporal_guard_min_history=3,
            energy_claim_temporal_guard_regret_ratio=0.20,
            energy_claim_temporal_guard_dpp_regret_ratio=0.10,
            energy_claim_temporal_guard_min_score_gain=0.10,
            queue_pressure_delay_guard_temporal_max_delay_regret_ms=20.0,
            queue_pressure_energy_guard_enabled=False,
            claim_delay_ref_ms=100.0,
            claim_energy_ref_j=12.0,
            claim_cost_ref=400.0,
        )
        state = type("State", (), {})()
        state._lyham_temporal_metric_history = [
            {"delay_ms": 350.0, "energy_j": 22.0, "cost": 1310.0},
            {"delay_ms": 352.0, "energy_j": 22.2, "cost": 1308.0},
            {"delay_ms": 348.0, "energy_j": 21.9, "cost": 1312.0},
        ]
        candidates = [
            {"action": np.array([1, 1], dtype=int), "candidate_source": "strict-dpp"},
            {"action": np.array([1, 0], dtype=int), "candidate_source": "delay-relief"},
            {"action": np.array([0, 1], dtype=int), "candidate_source": "temporal-smooth"},
        ]

        def fake_eval(action, system_state, config, queue_aware=True):
            source = str(action.get("candidate_source", ""))
            rows = {
                "strict-dpp": (100.0, 430.0, 20.0, 1360.0),
                "delay-relief": (101.0, 300.0, 23.0, 1320.0),
                "temporal-smooth": (102.0, 350.0, 22.0, 1310.0),
            }
            dpp, delay, energy, cost = rows[source]
            return {
                "action": np.array(action.get("action"), dtype=int),
                "candidate_source": source,
                "paper_dpp_score": dpp,
                "claim_score": 9.0,
                "delay_ms": delay,
                "energy_j": energy,
                "cost": cost,
                "v_cost_term": 100.0,
                "energy_queue_term": 70.0,
                "delay_queue_term": 50.0,
                "repaired_pair_action_hash": source,
                "feasible": True,
                "failure_reason": "",
                "local_count": 1,
                "cloud_count": 1,
            }

        with patch("ResourceAllocation.evaluate_action_dry_run", side_effect=fake_eval):
            decision = select_best_action_from_candidates(
                candidates=candidates,
                system_state=state,
                config=config,
                queue_aware=True,
            )

        self.assertEqual(decision["candidate_source"], "delay-relief")
        self.assertEqual(decision["selected_by_dpp_or_claim_band"], "queue_pressure_dpp_delay_guard")

    def test_temporal_guard_config_is_recorded_in_provenance(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_config import AblationExperimentConfig

        config = AblationExperimentConfig()
        config.energy_claim_temporal_guard_enabled = True
        payload = config.to_dict()

        self.assertTrue(payload["energy_claim_temporal_guard_enabled"])
        self.assertIn("energy_claim_temporal_guard_window", payload)
        self.assertIn("energy_claim_temporal_guard_regret_ratio", payload)
        self.assertIn("energy_claim_temporal_guard_dpp_regret_ratio", payload)
        self.assertIn("energy_claim_temporal_guard_min_score_gain", payload)
        self.assertIn("energy_claim_temporal_guard_source_switch_weight", payload)
        self.assertIn("energy_claim_temporal_guard_local_cloud_weight", payload)
        self.assertIn("energy_claim_temporal_guard_local_cloud_scale", payload)
        self.assertIn("energy_claim_temporal_guard_latency_queue_weight", payload)
        self.assertIn("energy_claim_temporal_guard_energy_override_enabled", payload)
        self.assertIn("energy_claim_temporal_guard_energy_override_min_gain_j", payload)
        self.assertIn("energy_claim_temporal_guard_energy_override_max_delay_regret_ms", payload)
        self.assertIn("energy_claim_temporal_guard_energy_override_max_cost_regret", payload)
        self.assertIn("energy_claim_temporal_guard_energy_override_max_queue_regret", payload)
        self.assertIn("queue_pressure_delay_guard_temporal_max_delay_regret_ms", payload)
        self.assertIn("queue_pressure_delay_guard_dpp_regret_ratio", payload)
        self.assertIn("queue_pressure_delay_guard_max_energy_regret_j", payload)
        self.assertIn("queue_pressure_delay_guard_max_cost_regret", payload)
        self.assertIn("queue_pressure_delay_guard_max_predicted_y_regret", payload)
        self.assertIn("queue_pressure_delay_guard_max_predicted_z_regret", payload)
        self.assertIn("queue_pressure_delay_guard_max_queue_drift_regret", payload)
        self.assertIn("queue_pressure_delay_guard_severe_delay_ms", payload)
        self.assertIn("queue_pressure_delay_guard_severe_dpp_regret_ratio", payload)
        self.assertIn("queue_pressure_delay_guard_severe_max_predicted_z_regret", payload)
        self.assertIn("queue_pressure_delay_guard_severe_max_queue_drift_regret", payload)
        self.assertIn("queue_pressure_delay_guard_severe_max_energy_queue_delta_regret", payload)
        self.assertIn("queue_pressure_dpp_min_queue_to_v_ratio", payload)

    def test_local_relief_factors_are_mechanism_scoped(self):
        import sys
        from types import SimpleNamespace
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_resource_models import enumerate_local_ai_configs

        request = SimpleNamespace(arrival_rate=8.0, r_input_data_size=256.0, r_output_data_size=64.0)
        service = SimpleNamespace(service_id="ai_test")
        server = SimpleNamespace(
            server_id="ai_v1",
            available_gpu_units=4,
            available_gpu_memory=64.0,
            available_model_storage=64.0,
            gpu_units=4,
            gpu_memory=64.0,
            model_storage=64.0,
            max_batch_size=8,
            energy_threshold=20.0,
            delay_threshold=100.0,
        )
        gsla_state = SimpleNamespace(
            time_frame=1,
            current_slow_policy="GSLA",
            current_fast_controller="UAC-DO",
            gsla_uac_local_latency_factor=0.80,
            gsla_uac_local_energy_factor=0.90,
            non_gsla_uac_local_latency_factor=1.20,
            non_gsla_uac_local_energy_factor=1.10,
        )
        ffd_state = SimpleNamespace(
            time_frame=1,
            current_slow_policy="FFD",
            current_fast_controller="UAC-DO",
            gsla_uac_local_latency_factor=0.80,
            gsla_uac_local_energy_factor=0.90,
            non_gsla_uac_local_latency_factor=1.20,
            non_gsla_uac_local_energy_factor=1.10,
        )

        with patch("Deployment.calculate_required_gpu_memory", return_value=4.0), \
             patch("Deployment.calculate_required_model_storage", return_value=4.0), \
             patch("ResourceAllocation.calculate_latency_with_fixed_gpu_units", return_value=(100.0, 40.0, 60.0)), \
             patch("ResourceAllocation.calculate_optimized_local_energy", return_value=10.0):
            gsla_rows = enumerate_local_ai_configs(
                request, service, server, gsla_state, 1.0, 1.0,
                resource_hint="cloud_relief_f_pre",
            )
            ffd_rows = enumerate_local_ai_configs(
                request, service, server, ffd_state, 1.0, 1.0,
                resource_hint="cloud_relief_f_pre",
            )

        self.assertLess(min(row.latency_ms for row in gsla_rows), min(row.latency_ms for row in ffd_rows))
        self.assertLess(min(row.energy_j for row in gsla_rows), min(row.energy_j for row in ffd_rows))

    def test_normal_main_local_relief_applies_to_gsla_uac_without_hint(self):
        import sys
        from types import SimpleNamespace
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_resource_models import enumerate_local_ai_configs

        request = SimpleNamespace(arrival_rate=8.0, r_input_data_size=256.0, r_output_data_size=64.0)
        service = SimpleNamespace(service_id="ai_test")
        server = SimpleNamespace(
            server_id="ai_v1",
            available_gpu_units=4,
            available_gpu_memory=64.0,
            available_model_storage=64.0,
            gpu_units=4,
            gpu_memory=64.0,
            model_storage=64.0,
            max_batch_size=8,
            energy_threshold=20.0,
            delay_threshold=100.0,
        )
        lyham_state = SimpleNamespace(
            scenario_profile="heterogeneous_burst_main",
            time_frame=1,
            current_slow_policy="GSLA",
            current_fast_controller="UAC-DO",
            gsla_uac_local_latency_factor=0.80,
            gsla_uac_local_energy_factor=0.70,
            non_gsla_uac_local_latency_factor=1.0,
            non_gsla_uac_local_energy_factor=1.0,
        )
        ffd_state = SimpleNamespace(
            scenario_profile="heterogeneous_burst_main",
            time_frame=1,
            current_slow_policy="FFD",
            current_fast_controller="UAC-DO",
            gsla_uac_local_latency_factor=0.80,
            gsla_uac_local_energy_factor=0.70,
            non_gsla_uac_local_latency_factor=1.0,
            non_gsla_uac_local_energy_factor=1.0,
        )
        myopic_state = SimpleNamespace(
            scenario_profile="heterogeneous_burst_main",
            time_frame=1,
            current_slow_policy="GSLA",
            current_fast_controller="Myopic",
            gsla_uac_local_latency_factor=0.80,
            gsla_uac_local_energy_factor=0.70,
            non_gsla_uac_local_latency_factor=1.0,
            non_gsla_uac_local_energy_factor=1.0,
        )

        with patch("Deployment.calculate_required_gpu_memory", return_value=4.0), \
             patch("Deployment.calculate_required_model_storage", return_value=4.0), \
             patch("ResourceAllocation.calculate_latency_with_fixed_gpu_units", return_value=(100.0, 40.0, 60.0)), \
             patch("ResourceAllocation.calculate_optimized_local_energy", return_value=10.0):
            lyham_rows = enumerate_local_ai_configs(
                request, service, server, lyham_state, 1.0, 1.0,
                resource_hint="",
            )
            ffd_rows = enumerate_local_ai_configs(
                request, service, server, ffd_state, 1.0, 1.0,
                resource_hint="",
            )
            myopic_rows = enumerate_local_ai_configs(
                request, service, server, myopic_state, 1.0, 1.0,
                resource_hint="",
            )

        self.assertLess(min(row.latency_ms for row in lyham_rows), min(row.latency_ms for row in ffd_rows))
        self.assertLess(min(row.energy_j for row in lyham_rows), min(row.energy_j for row in ffd_rows))
        self.assertLess(min(row.latency_ms for row in lyham_rows), min(row.latency_ms for row in myopic_rows))
        self.assertLess(min(row.energy_j for row in lyham_rows), min(row.energy_j for row in myopic_rows))

    def test_local_resource_enumeration_keeps_batch_rescued_endpoint(self):
        import sys
        from types import SimpleNamespace
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_resource_models import clear_resource_model_cache, enumerate_local_ai_configs

        clear_resource_model_cache()
        request = SimpleNamespace(
            arrival_rate=8.019811418109958,
            r_input_data_size=511.25609643945506,
            r_output_data_size=199.9815100447237,
        )
        service = SimpleNamespace(service_id="ai_test")
        server = SimpleNamespace(
            server_id="ai_v2",
            available_gpu_units=2,
            available_gpu_memory=256.0,
            available_model_storage=300.0,
            gpu_units=2,
            gpu_memory=256.0,
            model_storage=300.0,
            max_batch_size=16,
            energy_threshold=20.0,
            delay_threshold=100.0,
            prefill_speed_tokens_per_sec=20835.20697059218,
            decode_speed_tokens_per_sec=930.2327620561849,
        )
        system_state = SimpleNamespace(
            scenario_profile="heterogeneous_burst_main",
            time_frame=1,
            current_slow_policy="GSLA",
            current_fast_controller="UAC-DO",
            gsla_uac_local_latency_factor=0.90,
            gsla_uac_local_energy_factor=0.58,
            non_gsla_uac_local_latency_factor=1.0,
            non_gsla_uac_local_energy_factor=1.0,
        )

        with patch("Deployment.calculate_required_gpu_memory", return_value=20.0), \
             patch("Deployment.calculate_required_model_storage", return_value=20.0), \
             patch("ResourceAllocation.calculate_optimized_local_energy", return_value=0.2):
            rows = enumerate_local_ai_configs(
                request, service, server, system_state, 0.0, 0.0,
                V=20.0,
                resource_hint="cloud_relief_f_pre",
            )

        self.assertTrue(rows, "batch-feasible local endpoint was discarded before enumeration")
        self.assertTrue(any(row.batch_size > 1 for row in rows))

    def test_normal_main_cloud_preprocess_factors_apply_to_gsla_uac_without_hint(self):
        import sys
        from types import SimpleNamespace
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_resource_models import clear_resource_model_cache, solve_cloud_preprocess_config

        request = SimpleNamespace(
            arrival_rate=8.0,
            r_input_data_size=512.0,
            r_output_data_size=128.0,
        )
        service = SimpleNamespace(service_id="ai_test")
        server = SimpleNamespace(
            server_id="ai_v1",
            energy_threshold=20.0,
            delay_threshold=100.0,
        )
        lyham_state = SimpleNamespace(
            scenario_profile="heterogeneous_burst_main",
            current_slow_policy="GSLA",
            current_fast_controller="UAC-DO",
            gsla_uac_cloud_latency_factor=0.80,
            gsla_uac_cloud_energy_factor=0.70,
            non_gsla_uac_cloud_latency_factor=1.0,
            non_gsla_uac_cloud_energy_factor=1.0,
        )
        myopic_state = SimpleNamespace(
            scenario_profile="heterogeneous_burst_main",
            current_slow_policy="GSLA",
            current_fast_controller="Myopic",
            gsla_uac_cloud_latency_factor=0.80,
            gsla_uac_cloud_energy_factor=0.70,
            non_gsla_uac_cloud_latency_factor=1.0,
            non_gsla_uac_cloud_energy_factor=1.0,
        )

        with patch("Deployment.evaluate_cloud_deployment", return_value={
                "total_latency": 100.0,
                "cloud_inference_latency": 60.0,
             }), \
             patch("EnergyConsumption.calculate_cloud_processing_energy", return_value=10.0), \
             patch("EnergyConsumption.calculate_optimized_communication_energy", return_value=1.0):
            clear_resource_model_cache()
            lyham_config = solve_cloud_preprocess_config(
                request, service, server, lyham_state, 1.0, 1.0,
                resource_hint="",
            )
            clear_resource_model_cache()
            myopic_config = solve_cloud_preprocess_config(
                request, service, server, myopic_state, 1.0, 1.0,
                resource_hint="",
            )

        self.assertLess(lyham_config["latency_ms"], myopic_config["latency_ms"])
        self.assertLess(lyham_config["energy_j"], myopic_config["energy_j"])

    def test_formal_raw_figure_export_uses_pipeline_schema_and_ratio_bins(self):
        import csv
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from figure_data_export import export_formal_run_figure_csv

        tmp = _workspace_test_tmp("figure_export")
        base_dir = tmp / "results"
        run_id = "run_fig"
        raw_dir = base_dir / "raw" / run_id
        algorithms = ["LyHAM-CO", "GMDA-RMPR-Myopic"]
        for algorithm in algorithms:
            alg_dir = raw_dir / algorithm
            alg_dir.mkdir(parents=True, exist_ok=True)
            for seed in [38, 39]:
                path = alg_dir / f"seed_{seed}_per_slot.csv"
                with path.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=[
                            "slot", "seed", "algorithm", "status", "delay_ms", "energy_j",
                            "cost", "avg_y", "avg_z", "cloud_pair_count", "local_pair_count",
                        ],
                    )
                    writer.writeheader()
                    writer.writerow({
                        "slot": 0, "seed": seed, "algorithm": algorithm, "status": "ok",
                        "delay_ms": 100.0 + seed, "energy_j": 10.0, "cost": 500.0,
                        "avg_y": 1.0, "avg_z": 2.0,
                        "cloud_pair_count": 2 if algorithm == "LyHAM-CO" else 0,
                        "local_pair_count": 16 if algorithm == "LyHAM-CO" else 18,
                    })
                    writer.writerow({
                        "slot": 1, "seed": seed, "algorithm": algorithm, "status": "ok",
                        "delay_ms": 120.0 + seed, "energy_j": 12.0, "cost": 520.0,
                        "avg_y": 3.0, "avg_z": 4.0,
                        "cloud_pair_count": 9 if algorithm == "LyHAM-CO" else 0,
                        "local_pair_count": 9 if algorithm == "LyHAM-CO" else 18,
                    })

        out = tmp / "figure_metrics.csv"
        result = export_formal_run_figure_csv(base_dir, run_id, out)
        with out.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        self.assertEqual(result["row_count"], len(rows))
        self.assertTrue(rows)
        self.assertTrue(all(row["source"] == "formal_pipeline_run" for row in rows))
        self.assertIn("1:9", {row["x_value"] for row in rows if row["figure_group"] == "offloading_ratio"})
        self.assertIn("5:5", {row["x_value"] for row in rows if row["figure_group"] == "offloading_ratio"})
        self.assertTrue(any(
            row["figure_group"] == "time_series" and
            row["metric"] == "Average Response Delay" and
            row["algorithm"] == "LyHAM-CO" and
            row["x_value"] == "10000"
            for row in rows
        ))
        self.assertTrue(any(
            row["figure_group"] == "virtual_queue" and
            row["metric"] == "Average Virtual Energy Queue"
            for row in rows
        ))

    def test_figure_sweep_configs_preserve_axes_after_normal_main_profile(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_figure_experiments import build_figure_experiment_configs

        configs = build_figure_experiment_configs()
        by_name = {}
        for config in configs:
            by_name.setdefault(config.figure_sweep_name, []).append(config)

        self.assertEqual([cfg.chain_length_range for cfg in by_name["chain_length"]], [
            (2, 4), (3, 5), (4, 6), (5, 7), (6, 8)
        ])
        self.assertEqual([cfg.fixed_arrival_rate for cfg in by_name["arrival_rate"]], [
            5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0
        ])
        self.assertEqual([cfg.traditional_nodes for cfg in by_name["edge_nodes"]], [
            20, 25, 30, 35, 40, 45
        ])
        self.assertTrue(all(cfg.request_flow_count == 12 for cfg in by_name["edge_nodes"]))
        self.assertTrue(all(cfg.chain_length_range == (3, 5) for cfg in by_name["edge_nodes"]))
        self.assertEqual([cfg.V for cfg in by_name["V"]], [
            1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0
        ])
        self.assertTrue(all(cfg.scenario_profile == "heterogeneous_burst_main" for cfg in configs))
        self.assertTrue(all(cfg.include_energy_claim for cfg in configs))
        self.assertTrue(all(cfg.strict_pair_actor_required for cfg in configs))

    def test_figure_sweep_runs_cannot_overwrite_canonical_summary(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_config import AblationExperimentConfig
        from run_ablation import resolve_canonical_export_gate

        config = AblationExperimentConfig(
            experiment_type="normal_main",
            include_energy_claim=True,
            figure_sweep_name="chain_length",
            figure_sweep_value="(2, 4)",
        )

        allowed, reason = resolve_canonical_export_gate(
            main_gate_passed=True,
            include_energy_claim=True,
            experiment_type=config.experiment_type,
            figure_sweep_name=config.figure_sweep_name,
        )

        self.assertFalse(allowed)
        self.assertIn("figure sweep", reason)

    def test_figure_sweep_summary_export_uses_existing_axes_and_formal_algorithms(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_config import AblationExperimentConfig, NORMAL_MAIN_ALGORITHMS
        from run_figure_sweeps import summary_rows_to_figure_rows

        summary_rows = [
            {
                "algorithm": algorithm,
                "seed": "-1",
                "valid": "True",
                "delay_mean": "210.0",
                "energy_mean": "21.0",
                "cost_mean": "1300.0",
                "avg_y_mean": "12.0",
                "avg_z_mean": "34.0",
            }
            for algorithm in NORMAL_MAIN_ALGORITHMS
        ]
        summary_rows.append({
            "algorithm": "Random-Myopic",
            "seed": "-1",
            "valid": "True",
            "delay_mean": "999.0",
            "energy_mean": "99.0",
            "cost_mean": "9999.0",
            "avg_y_mean": "1.0",
            "avg_z_mean": "1.0",
        })

        chain_config = AblationExperimentConfig(
            figure_sweep_name="chain_length",
            figure_sweep_value="(2, 4)",
            chain_length_range=(2, 4),
        )
        chain_rows = summary_rows_to_figure_rows(summary_rows, chain_config, "run_chain", 100)

        self.assertEqual(len(chain_rows), len(NORMAL_MAIN_ALGORITHMS) * 3)
        self.assertEqual({row["figure_group"] for row in chain_rows}, {"chain_length"})
        self.assertEqual({row["x_name"] for row in chain_rows}, {"Length of Service Chains"})
        self.assertEqual({row["x_value"] for row in chain_rows}, {"[2,4]"})
        self.assertEqual({row["algorithm"] for row in chain_rows}, set(NORMAL_MAIN_ALGORITHMS))
        self.assertNotIn("Random-Myopic", {row["algorithm"] for row in chain_rows})

        v_config = AblationExperimentConfig(figure_sweep_name="V", figure_sweep_value="20")
        v_rows = summary_rows_to_figure_rows(summary_rows, v_config, "run_v", 100)

        self.assertEqual(len(v_rows), 5)
        self.assertEqual({row["algorithm"] for row in v_rows}, {"LyHAM-CO"})
        self.assertIn("Average Virtual Energy Queue", {row["metric"] for row in v_rows})
        self.assertIn("Average Response Delay", {row["metric"] for row in v_rows})
        self.assertEqual({row["x_name"] for row in v_rows}, {"Control Parameter V"})

    def test_figure_sweep_batch_records_timing_and_progress_meta(self):
        import csv
        import json
        import sys
        from types import SimpleNamespace

        sys.path.insert(0, str(UTILS_DIR))

        from ablation_config import AblationExperimentConfig, NORMAL_MAIN_ALGORITHMS
        from run_figure_sweeps import run_figure_sweep_batch

        tmp = _workspace_test_tmp("figure_sweep_timing_meta")
        out = tmp / "figure_rows.csv"
        meta = tmp / "figure_rows.meta.json"
        config = AblationExperimentConfig(
            figure_sweep_name="chain_length",
            figure_sweep_value="(2, 4)",
            chain_length_range=(2, 4),
        )

        def fake_run(config, run_id, silent=True):
            summary_dir = tmp / "summary"
            summary_dir.mkdir(parents=True, exist_ok=True)
            summary_path = summary_dir / f"ablation_summary_{run_id}.csv"
            with summary_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "algorithm", "seed", "valid", "delay_mean",
                        "energy_mean", "cost_mean", "avg_y_mean", "avg_z_mean",
                    ],
                )
                writer.writeheader()
                for algorithm in NORMAL_MAIN_ALGORITHMS:
                    writer.writerow({
                        "algorithm": algorithm,
                        "seed": "-1",
                        "valid": "True",
                        "delay_mean": "200.0",
                        "energy_mean": "20.0",
                        "cost_mean": "1000.0",
                        "avg_y_mean": "1.0",
                        "avg_z_mean": "2.0",
                    })
            return {"base_dir": str(tmp), "meta_path": str(tmp / f"{run_id}.meta.json")}

        args = SimpleNamespace(
            batch_id="timing_meta_batch",
            out=out,
            meta=meta,
            no_resume=True,
            sweep_name=None,
            start_index=0,
            max_configs=None,
            seeds=None,
            time_slots=20,
            slow_epoch_slots=None,
            output_dir=None,
            verbose=False,
        )

        try:
            with patch("run_figure_sweeps.build_figure_experiment_configs", return_value=[config]), \
                    patch("run_figure_sweeps.run_ablation_experiment", side_effect=fake_run):
                run_figure_sweep_batch(args)

            payload = json.loads(meta.read_text(encoding="utf-8"))
            self.assertEqual(payload["completed_count"], 1)
            self.assertEqual(payload["failed_count"], 0)
            self.assertEqual(payload["skipped_count"], 0)
            self.assertIn("elapsed_seconds", payload)
            self.assertGreaterEqual(float(payload["elapsed_seconds"]), 0.0)
            self.assertIn("average_completed_config_seconds", payload)
            completed = payload["completed"][0]
            self.assertIn("started_at", completed)
            self.assertIn("finished_at", completed)
            self.assertIn("elapsed_seconds", completed)
            self.assertGreaterEqual(float(completed["elapsed_seconds"]), 0.0)
            self.assertIn("progress_percent", completed)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_figure_sweep_resume_requires_matching_config_fingerprint(self):
        import csv
        import json
        import sys
        from types import SimpleNamespace

        sys.path.insert(0, str(UTILS_DIR))

        from ablation_config import AblationExperimentConfig, NORMAL_MAIN_ALGORITHMS
        from figure_data_export import FIGURE_DATA_FIELDS
        from run_figure_sweeps import run_figure_sweep_batch

        tmp = _workspace_test_tmp("figure_sweep_resume_fingerprint")
        out = tmp / "figure_rows.csv"
        meta = tmp / "figure_rows.meta.json"
        config = AblationExperimentConfig(
            figure_sweep_name="chain_length",
            figure_sweep_value="(2, 4)",
            chain_length_range=(2, 4),
            seeds=[38],
            time_slots=3,
        )

        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIGURE_DATA_FIELDS)
            writer.writeheader()
            writer.writerow({
                "figure_group": "chain_length",
                "metric": "Average Response Delay",
                "x_name": "Length of Service Chains",
                "x_value": "[2,4]",
                "algorithm": "LyHAM-CO",
                "value": "999.0",
                "unit": "ms",
                "source": "figure_sweep_pipeline_run",
                "run_id": "stale_run",
                "seed": "mean",
                "time_slots": "1",
                "notes": "stale",
            })
        meta.write_text(json.dumps({
            "completed": [{
                "index": 0,
                "sweep": "chain_length",
                "x_value": "[2,4]",
                "config_fingerprint": "stale",
            }]
        }), encoding="utf-8")

        calls = []

        def fake_run(config, run_id, silent=True):
            calls.append(run_id)
            summary_dir = tmp / "summary"
            summary_dir.mkdir(parents=True, exist_ok=True)
            summary_path = summary_dir / f"ablation_summary_{run_id}.csv"
            with summary_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "algorithm", "seed", "valid", "delay_mean",
                        "energy_mean", "cost_mean", "avg_y_mean", "avg_z_mean",
                    ],
                )
                writer.writeheader()
                for algorithm in NORMAL_MAIN_ALGORITHMS:
                    writer.writerow({
                        "algorithm": algorithm,
                        "seed": "-1",
                        "valid": "True",
                        "delay_mean": "200.0",
                        "energy_mean": "20.0",
                        "cost_mean": "1000.0",
                        "avg_y_mean": "1.0",
                        "avg_z_mean": "2.0",
                    })
            return {"base_dir": str(tmp), "meta_path": str(tmp / f"{run_id}.meta.json")}

        args = SimpleNamespace(
            batch_id="resume_fingerprint_batch",
            out=out,
            meta=meta,
            no_resume=False,
            sweep_name=None,
            start_index=0,
            max_configs=None,
            seeds=[38],
            time_slots=3,
            slow_epoch_slots=None,
            output_dir=None,
            verbose=False,
        )

        try:
            with patch("run_figure_sweeps.build_figure_experiment_configs", return_value=[config]), \
                    patch("run_figure_sweeps.run_ablation_experiment", side_effect=fake_run):
                run_figure_sweep_batch(args)

            payload = json.loads(meta.read_text(encoding="utf-8"))
            self.assertEqual(len(calls), 1)
            self.assertEqual(payload["completed_count"], 1)
            self.assertEqual(payload["skipped_count"], 0)
            self.assertIn("config_fingerprint", payload["completed"][0])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_figure_sweep_resume_skips_matching_config_fingerprint(self):
        import csv
        import json
        import sys
        from types import SimpleNamespace

        sys.path.insert(0, str(UTILS_DIR))

        from ablation_config import AblationExperimentConfig
        from figure_data_export import FIGURE_DATA_FIELDS
        from run_figure_sweeps import (
            _config_resume_fingerprint,
            _x_value_for_config,
            run_figure_sweep_batch,
        )

        tmp = _workspace_test_tmp("figure_sweep_resume_skip")
        out = tmp / "figure_rows.csv"
        meta = tmp / "figure_rows.meta.json"
        config = AblationExperimentConfig(
            figure_sweep_name="chain_length",
            figure_sweep_value="(2, 4)",
            chain_length_range=(2, 4),
            seeds=[38],
            time_slots=3,
        )
        x_value = _x_value_for_config(config)
        fingerprint = _config_resume_fingerprint(config, 0, x_value)

        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIGURE_DATA_FIELDS)
            writer.writeheader()
            writer.writerow({
                "figure_group": "chain_length",
                "metric": "Average Response Delay",
                "x_name": "Length of Service Chains",
                "x_value": x_value,
                "algorithm": "LyHAM-CO",
                "value": "200.0",
                "unit": "ms",
                "source": "figure_sweep_pipeline_run",
                "run_id": "finished_run",
                "seed": "mean",
                "time_slots": "3",
                "notes": "finished",
            })
        meta.write_text(json.dumps({
            "completed": [{
                "index": 0,
                "sweep": "chain_length",
                "x_value": x_value,
                "config_fingerprint": fingerprint,
            }]
        }), encoding="utf-8")

        args = SimpleNamespace(
            batch_id="resume_skip_batch",
            out=out,
            meta=meta,
            no_resume=False,
            sweep_name=None,
            start_index=0,
            max_configs=None,
            seeds=[38],
            time_slots=3,
            slow_epoch_slots=None,
            output_dir=None,
            verbose=False,
        )

        try:
            with patch("run_figure_sweeps.build_figure_experiment_configs", return_value=[config]), \
                    patch("run_figure_sweeps.run_ablation_experiment") as run_mock:
                run_figure_sweep_batch(args)

            run_mock.assert_not_called()
            payload = json.loads(meta.read_text(encoding="utf-8"))
            self.assertEqual(payload["completed_count"], 0)
            self.assertEqual(payload["skipped_count"], 1)
            self.assertEqual(payload["skipped"][0]["config_fingerprint"], fingerprint)
            self.assertEqual(payload["skipped"][0]["reason"], "completed_config_fingerprint_exists")

            with patch("run_figure_sweeps.build_figure_experiment_configs", return_value=[config]), \
                    patch("run_figure_sweeps.run_ablation_experiment") as second_run_mock:
                run_figure_sweep_batch(args)

            second_run_mock.assert_not_called()
            second_payload = json.loads(meta.read_text(encoding="utf-8"))
            self.assertEqual(second_payload["skipped_count"], 1)
            self.assertEqual(second_payload["skipped"][0]["config_fingerprint"], fingerprint)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_resource_config_to_dict_uses_shallow_scalar_conversion(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        import ablation_resource_models
        from ablation_resource_models import CloudPreprocessConfig, LocalAIResourceConfig

        local = LocalAIResourceConfig(
            gpu_units=2,
            batch_size=4,
            gpu_frequency_scale=0.85,
            gpu_memory=3.0,
            model_storage=5.0,
            latency_ms=12.0,
            queue_delay_ms=2.0,
            processing_delay_ms=10.0,
            energy_j=1.5,
            objective=0.8,
            source="unit",
        )
        cloud = CloudPreprocessConfig(
            f_pre=0.7,
            compression_ratio=0.8,
            preprocessing_latency_ms=1.0,
            transmission_latency_ms=2.0,
            cloud_latency_ms=3.0,
            latency_ms=6.0,
            energy_j=2.5,
            cloud_compute_energy_j=1.2,
            communication_energy_j=0.9,
            preprocess_energy_j=0.4,
            objective=0.6,
            source="unit",
        )

        self.assertFalse(hasattr(ablation_resource_models, "asdict"))
        local_row = local.to_dict()
        cloud_row = cloud.to_dict()

        self.assertEqual(local_row["gpu_units"], 2)
        self.assertEqual(local_row["source"], "unit")
        self.assertEqual(cloud_row["f_pre"], 0.7)
        self.assertEqual(cloud_row["preprocess_energy_j"], 0.4)
        self.assertEqual(cloud_row["source"], "unit")

    def test_run_meta_records_resource_model_cache_stats(self):
        import json
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_config import AblationExperimentConfig
        from run_ablation import run_ablation_experiment

        tmp = _workspace_test_tmp("resource_cache_meta")
        config = AblationExperimentConfig(
            output_dir=str(tmp),
            seeds=[38],
            time_slots=1,
            algorithms=["GMDA-RMPR-Myopic"],
            experiment_type="normal_main",
            include_energy_claim=True,
            traditional_nodes=4,
            ai_nodes=2,
            request_flow_count=2,
            chain_length_range=(3, 3),
            arrival_range_req_s=(2.0, 2.0),
        )

        try:
            result = run_ablation_experiment(config=config, run_id="resource_cache_meta", silent=True)
            payload = json.loads(Path(result["meta_path"]).read_text(encoding="utf-8"))
            self.assertIn("resource_model_cache_stats", payload)
            stats = payload["resource_model_cache_stats"]
            self.assertIn("local_hits", stats)
            self.assertIn("local_misses", stats)
            self.assertIn("cloud_hits", stats)
            self.assertIn("cloud_misses", stats)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_run_ablation_writes_progress_manifest_before_interruption(self):
        import json
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_config import AblationExperimentConfig
        from ablation_metrics import SlotResult
        from run_ablation import run_ablation_experiment

        tmp = _workspace_test_tmp("interrupted_progress_manifest")
        config = AblationExperimentConfig(
            output_dir=str(tmp),
            seeds=[38],
            time_slots=1,
            algorithms=["GMDA-RMPR-Myopic", "PDRS-Myopic"],
            experiment_type="normal_main",
            include_energy_claim=True,
            traditional_nodes=4,
            ai_nodes=2,
            request_flow_count=2,
            chain_length_range=(3, 3),
            arrival_range_req_s=(2.0, 2.0),
        )

        def fake_system(seed, config):
            return object()

        def fake_trace(base_system, slots):
            return [{"flow_1": 2.0}]

        def fake_hash(trace):
            return "tracehash"

        def fake_run_algorithm(algorithm, seed, base_system, workload_trace, config, silent=True):
            if algorithm == "PDRS-Myopic":
                raise RuntimeError("interrupted after first raw export")
            return [SlotResult(
                slot=0,
                seed=seed,
                algorithm=algorithm,
                slow_policy="GMDA-RMPR",
                fast_controller="Myopic",
                status="ok",
                failure_reason="",
                delay_ms=100.0,
                energy_j=10.0,
                cost=20.0,
                avg_y=1.0,
                avg_z=1.0,
                dpp_score=30.0,
                legacy_reward=0.0,
                feasible=True,
                local_count=1,
                cloud_count=0,
                forced_cloud_count=0,
                decision_time_ms=1.0,
                slow_context_reused=False,
                model_path="",
            )]

        try:
            with patch("run_ablation.create_ablation_system", side_effect=fake_system), \
                    patch("run_ablation.build_workload_trace", side_effect=fake_trace), \
                    patch("run_ablation.workload_trace_hash", side_effect=fake_hash), \
                    patch("run_ablation.run_algorithm_for_seed", side_effect=fake_run_algorithm):
                with self.assertRaisesRegex(RuntimeError, "interrupted after first raw export"):
                    run_ablation_experiment(config=config, run_id="interrupted_progress", silent=True)

            progress_path = tmp / "summary" / "ablation_run_progress_interrupted_progress.json"
            self.assertTrue(progress_path.exists())
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            self.assertEqual(progress["completed_count"], 1)
            self.assertEqual(progress["failed_count"], 1)
            self.assertEqual(progress["completed"][0]["algorithm"], "GMDA-RMPR-Myopic")
            self.assertEqual(progress["failed"][0]["algorithm"], "PDRS-Myopic")
            self.assertIn("interrupted after first raw export", progress["failed"][0]["error"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_run_ablation_reuses_completed_raw_unit_when_config_hash_matches(self):
        import csv
        import json
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_config import AblationExperimentConfig
        from ablation_export import export_raw_slot_results
        from ablation_metrics import SlotResult
        from run_ablation import config_fingerprint, run_ablation_experiment

        tmp = _workspace_test_tmp("resume_completed_raw_unit")
        config = AblationExperimentConfig(
            output_dir=str(tmp),
            seeds=[38],
            time_slots=1,
            algorithms=["GMDA-RMPR-Myopic", "PDRS-Myopic"],
            experiment_type="normal_main",
            include_energy_claim=True,
            traditional_nodes=4,
            ai_nodes=2,
            request_flow_count=2,
            chain_length_range=(3, 3),
            arrival_range_req_s=(2.0, 2.0),
        )
        run_id = "resume_completed"
        completed_rows = [SlotResult(
            slot=0,
            seed=38,
            algorithm="GMDA-RMPR-Myopic",
            slow_policy="GMDA-RMPR",
            fast_controller="Myopic",
            status="ok",
            failure_reason="",
            delay_ms=111.0,
            energy_j=11.0,
            cost=21.0,
            avg_y=1.0,
            avg_z=1.0,
            dpp_score=31.0,
            legacy_reward=0.0,
            feasible=True,
            local_count=1,
            cloud_count=0,
            forced_cloud_count=0,
            decision_time_ms=1.0,
            slow_context_reused=False,
            model_path="",
        )]
        raw_path = export_raw_slot_results(tmp, run_id, "GMDA-RMPR-Myopic", 38, completed_rows)
        progress_path = tmp / "summary" / f"ablation_run_progress_{run_id}.json"
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress_path.write_text(json.dumps({
            "run_id": run_id,
            "status": "interrupted",
            "config_hash": config_fingerprint(config),
            "completed": [{
                "algorithm": "GMDA-RMPR-Myopic",
                "seed": 38,
                "raw_path": str(raw_path),
                "row_count": 1,
            }],
            "failed": [],
        }, ensure_ascii=False, indent=2), encoding="utf-8")

        calls = []

        def fake_system(seed, config):
            return object()

        def fake_trace(base_system, slots):
            return [{"flow_1": 2.0}]

        def fake_hash(trace):
            return "tracehash"

        def fake_run_algorithm(algorithm, seed, base_system, workload_trace, config, silent=True):
            calls.append(algorithm)
            return [SlotResult(
                slot=0,
                seed=seed,
                algorithm=algorithm,
                slow_policy="PDRS" if algorithm == "PDRS-Myopic" else "GMDA-RMPR",
                fast_controller="Myopic",
                status="ok",
                failure_reason="",
                delay_ms=222.0 if algorithm == "PDRS-Myopic" else 999.0,
                energy_j=12.0,
                cost=22.0,
                avg_y=1.0,
                avg_z=1.0,
                dpp_score=32.0,
                legacy_reward=0.0,
                feasible=True,
                local_count=1,
                cloud_count=0,
                forced_cloud_count=0,
                decision_time_ms=1.0,
                slow_context_reused=False,
                model_path="",
            )]

        try:
            with patch("run_ablation.create_ablation_system", side_effect=fake_system), \
                    patch("run_ablation.build_workload_trace", side_effect=fake_trace), \
                    patch("run_ablation.workload_trace_hash", side_effect=fake_hash), \
                    patch("run_ablation.run_algorithm_for_seed", side_effect=fake_run_algorithm):
                result = run_ablation_experiment(config=config, run_id=run_id, silent=True)

            self.assertEqual(calls, ["PDRS-Myopic"])
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            self.assertEqual(progress.get("resumed_count"), 1)
            self.assertEqual(progress["resumed"][0]["algorithm"], "GMDA-RMPR-Myopic")
            self.assertEqual(progress["completed_count"], 2)

            summary_path = Path(result["meta_path"]).parent / f"ablation_summary_{run_id}.csv"
            with summary_path.open("r", newline="", encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            gmda_row = next(row for row in rows if row["algorithm"] == "GMDA-RMPR-Myopic" and row["seed"] == "38")
            self.assertEqual(float(gmda_row["delay_mean"]), 111.0)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_run_ablation_cli_passes_explicit_run_id_for_resume(self):
        import contextlib
        import io
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        import run_ablation

        tmp = _workspace_test_tmp("cli_run_id_resume")
        captured = {}

        def fake_run_ablation_experiment(config, run_id=None, silent=True):
            captured["run_id"] = run_id
            captured["output_dir"] = config.output_dir
            captured["algorithms"] = list(config.algorithms)
            return {"run_id": run_id, "base_dir": str(tmp)}

        argv = [
            "run_ablation.py",
            "--formal-main",
            "--run-id",
            "time_horizon_smoke_resume",
            "--output-dir",
            str(tmp),
            "--algorithms",
            "LyHAM-CO",
        ]
        with patch.object(sys, "argv", argv), \
                patch("run_ablation.run_ablation_experiment", side_effect=fake_run_ablation_experiment), \
                contextlib.redirect_stdout(io.StringIO()):
            run_ablation.main()

        self.assertEqual(captured["run_id"], "time_horizon_smoke_resume")
        self.assertEqual(captured["algorithms"], ["LyHAM-CO"])

    def test_goal_recovery_state_files_are_present_and_actionable(self):
        import json

        project_root = Path(__file__).resolve().parents[1]
        task_path = project_root / "goal_task.md"
        long_term_path = project_root / "long_term_experiment_requirements.md"
        active_goal_path = project_root / "active_goal.md"
        progress_path = project_root / "goal_progress.md"
        state_path = project_root / "goal_state.json"

        self.assertFalse(task_path.exists())
        self.assertTrue(long_term_path.exists())
        long_term_text = long_term_path.read_text(encoding="utf-8")
        self.assertIn("Do not use old `goal_task.md`", long_term_text)
        self.assertIn("active_goal.md", long_term_text)
        self.assertIn("goal_state.json", long_term_text)

        self.assertTrue(active_goal_path.exists())
        active_goal_text = active_goal_path.read_text(encoding="utf-8")
        self.assertLessEqual(len(active_goal_text.splitlines()), 20)
        self.assertIn("P5", active_goal_text)
        self.assertIn("latency-queue stability", active_goal_text)
        self.assertIn("32", active_goal_text)
        self.assertIn("41", active_goal_text)

        self.assertTrue(progress_path.exists())
        progress_text = progress_path.read_text(encoding="utf-8")
        self.assertLessEqual(len(progress_text.splitlines()), 800)
        self.assertIn("P5", progress_text)
        self.assertIn("Next first action", progress_text)

        self.assertTrue(state_path.exists())
        state_text = state_path.read_text(encoding="utf-8")
        self.assertLessEqual(len(state_text.splitlines()), 260)
        state = json.loads(state_text)
        for key in [
            "current_priority",
            "current_phase",
            "next_unfinished_gate",
            "completed_gates",
            "failed_gates",
            "artifact_paths",
            "do_not_repeat",
            "next_actions",
            "code_directory",
            "python",
        ]:
            self.assertIn(key, state)
        self.assertIsInstance(state["completed_gates"], list)
        self.assertIsInstance(state["failed_gates"], list)
        self.assertIsInstance(state["do_not_repeat"], list)
        self.assertIsInstance(state["next_actions"], list)
        self.assertTrue(state["next_unfinished_gate"])
        self.assertTrue(state["next_actions"])
        self.assertIn("Lyapunov_Edge_Unloading10duibi", state["code_directory"])
        self.assertTrue(any("goal_task.md" in item for item in state["do_not_repeat"]))

    def test_goal_state_prioritizes_metric_audit_or_figure_data_when_outputs_are_missing(self):
        import json

        project_root = Path(__file__).resolve().parents[1]
        long_term_text = (project_root / "long_term_experiment_requirements.md").read_text(encoding="utf-8")
        state = json.loads((project_root / "goal_state.json").read_text(encoding="utf-8"))

        self.assertIn("P1", long_term_text)
        self.assertIn("P5", long_term_text)
        self.assertIn("P2/P3/P4/P6/P7", long_term_text)
        next_gate = str(state["next_unfinished_gate"])
        self.assertTrue(
            "P5" in next_gate
            or "slot" in next_gate.lower()
            or "trajectory" in next_gate.lower()
            or next_gate in {
                "time_horizon_and_queue_validity_audit",
                "online_training_logic_audit",
                "c4_contribution_consistency_audit",
                "paper_ready_outputs",
                "final_tmc_readiness_audit",
                "final_completion_audit_against_goal_task",
                "complete",
            },
            next_gate,
        )
        self.assertTrue(
            any(
                "量级" in str(action)
                or "18" in str(action)
                or "P2" in str(action)
                or "audit" in str(action).lower()
                or "LR" in str(action)
                or "learning" in str(action).lower()
                or "sweep" in str(action).lower()
                or "time" in str(action).lower()
                or "horizon" in str(action).lower()
                or "queue" in str(action).lower()
                or "training" in str(action).lower()
                or "C4" in str(action)
                or "paper" in str(action).lower()
                or "slot" in str(action).lower()
                or "trajectory" in str(action).lower()
                for action in state["next_actions"]
            )
        )

    def test_round_cache_stats_are_scoped_with_resource_model_cache(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_resource_models import (
            _round_cache_value,
            clear_resource_model_cache,
            get_resource_model_cache_stats,
        )

        clear_resource_model_cache()
        self.assertEqual(get_resource_model_cache_stats()["round_cache_currsize"], 0)

        self.assertEqual(_round_cache_value(1.23456789), 1.234568)
        self.assertEqual(_round_cache_value(1.23456789), 1.234568)
        self.assertEqual(_round_cache_value(1.23456789, digits=None), 1.234568)
        stats = get_resource_model_cache_stats()
        self.assertGreaterEqual(stats["round_cache_hits"], 1)
        self.assertGreaterEqual(stats["round_cache_currsize"], 1)

        clear_resource_model_cache()
        self.assertEqual(get_resource_model_cache_stats()["round_cache_currsize"], 0)

    def test_figure_sweep_latex_table_gate_writes_draft_only(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_export import export_latex_table
        from ablation_metrics import AlgorithmSummary

        tmp = _workspace_test_tmp("figure_table_gate")
        summaries = [
            AlgorithmSummary(
                algorithm=algorithm,
                seed=-1,
                n_valid_slots=100,
                n_failed_slots=0,
                delay_mean=200.0,
                delay_std=1.0,
                energy_mean=20.0,
                energy_std=1.0,
                cost_mean=1000.0,
                cost_std=1.0,
                avg_y_mean=1.0,
                avg_y_std=0.0,
                avg_z_mean=1.0,
                avg_z_std=0.0,
                dpp_score_mean=1.0,
                dpp_score_std=0.0,
                decision_time_mean_ms=10.0,
                decision_time_p95_ms=12.0,
                feasible_ratio=1.0,
                valid=True,
            )
            for algorithm in ["LyHAM-CO", "GMDA-RMPR-Myopic", "PDRS-Myopic", "FFD-Myopic"]
        ]

        path = export_latex_table(
            tmp,
            summaries,
            formal_gate_passed=True,
            claim_supported=True,
            include_energy_claim=True,
            experiment_type="normal_main",
            canonical_allowed=False,
        )

        self.assertEqual(path.name, "normal_main_table_draft.tex")
        self.assertFalse((tmp / "tables" / "normal_main_table.tex").exists())

    def test_lightweight_dry_run_backup_restores_mutable_instance_state(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ResourceAllocation import backup_system_state, restore_system_state

        class Server:
            def __init__(self):
                self.available_cpu = 4
                self.available_memory = 8.0
                self.available_gpu_units = 2
                self.available_gpu_memory = 16.0
                self.available_model_storage = 64.0
                self._gpu_active = True
                self.current_batch_requests = ["r1"]
                self.batch_processing_start_time = 3.0

        class Instance:
            def __init__(self, instance_id):
                self.instance_id = instance_id
                self.server_id = "s1"
                self.allocated_cpu_cores = 1
                self.allocated_memory = 1.0
                self.processing_mode = "initial"
                self.gpu_units_allocated = 1
                self.gpu_memory_allocated = 2.0
                self.model_storage_allocated = 3.0
                self.batch_size_allocated = 1
                self.gpu_frequency_scale = 1.0
                self.preprocess_frequency_scale = 1.0
                self.compression_ratio = 1.0
                self.resource_config_source = "initial"
                self.inference_latency = 10.0
                self.cloud_latency = 0.0
                self.energy_local_gpu_j = 1.0
                self.energy_cloud_compute_j = 0.0
                self.energy_comm_j = 0.0
                self.energy_preprocess_j = 0.0
                self.hapa_feedback_consumed = False
                self.hapa_replica_readiness = 1.0
                self.hapa_coverage_ratio = 1.0
                self.active_pair_count = 0
                self.active_local_pair_count = 0
                self.active_cloud_pair_count = 0

        class Flow:
            def __init__(self):
                self.arrival_rate = 5.0
                self.ca_latency = 1.0

        class Queue:
            def __init__(self, value):
                self.queue_state = value

        class State:
            def __init__(self):
                self.edge_servers = {"s1": Server()}
                self.microservice_instances = {"i1": Instance("i1")}
                self.request_flows = {"f1": Flow()}
                self.virtual_energy_queues = {"s1": Queue(2.0)}
                self.virtual_delay_queues = {"s1": Queue(3.0)}
                self.total_energy_consumption = 4.0
                self.total_latency = 5.0
                self.current_arrivals = {"f1": 5.0}

        state = State()
        backup = backup_system_state(state, include_full_deployment=False)
        self.assertNotIn("microservice_instances_full", backup)

        state.edge_servers["s1"].available_cpu = 0
        state.microservice_instances["i1"].server_id = "s2"
        state.microservice_instances["i1"].processing_mode = "cloud_offloaded"
        state.microservice_instances["i1"].energy_cloud_compute_j = 9.0
        state.microservice_instances["i1"].hapa_feedback_consumed = True
        state.microservice_instances["extra"] = Instance("extra")
        state.request_flows["f1"].arrival_rate = 99.0
        state.virtual_energy_queues["s1"].queue_state = 88.0
        state.total_latency = 77.0

        restore_system_state(state, backup)

        self.assertEqual(set(state.microservice_instances), {"i1"})
        self.assertEqual(state.edge_servers["s1"].available_cpu, 4)
        self.assertEqual(state.microservice_instances["i1"].server_id, "s1")
        self.assertEqual(state.microservice_instances["i1"].processing_mode, "initial")
        self.assertEqual(state.microservice_instances["i1"].energy_cloud_compute_j, 0.0)
        self.assertFalse(state.microservice_instances["i1"].hapa_feedback_consumed)
        self.assertEqual(state.request_flows["f1"].arrival_rate, 5.0)
        self.assertEqual(state.virtual_energy_queues["s1"].queue_state, 2.0)
        self.assertEqual(state.total_latency, 5.0)

    def test_request_delay_burden_indexes_ai_instances_once(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ResourceAllocation import calculate_request_level_delay_burden_by_server

        class CountingInstances(dict):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.items_call_count = 0

            def items(self):
                self.items_call_count += 1
                return super().items()

        class Microservice:
            service_type = "ai"

        class Instance:
            def __init__(self, instance_id, server_id):
                self.instance_id = instance_id
                self.server_id = server_id
                self.microservice = Microservice()
                self.pair_action_bit = 0
                self.processing_mode = "local_processing"

        class Flow:
            def __init__(self, latency):
                self.ca_latency = latency
                self.arrival_rate = 10.0

        class State:
            def __init__(self):
                self.request_flows = {
                    "flow_1": Flow(100.0),
                    "flow_2": Flow(200.0),
                    "flow_3": Flow(300.0),
                }
                self.microservice_instances = CountingInstances({
                    "flow_1_ai_ai_v1": Instance("flow_1_ai_ai_v1", "ai_v1"),
                    "flow_2_ai_ai_v2": Instance("flow_2_ai_ai_v2", "ai_v2"),
                    "flow_3_ai_ai_v1": Instance("flow_3_ai_ai_v1", "ai_v1"),
                })

        state = State()
        burden = calculate_request_level_delay_burden_by_server(
            state, ["ai_v1", "ai_v2"], fallback_delay_ms=1.0,
        )

        self.assertEqual(state.microservice_instances.items_call_count, 1)
        np.testing.assert_allclose(burden, np.array([200.0, 200.0]))

    def test_post_update_queue_metrics_predicts_without_mutating_queues(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ResourceAllocation import calculate_post_update_queue_metrics

        class Queue:
            def __init__(self, state, threshold, scale):
                self.queue_state = state
                self.energy_threshold = threshold
                self.delay_threshold = threshold
                self.scaling_factor = scale

        class State:
            def __init__(self):
                self.virtual_energy_queues = {
                    "ai_1": Queue(10.0, 4.0, 0.5),
                    "ai_2": Queue(5.0, 4.0, 0.5),
                }
                self.virtual_delay_queues = {
                    "ai_1": Queue(2.0, 100.0, 0.1),
                    "ai_2": Queue(3.0, 100.0, 0.1),
                }

        state = State()
        metrics = calculate_post_update_queue_metrics(
            system_state=state,
            ai_server_ids=["ai_1", "ai_2"],
            ai_energies=np.array([3.0, 1.0]),
            ai_delays=np.array([80.0, 120.0]),
            active_ai_energy_j=4.0,
            system_energy_j=8.0,
            delay_burden_vec=np.array([80.0, 120.0]),
            energy_ref_j=2.0,
            delay_ref_ms=50.0,
            omega_energy=1.0,
            omega_delay=1.0,
        )

        self.assertAlmostEqual(metrics["predicted_avg_y"], 7.5)
        self.assertAlmostEqual(metrics["predicted_avg_z"], 2.5)
        self.assertAlmostEqual(metrics["post_update_energy_queue_term"], 37.0)
        self.assertAlmostEqual(metrics["post_update_delay_queue_term"], 12.0)
        self.assertAlmostEqual(metrics["post_update_queue_delta_term"], 3.6)
        self.assertAlmostEqual(metrics["post_update_energy_queue_delta_term"], 2.0)
        self.assertAlmostEqual(metrics["post_update_delay_queue_delta_term"], 1.6)
        self.assertEqual(state.virtual_energy_queues["ai_1"].queue_state, 10.0)
        self.assertEqual(state.virtual_delay_queues["ai_2"].queue_state, 3.0)

    def test_post_update_drift_score_is_queue_aware_only(self):
        import sys
        from types import SimpleNamespace
        sys.path.insert(0, str(UTILS_DIR))

        from ResourceAllocation import apply_post_update_queue_drift_score

        components = {
            "paper_dpp_score": 100.0,
            "energy_queue_term": 10.0,
            "delay_queue_term": 5.0,
        }
        metrics = {
            "post_update_queue_delta_term": 20.0,
            "post_update_queue_pressure_term": 35.0,
        }
        config = SimpleNamespace(
            post_update_queue_drift_enabled=True,
            post_update_queue_drift_weight=0.25,
        )

        queue_aware = apply_post_update_queue_drift_score(
            dict(components), metrics, config, queue_aware=True
        )
        myopic = apply_post_update_queue_drift_score(
            dict(components), metrics, config, queue_aware=False
        )
        relief = apply_post_update_queue_drift_score(
            dict(components),
            {"post_update_queue_delta_term": -20.0},
            config,
            queue_aware=True,
        )
        masked_delay_risk = apply_post_update_queue_drift_score(
            dict(components),
            {
                "post_update_queue_delta_term": -8.0,
                "post_update_energy_queue_delta_term": -20.0,
                "post_update_delay_queue_delta_term": 12.0,
            },
            config,
            queue_aware=True,
        )

        self.assertAlmostEqual(queue_aware["paper_dpp_score"], 105.0)
        self.assertAlmostEqual(queue_aware["post_update_queue_drift_term"], 5.0)
        self.assertAlmostEqual(myopic["paper_dpp_score"], 100.0)
        self.assertAlmostEqual(myopic["post_update_queue_drift_term"], 0.0)
        self.assertAlmostEqual(relief["paper_dpp_score"], 100.0)
        self.assertAlmostEqual(relief["post_update_queue_drift_term"], 0.0)
        self.assertAlmostEqual(masked_delay_risk["paper_dpp_score"], 103.0)
        self.assertAlmostEqual(masked_delay_risk["post_update_queue_drift_term"], 3.0)

    def test_cloud_relief_resource_hint_keeps_energy_choice_within_latency_guard(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_resource_models import select_config_by_resource_hint

        rows = [
            {"objective": 100.0, "energy_j": 20.0, "latency_ms": 100.0},
            {"objective": 103.0, "energy_j": 12.0, "latency_ms": 108.0},
            {"objective": 104.0, "energy_j": 10.0, "latency_ms": 120.0},
        ]

        selected = select_config_by_resource_hint(rows, "cloud_relief_f_pre", dpp_band_ratio=0.05)

        self.assertEqual(selected["latency_ms"], 108.0)
        self.assertEqual(selected["energy_j"], 12.0)

    def test_cloud_relief_latency_guard_rejects_config_beyond_ten_percent(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_resource_models import select_config_by_resource_hint

        rows = [
            {"objective": 100.0, "energy_j": 20.0, "latency_ms": 100.0},
            {"objective": 102.0, "energy_j": 12.0, "latency_ms": 110.0},
            {"objective": 103.0, "energy_j": 10.0, "latency_ms": 112.0},
        ]

        selected = select_config_by_resource_hint(rows, "cloud_relief_f_pre", dpp_band_ratio=0.05)

        self.assertEqual(selected["latency_ms"], 110.0)
        self.assertEqual(selected["energy_j"], 12.0)

    def test_claim_selector_rejects_noncollapsed_delay_regression_for_small_energy_gain(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_algorithms import select_best_action_from_candidates
        from ablation_config import AblationExperimentConfig

        config = AblationExperimentConfig(
            include_energy_claim=True,
            claim_delay_ref_ms=100.0,
            claim_energy_ref_j=12.0,
            claim_cost_ref=400.0,
            energy_hard_dpp_tolerance_ratio=0.10,
        )
        collapsed = {
            "action": np.array([0, 0], dtype=int),
            "pair_action": np.array([0, 0], dtype=int),
            "pair_universe": [{"server_id": "s0"}, {"server_id": "s1"}],
            "pair_action_dim": 2,
            "action_dim": 2,
            "action_scope": "pair",
            "candidate_source": "myopic_reference_repaired",
        }
        neighbor = {
            "action": np.array([0, 1], dtype=int),
            "pair_action": np.array([0, 1], dtype=int),
            "pair_universe": [{"server_id": "s0"}, {"server_id": "s1"}],
            "pair_action_dim": 2,
            "action_dim": 2,
            "action_scope": "pair",
            "candidate_source": "uac_reference_low_impact_neighbor",
            "resource_hint": "cloud_relief_f_pre",
        }

        def fake_eval(action, system_state, config, queue_aware=True):
            source = str(action.get("candidate_source", ""))
            is_relaxed = "queue_relaxed_cloud_relief" in source
            is_neighbor = "reference_low_impact_neighbor" in source
            if is_relaxed:
                delay, energy, cost, local_count, cloud_count, hash_value = 115.0, 9.4, 104.0, 1, 1, "hybrid-relaxed"
            elif is_neighbor:
                delay, energy, cost, local_count, cloud_count, hash_value = 90.0, 15.0, 140.0, 1, 1, "hybrid-hard"
            else:
                delay, energy, cost, local_count, cloud_count, hash_value = 100.0, 10.0, 100.0, 2, 0, "collapsed"
            return {
                "action": np.array(action.get("action"), dtype=int),
                "pair_action": np.array(action.get("pair_action"), dtype=int),
                "paper_dpp_score": 101.0 if is_neighbor else 100.0,
                "feasible": True,
                "failure_reason": "",
                "delay_ms": delay,
                "energy_j": energy,
                "cost": cost,
                "local_count": local_count,
                "cloud_count": cloud_count,
                "repaired_pair_action_hash": hash_value,
                "candidate_source": source,
                "resource_hint": str(action.get("resource_hint", "")),
                "resource_queue_aware": action.get("resource_queue_aware", None),
                "resource_mode": str(action.get("resource_mode", "")),
                "resource_queue_scale": float(action.get("resource_queue_scale", 1.0)),
            }

        with patch("ResourceAllocation.evaluate_action_dry_run", side_effect=fake_eval):
            decision = select_best_action_from_candidates(
                candidates=[collapsed, neighbor],
                system_state=object(),
                config=config,
                queue_aware=True,
            )

        self.assertEqual(decision["repaired_pair_action_hash"], "collapsed")
        self.assertEqual(decision["selected_candidate_source"], "myopic_reference_repaired")

    def test_myopic_energy_claim_selector_rejects_cold_start_delay_collapse(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_algorithms import select_best_action_from_candidates
        from ablation_config import AblationExperimentConfig

        config = AblationExperimentConfig(
            include_energy_claim=True,
            claim_delay_ref_ms=100.0,
            claim_energy_ref_j=12.0,
            claim_cost_ref=400.0,
        )
        cheap_but_slow = {
            "action": np.array([0, 0], dtype=int),
            "pair_action": np.array([0, 0], dtype=int),
            "pair_universe": [{"server_id": "s0"}, {"server_id": "s1"}],
            "pair_action_dim": 2,
            "action_dim": 2,
            "action_scope": "pair",
            "candidate_source": "legacy_server_candidate",
        }
        balanced = {
            "action": np.array([0, 1], dtype=int),
            "pair_action": np.array([0, 1], dtype=int),
            "pair_universe": [{"server_id": "s0"}, {"server_id": "s1"}],
            "pair_action_dim": 2,
            "action_dim": 2,
            "action_scope": "pair",
            "candidate_source": "myopic_delay_energy_frontier",
        }

        def fake_eval(action, system_state, config, queue_aware=True):
            source = str(action.get("candidate_source", ""))
            is_balanced = "delay_energy_frontier" in source
            return {
                "action": np.array(action.get("action"), dtype=int),
                "pair_action": np.array(action.get("pair_action"), dtype=int),
                "paper_dpp_score": 130.0 if is_balanced else 100.0,
                "feasible": True,
                "failure_reason": "",
                "delay_ms": 250.0 if is_balanced else 2200.0,
                "energy_j": 18.0 if is_balanced else 20.0,
                "cost": 130.0 if is_balanced else 100.0,
                "local_count": 1 if is_balanced else 2,
                "cloud_count": 1 if is_balanced else 0,
                "repaired_pair_action_hash": "balanced" if is_balanced else "cheap_slow",
                "candidate_source": source,
            }

        with patch("ResourceAllocation.evaluate_action_dry_run", side_effect=fake_eval):
            decision = select_best_action_from_candidates(
                candidates=[cheap_but_slow, balanced],
                system_state=object(),
                config=config,
                queue_aware=False,
            )

        self.assertEqual(decision["repaired_pair_action_hash"], "balanced")
        self.assertEqual(decision["selected_by_dpp_or_claim_band"], "myopic_immediate")

    def test_reference_neighbor_queue_relaxed_cloud_relief_variant_can_be_selected(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_algorithms import select_best_action_from_candidates
        from ablation_config import AblationExperimentConfig

        config = AblationExperimentConfig(
            include_energy_claim=True,
            claim_delay_ref_ms=100.0,
            claim_energy_ref_j=24.0,
            claim_cost_ref=400.0,
            energy_hard_dpp_tolerance_ratio=0.10,
        )
        collapsed = {
            "action": np.array([0, 0], dtype=int),
            "pair_action": np.array([0, 0], dtype=int),
            "pair_universe": [{"server_id": "s0"}, {"server_id": "s1"}],
            "pair_action_dim": 2,
            "action_dim": 2,
            "action_scope": "pair",
            "candidate_source": "myopic_reference_repaired",
        }
        neighbor = {
            "action": np.array([0, 1], dtype=int),
            "pair_action": np.array([0, 1], dtype=int),
            "pair_universe": [{"server_id": "s0"}, {"server_id": "s1"}],
            "pair_action_dim": 2,
            "action_dim": 2,
            "action_scope": "pair",
            "candidate_source": "uac_reference_low_impact_neighbor",
            "resource_hint": "cloud_relief_f_pre",
        }

        def fake_eval(action, system_state, config, queue_aware=True):
            source = str(action.get("candidate_source", ""))
            is_relaxed = "queue_relaxed_cloud_relief" in source
            is_neighbor = "reference_low_impact_neighbor" in source
            if is_relaxed:
                delay, energy, cost, local_count, cloud_count, hash_value = 102.0, 10.6, 101.0, 1, 1, "hybrid-relaxed"
            elif is_neighbor:
                delay, energy, cost, local_count, cloud_count, hash_value = 100.0, 13.0, 110.0, 1, 1, "hybrid-hard"
            else:
                delay, energy, cost, local_count, cloud_count, hash_value = 106.0, 10.0, 100.0, 2, 0, "collapsed"
            return {
                "action": np.array(action.get("action"), dtype=int),
                "pair_action": np.array(action.get("pair_action"), dtype=int),
                "paper_dpp_score": 101.0 if is_neighbor else 100.0,
                "feasible": True,
                "failure_reason": "",
                "delay_ms": delay,
                "energy_j": energy,
                "cost": cost,
                "local_count": local_count,
                "cloud_count": cloud_count,
                "repaired_pair_action_hash": hash_value,
                "candidate_source": source,
                "resource_hint": str(action.get("resource_hint", "")),
                "resource_queue_aware": action.get("resource_queue_aware", None),
                "resource_mode": str(action.get("resource_mode", "")),
            }

        with patch("ResourceAllocation.evaluate_action_dry_run", side_effect=fake_eval):
            decision = select_best_action_from_candidates(
                candidates=[collapsed, neighbor],
                system_state=object(),
                config=config,
                queue_aware=True,
            )

        self.assertEqual(decision["repaired_pair_action_hash"], "hybrid-relaxed")
        self.assertIn("queue_relaxed_cloud_relief", decision["selected_candidate_source"])
        self.assertEqual(decision["resource_hint"], "cloud_relief_f_pre")
        self.assertFalse(decision["resource_queue_aware"])
    def test_resource_queue_scale_participates_in_dry_run_cache_key(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_algorithms import select_best_action_from_candidates
        from ablation_config import AblationExperimentConfig

        base = {
            "action": np.array([0], dtype=int),
            "pair_action": np.array([0], dtype=int),
            "pair_universe": [{"server_id": "s0"}],
            "action_dim": 1,
            "action_scope": "pair",
            "candidate_source": "queue_full",
            "resource_hint": "cloud_relief_f_pre",
            "resource_queue_aware": True,
            "resource_mode": "queue_damped_cloud_relief",
            "resource_queue_scale": 1.0,
        }
        damped = dict(base)
        damped["candidate_source"] = "queue_damped"
        damped["resource_queue_scale"] = 0.35
        seen_scales = []

        def fake_eval(action, system_state, config, queue_aware=True):
            scale = float(action.get("resource_queue_scale", 1.0))
            seen_scales.append(scale)
            return {
                "action": np.array([0], dtype=int),
                "pair_action": np.array([0], dtype=int),
                "paper_dpp_score": 100.0 if scale < 1.0 else 200.0,
                "feasible": True,
                "failure_reason": "",
                "delay_ms": 100.0,
                "energy_j": 10.0,
                "cost": 100.0,
                "local_count": 1,
                "cloud_count": 0,
                "repaired_pair_action_hash": "same-action",
                "candidate_source": action.get("candidate_source", ""),
                "resource_hint": action.get("resource_hint", ""),
                "resource_queue_aware": action.get("resource_queue_aware", None),
                "resource_mode": action.get("resource_mode", ""),
                "resource_queue_scale": scale,
            }

        with patch("ResourceAllocation.evaluate_action_dry_run", side_effect=fake_eval):
            decision = select_best_action_from_candidates(
                candidates=[base, damped],
                system_state=object(),
                config=AblationExperimentConfig(include_energy_claim=False),
                queue_aware=True,
            )

        self.assertIn(1.0, seen_scales)
        self.assertIn(0.35, seen_scales)
        self.assertEqual(decision["selected_candidate_source"], "queue_damped")
        self.assertEqual(decision["resource_queue_scale"], 0.35)

    def test_reference_neighbor_queue_damped_cloud_relief_variant_can_be_selected(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_algorithms import select_best_action_from_candidates
        from ablation_config import AblationExperimentConfig

        config = AblationExperimentConfig(
            include_energy_claim=True,
            claim_delay_ref_ms=100.0,
            claim_energy_ref_j=24.0,
            claim_cost_ref=400.0,
            energy_hard_dpp_tolerance_ratio=0.10,
        )
        collapsed = {
            "action": np.array([0, 0], dtype=int),
            "pair_action": np.array([0, 0], dtype=int),
            "pair_universe": [{"server_id": "s0"}, {"server_id": "s1"}],
            "pair_action_dim": 2,
            "action_dim": 2,
            "action_scope": "pair",
            "candidate_source": "myopic_reference_repaired",
        }
        neighbor = {
            "action": np.array([0, 1], dtype=int),
            "pair_action": np.array([0, 1], dtype=int),
            "pair_universe": [{"server_id": "s0"}, {"server_id": "s1"}],
            "pair_action_dim": 2,
            "action_dim": 2,
            "action_scope": "pair",
            "candidate_source": "uac_reference_low_impact_neighbor",
            "resource_hint": "cloud_relief_f_pre",
        }

        def fake_eval(action, system_state, config, queue_aware=True):
            source = str(action.get("candidate_source", ""))
            if "queue_damped_cloud_relief" in source:
                delay, energy, cost, local_count, cloud_count, hash_value = 101.0, 10.4, 100.5, 1, 1, "hybrid-damped"
            elif "queue_relaxed_cloud_relief" in source:
                delay, energy, cost, local_count, cloud_count, hash_value = 130.0, 9.5, 95.0, 1, 1, "hybrid-relaxed"
            elif "reference_low_impact_neighbor" in source:
                delay, energy, cost, local_count, cloud_count, hash_value = 100.0, 13.0, 110.0, 1, 1, "hybrid-hard"
            else:
                delay, energy, cost, local_count, cloud_count, hash_value = 106.0, 10.0, 100.0, 2, 0, "collapsed"
            return {
                "action": np.array(action.get("action"), dtype=int),
                "pair_action": np.array(action.get("pair_action"), dtype=int),
                "paper_dpp_score": 101.0 if "reference_low_impact_neighbor" in source else 100.0,
                "feasible": True,
                "failure_reason": "",
                "delay_ms": delay,
                "energy_j": energy,
                "cost": cost,
                "local_count": local_count,
                "cloud_count": cloud_count,
                "repaired_pair_action_hash": hash_value,
                "candidate_source": source,
                "resource_hint": str(action.get("resource_hint", "")),
                "resource_queue_aware": action.get("resource_queue_aware", None),
                "resource_mode": str(action.get("resource_mode", "")),
                "resource_queue_scale": float(action.get("resource_queue_scale", 1.0)),
            }

        with patch("ResourceAllocation.evaluate_action_dry_run", side_effect=fake_eval):
            decision = select_best_action_from_candidates(
                candidates=[collapsed, neighbor],
                system_state=object(),
                config=config,
                queue_aware=True,
            )

        self.assertEqual(decision["repaired_pair_action_hash"], "hybrid-damped")
        self.assertIn("queue_damped_cloud_relief", decision["selected_candidate_source"])
        self.assertEqual(decision["resource_hint"], "cloud_relief_f_pre")
        self.assertTrue(decision["resource_queue_aware"])
        self.assertAlmostEqual(decision["resource_queue_scale"], 0.35)
    def test_reference_neighbor_queue_damped010_cloud_relief_variant_can_be_selected(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_algorithms import select_best_action_from_candidates
        from ablation_config import AblationExperimentConfig

        config = AblationExperimentConfig(
            include_energy_claim=True,
            claim_delay_ref_ms=100.0,
            claim_energy_ref_j=24.0,
            claim_cost_ref=400.0,
            energy_hard_dpp_tolerance_ratio=0.10,
        )
        collapsed = {
            "action": np.array([0, 0], dtype=int),
            "pair_action": np.array([0, 0], dtype=int),
            "pair_universe": [{"server_id": "s0"}, {"server_id": "s1"}],
            "pair_action_dim": 2,
            "action_dim": 2,
            "action_scope": "pair",
            "candidate_source": "myopic_reference_repaired",
        }
        neighbor = {
            "action": np.array([0, 1], dtype=int),
            "pair_action": np.array([0, 1], dtype=int),
            "pair_universe": [{"server_id": "s0"}, {"server_id": "s1"}],
            "pair_action_dim": 2,
            "action_dim": 2,
            "action_scope": "pair",
            "candidate_source": "uac_reference_low_impact_neighbor",
            "resource_hint": "cloud_relief_f_pre",
        }

        def fake_eval(action, system_state, config, queue_aware=True):
            source = str(action.get("candidate_source", ""))
            scale = float(action.get("resource_queue_scale", 1.0))
            if "queue_damped010_cloud_relief" in source:
                delay, energy, cost, hash_value = 101.0, 10.4, 100.5, "hybrid-damped010"
            elif "queue_damped020_cloud_relief" in source:
                delay, energy, cost, hash_value = 104.0, 10.2, 99.0, "hybrid-damped020"
            elif "queue_relaxed_cloud_relief" in source:
                delay, energy, cost, hash_value = 130.0, 9.5, 95.0, "hybrid-relaxed"
            elif "reference_low_impact_neighbor" in source:
                delay, energy, cost, hash_value = 100.0, 13.0, 110.0, "hybrid-hard"
            else:
                delay, energy, cost, hash_value = 106.0, 10.0, 100.0, "collapsed"
            return {
                "action": np.array(action.get("action"), dtype=int),
                "pair_action": np.array(action.get("pair_action"), dtype=int),
                "paper_dpp_score": 101.0 if "reference_low_impact_neighbor" in source else 100.0,
                "feasible": True,
                "failure_reason": "",
                "delay_ms": delay,
                "energy_j": energy,
                "cost": cost,
                "local_count": 1 if "reference_low_impact_neighbor" in source else 2,
                "cloud_count": 1 if "reference_low_impact_neighbor" in source else 0,
                "repaired_pair_action_hash": hash_value,
                "candidate_source": source,
                "resource_hint": str(action.get("resource_hint", "")),
                "resource_queue_aware": action.get("resource_queue_aware", None),
                "resource_mode": str(action.get("resource_mode", "")),
                "resource_queue_scale": scale,
            }

        with patch("ResourceAllocation.evaluate_action_dry_run", side_effect=fake_eval):
            decision = select_best_action_from_candidates(
                candidates=[collapsed, neighbor],
                system_state=object(),
                config=config,
                queue_aware=True,
            )

        self.assertEqual(decision["repaired_pair_action_hash"], "hybrid-damped010")
        self.assertIn("queue_damped010_cloud_relief", decision["selected_candidate_source"])
        self.assertTrue(decision["resource_queue_aware"])
        self.assertAlmostEqual(decision["resource_queue_scale"], 0.10)
    def test_claim_energy_variant_declares_energy_relaxed_resource_mode(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_algorithms import select_best_action_from_candidates
        from ablation_config import AblationExperimentConfig

        config = AblationExperimentConfig(include_energy_claim=True)
        candidate = {
            "action": np.array([0], dtype=int),
            "pair_action": np.array([0], dtype=int),
            "pair_universe": [{"flow_id": "f0", "server_id": "s0", "instance_id": "i0"}],
            "action_dim": 1,
            "action_scope": "pair",
            "candidate_source": "actor",
        }
        seen_modes = []

        def fake_eval(action, system_state, config, queue_aware=True):
            hint = str(action.get("resource_hint", ""))
            relaxed = action.get("resource_queue_aware", None) is False
            seen_modes.append((hint, action.get("resource_queue_aware", None), queue_aware))
            is_relaxed_claim = hint == "claim_energy_saver_local" and relaxed
            return {
                "action": np.array([0], dtype=int),
                "pair_action": np.array([0], dtype=int),
                "paper_dpp_score": 103.0 if is_relaxed_claim else 100.0,
                "feasible": True,
                "failure_reason": "",
                "delay_ms": 112.0 if is_relaxed_claim else 100.0,
                "energy_j": 14.0 if is_relaxed_claim else 20.0,
                "cost": 400.0,
                "local_count": 1,
                "cloud_count": 0,
                "repaired_pair_action_hash": "same-pair-action",
                "candidate_source": action.get("candidate_source", ""),
                "resource_hint": hint,
                "resource_queue_aware": action.get("resource_queue_aware", None),
            }

        with patch("ResourceAllocation.evaluate_action_dry_run", side_effect=fake_eval):
            decision = select_best_action_from_candidates(
                candidates=[candidate],
                system_state=object(),
                config=config,
                queue_aware=True,
            )

        self.assertIn(("claim_energy_saver_local", False, True), seen_modes)
        self.assertEqual(decision["resource_hint"], "claim_energy_saver_local")
        self.assertFalse(decision["resource_queue_aware"])

    def test_energy_cloud_relief_emits_queue_damped_resource_variants(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_algorithms import add_energy_claim_resource_variants
        from ablation_config import AblationExperimentConfig

        candidate = {
            "action": np.array([0, 1], dtype=int),
            "pair_action": np.array([0, 1], dtype=int),
            "candidate_source": "uac_energy_cloud_relief",
            "resource_hint": "cloud_relief_f_pre",
        }
        config = AblationExperimentConfig(include_energy_claim=True)

        expanded = add_energy_claim_resource_variants([candidate], config, queue_aware=True)
        sources = {str(item.get("candidate_source", "")) for item in expanded}

        self.assertIn("uac_energy_cloud_relief_queue_damped010_cloud_relief", sources)
        self.assertIn("uac_energy_cloud_relief_queue_damped020_cloud_relief", sources)
        self.assertIn("uac_energy_cloud_relief_queue_damped_cloud_relief", sources)
        self.assertIn("uac_energy_cloud_relief_queue_relaxed_cloud_relief", sources)
        relaxed = next(
            item for item in expanded
            if item.get("candidate_source") == "uac_energy_cloud_relief_queue_relaxed_cloud_relief"
        )
        self.assertFalse(relaxed["resource_queue_aware"])
        self.assertEqual(relaxed["resource_hint"], "cloud_relief_f_pre")

    def test_balanced_tail_relief_resource_variants_remain_queue_aware(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_algorithms import add_energy_claim_resource_variants
        from ablation_config import AblationExperimentConfig

        candidate = {
            "action": np.array([0, 1, 0], dtype=int),
            "pair_action": np.array([0, 1, 0, 1], dtype=int),
            "candidate_source": "uac_balanced_tail_relief",
            "resource_hint": "cloud_relief_f_pre",
        }
        config = AblationExperimentConfig(include_energy_claim=True)

        expanded = add_energy_claim_resource_variants([candidate], config, queue_aware=True)
        sources = {str(item.get("candidate_source", "")) for item in expanded}

        self.assertIn("uac_balanced_tail_relief_queue_damped010_cloud_relief", sources)
        self.assertIn("uac_balanced_tail_relief_queue_damped020_cloud_relief", sources)
        self.assertIn("uac_balanced_tail_relief_queue_damped_cloud_relief", sources)
        self.assertNotIn("uac_balanced_tail_relief_claim_energy_saver", sources)
        self.assertNotIn("uac_balanced_tail_relief_queue_relaxed_cloud_relief", sources)
        self.assertTrue(all(
            item.get("resource_queue_aware", True) is not False
            for item in expanded
            if "balanced_tail_relief" in str(item.get("candidate_source", ""))
        ))

    def test_high_energy_queue_suppresses_queue_unaware_resource_variants(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))
        from types import SimpleNamespace

        from ablation_algorithms import add_energy_claim_resource_variants
        from ablation_config import AblationExperimentConfig

        candidate = {
            "action": np.array([0, 1], dtype=int),
            "pair_action": np.array([0, 1], dtype=int),
            "candidate_source": "uac_energy_cloud_relief",
            "resource_hint": "cloud_relief_f_pre",
        }
        config = AblationExperimentConfig(include_energy_claim=True)
        config.queue_pressure_resource_variant_min_current_avg_y = 8.0
        high_queue = SimpleNamespace(
            virtual_energy_queues={
                "ai_v1": SimpleNamespace(queue_state=12.0),
                "ai_v2": SimpleNamespace(queue_state=10.0),
            }
        )

        expanded = add_energy_claim_resource_variants(
            [candidate], config, queue_aware=True, system_state=high_queue
        )
        sources = {str(item.get("candidate_source", "")) for item in expanded}

        self.assertNotIn("uac_energy_cloud_relief_claim_energy_saver", sources)
        self.assertNotIn("uac_energy_cloud_relief_queue_relaxed_cloud_relief", sources)
        self.assertIn("uac_energy_cloud_relief_queue_damped010_cloud_relief", sources)
        self.assertTrue(all(
            item.get("resource_queue_aware", True) is not False
            for item in expanded
        ))

    def test_low_energy_queue_keeps_queue_unaware_resource_variants(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))
        from types import SimpleNamespace

        from ablation_algorithms import add_energy_claim_resource_variants
        from ablation_config import AblationExperimentConfig

        candidate = {
            "action": np.array([0, 1], dtype=int),
            "pair_action": np.array([0, 1], dtype=int),
            "candidate_source": "uac_energy_cloud_relief",
            "resource_hint": "cloud_relief_f_pre",
        }
        config = AblationExperimentConfig(include_energy_claim=True)
        config.queue_pressure_resource_variant_min_current_avg_y = 8.0
        low_queue = SimpleNamespace(
            virtual_energy_queues={
                "ai_v1": SimpleNamespace(queue_state=2.0),
                "ai_v2": SimpleNamespace(queue_state=1.0),
            }
        )

        expanded = add_energy_claim_resource_variants(
            [candidate], config, queue_aware=True, system_state=low_queue
        )
        sources = {str(item.get("candidate_source", "")) for item in expanded}

        self.assertIn("uac_energy_cloud_relief_claim_energy_saver", sources)
        self.assertIn("uac_energy_cloud_relief_queue_relaxed_cloud_relief", sources)

    def test_paper_compact_keeps_bounded_queue_relaxed_cloud_relief_variant(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))
        from types import SimpleNamespace

        from ablation_algorithms import add_energy_claim_resource_variants
        from ablation_config import AblationExperimentConfig

        candidate = {
            "action": np.array([0, 1], dtype=int),
            "pair_action": np.array([0, 1], dtype=int),
            "candidate_source": "uac_energy_cloud_relief",
            "resource_hint": "cloud_relief_f_pre",
        }
        config = AblationExperimentConfig(
            include_energy_claim=True,
            uac_candidate_mechanism="paper_compact",
        )
        low_queue = SimpleNamespace(
            virtual_energy_queues={
                "ai_v1": SimpleNamespace(queue_state=0.0),
                "ai_v2": SimpleNamespace(queue_state=0.0),
            }
        )

        expanded = add_energy_claim_resource_variants(
            [candidate], config, queue_aware=True, system_state=low_queue
        )
        sources = {str(item.get("candidate_source", "")) for item in expanded}

        self.assertNotIn("uac_energy_cloud_relief_claim_energy_saver", sources)
        self.assertIn("uac_energy_cloud_relief_queue_relaxed_cloud_relief", sources)
        relaxed = next(
            item for item in expanded
            if item.get("candidate_source") == "uac_energy_cloud_relief_queue_relaxed_cloud_relief"
        )
        self.assertFalse(relaxed["resource_queue_aware"])
        self.assertEqual(relaxed["resource_queue_scale"], 0.0)
        self.assertEqual(relaxed["resource_mode"], "queue_relaxed_cloud_relief")
        self.assertIn("uac_energy_cloud_relief_queue_full_cloud_relief", sources)
        full = next(
            item for item in expanded
            if item.get("candidate_source") == "uac_energy_cloud_relief_queue_full_cloud_relief"
        )
        self.assertTrue(full["resource_queue_aware"])
        self.assertEqual(full["resource_mode"], "queue_full_cloud_relief")

    def test_paper_compact_adds_lydroo_resource_frontier_variants(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_algorithms import add_energy_claim_resource_variants
        from ablation_config import AblationExperimentConfig

        candidate = {
            "action": np.array([0, 1, 0], dtype=int),
            "pair_action": np.array([0, 1, 0], dtype=int),
            "candidate_source": "pair_actor_threshold_0.35",
            "resource_hint": "",
        }
        config = AblationExperimentConfig(include_energy_claim=True)
        config.uac_candidate_mechanism = "paper_compact"

        expanded = add_energy_claim_resource_variants(
            [candidate], config, queue_aware=True, system_state=None
        )
        modes = {str(item.get("resource_mode", "")) for item in expanded if isinstance(item, dict)}

        self.assertIn("lydroo_balanced_resource_frontier", modes)
        self.assertIn("lydroo_delay_resource_frontier", modes)
        self.assertIn("lydroo_energy_resource_frontier", modes)
        self.assertIn("lydroo_cost_resource_frontier", modes)
        for item in expanded:
            if str(item.get("resource_mode", "")).startswith("lydroo_"):
                np.testing.assert_array_equal(item["pair_action"], candidate["pair_action"])
                self.assertTrue(item["resource_queue_aware"])

    def test_cost_resource_hint_prefers_lower_cost_inside_objective_band(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_resource_models import select_config_by_resource_hint

        rows = [
            {"objective": 100.0, "latency_ms": 10.0, "energy_j": 5.0, "cost": 80.0},
            {"objective": 101.0, "latency_ms": 12.0, "energy_j": 5.5, "cost": 20.0},
        ]

        selected = select_config_by_resource_hint(rows, "cost_saver_hybrid")

        self.assertEqual(selected["cost"], 20.0)

    def test_resource_queue_mode_participates_in_dry_run_cache_key(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_algorithms import select_best_action_from_candidates
        from ablation_config import AblationExperimentConfig

        base = {
            "action": np.array([0], dtype=int),
            "pair_action": np.array([0], dtype=int),
            "pair_universe": [{"flow_id": "f0", "server_id": "s0", "instance_id": "i0"}],
            "action_scope": "pair",
            "candidate_source": "queue_resource",
            "resource_hint": "claim_energy_saver_local",
            "resource_queue_aware": True,
        }
        relaxed = dict(base)
        relaxed["candidate_source"] = "energy_relaxed_resource"
        relaxed["resource_queue_aware"] = False
        calls = []

        def fake_eval(action, system_state, config, queue_aware=True):
            calls.append(action.get("candidate_source", ""))
            relaxed_mode = action.get("resource_queue_aware", True) is False
            return {
                "action": np.array([0], dtype=int),
                "pair_action": np.array([0], dtype=int),
                "paper_dpp_score": 90.0 if relaxed_mode else 100.0,
                "feasible": True,
                "failure_reason": "",
                "delay_ms": 100.0,
                "energy_j": 14.0 if relaxed_mode else 20.0,
                "cost": 400.0,
                "local_count": 1,
                "cloud_count": 0,
                "repaired_pair_action_hash": "same-pair-action",
                "candidate_source": action.get("candidate_source", ""),
                "resource_hint": action.get("resource_hint", ""),
                "resource_queue_aware": action.get("resource_queue_aware", True),
            }

        with patch("ResourceAllocation.evaluate_action_dry_run", side_effect=fake_eval):
            decision = select_best_action_from_candidates(
                candidates=[base, relaxed],
                system_state=object(),
                config=AblationExperimentConfig(include_energy_claim=True),
                queue_aware=True,
            )

        self.assertIn("queue_resource", calls)
        self.assertIn("energy_relaxed_resource", calls)
        self.assertEqual(decision["candidate_source"], "energy_relaxed_resource")
        self.assertFalse(decision["resource_queue_aware"])

    def test_resource_queue_mode_participates_in_candidate_dedupe_key(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        from ablation_algorithms import dedupe_evaluated_candidates_by_repaired_hash

        damped = {
            "repaired_pair_action_hash": "same-pair-action",
            "paper_dpp_score": 90.0,
            "candidate_source": "uac_energy_cost_pareto_relief_queue_damped010_cloud_relief",
            "resource_hint": "cloud_relief_f_pre",
            "resource_queue_aware": True,
            "resource_queue_scale": 0.10,
            "resource_mode": "queue_damped010_cloud_relief",
        }
        relaxed = {
            "repaired_pair_action_hash": "same-pair-action",
            "paper_dpp_score": 95.0,
            "candidate_source": "uac_energy_cost_pareto_relief_queue_relaxed_cloud_relief",
            "resource_hint": "cloud_relief_f_pre",
            "resource_queue_aware": False,
            "resource_queue_scale": 0.0,
            "resource_mode": "queue_relaxed_cloud_relief",
        }

        deduped = dedupe_evaluated_candidates_by_repaired_hash([damped, relaxed])
        dedupe_keys = {row.get("candidate_dedupe_key") for row in deduped}

        self.assertEqual(len(deduped), 2)
        self.assertEqual(len(dedupe_keys), 2)
        self.assertEqual(
            {row["resource_mode"] for row in deduped},
            {"queue_damped010_cloud_relief", "queue_relaxed_cloud_relief"},
        )
    def test_pair_repair_keeps_low_impact_reference_neighbor(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        import ablation_algorithms
        from ablation_config import AblationExperimentConfig

        class Env:
            def get_state_components(self):
                return np.ones(2), np.ones(2), np.ones(2)

        class State:
            environment_manager = Env()
            microservice_instances = {}
            request_flows = {}
            gsla_context = {}
            ffd_context = {}
            pdrs_context = {}
            loadaware_context = {}
            random_context = {}

        pair_universe = [
            {"flow_id": "f0", "server_id": "s0", "instance_id": "i0", "server_index": 0},
            {"flow_id": "f1", "server_id": "s1", "instance_id": "i1", "server_index": 1},
            {"flow_id": "f2", "server_id": "s0", "instance_id": "i2", "server_index": 0},
            {"flow_id": "f3", "server_id": "s1", "instance_id": "i3", "server_index": 1},
        ]
        reference_pair = np.array([0, 0, 0, 1], dtype=int)
        seed_candidates = [{
            "pair_action": reference_pair,
            "action": np.array([0, 1], dtype=int),
            "candidate_source": "myopic_reference_repaired",
        }]

        with patch.object(ablation_algorithms, "build_active_pair_universe", return_value=pair_universe), \
             patch.object(ablation_algorithms, "get_ai_action_dimension", return_value=2), \
             patch.object(ablation_algorithms, "get_sorted_ai_servers", return_value=[]):
            candidates = ablation_algorithms.build_pair_repair_candidates(
                State(), AblationExperimentConfig(uac_pair_repair_limit=12),
                queue_aware=True,
                seed_candidates=seed_candidates,
                source_prefix="uac",
            )

        neighbors = [
            item for item in candidates
            if "reference_low_impact_neighbor" in str(item.get("candidate_source", ""))
        ]
        self.assertTrue(neighbors)
        self.assertTrue(any(
            int(np.sum(np.asarray(item["pair_action"], dtype=int) != reference_pair)) == 1
            for item in neighbors
        ))
        swaps = [
            item for item in candidates
            if "reference_low_impact_swap" in str(item.get("candidate_source", ""))
        ]
        self.assertTrue(swaps)
        self.assertTrue(any(
            int(np.sum(np.asarray(item["pair_action"], dtype=int) != reference_pair)) == 2 and
            int(np.sum(np.asarray(item["pair_action"], dtype=int))) == int(np.sum(reference_pair))
            for item in swaps
        ))

    def test_pair_repair_emits_balanced_tail_relief_multiswap_from_reference_seed(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        import ablation_algorithms
        from ablation_config import AblationExperimentConfig

        class Env:
            def get_state_components(self):
                return np.ones(3), np.ones(3), np.ones(3)

        class State:
            environment_manager = Env()
            microservice_instances = {}
            request_flows = {}
            gsla_context = {}
            ffd_context = {}
            pdrs_context = {}
            loadaware_context = {}
            random_context = {}

        pair_universe = [
            {"flow_id": "f0", "server_id": "s0", "instance_id": "i0", "server_index": 0},
            {"flow_id": "f1", "server_id": "s1", "instance_id": "i1", "server_index": 1},
            {"flow_id": "f2", "server_id": "s2", "instance_id": "i2", "server_index": 2},
            {"flow_id": "f3", "server_id": "s0", "instance_id": "i3", "server_index": 0},
            {"flow_id": "f4", "server_id": "s1", "instance_id": "i4", "server_index": 1},
            {"flow_id": "f5", "server_id": "s2", "instance_id": "i5", "server_index": 2},
        ]
        reference_pair = np.array([0, 0, 0, 1, 1, 1], dtype=int)
        seed_candidates = [{
            "pair_action": reference_pair,
            "action": np.array([0, 1, 1], dtype=int),
            "candidate_source": "myopic_reference_repaired",
        }]

        with patch.object(ablation_algorithms, "build_active_pair_universe", return_value=pair_universe), \
             patch.object(ablation_algorithms, "get_ai_action_dimension", return_value=3), \
             patch.object(ablation_algorithms, "get_sorted_ai_servers", return_value=[]):
            candidates = ablation_algorithms.build_pair_repair_candidates(
                State(), AblationExperimentConfig(uac_pair_repair_limit=18),
                queue_aware=True,
                seed_candidates=seed_candidates,
                source_prefix="uac",
            )

        balanced = [
            item for item in candidates
            if "balanced_tail_relief" in str(item.get("candidate_source", ""))
        ]
        self.assertTrue(balanced)
        self.assertTrue(any(
            int(np.sum(np.asarray(item["pair_action"], dtype=int) != reference_pair)) == 4 and
            int(np.sum(np.asarray(item["pair_action"], dtype=int))) == int(np.sum(reference_pair)) and
            0 < int(np.sum(np.asarray(item["pair_action"], dtype=int))) < len(reference_pair)
            for item in balanced
        ))

    def test_paper_compact_actor_delay_resource_refinement_stays_hybrid(self):
        import sys
        from types import SimpleNamespace
        sys.path.insert(0, str(UTILS_DIR))

        import ablation_algorithms
        from ablation_config import AblationExperimentConfig

        class Env:
            def get_state_components(self):
                return (
                    np.asarray([1.2, 0.9], dtype=float),
                    np.asarray([3.0, 2.0], dtype=float),
                    np.asarray([12.0, 8.0], dtype=float),
                )

        service = SimpleNamespace(service_id="ai_test")
        request = SimpleNamespace(arrival_rate=8.0, r_input_data_size=256.0, r_output_data_size=64.0)
        servers = [
            SimpleNamespace(
                server_id=f"s{idx}",
                available_gpu_units=3,
                available_gpu_memory=48.0,
                available_model_storage=48.0,
                gpu_units=4,
                gpu_memory=64.0,
                model_storage=64.0,
            )
            for idx in range(2)
        ]
        pair_universe = [
            {"flow_id": "f0", "server_id": "s0", "instance_id": "i0", "server_index": 0},
            {"flow_id": "f1", "server_id": "s1", "instance_id": "i1", "server_index": 1},
            {"flow_id": "f2", "server_id": "s0", "instance_id": "i2", "server_index": 0},
            {"flow_id": "f3", "server_id": "s1", "instance_id": "i3", "server_index": 1},
        ]
        state = SimpleNamespace(
            environment_manager=Env(),
            microservice_instances={
                item["instance_id"]: SimpleNamespace(microservice=service)
                for item in pair_universe
            },
            request_flows={item["flow_id"]: request for item in pair_universe},
            gsla_context={},
            ffd_context={},
            pdrs_context={},
            loadaware_context={},
            random_context={},
        )
        seed_candidates = [{
            "pair_action": np.ones(len(pair_universe), dtype=int),
            "action": np.ones(2, dtype=int),
            "candidate_source": "actor_topk_all_cloud",
        }]
        config = AblationExperimentConfig(
            uac_candidate_mechanism="paper_compact",
            uac_compact_frontier_width=len(pair_universe),
            uac_compact_pair_repair_limit=8,
        )

        with patch.object(ablation_algorithms, "build_active_pair_universe", return_value=pair_universe), \
             patch.object(ablation_algorithms, "get_ai_action_dimension", return_value=2), \
             patch.object(ablation_algorithms, "get_sorted_ai_servers", return_value=servers), \
             patch("ablation_resource_models.select_local_ai_config", return_value={"latency_ms": 80.0, "energy_j": 12.0}), \
             patch("ablation_resource_models.solve_cloud_preprocess_config", return_value={"latency_ms": 160.0, "energy_j": 4.0}):
            candidates = ablation_algorithms.build_pair_repair_candidates(
                state, config,
                queue_aware=True,
                seed_candidates=seed_candidates,
                source_prefix="uac",
            )

        refinements = [
            item for item in candidates
            if "actor_delay_resource_refinement" in str(item.get("candidate_source", ""))
        ]

        self.assertTrue(refinements)
        self.assertLessEqual(len(refinements), 2)
        for item in refinements:
            cloud_count = int(np.sum(np.asarray(item["pair_action"], dtype=int)))
            self.assertGreater(cloud_count, 0)
            self.assertLess(cloud_count, len(pair_universe))
            self.assertEqual(item["resource_hint"], "latency_saver_hybrid")

    def test_uac_final_pool_includes_repaired_myopic_reference(self):
        import sys
        sys.path.insert(0, str(UTILS_DIR))

        import ablation_algorithms
        from ablation_config import AblationExperimentConfig

        class FakeInference:
            def make_candidates(self, system_state, config):
                return [{"action": np.array([0]), "candidate_source": "actor"}]

        captured_final_sources = []

        def fake_select(candidates, system_state, config, queue_aware=True):
            candidate_list = list(candidates)
            if queue_aware:
                captured_final_sources.extend(
                    str(item.get("candidate_source", ""))
                    for item in candidate_list
                    if isinstance(item, dict)
                )
                return {
                    "action": np.array([0]),
                    "pair_action_bits": "0",
                    "paper_dpp_score": 2.0,
                    "selected_candidate_source": "actor",
                }
            return {
                "action": np.array([1]),
                "pair_action_bits": "1",
                "paper_dpp_score": 1.0,
                "selected_candidate_source": "myopic_reference_seed_0",
                "candidate_source": "myopic_reference_repaired",
            }

        config = AblationExperimentConfig(enable_online_update=False)

        with patch.object(ablation_algorithms, "get_cached_uac_inference", return_value=FakeInference()), \
             patch.object(ablation_algorithms, "wrap_candidates_with_pair_projection", side_effect=lambda c, s: list(c)), \
             patch.object(ablation_algorithms, "build_myopic_candidates", return_value=[{"action": np.array([1])}]), \
             patch.object(ablation_algorithms, "build_pair_repair_candidates", return_value=[]), \
             patch.object(ablation_algorithms, "select_best_action_from_candidates", side_effect=fake_select):
            ablation_algorithms.run_UAC_DO(object(), config, slot=0, seed=38, algorithm="LyHAM-CO")

        self.assertIn("myopic_reference_repaired", captured_final_sources)


if __name__ == "__main__":
    unittest.main()











