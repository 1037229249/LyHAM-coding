"""
训练好的AI模型推理器
用于加载训练好的模型并进行即时决策
"""
import torch
import numpy as np
import json
import hashlib
from memory import AIOffloadingMemoryDNN
from typing import TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from Constant import SystemState





def validate_pair_actor_checkpoint_metadata(model_path: str, require_pair_actor: bool = False):
    """校验pair actor checkpoint及训练meta，formal/energy运行禁止旧server-level fallback。"""
    path = Path(model_path).expanduser().resolve()
    meta = {"model_path": str(path), "pair_actor": False}
    if not path.exists():
        return False, f"模型文件不存在: {path}", meta
    try:
        checkpoint = torch.load(str(path), map_location="cpu")
    except Exception as exc:
        return False, f"无法读取pair actor checkpoint: {exc}", meta
    pair_like = isinstance(checkpoint, dict) and "state_dict" in checkpoint and "feature_dim" in checkpoint
    meta.update({
        "pair_actor": bool(isinstance(checkpoint, dict) and checkpoint.get("pair_actor", False)),
        "feature_dim": int(checkpoint.get("feature_dim", 0)) if isinstance(checkpoint, dict) else 0,
        "training_dataset_hash": str(checkpoint.get("training_dataset_hash", "")) if isinstance(checkpoint, dict) else "",
        "train_config_hash": str(checkpoint.get("train_config_hash", "")) if isinstance(checkpoint, dict) else "",
    })
    meta_path = path.with_suffix(".meta.json")
    meta["meta_path"] = str(meta_path)
    if meta_path.exists():
        try:
            disk_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta.update({
                "pair_actor": bool(meta.get("pair_actor", False) or disk_meta.get("pair_actor", False)),
                "formal_seed_excluded": bool(disk_meta.get("formal_seed_excluded", False)),
                "model_mutated": bool(disk_meta.get("model_mutated", False)),
                "training_dataset_hash": str(meta.get("training_dataset_hash") or disk_meta.get("training_dataset_hash", "")),
                "train_config_hash": str(meta.get("train_config_hash") or disk_meta.get("train_config_hash", "")),
            })
        except Exception as exc:
            if require_pair_actor:
                return False, f"pair actor meta读取失败: {exc}", meta
    pair_ok = pair_like and bool(meta.get("pair_actor", False)) and bool(meta.get("training_dataset_hash", ""))
    if require_pair_actor:
        if not pair_ok:
            return False, "checkpoint不是可验证的pair actor，禁止server-level fallback", meta
        if not meta_path.exists():
            return False, "pair actor缺少训练meta，无法证明formal seed隔离", meta
        if not meta.get("formal_seed_excluded", False):
            return False, "pair actor meta未证明formal seed排除", meta
        if meta.get("model_mutated", False):
            return False, "pair actor meta显示模型被在线修改", meta
    return bool(pair_ok or not require_pair_actor), "", meta


class PairUACActorNetwork(torch.nn.Module):
    """
    pair-level UAC actor网络
    输入是active (flow, AI service, server) pair特征，输出该pair选择云端的logit。
    """

    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(int(feature_dim), int(hidden_dim)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(hidden_dim), int(hidden_dim)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, x):
        return self.net(x)

class TrainedAIInference:
    """AI推理器"""

    def __init__(self, model_path: str, strict_pair_actor: bool = False):
        """
        初始化推理器
        Args:
            model_path: 训练好的模型文件路径
        """
        self.model_path = model_path
        self.strict_pair_actor = bool(strict_pair_actor)
        self.checkpoint_meta = {}
        self.model = None
        self.is_loaded = False

        self.model_kind = "server_actor"
        self.training_dataset_hash = ""
        self.online_replay_buffer = []
        self.online_replay_records = []
        self.online_update_step = 0
        self.online_optimizer = None
        self.model_mutated_during_run = False

    def load_model(self):
        """加载训练好的模型，优先识别pair-level actor checkpoint。"""
        if self.strict_pair_actor:
            ok, reason, meta = validate_pair_actor_checkpoint_metadata(self.model_path, require_pair_actor=True)
            self.checkpoint_meta = dict(meta)
            if not ok:
                raise RuntimeError(reason)
        try:
            checkpoint = torch.load(self.model_path, map_location="cpu")
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint and "feature_dim" in checkpoint:
                feature_dim = int(checkpoint.get("feature_dim", 15))
                hidden_dim = int(checkpoint.get("hidden_dim", 128))
                model = PairUACActorNetwork(feature_dim=feature_dim, hidden_dim=hidden_dim)
                pair_state = checkpoint["state_dict"]
                try:
                    model.load_state_dict(pair_state)
                except RuntimeError:
                    # 训练脚本保存的是裸Sequential权重时，键名没有net.前缀。
                    model.net.load_state_dict(pair_state)
                model.eval()
                self.model = model
                self.model_kind = "pair_actor"
                self.checkpoint_state_dim = feature_dim
                self.checkpoint_num_ai_servers = 0
                self.training_dataset_hash = str(checkpoint.get("training_dataset_hash", "") or self.checkpoint_meta.get("training_dataset_hash", ""))
                self.is_loaded = True
                print(f"✓ Pair-level UAC模型加载成功: {self.model_path}")
                return
        except Exception:
            # 不是pair actor格式时，非formal兼容旧server-level checkpoint。
            if self.strict_pair_actor:
                raise
            pass

        try:
            # 创建模型架构（需要与训练时一致）
            state_dim = 10 * 3  # 10个AI服务器 * 3个状态组件
            num_ai_servers = 10
            if self.strict_pair_actor:
                raise RuntimeError("strict_pair_actor=True时禁止加载server-level checkpoint")
            self.checkpoint_state_dim = state_dim
            self.checkpoint_num_ai_servers = num_ai_servers
            self.model_kind = "server_actor"

            self.model = AIOffloadingMemoryDNN(
                state_dim=state_dim,
                num_ai_servers=num_ai_servers,
                learning_rate=0.008,
                training_interval=10,
                batch_size=256,
                memory_size=1024
            )

            # 加载训练好的参数
            self.model.load_model(self.model_path)
            self.is_loaded = True
            print(f"✓ AI模型加载成功: {self.model_path}")

        except Exception as e:
            print(f"✗ AI模型加载失败: {e}")
            self.is_loaded = False

    def make_candidates(self, system_state: 'SystemState', config=None,
                        decoder_mode: str = 'OPN') -> list:
        """
        使用训练好的AI生成候选动作集合
        UAC-DO新入口会逐候选调用Lyapunov evaluator，而不是直接取第一个动作。
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError(f"AI模型未加载，不能执行UAC-DO决策: {self.model_path}")

        if self.model_kind == "pair_actor":
            return self._make_pair_actor_candidates(system_state, config)

        # 获取当前系统状态
        env_manager = system_state.environment_manager
        SH, SQ, SZ = env_manager.get_state_components()

        # 组合状态向量 [SH, SQ, SZ]
        state_vector = np.concatenate((SH, SQ, SZ))
        expected_dim = int(getattr(self, "checkpoint_state_dim", 30))
        state_vector = self.fit_state_vector_to_checkpoint(state_vector, expected_dim)

        # 使用训练好的模型生成候选集合
        base_k = int(getattr(config, "uac_candidate_count", 10)) if config is not None else 10
        queue_spread = float(np.std(SQ) + np.std(SZ)) if len(SQ) else 0.0
        k = min(max(base_k + int(queue_spread > 1.0) * 2, 1), 32)
        candidate_actions = self.model.decode(state_vector, k=max(k, 1), mode=decoder_mode)

        n = len(SH)
        active_mask = self._active_ai_server_mask(system_state, n)
        candidates = []
        def add_candidate(action, source):
            candidates.append({
                "action": self._apply_active_mask(self._normalize_action(action, n), active_mask),
                "candidate_source": source,
            })

        for action in candidate_actions:
            add_candidate(action, "actor_topk")

        # 保护性候选，避免actor候选单一导致搜索退化
        add_candidate(np.zeros(n, dtype=int), "guard_all_local")
        add_candidate(np.ones(n, dtype=int), "guard_all_cloud")
        sh_min = float(np.min(SH)) if len(SH) else 0.0
        sh_span = float(np.max(SH) - sh_min) if len(SH) else 1.0
        sh_span = sh_span or 1.0
        normalized_sh = (np.asarray(SH, dtype=float) - sh_min) / sh_span if len(SH) else np.array([])
        for threshold in [0.35, 0.5, 0.65]:
            add_candidate((normalized_sh < threshold).astype(int), f"actor_threshold_{threshold:.2f}")

        # 固定种子Gumbel扰动，扩展Actor候选边界但保持复现性
        gumbel_count = int(getattr(config, "uac_gumbel_count", 4)) if config is not None else 4
        if n and gumbel_count > 0:
            seed_payload = np.asarray(state_vector, dtype=float)
            weights = np.arange(1, len(seed_payload) + 1, dtype=float)
            seed_value = int(abs(np.sum(np.round(seed_payload * 1000.0) * weights))) % (2 ** 32)
            rng = np.random.default_rng(seed_value)
            sq_norm = self._normalize_vector(SQ)
            sz_norm = self._normalize_vector(SZ)
            local_score = normalized_sh - 0.15 * sq_norm[:n] - 0.15 * sz_norm[:n]
            for _ in range(gumbel_count):
                noise = rng.gumbel(0.0, 0.15, size=n)
                threshold = float(np.median(local_score + noise))
                add_candidate((local_score + noise < threshold).astype(int), "actor_gumbel")

        unique_candidates = []
        seen = set()
        for candidate in candidates:
            action = candidate["action"]
            key = tuple(int(x) for x in action)
            if key not in seen:
                seen.add(key)
                unique_candidates.append(candidate)
        return unique_candidates

    def _make_pair_actor_candidates(self, system_state: 'SystemState', config=None) -> list:
        """使用离线pair actor生成active pair级候选动作。"""
        from ablation_pair_actions import (
            project_pair_action_to_server_action,
        )

        features, pair_universe, action_dim = self._pair_actor_feature_matrix(system_state, config)
        if not pair_universe:
            return []
        with torch.no_grad():
            logits = self.model(torch.tensor(features, dtype=torch.float32)).reshape(-1)
            probs = torch.sigmoid(logits).cpu().numpy()

        candidates = []
        seen = set()

        def add_pair(bits, source):
            pair_action = np.asarray(bits, dtype=int).reshape(-1)
            if len(pair_action) != len(pair_universe):
                pair_action = self._normalize_action(pair_action, len(pair_universe))
            key = tuple(int(x) for x in pair_action)
            if key in seen:
                return
            seen.add(key)
            server_action = project_pair_action_to_server_action(pair_action, pair_universe, action_dim)
            candidates.append({
                "action": server_action,
                "pair_action": pair_action,
                "pair_universe": pair_universe,
                "action_dim": int(action_dim),
                "action_scope": "pair",
                "candidate_source": source,
                "pair_actor": True,
                "training_dataset_hash": self.training_dataset_hash,
            })

        for threshold in [0.35, 0.5, 0.65]:
            add_pair((probs >= threshold).astype(int), f"pair_actor_threshold_{threshold:.2f}")
        for size in sorted(set([1, 2, 4, max(1, len(probs) // 3), max(1, len(probs) // 2)])):
            bits = np.ones(len(probs), dtype=int)
            local_idx = np.argsort(probs)[:min(size, len(probs))]
            bits[local_idx] = 0
            add_pair(bits, f"pair_actor_topk_local_{size}")
        add_pair(np.zeros(len(probs), dtype=int), "pair_actor_guard_all_local")
        add_pair(np.ones(len(probs), dtype=int), "pair_actor_guard_all_cloud")

        gumbel_count = int(getattr(config, "uac_gumbel_count", 4)) if config is not None else 4
        if gumbel_count > 0:
            seed_value = int(abs(np.sum(np.round(probs * 100000.0) * np.arange(1, len(probs) + 1)))) % (2 ** 32)
            rng = np.random.default_rng(seed_value)
            logits_np = np.log(np.clip(probs, 1e-6, 1 - 1e-6)) - np.log(np.clip(1 - probs, 1e-6, 1.0))
            for _ in range(gumbel_count):
                noise = rng.gumbel(0.0, 0.35, size=len(probs))
                add_pair((logits_np + noise >= 0.0).astype(int), "pair_actor_gumbel")
        return candidates

    def _pair_actor_feature_matrix(self, system_state: 'SystemState', config=None):
        """构造pair actor特征矩阵，并按checkpoint维度补齐或截断。"""
        from ablation_pair_actions import build_active_pair_universe, get_ai_action_dimension

        pair_universe = build_active_pair_universe(system_state)
        action_dim = get_ai_action_dimension(system_state)
        if not pair_universe:
            return np.zeros((0, int(getattr(self, "checkpoint_state_dim", 0))), dtype=np.float32), pair_universe, action_dim
        feature_rows = [self._extract_pair_actor_features(item, system_state, config) for item in pair_universe]
        features = np.asarray(feature_rows, dtype=np.float32)
        expected_dim = int(getattr(self, "checkpoint_state_dim", features.shape[1]))
        if features.shape[1] < expected_dim:
            pad = np.zeros((features.shape[0], expected_dim - features.shape[1]), dtype=np.float32)
            features = np.hstack([features, pad])
        elif features.shape[1] > expected_dim:
            features = features[:, :expected_dim]
        return features, pair_universe, action_dim

    def model_state_hash(self, length: int = 16) -> str:
        """计算当前内存模型参数哈希，用于formal在线更新追踪。"""
        if self.model is None or not hasattr(self.model, "state_dict"):
            return ""
        digest = hashlib.sha256()
        with torch.no_grad():
            for name, tensor in sorted(self.model.state_dict().items()):
                digest.update(str(name).encode("utf-8"))
                arr = tensor.detach().cpu().contiguous().numpy()
                digest.update(arr.tobytes())
        return digest.hexdigest()[:int(length)]

    def online_update_from_feature_batch(self, features, repaired_pair_action, config,
                                         seed: int, slot: int) -> dict:
        """用当前及历史repaired pair标签在线监督更新pair actor。"""
        if not bool(getattr(config, "enable_online_update", False)):
            return {"replay_written": False, "online_update_step": self.online_update_step, "model_mutated": False}
        if self.model_kind != "pair_actor" or self.model is None:
            raise RuntimeError("online UAC-DO requires pair_actor model")

        x = np.asarray(features, dtype=np.float32)
        y = np.asarray(repaired_pair_action, dtype=np.float32).reshape(-1)
        if x.ndim != 2 or len(x) == 0:
            return {"replay_written": False, "online_update_step": self.online_update_step, "model_mutated": False}
        if len(y) < x.shape[0]:
            y = np.pad(y, (0, x.shape[0] - len(y)), constant_values=1.0)
        elif len(y) > x.shape[0]:
            y = y[:x.shape[0]]

        record = {"seed": int(seed), "slot": int(slot), "sample_count": int(x.shape[0])}
        self.online_replay_buffer.append((x.copy(), y.copy()))
        self.online_replay_records.append(record)
        capacity = max(int(getattr(config, "online_replay_capacity", 512)), 1)
        if len(self.online_replay_buffer) > capacity:
            self.online_replay_buffer = self.online_replay_buffer[-capacity:]
            self.online_replay_records = self.online_replay_records[-capacity:]

        interval = max(int(getattr(config, "online_update_interval", 1)), 1)
        if len(self.online_replay_buffer) % interval != 0:
            return {"replay_written": True, "online_update_step": self.online_update_step, "model_mutated": False}

        batch_size = max(int(getattr(config, "online_update_batch_size", 16)), 1)
        batch = self.online_replay_buffer[-batch_size:]
        batch_x = torch.tensor(np.vstack([item[0] for item in batch]), dtype=torch.float32)
        batch_y = torch.tensor(np.concatenate([item[1] for item in batch]), dtype=torch.float32)
        if self.online_optimizer is None:
            lr = float(getattr(config, "online_learning_rate", 1e-4))
            self.online_optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        self.online_optimizer.zero_grad()
        logits = self.model(batch_x).reshape(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch_y)
        loss.backward()
        self.online_optimizer.step()
        self.model.eval()
        self.online_update_step += 1
        self.model_mutated_during_run = True
        return {
            "replay_written": True,
            "online_update_step": int(self.online_update_step),
            "model_mutated": True,
            "online_loss": float(loss.detach().cpu().item()),
            "final_model_hash": self.model_state_hash(),
        }

    def online_update_from_decision(self, system_state: 'SystemState', config, decision: dict,
                                    seed: int, slot: int) -> dict:
        """从shared evaluator选择的repaired pair action写replay并在线更新。"""
        features, pair_universe, _ = self._pair_actor_feature_matrix(system_state, config)
        repaired = decision.get("pair_action")
        if repaired is None or len(np.asarray(repaired).reshape(-1)) == 0:
            bits = str(decision.get("pair_action_bits", ""))
            if bits:
                repaired = [int(ch) for ch in bits if ch in {"0", "1"}]
        if repaired is None:
            repaired = np.ones(len(pair_universe), dtype=int)
        return self.online_update_from_feature_batch(features, repaired, config, seed=seed, slot=slot)

    def _extract_pair_actor_features(self, item: dict, system_state: 'SystemState', config=None) -> list:
        """提取与离线训练一致的pair特征，单位做轻量归一化。"""
        from ResourceAllocation import _pair_hapa_features

        ai_nodes = float(getattr(config, "ai_nodes", 10)) if config is not None else 10.0
        server = system_state.edge_servers.get(item.get("server_id"))
        flow = system_state.request_flows.get(item.get("flow_id"))
        hapa = _pair_hapa_features(item, system_state)
        y_queue = getattr(system_state.virtual_energy_queues.get(item.get("server_id")), "queue_state", 0.0)
        z_queue = getattr(system_state.virtual_delay_queues.get(item.get("server_id")), "queue_state", 0.0)
        return [
            float(item.get("server_index", 0)) / max(ai_nodes, 1.0),
            float(getattr(flow, "arrival_rate", 0.0)) / 20.0 if flow is not None else 0.0,
            float(getattr(flow, "r_input_data_size", 0.0)) / 1024.0 if flow is not None else 0.0,
            float(getattr(flow, "r_output_data_size", 0.0)) / 512.0 if flow is not None else 0.0,
            float(y_queue) / 1000.0,
            float(z_queue) / 1000.0,
            float(getattr(server, "available_gpu_units", 0.0)) / 16.0 if server is not None else 0.0,
            float(getattr(server, "available_gpu_memory", 0.0)) / 128.0 if server is not None else 0.0,
            float(getattr(server, "available_model_storage", 0.0)) / 512.0 if server is not None else 0.0,
            float(hapa.get("coverage_ratio", 1.0)),
            float(hapa.get("replica_readiness", 1.0)),
            float(hapa.get("hapa_psi", 1.0)),
            float(hapa.get("hapa_d_loc", 1.0)),
            float(hapa.get("reservation_pressure", 0.0)),
            1.0 if float(hapa.get("has_hapa_feedback", 0.0)) > 0.0 else 0.0,
        ]

    def make_decision(self, system_state: 'SystemState', decoder_mode: str = 'OPN') -> np.ndarray:
        """
        使用训练好的AI进行即时决策
        兼容旧入口，新消融入口会对候选集合逐个评分。
        """
        first = self.make_candidates(system_state, None, decoder_mode=decoder_mode)[0]
        return np.asarray(first.get("action", first), dtype=int)

    @staticmethod
    def _normalize_action(action, expected_len: int) -> np.ndarray:
        """整理二进制动作维度"""
        arr = np.array(action, dtype=float).reshape(-1)
        if len(arr) < expected_len:
            arr = np.pad(arr, (0, expected_len - len(arr)), constant_values=1)
        elif len(arr) > expected_len:
            arr = arr[:expected_len]
        return (arr >= 0.5).astype(int)

    @staticmethod
    def fit_state_vector_to_checkpoint(state_vector, expected_dim: int = 30) -> np.ndarray:
        """
        将当前场景状态投影到checkpoint输入维度
        小规模smoke会补零，正式10 AI节点场景保持原维度。
        """
        arr = np.asarray(state_vector, dtype=float).reshape(-1)
        expected_dim = int(max(expected_dim, 0))
        if expected_dim == 0:
            return arr
        if len(arr) < expected_dim:
            arr = np.pad(arr, (0, expected_dim - len(arr)), constant_values=0.0)
        elif len(arr) > expected_dim:
            arr = arr[:expected_dim]
        return arr

    @staticmethod
    def _normalize_vector(values) -> np.ndarray:
        """标准化向量到0-1区间"""
        arr = np.asarray(values, dtype=float)
        if len(arr) == 0:
            return arr
        span = float(np.max(arr) - np.min(arr)) or 1.0
        return (arr - float(np.min(arr))) / span

    @staticmethod
    def _active_ai_server_mask(system_state: 'SystemState', expected_len: int) -> np.ndarray:
        """标记当前慢层中已有AI实例的服务器"""
        ai_servers = sorted([
            server for server in system_state.edge_servers.values()
            if server.server_type.value == "ai_capable"
        ], key=lambda server: server.server_id)
        mask = np.zeros(expected_len, dtype=bool)
        for idx, server in enumerate(ai_servers[:expected_len]):
            for instance in system_state.microservice_instances.values():
                if (instance.server_id == server.server_id and
                        instance.microservice.service_type == "ai"):
                    mask[idx] = True
                    break
        return mask

    @staticmethod
    def _apply_active_mask(action: np.ndarray, active_mask: np.ndarray) -> np.ndarray:
        """无AI实例的服务器只能保留云端动作"""
        if len(active_mask) == 0:
            return action
        masked = np.array(action, dtype=int).copy()
        masked[~active_mask] = 1
        return masked

    def get_model_info(self):
        """获取模型信息"""
        if not self.is_loaded:
            return "模型未加载"
        if self.model_kind == "pair_actor":
            return f"PairUACActor(feature_dim={self.checkpoint_state_dim}, dataset={self.training_dataset_hash})"
        return self.model.get_model_summary()


def resolve_default_model_path() -> str:
    """解析默认模型路径"""
    return str(Path(__file__).resolve().parent.parent / "Training results" / "Training results1" / "trained_ai_offloading_model_1.2.pth")


def create_trained_ai_inference(model_path: str = None, strict: bool = False, strict_pair_actor: bool = False) -> TrainedAIInference:
    """
    创建训练好的AI推理器
    Args:
        model_path: 模型文件路径
    Returns:
        TrainedAIInference: AI推理器实例
    """
    if model_path is None:
        model_path = resolve_default_model_path()

    if strict and not Path(model_path).exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    inference = TrainedAIInference(model_path, strict_pair_actor=strict_pair_actor)
    inference.load_model()
    if strict and not inference.is_loaded:
        raise RuntimeError(f"模型加载失败: {model_path}")
    return inference




