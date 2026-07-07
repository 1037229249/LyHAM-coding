"""
简化的Myopic贪心算法 - 基于阈值的简单决策
核心思想：性能因子 > 阈值(0.08) → 本地处理，否则 → 云端卸载
"""
import numpy as np
from typing import Dict, List, Optional
import random


class SimpleMyopicOptimizer:
    """
    简化的短视贪心优化器
    """
    def __init__(self):
        self.name = "Simple Myopic Optimizer"
        self.decision_history = []

        # 核心参数：性能因子阈值
        self.performance_threshold = 0.5

        # 简单的随机性
        self.randomness_rate = 0.02  # 10%的决策加入随机性

    def make_offloading_decision(self,
                                 SH: np.ndarray,  # AI服务器性能因子
                                 system_state: 'SystemState',
                                 weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        简单的阈值贪心决策
        """
        # 获取AI服务器列表
        ai_servers = [server for server in system_state.edge_servers.values()
                      if server.server_type.value == "ai_capable"]
        ai_server_ids = sorted([server.server_id for server in ai_servers])
        N = len(ai_server_ids)

        # 初始化决策向量
        offloading_decisions = np.zeros(N, dtype=int)

        print(f"\n=== {self.name} 简单阈值决策 ===")
        print(f"性能阈值: {self.performance_threshold}")
        print(f"性能因子: {SH}")

        # 为每个AI服务器做决策
        for i, server_id in enumerate(ai_server_ids):
            performance_factor = SH[i]

            # 1. 基础阈值决策
            if performance_factor > self.performance_threshold:
                base_decision = 0  # 本地处理
                reason = f"性能好({performance_factor:.3f} > {self.performance_threshold})"
            else:
                base_decision = 1  # 云端卸载
                reason = f"性能差({performance_factor:.3f} <= {self.performance_threshold})"

            # 2. 检查资源充足性（如果选择本地处理）
            if base_decision == 0:
                resource_ok = self._check_simple_resource(server_id, system_state)
                if not resource_ok:
                    base_decision = 1  # 强制云端卸载
                    reason = "资源不足，强制云端"

            # 3. 添加少量随机性
            final_decision = base_decision
            if random.random() < self.randomness_rate:
                final_decision = 1 - base_decision  # 翻转决策
                reason += " (随机翻转)"

            offloading_decisions[i] = final_decision

            decision_type = "云端卸载" if final_decision == 1 else "本地处理"
            print(f"  AI服务器 {i+1}: {decision_type} - {reason}")

        # 统计结果
        local_count = np.sum(offloading_decisions == 0)
        cloud_count = N - local_count

        # 记录历史
        self.decision_history.append({
            'decisions': offloading_decisions.copy(),
            'SH': SH.copy(),
            'local_count': local_count,
            'cloud_count': cloud_count
        })

        print(f"\n决策结果: 本地{local_count}个, 云端{cloud_count}个")
        print(f"决策向量: {offloading_decisions}")

        return offloading_decisions

    def _check_simple_resource(self, server_id: str, system_state: 'SystemState') -> bool:
        """
        简单的资源检查：只检查基本可用性
        """
        try:
            from ResourceAllocation import check_local_resource_sufficiency
            return check_local_resource_sufficiency(server_id, system_state)
        except:
            # 如果检查失败，假设资源充足
            return True

    def get_statistics(self) -> Dict:
        """获取简单统计信息"""
        if not self.decision_history:
            return {}

        recent = self.decision_history[-5:]  # 最近5次
        total_decisions = sum(len(hist['decisions']) for hist in recent)
        local_decisions = sum(hist['local_count'] for hist in recent)

        return {
            'algorithm': self.name,
            'threshold': self.performance_threshold,
            'recent_slots': len(recent),
            'local_ratio': local_decisions / max(total_decisions, 1),
            'total_decisions': total_decisions
        }

    def reset(self):
        """重置状态"""
        self.decision_history.clear()
        print(f"{self.name} 已重置")


def create_myopic_optimizer() -> SimpleMyopicOptimizer:
    """
    创建简化的短视优化器
    """
    return SimpleMyopicOptimizer()


def run_Myopic(SH: np.ndarray, system_state: 'SystemState',
               randomness_rate: float = 0.0) -> np.ndarray:
    """
    论文命名的Myopic快层入口
    消融实验默认关闭随机翻转，只保留短视阈值决策。
    """
    optimizer = create_myopic_optimizer()
    optimizer.randomness_rate = randomness_rate
    return optimizer.make_offloading_decision(SH, system_state)
