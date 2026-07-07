"""
AI微服务计算卸载的深度神经网络记忆模块
主要功能：
1. 深度神经网络构建和训练
2. 经验存储和回放机制
3. AI服务器卸载决策生成（二进制向量）
4. 支持多种解码模式（OP, KNN, OPN）

状态输入：[SH, SQ] 其中
- SH: AI服务器性能环境因子向量
- SQ: AI服务器虚拟能耗队列状态向量

动作输出：N维二进制向量，N为AI服务器数量
- 0: 本地处理（local processing）
- 1: 云端卸载（cloud offloading）
"""

from __future__ import print_function

import os

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import itertools
from typing import List, Optional


class AIOffloadingMemoryDNN:
    """
    AI微服务计算卸载的深度神经网络记忆模块
    类比LyDROO中的MemoryDNN，但适配AI微服务卸载场景
    """

    def __init__(
            self,
            state_dim: int,  # 状态维度 = len(SH) + len(SQ) = 2*N_ai_servers
            num_ai_servers: int,  # AI服务器数量，决定动作维度
            learning_rate: float = 0.01,
            training_interval: int = 10,
            batch_size: int = 128,
            memory_size: int = 1024,
            output_graph: bool = False
    ):

        self.state_dim = state_dim
        self.num_ai_servers = num_ai_servers
        self.action_dim = num_ai_servers  # 动作维度等于AI服务器数量

        # 输入：状态向量 [SH, SQ]
        # 输出：AI服务器卸载决策向量 [0,1]^N
        self.net = [state_dim, 256, 128, num_ai_servers]

        # 训练参数
        self.training_interval = training_interval
        self.lr = learning_rate
        self.batch_size = batch_size
        self.memory_size = memory_size

        # 枚举所有可能的二进制动作（用于KNN解码）
        self.enumerate_actions = []

        # 记忆计数器
        self.memory_counter = 1

        # 训练损失历史
        self.cost_his = []

        # 初始化记忆缓冲区 [state, action]
        # state_dim + action_dim
        self.memory = np.zeros((self.memory_size, self.state_dim + self.action_dim))

        # 构建神经网络
        self._build_net()

        print(f"AI卸载记忆网络初始化完成:")
        print(f"  状态维度: {self.state_dim}")
        print(f"  AI服务器数量: {self.num_ai_servers}")
        print(f"  动作维度: {self.action_dim}")
        print(f"  网络结构: {self.net}")

    def _build_net(self):
        """构建深度神经网络"""
        self.model = nn.Sequential(
            nn.Linear(self.net[0], self.net[1]),
            nn.LeakyReLU(0.1),
            nn.Linear(self.net[1], self.net[2]),
            nn.LeakyReLU(0.1),
            nn.Linear(self.net[2], self.net[3]),
            nn.Sigmoid()  # 输出0-1之间，便于二进制解码
        )

        # 初始化网络权重
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def remember(self, state: np.ndarray, action: np.ndarray):
        """
        存储经验到记忆缓冲区
        Args:
            state: 状态向量 [SH, SQ]
            action: 动作向量 [0,1]^N
        """
        # 替换旧记忆
        idx = self.memory_counter % self.memory_size
        self.memory[idx, :] = np.hstack((state, action))
        self.memory_counter += 1

    def encode(self, state: np.ndarray, action: np.ndarray):
        """
        编码经验并触发训练
        Args:
            state: 状态向量
            action: 最优动作向量
        """
        # 存储经验
        self.remember(state, action)

        # 定期训练网络
        if self.memory_counter % self.training_interval == 0:
            self.learn()

    def learn(self):
        """训练深度神经网络"""
        # 采样批次记忆
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)

        batch_memory = self.memory[sample_index, :]

        # 分离状态和动作
        state_train = torch.Tensor(batch_memory[:, 0:self.state_dim])
        action_train = torch.Tensor(batch_memory[:, self.state_dim:])

        # 学习率衰减策略
        current_episode = len(self.cost_his)
        if current_episode < 100:
            current_lr = self.lr  # 前100次训练使用原始学习率

        elif current_episode < 200:
            current_lr = self.lr * 0.8  # 100-300次训练使用0.8倍学习率
        elif current_episode < 400:
            current_lr = self.lr * 0.6  # 300-600次训练使用0.6倍学习率
        elif current_episode < 600:
            current_lr = self.lr * 0.4  # 600-1000次训练使用0.4倍学习率
        else:
            current_lr = self.lr * 0.2  # 1000次后使用0.2倍学习率

        # 使用动态学习率创建优化器
        optimizer = optim.Adam(self.model.parameters(), lr=current_lr,
                               betas=(0.09, 0.999), weight_decay=0.0001)

        criterion = nn.BCELoss()

        self.model.train()
        optimizer.zero_grad()

        # 前向传播
        predict = self.model(state_train)
        loss = criterion(predict, action_train)

        # 反向传播
        loss.backward()

        # 添加梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)

        optimizer.step()

        # 记录损失
        self.cost = loss.item()
        assert (self.cost > 0)
        self.cost_his.append(self.cost)

    def decode(self, state: np.ndarray, k: int = 10, mode: str = 'OPN') -> List[np.ndarray]:
        """
        解码生成AI服务器卸载决策
        Args:
            state: 状态向量 [SH, SQ]
            k: 生成k个候选动作
            mode: 解码模式 'OP', 'KNN', 'OPN'
        Returns:
            List[np.ndarray]: k个二进制卸载决策向量
        """
        # 添加批次维度
        state_tensor = torch.Tensor(state[np.newaxis, :])

        self.model.eval()
        with torch.no_grad():
            action_pred = self.model(state_tensor)

        action_pred = action_pred.detach().numpy()[0]

        if mode == 'OP':
            return self.order_preserving_decode(action_pred, k)
        elif mode == 'KNN':
            return self.knn_decode(action_pred, k)
        elif mode == 'OPN':
            return self.order_preserving_with_noise_decode(action_pred, k)
        else:
            raise ValueError("解码模式必须是 'OP', 'KNN' 或 'OPN'")

    def order_preserving_decode(self, prediction: np.ndarray, k: int = 1) -> List[np.ndarray]:
        """
        保序解码：生成k个二进制卸载决策
        算法：基于预测值与0.5的距离生成多样化动作
        """
        action_list = []

        # 生成第一个二进制决策（基于0.5阈值）
        first_action = (prediction > 0.5).astype(int)
        action_list.append(first_action)

        if k > 1:
            # 生成剩余k-1个决策（基于与0.5的距离）
            distance_from_half = np.abs(prediction - 0.5)
            sorted_indices = np.argsort(distance_from_half)

            for i in range(min(k - 1, self.num_ai_servers)):
                idx = sorted_indices[i]

                # 创建新动作：翻转最不确定的服务器决策
                new_action = first_action.copy()
                if prediction[idx] > 0.5:
                    # 原本是卸载(1)，改为本地处理(0)
                    new_action[idx] = 0
                else:
                    # 原本是本地处理(0)，改为卸载(1)
                    new_action[idx] = 1

                action_list.append(new_action)

        return action_list

    def knn_decode(self, prediction: np.ndarray, k: int = 1) -> List[np.ndarray]:
        """
        K近邻解码：从所有可能的二进制动作中选择最相似的k个
        """
        # 懒加载：生成所有可能的二进制动作
        if len(self.enumerate_actions) == 0:
            self.enumerate_actions = np.array(
                list(itertools.product([0, 1], repeat=self.num_ai_servers))
            )

        # 计算欧几里得距离
        distances = np.sum((self.enumerate_actions - prediction) ** 2, axis=1)

        # 选择距离最小的k个动作
        nearest_indices = np.argsort(distances)[:k]

        return [self.enumerate_actions[idx] for idx in nearest_indices]


    def order_preserving_with_noise_decode(self, prediction: np.ndarray, k: int ) -> List[np.ndarray]:
        """
        保序加噪声解码：与LyDROO完全一致的OPN实现
        Args:
            prediction: 神经网络的原始输出预测 [0,1]^N
            k: 每种类型（原始+噪声）生成的候选数量
        Returns:
            List[np.ndarray]: 候选动作列表，总数为2*k个（与LyDROO一致）
        """

        # 第一步：对原始预测使用保序解码，生成k个候选
        original_actions = self.order_preserving_decode(prediction, k)

        # 第二步：添加高斯噪声（标准差为1.0）
        noisy_prediction = prediction + np.random.normal(0, 1.0, len(prediction))

        # 第三步：通过sigmoid压缩到[0,1]范围
        noisy_prediction = 1 / (1 + np.exp(-noisy_prediction))

        # 第四步：对噪声版本也使用保序解码，生成k个候选
        noisy_actions = self.order_preserving_decode(noisy_prediction, k)

        print(f"OPN解码详情:")
        print(f"  原始预测: {prediction}")
        print(f"  噪声预测: {noisy_prediction}")
        print(f"  原始{k}个候选: {[action.tolist() for action in original_actions]}")
        print(f"  噪声{k}个候选: {[action.tolist() for action in noisy_actions]}")
        print(f"  总计{len(original_actions + noisy_actions)}个候选")

        # 返回2*k个候选：原始k个 + 噪声k个
        return original_actions + noisy_actions

    def get_action_space_size(self) -> int:
        """获取动作空间大小"""
        return 2 ** self.num_ai_servers

    def get_network_output_size(self) -> int:
        """获取网络输出维度"""
        return self.num_ai_servers

    def plot_cost(self):
        """绘制训练损失曲线"""
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.plot(np.arange(len(self.cost_his)) * self.training_interval, self.cost_his)
            plt.ylabel('Training Loss')
            plt.xlabel('Time Frames')
            plt.title('AI Offloading DNN Training Loss')
            plt.grid(True)
            plt.show()
        except ImportError:
            print("matplotlib未安装，无法绘制损失曲线")
            print(f"训练损失历史: {self.cost_his[-10:]}")  # 显示最近10个损失值

    def save_model(self, filepath: str):
        """保存模型（改进版）"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            # 保存模型
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'memory': self.memory,
                'memory_counter': self.memory_counter,
                'cost_his': self.cost_his,
                'net_config': self.net
            }, filepath)
            print(f"✓ 模型成功保存到: {filepath}")
            return True

        except Exception as e:
            print(f"✗ 模型保存失败: {e}")
            return False

    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.memory = checkpoint['memory']
        self.memory_counter = checkpoint['memory_counter']
        self.cost_his = checkpoint['cost_his']
        print(f"模型已从 {filepath} 加载")

    def get_model_summary(self) -> dict:
        """获取模型摘要信息"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        # 修正记忆使用计算
        actual_memory_used = min(self.memory_counter - 1, self.memory_size) if self.memory_counter > 1 else 0

        return {
            'network_structure': self.net,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'memory_usage': f"{actual_memory_used}/{self.memory_size}",
            'training_episodes': len(self.cost_his),
            'current_loss': self.cost_his[-1] if self.cost_his else None
        }

    def print_model_info(self):
        """打印模型信息 - 精简版"""
        summary = self.get_model_summary()
        print(f"\n=== AI卸载DNN模型信息 ===")
        print(f"网络结构: {summary['network_structure']}")
        print(f"总参数量: {summary['total_parameters']:,}")
        print(f"记忆使用: {summary['memory_usage']}")
        print(f"训练轮次: {summary['training_episodes']}")
        if summary['current_loss']:
            print(f"当前损失: {summary['current_loss']:.6f}")




