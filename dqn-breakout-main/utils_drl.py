from typing import (
    Optional,
)

import random

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils_types import (
    TensorStack4,
    TorchDevice,
)

from utils_memory import ReplayMemory
from utils_model import DQN


class Agent(object):

    def __init__(
            self,
            action_dim: int,
            device: TorchDevice,
            gamma: float,
            seed: int,

            eps_start: float,
            eps_final: float,
            eps_decay: float,

            restore: Optional[str] = None,
    ) -> None:
        self.__action_dim = action_dim
        self.__device = device
        self.__gamma = gamma

        self.__eps_start = eps_start
        self.__eps_final = eps_final
        self.__eps_decay = eps_decay

        self.__eps = eps_start
        self.__r = random.Random()
        self.__r.seed(seed)

        self.__policy = DQN(action_dim, device).to(device)
        self.__target = DQN(action_dim, device).to(device)
        if restore is None:
            self.__policy.apply(DQN.init_weights)
        else:
            self.__policy.load_state_dict(torch.load(restore))
        self.__target.load_state_dict(self.__policy.state_dict())
        self.__optimizer = optim.Adam(
            self.__policy.parameters(),
            lr=0.0000625,
            eps=1.5e-4,
        )
        self.__target.eval()

    # ε-greedy 选择action
    def run(self, state: TensorStack4, training: bool = False) -> int:
        """run suggests an action for the given state."""
        if training:
            # 动态衰减ε-greedy的ε
            self.__eps -= \
                (self.__eps_start - self.__eps_final) / self.__eps_decay
            self.__eps = max(self.__eps, self.__eps_final)

        # 如果没碰上ε，选择使值函数最大的action，注意使用的是“max(1).indices”
        if self.__r.random() > self.__eps:
            with torch.no_grad():
                return self.__policy(state).max(1).indices.item() # .indices: 类比于argmax，返回使值函数最大的action
        # 如果ε，则随机选择
        return self.__r.randint(0, self.__action_dim - 1)

    # 训练网络
    def learn(self, memory: ReplayMemory, batch_size: int) -> float:
        """learn trains the value network via TD-learning."""
        # 从 Replay Buffer 中采样
        state_batch, action_batch, reward_batch, next_batch, done_batch = \
            memory.sample(batch_size)
        # 使用行为网络计算值函数 Q_j
        values = self.__policy(state_batch.float()).gather(1, action_batch)

        # 使用目标网络计算 Q_{j+1} 并计算 y_j = r_{j+1} + γ max_{a'} Q_{j+1} 
        # 其中(1-done_batch)用于判断是否terminal，若是则退化为 y_j = r_{j+1} 
        # 这里实现的应该是 DQN 而不是 PPT 上说的 Double DQN，两个 github 上说的也是 DQN
        values_next = self.__target(next_batch.float()).max(1).values.detach()  # max(1).values：按行返回最大值
        expected = (self.__gamma * values_next.unsqueeze(1)) * \
            (1. - done_batch) + reward_batch

        # 根据目标函数 (Q_j - y_{j})^2 进行梯度下降
        loss = F.smooth_l1_loss(values, expected)
        self.__optimizer.zero_grad()
        loss.backward()
        for param in self.__policy.parameters():
            param.grad.data.clamp_(-1, 1)
        self.__optimizer.step()

        return loss.item()
        
    # 同步行为网络和目标网络
    def sync(self) -> None:
        """sync synchronizes the weights from the policy network to the target
        network."""
        self.__target.load_state_dict(self.__policy.state_dict())

    # 保存行为网络
    def save(self, path: str) -> None:
        """save saves the state dict of the policy network."""
        torch.save(self.__policy.state_dict(), path)
