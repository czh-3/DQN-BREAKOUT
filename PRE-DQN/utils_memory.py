import random
import numpy as np
import torch

from typing import (
    Tuple, )

from utils_types import (
    BatchAction,
    BatchDone,
    BatchNext,
    BatchReward,
    BatchState,
    BatchWeight,
    TensorStack5,
    TorchDevice,
)


class Node:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        if self.a == self.b:
            self.sum = 0
            self.min = 0
        else:
            self.mid = (a + b) // 2
            self.left = Node(a, self.mid)
            self.right = Node(self.mid + 1, b)
            self.update()

    def update(self):
        self.sum = self.left.sum + self.right.sum
        self.min = min(self.left.min, self.right.min)

    def change(self, p, val):
        if self.a == self.b:
            self.sum = val
            self.min = val
        else:
            if p <= self.mid:
                self.left.change(p, val)
            else:
                self.right.change(p, val)
            self.update()

    def select(self, val):
        if self.a == self.b:
            return self.a, self.sum
        else:
            if val <= self.left.sum:
                return self.left.select(val)
            else:
                return self.right.select(self.left.sum - val)


class SegmentTree:
    def __init__(self, a, b):
        self.root = Node(a, b)

    def change(self, p, val):
        self.root.change(p, val)

    def random_select(self):
        return self.root.select(random.uniform(0, self.root.sum))

    def get_min(self):
        return self.root.min


class ReplayMemory(object):
    def __init__(
            self,
            channels: int,
            capacity: int,
            device: TorchDevice,
            eps,
            a,
            b,
            db,
    ) -> None:
        self.__device = device
        self.__capacity = capacity
        self.__size = 0
        self.__pos = 0

        self.__eps = eps
        self.__a = a
        self.__b = b
        self.__db = db

        self.__m_states = torch.zeros((capacity, channels, 84, 84),
                                      dtype=torch.uint8)
        self.__m_actions = torch.zeros((capacity, 1), dtype=torch.long)
        self.__m_rewards = torch.zeros((capacity, 1), dtype=torch.int8)
        self.__m_dones = torch.zeros((capacity, 1), dtype=torch.bool)
        self.__SegmentTree = SegmentTree(1, capacity)

    def push(
            self,
            folded_state: TensorStack5,
            action: int,
            reward: int,
            done: bool,
            agent,
    ) -> None:
        self.__m_states[self.__pos] = folded_state
        self.__m_actions[self.__pos, 0] = action
        self.__m_rewards[self.__pos, 0] = reward
        self.__m_dones[self.__pos, 0] = done

        indices = [self.__pos]
        state = self.__m_states[indices, :4].to(self.__device).float()
        next = self.__m_states[indices, 1:].to(self.__device).float()
        action = self.__m_actions[indices].to(self.__device)
        reward = self.__m_rewards[indices].to(self.__device).float()
        done = self.__m_dones[indices].to(self.__device).float()
        priority = agent.GetPriority(state, action, reward, done, next)
        priority = priority.cpu().reshape((1))
        priority = (priority + self.__eps)**self.__a
        self.__SegmentTree.change(self.__pos + 1, priority)
        self.__pos = (self.__pos + 1) % self.__capacity
        self.__size = max(self.__size, self.__pos)

    def sample(self, batch_size: int):
        self.__b = np.min([1.0, self.__b + self.__db])
        temp = np.array(
            [self.__SegmentTree.random_select() for i in range(batch_size)])
        indices = temp[:, 0]
        priority = torch.Tensor(temp[:, 1])
        min_pri = max(self.__SegmentTree.get_min(), 0.0001)
        weight = (priority / min_pri).pow(-self.__b)
        state = self.__m_states[indices, :4].to(self.__device).float()
        next = self.__m_states[indices, 1:].to(self.__device).float()
        action = self.__m_actions[indices].to(self.__device)
        reward = self.__m_rewards[indices].to(self.__device).float()
        done = self.__m_dones[indices].to(self.__device).float()
        return state, action, reward, next, weight, done

    def __len__(self) -> int:
        return self.__size
