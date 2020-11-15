import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self, action_dim, device):
        super(DQN, self).__init__()
        self.__conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, bias=False)
        self.__conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)
        self.__conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)

        # 将网络末端拆成 值网络 和 优势网络，对应 Dueling DQN 中的 基于状态的值函数 和 优势函数
        # q(s_t,a_t) = v(s_t) + A(s_t,a_t)
        self.__value = nn.Linear(64*7*7, 512)
        self.__value2 = nn.Linear(512, 1)
        self.__advantage = nn.Linear(64*7*7, 512)
        self.__advantage2 = nn.Linear(512, action_dim)

        # 原本代码
        # self.__fc1 = nn.Linear(64*7*7, 512)
        # self.__fc2 = nn.Linear(512, action_dim)
        self.__device = device

    def forward(self, x):
        x = x / 255.
        x = F.relu(self.__conv1(x))
        x = F.relu(self.__conv2(x))
        x = F.relu(self.__conv3(x))
        # x = F.relu(self.__fc1(x.view(x.size(0), -1)))
        # return self.__fc2(x)

        # q(s_t,a_t) = v(s_t) + ( A(s_t,a_t) - \frac{1}{|A|} \sum_{a'} A(s_t,a_t') )
        value = F.relu(self.__value(x.view(x.size(0), -1)))
        value = self.__value2(value)
        advantage = F.relu(self.__advantage(x.view(x.size(0), -1)))
        advantage = self.__advantage2(advantage)
        return value + advantage - advantage.mean(dim=1, keepdim=True)

    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            module.bias.data.fill_(0.0)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
