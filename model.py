"""
Module for DQN Model in Ape-X.
"""
import random
import torch
import torch.nn as nn


class DuelingDQN(nn.Module):
    """
    Dueling Network Architectures for Deep Reinforcement Learning
    https://arxiv.org/abs/1511.06581
    """
    def __init__(self, env):
        super(DuelingDQN, self).__init__()

        self.input_shape = env.observation_space.shape
        self.num_actions = env.action_space.n

        self.flatten = Flatten()

        self.features = nn.Sequential(
            init(nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU()
        )

        self.advantage = nn.Sequential(
            init(nn.Linear(self._feature_size(), 512)),
            nn.ReLU(),
            init(nn.Linear(512, self.num_actions))
        )

        self.value = nn.Sequential(
            init(nn.Linear(self._feature_size(), 512)),
            nn.ReLU(),
            init(nn.Linear(512, 1))
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean(1, keepdim=True)

    def _feature_size(self):
        return self.features(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)

    def act(self, state, epsilon):
        """
        Return action, max_q_value for given state
        """
        with torch.no_grad():
            state = state.unsqueeze(0)
            q_values = self.forward(state)

            if random.random() > epsilon:
                action = q_values.max(1)[1].item()
            else:
                action = random.randrange(self.num_actions)
        return action, q_values.numpy()[0]


class Flatten(nn.Module):
    """
    Simple module for flattening parameters
    """
    def forward(self, x):
        return x.view(x.size(0), -1)


def init_(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def init(module):
    return init_(module,
                 nn.init.orthogonal_,
                 lambda x: nn.init.constant_(x, 0),
                 nn.init.calculate_gain('relu'))
