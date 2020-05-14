import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_size, action_size, fc1=256, fc2=128, leak=0.01, seed=0):
        """

        :param state_size: State dimension
        :param action_size: Action dimension
        :param fc1: #hidden neurons in first fc layer
        :param fc2: #hidden neurons in second fc layer
        :param leak: leaky Relu size
        :param seed: Random seed
        """

        super(Actor, self).__init__()
        self.seed = seed
        self.leak = leak

        self.fc1 = nn.Linear(state_size, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.fc3 = nn.Linear(fc2, action_size)

        self.bn = nn.BatchNorm1d(state_size)
        self.initialize_weights()

    def initialize_weights(self):
        """
        Initialize weights
        :return:
        """

        torch.nn.init.kaiming_normal_(self.fc1.weight.data, a=self.leak, mode='fan_in')
        torch.nn.init.kaiming_normal_(self.fc2.weight.data, a=self.leak, mode='fan_in')
        torch.nn.init.uniform_(self.fc3.weight.data, -3e-3, 3e-3)

    def forward(self, state):
        """
        Forward pass of the network
        :param state:
        :return:
        """

        state = self.bn(state)
        x = F.leaky_relu(self.fc1(state), negative_slope=self.leak)
        x = F.leaky_relu(self.fc2(x), negative_slope=self.leak)
        x = F.tanh(self.fc3(x))

        return x


class Critic(nn.Module):
    def __init__(self, state_size, action_size, fc1=256, fc2=128, fc3=128, leak=0.01, seed=0):
        """
        :param state_size: State dimesion
        :param action_size: Action dimension
        :param fc1: #hidden neurons in fc1
        :param fc2: #hidden neurons in fc2
        :param fc3: #hidden neurons in fc3
        :param leak: #Leaky Relu slope
        :param seed: Random seed
        """
        super(Critic, self).__init__()
        self.leak = leak
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, fc1)
        self.fc2 = nn.Linear(fc1 + action_size, fc2)
        self.fc3 = nn.Linear(fc2, fc3)
        self.fc4 = nn.Linear(fc3, 1)

        self.bn = nn.BatchNorm1d(state_size)

        self.initialize_weights()

    def initialize_weights(self):
        """
        Initialize weights
        :return:
        """

        torch.nn.init.kaiming_normal_(self.fc1.weight.data, a=self.leak, mode='fan_in')
        torch.nn.init.kaiming_normal_(self.fc2.weight.data, a=self.leak, mode='fan_in')
        torch.nn.init.uniform_(self.fc3.weight.data, -3e-3, 3e-3)

    def forward(self, state, action):
        """

        :param state:
        :param action:
        :return:
        """

        state = self.bn(state)
        x = F.leaky_relu(self.fc1(state), negative_slope=self.leak)
        x = torch.cat((x, action), dim=1)
        x = F.leaky_relu(self.fc2(x), negative_slope=self.leak)
        x = F.leaky_relu(self.fc3(x), negative_slope=self.leak)
        x = self.fc4(x)

        return x
