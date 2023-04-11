from __future__ import division
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from dlrl.utils import norm_col_init, weights_init, weights_init_mlp

class A3C_MLP(torch.nn.Module):
    def __init__(self, num_inputs, action_space, size=24):
        UNITS=size ** 2

        super(A3C_MLP, self).__init__()
        self.fc1 = nn.Linear(num_inputs, UNITS)
        self.lrelu1 = nn.LeakyReLU(0.1)
        self.fc2 = nn.Linear(UNITS, UNITS)
        self.lrelu2 = nn.LeakyReLU(0.1)
        self.fc3 = nn.Linear(UNITS, int(UNITS/2))
        self.lrelu3 = nn.LeakyReLU(0.1)
        self.fc4 = nn.Linear(int(UNITS/2), int(UNITS/2))
        self.lrelu4 = nn.LeakyReLU(0.1)

        num_outputs = action_space.shape[0]
        self.critic_linear = nn.Linear(int(UNITS/2), 1)
        self.actor_linear = nn.Linear(int(UNITS/2), num_outputs)
        self.actor_linear2 = nn.Linear(int(UNITS/2), num_outputs)

        self.apply(weights_init_mlp)
        lrelu = nn.init.calculate_gain('leaky_relu')
        self.fc1.weight.data.mul_(lrelu)
        self.fc2.weight.data.mul_(lrelu)
        self.fc3.weight.data.mul_(lrelu)
        self.fc4.weight.data.mul_(lrelu)

        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.actor_linear2.weight.data = norm_col_init(
            self.actor_linear2.weight.data, 0.01)
        self.actor_linear2.bias.data.fill_(0)
        self.critic_linear.weight.data = norm_col_init(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.train()

    def forward(self, inputs):
        x = inputs

        x = self.lrelu1(self.fc1(x))
        x = self.lrelu2(self.fc2(x))
        x = self.lrelu3(self.fc3(x))
        x = self.lrelu4(self.fc4(x))

        return self.critic_linear(x), F.softsign(self.actor_linear(x)), self.actor_linear2(x)
