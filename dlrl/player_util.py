from __future__ import division
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from dlrl.utils import normal  # , pi


class Agent(object):
    def __init__(self, model, env, state):
        self.model = model
        self.env = env
        self.state = state
        self.hx = None
        self.cx = None
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.done = True
        self.info = None
        self.reward = 0
        self.gpu_id = -1

    def action_test(self):
        with torch.no_grad():
            value, mu, sigma = self.model(
                Variable(self.state))
        mu = torch.clamp(mu.data, -1.0, 1.0)
        action = mu.cpu().numpy()# [0]
        state, self.reward, self.done, self.info = self.env.step(action)
        self.state = torch.from_numpy(state).float()
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.state = self.state.cuda()
        return self

    def clear_actions(self):
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        return self
