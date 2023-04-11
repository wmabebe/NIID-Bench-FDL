from __future__ import division
from setproctitle import setproctitle as ptitle
import numpy as np
import torch
from environment import create_env
from utils import setup_logger
from model import A3C_CONV, A3C_MLP
from player_util import Agent
from torch.autograd import Variable
import time
import logging
import gym

from dlrl import DLRLEnv

player = Agent(

def gencliq(args, shared_model):
    player.model = A3C_MLP(
        args.size ** 2 + args.size, player.env.action_space, args.stack_frames)

    player.state = player.env.reset()
    player.state = torch.from_numpy(player.state).float()
    player.model.eval()

    while not player.done:
        player.action_test()
        reward_sum += player.reward


                state = player.env.reset()
                time.sleep(10)
                player.state = torch.from_numpy(state).float()
                if gpu_id >= 0:
                    with torch.cuda.device(gpu_id):
                        player.state = player.state.cuda()
