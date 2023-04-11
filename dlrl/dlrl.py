"""
    Methods and structures for the DLRL environment
"""

import gym
from gym import spaces, Env
import numpy as np
import random
from dlrl.model import A3C_MLP
import torch

from dlrl.player_util import Agent
from dlrl.dlrl_env import DLRLEnv

SIZE = 24       # the model must be regenerated for new matrix sizes
CLIQUE_SIZE = 5 # likewise, might not perform well if this is changed

env = DLRLEnv(size=SIZE, clique_size=CLIQUE_SIZE)

model = A3C_MLP(SIZE ** 2 + SIZE, env.action_space, size=SIZE)
params = torch.load('dlrl/clique_mlp.dat')
model.load_state_dict(params)

player = Agent(model, env, None)

def gen_cliques(matrix,c):
    """Given a similarity matrix (2-d), generate
        correspondig set of c cliques madeup of
        heterogeneous nodes. A clique contains the ids
        of the nodes in the clique.

    Args:
        matrix (list): 2-d simiarity matrix. Matrix row index = node id.
        c (int): Number of cliques. Individual clique size ~ len(matrix) / c.
    """

    player.state = player.env.reset(mat=matrix)
    player.state = torch.from_numpy(player.state).float()
    player.env.clique_size = c
    player.done = False
    player.model.eval()

    while not player.done:
        player.action_test()

    return player.env.cliques

def get_clique_value(clique,matrix):
    """Given a clique, compute the clique value by.
        summing the similarities between all pairs in the.
        clique.

    Args:
        clique (list): List of node id's in a clique.
        matrix (list): 2-d simiarity matrix. Matrix row index = node id.

    Returns:
        float: Clique heterogeneous value.
    """
    val = 0
    for n1 in clique:
        for n2 in clique:
            val += matrix[n1][n2]
    return val / 2

def generate_matrix(n,m):
    """Generate a simulated similarity matrix of size n with
    uniformly random values (0,m).

    Args:
        n (int): Matrix size.
        m (int): Max value for random values.

    Returns:
        list: Similarity matrix.
    """
    matrix = [[None for i in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i][j] = 0
                matrix[j][i] = 0
            elif matrix[i][j] == None and matrix[j][i] == None:
                matrix[i][j] = random.uniform(0,m)
                lower = max(0,0.5 * matrix[i][j])
                upper = min(m,matrix[i][j] * 1.5)
                matrix[j][i] = random.uniform(lower,upper)
    return matrix
