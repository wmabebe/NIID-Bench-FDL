"""
    Methods and structures for the DLRL environment
"""

import gym
from gym import spaces, Env
import numpy as np
import random

# TODO: proper seeding

def gen_cliques(matrix,c):
    """Given a similarity matrix (2-d), generate
        correspondig set of c cliques madeup of
        heterogeneous nodes. A clique contains the ids
        of the nodes in the clique.

    Args:
        matrix (list): 2-d simiarity matrix. Matrix row index = node id.
        c (int): Number of cliques. Individual clique size ~ len(matrix) / c.
    """
    pass

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

class DLRLEnv(gym.Env):
    def __init__(self, size=24, clique_size=5):
        super().__init__()

        # The observation space consists of a similarity matrix
        self.observation_space = spaces.Box(0, 5, (size ** 2 + size,), dtype=float)

        # The action space is a vector with a clique-preference value for each node.
        self.action_space = spaces.Box(0, 1, (size,), dtype=float)

        # keep the clique size and mat size for later
        self.size = size
        self.clique_size = clique_size

        self.rng = np.random.default_rng()


    def observe(self):
        return np.concatenate((np.reshape(self.mat, (self.size**2,)), self.mask))

    def seed(self, seed):
        self.rng = np.random.default_rng(seed=seed)

    def reset(self, mat):
        # We generate a new similarity matrix
        #self.mat = generate_matrix(self.size, self.size) if len(mat) == 0 or mat == None else mat
        self.mat = mat
        self.mask = np.zeros((self.size,))
        self.cliques = []

        return self.observe()

    def step(self, action):
        # mask out the unavailable cliques
        remaining_size = int(self.size - np.sum(self.mask))
        local_clique_size = min(remaining_size, self.clique_size)
        preferred_nodes = np.argsort(action)
        picked_nodes = []

        #print('action ', action)

        for i in preferred_nodes:
            if self.mask[i] == 0:
                picked_nodes.append(i)
                self.mask[i] = 1

            if len(picked_nodes) == local_clique_size:
                break

        #print('pick', picked_nodes, 'remaining', remaining_size, 'mask', self.mask)

        if len(picked_nodes) != local_clique_size:
            raise RuntimeError('unexpected local clique size ' + str(len(picked_nodes)) + ', expected ' + str(local_clique_size))

        self.cliques.append(picked_nodes)

        term = False
        if len(picked_nodes) == remaining_size:
            term = True

        info = {}

        # we return the immediate clique score as the reward
        reward = get_clique_value(picked_nodes, self.mat)

        return self.observe(), reward, term, info
