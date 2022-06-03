import numpy as np
from numpy import dot
from numpy.linalg import norm
import networkx as nx
import matplotlib.pyplot as plt
import math
import random
from utils import flatten_layers, get_radians

def build_graph(adj_list,graph):
    edges = []
    for node in adj_list:
        for neighbor in node.neighbors:
            if [node.id, neighbor.id] not in edges:
                edges.append([node.id, neighbor.id])
                graph.add_edge(node.id, neighbor.id)
    return graph

def color_graph(adj_list,global_gradient=None):
    colors = []
    if global_gradient != None:
        max_rad = float('-inf')
        for node in adj_list:
            g1 = flatten_layers(node.model.grads)
            g2 = flatten_layers(global_gradient)
            radian = get_radians(g1, g2)
            colors.append(radian)
            max_rad = radian if radian > max_rad else max_rad
        colors = [c/max_rad for c in colors]
    else:
        for node in adj_list:
            color = round(node.id / len(adj_list),2)
            colors.append(color)
    return colors

def draw_graph(graph,fname,colors=None):
    f = plt.figure()
    nx.draw_networkx(graph,node_color=colors)
    f.savefig(fname,bbox_inches='tight')

class Node:
    def __init__(self,id,model,data=None,max_peers=2):
        self.id = id
        self.model = model
        self.data = data
        self.neighbors = []
        self.max_peers = max_peers
        self.candidates = []
    
    def add_neighbors(self,nodes):
        if len(self.neighbors) + len(nodes) <= self.max_peers:
            self.neighbors += nodes
        else:
            raise Exception("Max number of peers ("+ str(self.max_peers) +") already reached!")
    
    def add_neighbor(self,node):
        if len(self.neighbors) < self.max_peers:
            self.neighbors.append(node)
        else:
            raise Exception("Max number of peers ("+ str(self.max_peers) +") already reached!")

    def __str__(self):
        return "N:" + str(self.id)
    
    def describe(self):
        return str(self) + " - " +  str([ str(n) for n in self.neighbors])
    
    def next_candidates(self,sort="grads"):
        self.candidates = [n for n in self.neighbors if n != self]
        for neighbor in self.neighbors:
            for n in neighbor.neighbors:
                if n not in self.candidates and n != self:
                    self.candidates.append(n)
        #If sort by grads, flatten node grads and compare degree similarity
        if sort == "grads": 
            self.candidates.sort(key=lambda n: abs( get_radians(flatten_layers(self.model.grads),
                                      flatten_layers(n.model.grads))) , reverse=True)
        #Sort by id, is a test feature to see how clumping works
        elif sort == "id":
            self.candidates.sort(key=lambda n: abs(self.id - n.id))

    #non_iid_frac represents the fraction of non_iid peers
    #we want to include in the neighbors list.
    #iid_strategy could be either 'random', or 'worst'.
    #Random strategy, grafts random non_iid peers when 
    #non_iid_frac > 0. Worst strategy, picks the most disimilar
    #peers when non_iid_frac > 0.
    def next_peers(self,non_iid_frac=0,non_iid_strategy="random"):
        if non_iid_strategy not in ["random","extreme"]:
            raise Exception("non_iid_strategy should be either 'random' or 'extreme'!")
        if non_iid_frac > 1 or non_iid_frac < 0:
            raise Exception("non_iid_frac has to be a value between [0,1]. " + str(non_iid_frac) + " given!")
        if len(self.candidates) <= 0:
            raise Exception("No candidates to pick next neighbors from!")
        
        self.neighbors = self.candidates[:self.max_peers]
        if non_iid_frac > 0:
            non_iid_size = math.ceil(non_iid_frac * self.max_peers)
            if non_iid_size <= len(self.candidates[self.max_peers:]):
                #Graft most disimilar non-iid peer
                if non_iid_strategy == "extreme":
                    self.neighbors[self.max_peers - non_iid_size:] = self.candidates[-non_iid_size:]
                #Graft random non-iid peer
                elif non_iid_strategy == "random":
                    self.neighbors[self.max_peers - non_iid_size:] = random.sample(self.candidates[self.max_peers:],k=non_iid_size)
            
        