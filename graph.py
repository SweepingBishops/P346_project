#!/usr/bin/env python
from collections.abc import Sequence
import numpy as np
from random import random as random_
from time import time
from draw import *
N = 50
P = 0.52

class Node(Sequence):
  def __init__(self,pos):
    self.pos = pos
    self.neighbors = list()
    self.clustered = False

  def __getitem__(self,i):
    return self.pos[i]
  def __len__(self):
    return 2  # If the length is not 2 something is seriously wrong.

class Graph:
  def __init__(self,nodes):
    self.nodes = nodes
    self.clusters = list()
    self.GenerateEdges()

  def GenerateEdges(self):
    for pos in self.nodes:
      x, y = pos
      neighbors = list()
      if random_() <= P:
        neighbors.append(((x+1)%N,y%N))
      if random_() <= P:
        neighbors.append((x%N,(y+1)%N))
      for neighbor_pos in neighbors:
        if self.nodes.get(neighbor_pos) is None:
          neighbors.remove(neighbor_pos)
      self.nodes.get(pos).neighbors.extend(neighbors)
      for neighbor in neighbors:
        self.nodes.get(neighbor).neighbors.append(pos)

  def GetNeighbors(self,pos):
    return self.nodes.get(pos).neighbors

  def GetNodes(self):
    return list(self.nodes.keys())

  def OldGetClusters(self):
    for pos in self.nodes:
      node = self.nodes[pos]
      if node.cluster is None:
        if len(self.clusters) == 0:
          new_cluster = 1
        else:
          new_cluster = list(self.clusters.keys())[-1] + 1
        self.clusters[new_cluster] = [pos]
        node.cluster = new_cluster
        for neighbor_pos in node.neighbors:
          self.nodes[neighbor_pos].cluster = new_cluster
        self.clusters[new_cluster].extend(node.neighbors)
      else:
        cluster = node.cluster
        for neighbor_pos in node.neighbors:
          self.nodes[neighbor_pos].cluster = cluster
        self.clusters[cluster].extend(node.neighbors)
    return list(self.clusters.values())

  def FindClusterFromNode(self,pos):
    cluster = [pos]
    self.nodes[pos].clustered = True
    next_neighbors = self.nodes[pos].neighbors
    while next_neighbors:
      cluster.extend(next_neighbors)
      current_neighbors = next_neighbors.copy()
      next_neighbors = list()
      for neighbor in current_neighbors:
        self.nodes[neighbor].clustered = True
        for next_neighbor in self.nodes[neighbor].neighbors:
          if not self.nodes[next_neighbor].clustered:
            next_neighbors.append(next_neighbor)
    return cluster

  def GetClusters(self):
    for pos in self.nodes:
      if not self.nodes[pos].clustered:
        self.clusters.append(self.FindClusterFromNode(pos))
    return self.clusters


nodes = dict()
start = time()
for i in range(N):
  for j in range(N):
    nodes[(i,j)] = Node((i,j))
graph = Graph(nodes)
clusters = graph.GetClusters()
end = time()
DrawSquareNetworkBonds(graph,nodelists=clusters,linewidth=2)
print(f"Time taken: {end-start}")
#graph.clusters = dict()
#for node in graph.nodes.values():
#  node.cluster = None
#clusters = graph.GetClusters()
