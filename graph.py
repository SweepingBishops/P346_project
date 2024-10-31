#!/usr/bin/env python
from collections.abc import Sequence
import numpy as np
from random import random as random_
from time import time
from draw import *

class Node(Sequence):
  '''Depreciated.'''
  def __init__(self,pos):
    self.pos = pos
    self.neighbors = list()
    self.clustered = False

  def __getitem__(self,i):
    return self.pos[i]
  def __len__(self):
    return 2  # If the length is not 2 something is seriously wrong.

class Graph:
  def __init__(self,nodes, P):
    self.nodes = set(nodes)
    self.not_clustered = set(nodes)
    self.clusters = list()
    self.neighbors = dict.fromkeys(nodes)
    self.GenerateEdges(P)

  def GenerateEdges(self,P):
    for node in self.nodes:
      x, y = node
      neighbors = list()
      if random_() <= P:
        neighbors.append(((x+1)%N,y%N))
      if random_() <= P:
        neighbors.append((x%N,(y+1)%N))
      for neighbor in neighbors:
        if neighbor not in self.nodes:
          neighbors.remove(neighbor)
      if self.neighbors[node]:
        self.neighbors[node].extend(neighbors)
      else:
        self.neighbors[node] = neighbors
      for neighbor in neighbors:
        if self.neighbors[neighbor]:
          self.neighbors[neighbor].append(node)
        else:
          self.neighbors[neighbor] = [node]

  def GetNeighbors(self,node):
    return self.neighbors[node]

  def GetNodes(self):
    return self.nodes

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

  def FindClusterFromNode(self,node):
    cluster = [node]
    self.not_clustered.remove(node)
    next_neighbors = self.neighbors[node]
    while next_neighbors:
      cluster.extend(next_neighbors)
      self.not_clustered.difference_update(next_neighbors)
      temp = set()
      for neighbor in next_neighbors:
        temp.update(self.neighbors[neighbor])
      next_neighbors = temp.intersection(self.not_clustered)
    return cluster

  def GetClusters(self):
    for node in self.nodes:
      if node in self.not_clustered:
        self.clusters.append(self.FindClusterFromNode(node))
    return self.clusters


nodes = list()
start = time()
N = 1000
for i in range(N):
  for j in range(N):
      nodes.append((i,j))
P=0.49
graph1 = Graph(nodes,P)
clusters = graph1.GetClusters()
P=0.51
graph2 = Graph(nodes,P)
clusters = graph2.GetClusters()
end = time()
DrawSquareNetworkBonds(graph1,nodelists=graph1.clusters,linewidth=2,imsize=600)
DrawSquareNetworkBonds(graph2,nodelists=graph2.clusters,linewidth=2,imsize=600)
#DrawSquareNetworkBonds(graph,linewidth=2,imsize=600)
print(f"Time taken: {end-start}")
#graph.clusters = dict()
#for node in graph.nodes.values():
#  node.cluster = None
#clusters = graph.GetClusters()
