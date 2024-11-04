#!/usr/bin/env python
from collections.abc import Sequence
import numpy as np
from random import random as random_
from time import time
from draw import *
import multiprocessing

class Node(Sequence):
  '''Depreciated'''
  def __init__(self,pos):
    self.pos = pos
    self.neighbors = list()
    self.clustered = False

  def __getitem__(self,i):
    return self.pos[i]
  def __len__(self):
    return 2  # If the length is not 2 something is seriously wrong.

class Graph:
  def __init__(self,nodes, P=0.25, random = True):
    self.nodes = set(nodes)
    self.not_clustered = set(nodes)
    self.clusters = None
    self.neighbors = dict.fromkeys(nodes)
    if random:
      self.GenerateEdges(P)
      self.GenerateClusters()

  def GenerateEdges(self,P):
    for node in self.nodes:
      x, y = node
      neighbors = list()
      if random_() <= P:
        neighbors.append(((x+1)%N,y%N))
      if random_() <= P:
        neighbors.append((x%N,(y+1)%N))
      temp = list()
      for neighbor in neighbors:
        if neighbor in self.nodes:
          temp.append(neighbor)
      neighbors = temp
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

  def GenerateClusters(self):
    self.clusters = list()
    for node in self.nodes:
      if node in self.not_clustered:
        self.clusters.append(self.FindClusterFromNode(node))
    return self.clusters

  @staticmethod
  def GenerateDual(graph):
    nodes = graph.nodes.copy()
    dual = Graph(nodes,random=False)
    for (x,y) in nodes:
      neighbors = list()
      if ((x+1)%N,y%N) not in graph.neighbors[(x,y)]:
        neighbors.append((x%N,(y-1)%N))
      if (x%N,(y+1)%N) not in graph.neighbors[(x,y)]:
        neighbors.append(((x-1)%N,(y)%N))

      for neighbor in neighbors.copy():
        if neighbor not in dual.nodes:
          neighbors.remove(neighbor)
      for neighbor in neighbors:
        assert neighbor in dual.nodes

      if dual.neighbors[(x,y)]:
        dual.neighbors[(x,y)].extend(neighbors)
      else:
        dual.neighbors[(x,y)] = neighbors
      for neighbor in neighbors:
        if dual.neighbors[neighbor]:
          dual.neighbors[neighbor].append((x,y))
        else:
          dual.neighbors[neighbor] = [(x,y)]
    return dual

class TriGraph:
  def __init__(self,nodes):
    self.nodes = set(nodes)
    self.not_clustered = set(nodes)
    self.clusters = None
    self.neighbors = dict.fromkeys(nodes)
    self.GenerateEdges()
    self.GenerateClusters()

  def GetNodes(self):
    return list(self.nodes)

  def GenerateEdges(self):
    for (x,y) in self.nodes:
      if self.neighbors[(x,y)] is None:
        self.neighbors[(x,y)] = list()
      if ((x+1)%N,y) in self.nodes:
        self.neighbors[(x,y)].append(((x+1)%N,y))
      if (x,(y+1)%N) in self.nodes:
        self.neighbors[(x,y)].append((x,(y+1)%N))
      if ((x-1)%N,(y+1)%N) in self.nodes:
        self.neighbors[(x,y)].append(((x-1)%N,(y+1)%N))

      for neighbor in self.neighbors[(x,y)]:
        if self.neighbors[neighbor] is None:
          self.neighbors[neighbor] = [(x,y)]
        else:
          self.neighbors[neighbor].append((x,y))

  def GetEdges(self,node):
    return self.neighbors[node]

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

  def GenerateClusters(self):
    self.clusters = list()
    for node in self.nodes:
      if node in self.not_clustered:
        self.clusters.append(self.FindClusterFromNode(node))
    return self.clusters

#  start = time()
#  nodes = list()
#  N = 1000
#  PN = 0.75
#  for i in range(N):
#    for j in range(N):
#        if random_() <= PN:
#          nodes.append((i,j))
#  
#  P=0.7
#  graph1 = Graph(nodes,P)
#  #P=0.75
#  #graph2 = Graph(nodes,P)
#  end = time()
#  print(f"Time taken: {end-start}")
#  DrawSquareNetworkBonds(graph1,nodelists=graph1.clusters,linewidth=2,imsize=500)
#  #DrawSquareNetworkBonds(graph2,nodelists=graph2.clusters,linewidth=2,imsize=500)
#  
#  #dual = Graph.GenerateDual(graph1)
#  #dual_clusters = dual.GetClusters()
#  #DrawSquareNetworkBonds(dual,nodelists=dual_clusters, imsize=600)

start = time()
nodes1 = list()
N = 400
PN = 0.51
for i in range(N):
  for j in range(N):
    if random_() <= PN:
      nodes1.append((i,j))
PN = 0.49
nodes2 = list()
for i in range(N):
  for j in range(N):
    if random_() <= PN:
      nodes2.append((i,j))
nodes = [nodes1,nodes2]
#tri_graph1 = TriGraph(nodes)
#tri_graph = TriGraph(nodes)
pool = multiprocessing.Pool(2)
graphs = pool.map_async(TriGraph,nodes)
pool.close()
pool.join()

end = time()
print(f"Time taken: {end-start}")
graphs = graphs.get()
start = time()

pool = multiprocessing.Pool()
for graph in graphs:
    pool.apply_async(DrawTriangularNetworkSites, args = [graph,graph.clusters], kwds = {'imsize':500, 'magnification':1})

pool.close()
pool.join()
end = time()
print(f"Time taken: {end-start}")
