import numpy as np

# Exercise 3 :
# Adjacency matrix of a graph

DEBUG: bool = False

# whole graph is a dict , the keys are the nodes and the values are a list
# of the nodes to which it is connected


# directed graph
graph: dict = {
    "N1": ["N2", "N4"],
    "N2": ["N3", "N4", "N4"],
    "N3": ["N4"],
    "N4": ["N5"],
    "N5": [],
}

print(graph.keys())
# example
# nodes = [ "N1", "B", "C", "D"]
# edges = [(A,B), (C,D), (B,C), (A,C)]


adj_mat: np.ndarray = np.zeros((len(list(graph.keys()))))
# [i][j]
for i in list(graph.keys()):
    for j in list(graph[i].values()):
        pass
