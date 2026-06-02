from re import X
import networkx as nx
import numpy as np
## Utility functions for graph processing and evaluation
## Some do not see use anymore from previous iterations
def find_complement(x_vec, universe_vec):
    x_set = set(x_vec)
    universe_set = set(universe_vec)

    # Find the complement by taking the difference of the sets
    complement_elements = universe_set.difference(x_set)
    
    return complement_elements

def find_common(x_vec, y_vec):
    x_set = set(x_vec)
    y_set = set(y_vec)

    # Find the common elements by taking the intersection of the sets
    common_elements = x_set.intersection(y_set)
    
    return common_elements

def find_num_common(x_vec, y_vec):
    # Find the common elements by taking the intersection of the sets
    common_elements = find_common(x_vec, y_vec)

    # Get the number of common elements
    num_common = len(common_elements)
    
    return num_common

def find_protected_portion(x_vec, protected_vec):
    num_common = find_num_common(x_vec, protected_vec)
    return num_common / len(x_vec)

def compute_r(S, protected_nodes_all,mu=1):
    r = (1/mu)*np.log(sum([np.exp(-mu*find_num_common(S.nodes(),protected_nodes_all[i])) for i in range(0,len(protected_nodes_all))]))

    return r  

def compute_density(S, G=None, weight=None):
    if type(S) is not nx.classes.graph.Graph:
        S = G.subgraph(S)

    return S.size(weight) / S.number_of_nodes()
    
