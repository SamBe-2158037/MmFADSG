import os
import copy
import numpy as np
import networkx as nx
from pathlib import Path
import utils
import csv

def init_graph(dataset_name):       
    if dataset_name == "polbooks":
        graph_path = "datasets/pol_books/polbooks.gml"
        G = nx.read_gml(graph_path)
    elif "cosponsor" in dataset_name:
        if "Bel" in dataset_name:
            graph_path = "datasets/cosponsor/Bel/net_be_ch2014.gml"
        elif "It" in dataset_name:
            graph_path = "datasets/cosponsor/It/net_it_ca2001.gml"
        with open(graph_path, encoding='utf-8') as f:
            G_ = nx.parse_gml(f.read(), label='id')
        G_ = G_.to_undirected()

        parties = sorted({d['party'] for _, d in G_.nodes(data=True) if d.get('party')})
        party_to_int = {p: i for i, p in enumerate(parties)}
        for node, data in G_.nodes(data=True):
            nx.set_node_attributes(G_, {node: party_to_int[data['party']]}, name='value')

        G = copy.deepcopy(G_)
        mapping = dict(zip(G, range(len(G.nodes()))))
        G = nx.relabel_nodes(G, mapping)
    # * ***Color Graph's Protected Nodes***
    if dataset_name == "polbooks":
        values_list = list(G.nodes(data='value'))
        protected_nodes = [[item[0] for item in values_list if item[1] == 'l'],
                           [item[0] for item in values_list if item[1] == 'n'],
                           [item[0] for item in values_list if item[1] == 'c']]
                            # protected = 'liberal'
    elif "cosponsor" in dataset_name:
        num_parties = len({d for _, d in G.nodes(data='value') if d is not None})
        colors_list = list(G.nodes(data='value'))
        protected_nodes = [
            [item[0] for item in colors_list if item[1] == i]
            for i in range(num_parties)
        ]   
    return G, protected_nodes
