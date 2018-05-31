#!/usr/bin/env python
"""

Utility functions to generate benchmark graphs

"""
#    Copyright (C) 2018 by
#    Carlo Nicolini <carlo.nicolini@iit.it>
#    All rights reserved.
#    BSD license.

import networkx as nx

def ring_of_cliques(n, r):
    """
    Returns a networkx graph of ring of cliques with n cliques, each with of r nodes
    
    args:
        n number of nodes in every clique
        r number of connected cliques
    """
    graph = nx.disjoint_union_all([nx.complete_graph(r) for i in range(0, n)])
    graph.add_edges_from([(u, u + r + 1) for u in range(0, r * n, r)])
    graph.remove_node(n * r + 1)
    graph.add_edge(r - 1, n * r - 1)
    return graph

def barthelemy_graph(n_er, prob, n_cliques, k_cliques):
    """
    Returns a networkx graph as described in the Barthelemy & Fortunato paper on PNAS
    Resolution limit in complex networks.

    args:
        n_er (int): number of nodes in the random graph
        prob (float): probability of linking in the random graph
        n_cliques (int): number of cliques
        k_cliques (int): number of nodes per clique
    """
    graph = nx.connected_component_subgraphs(
        nx.erdos_renyi_graph(n_er, prob))[0]
    graph_k = nx.disjoint_union_all(
        [nx.complete_graph(k_cliques) for i in range(0, n_cliques)])
    graph = nx.disjoint_union(graph, graph_k).copy()
    for i in range(0, n_cliques):
        graph.add_edge(np.random.randint(n_er), n_er + i * n_cliques)
    return graph

