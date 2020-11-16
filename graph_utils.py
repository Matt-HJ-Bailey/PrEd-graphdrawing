# -*- coding: utf-8 -*-
"""
Contains utility functions to operate on graphs.

This includes calculating faces, maximal planar embeddings,
and converting lists of nodes into edges.

Functions
---------
node_list_to_edges
    Convert a node list into a set of edges

calculate_faces
    Extract a set of faces of a planar graph

count_intersections_involving
    Count how often edges attached to a given node cross other edges

count_intersections
    Count how often two edges intersect

Variables
---------
Node
    A type for nodes, must be hashable to use as a dict key
Position
    A type for node positions

Created on Mon Nov 16 10:04:34 2020

@author: Matt
"""

from typing import Dict, Hashable, Iterable, List, Tuple, Collection, Union, Set

import networkx as nx
from numpy.random import MT19937, Generator

from line_utils import do_lines_intersect

Node = Union[Hashable]
Position = Union[Collection[float]]


def node_list_to_edges(node_list: Collection[Node], is_ring: bool = True):
    """
    Convert a node list into a set of edges.

    Takes a list of connected nodes, such that node[i] is connected to node[i - 1]
    and node[i + 1] and turn it into a set of edges.

    Parameters
    ----------
    node_list
        a list of nodes which must be a hashable type.
    is_ring
        is this a linked ring, i.e. is node[-1] connected to node[0]
    Returns
    -------
    edges
        a set of frozensets, each frozenset being one edge.
    """
    list_size = len(node_list)
    edges = set()

    # If this is a ring, iterate one over the size
    # of the list. If not, make sure to stop
    # before the end.
    if is_ring:
        offset = 0
    else:
        offset = -1
    for i in range(list_size + offset):
        next_index = (i + 1) % list_size
        edges.add(frozenset([node_list[i], node_list[next_index]]))
    return frozenset(edges)


def calculate_faces(G: nx.Graph) -> Set[Tuple[Node, ...]]:
    """
    Calculate the 'faces' of the graph.

    The faces can be perceived as rings in the drawing.

    Parameters
    ----------
    G
        The graph to identify faces in

    Returns
    -------
    faces
        A list of tuples, in each tuple is a list of nodes in one face.
    """
    is_planar, embedding = nx.check_planarity(G, counterexample=True)
    if not is_planar:
        raise RuntimeError("Expected a planar graph.")

    seen_edges: Iterable[Tuple[Node, Node]] = set()
    faces: Set[Tuple[Node, ...]] = set()
    for v, w in embedding.edges():
        if (v, w) in seen_edges:
            continue
        face = embedding.traverse_face(v, w, mark_half_edges=seen_edges)
        faces.add(tuple(face))
    return faces


def count_intersections_involving(
    G: nx.Graph, pos: Dict[Node, Position], node: Node
) -> int:
    """
    Count the number of edges that intersect with one another.

    Edges count as intersecting if they meet at one point, or are
    collinear and overlap.
    We do not count edges as intersecting if they meet at a node.
    Requires a specific embedding to be provided.

    Parameters
    ----------
    G
        The graph to check for edge intersections on.
    pos
        A dictionary mapping node indexes to positions in the 2D plane.
    node
        Check for any intersections involving this node.

    Returns
    -------
    num_intersections
        The number of edge intersections in this embedding.
    """
    num_intersections = 0
    edges_around_node = [(node, partner) for partner in G.neighbors(node)]
    for a, b in edges_around_node:
        p1, q1 = pos[a], pos[b]
        for u, v in G.edges():
            # Skip cases where these meet at a node.
            if a in (u, v) or b in (u, v):
                continue
            p2, q2 = pos[u], pos[v]
            if do_lines_intersect(p1, q1, p2, q2):
                num_intersections += 1
    return num_intersections


def count_intersections(G: nx.Graph, pos: Dict[Node, Position]) -> int:
    """
    Count the number of edges that intersect with one another.

    Edges count as intersecting if they meet at one point, or are
    collinear and overlap.
    We do not count edges as intersecting if they meet at a node.
    Requires a specific embedding to be provided.

    Parameters
    ----------
    G
        The graph to check for edge intersections on.
    pos
        A dictionary mapping node indexes to positions in the 2D plane.

    Returns
    -------
    num_intersections
        The number of edge intersections in this embedding.
    """
    edgelist = list(G.edges())
    num_intersections = 0
    for i in range(len(edgelist)):
        p1, q1 = pos[edgelist[i][0]], pos[edgelist[i][1]]
        for j in range(i):
            p2, q2 = pos[edgelist[j][0]], pos[edgelist[j][1]]
            # Skip cases where these meet at a node.
            if edgelist[i][0] in edgelist[j] or edgelist[i][1] in edgelist[j]:
                continue
            if do_lines_intersect(p1, q1, p2, q2):
                num_intersections += 1
    return num_intersections


def _maximal_planar_subgraph_removal(G: nx.Graph) -> nx.Graph:
    """
    Find a maximal planar subgraph by removing edges.

    Parameters
    ----------
    G
        A non-planar graph to remove edges from
    Returns
    -------
    planar_G
        An approximation to the maximal planar subgraph of G
    """
    new_graph = copy.deepcopy(G)
    rng = Generator(MT19937())

    is_planar, counterexample = nx.check_planarity(new_graph, counterexample=True)
    while not is_planar:
        edge_to_remove = rng.choice(counterexample.edges())
        new_graph.remove_edge(edge_to_remove)
        is_planar, counterexample = nx.check_planarity(new_graph)
    return new_graph

def _maximal_planar_subgraph_addition(G: nx.Graph, method:str="bfs") -> nx.Graph:
    """
    Find a maximal planar graph by adding edges one at a time.
    """
    new_graph = nx.create_empty_copy(G)

    VALID_METHODS = {"bfs", "dfs", "sorted", "default"}
    if method not in VALID_METHODS:
        raise NotImplementedError(f"Method {method} is not a valid choice. Pick one of ",
                             VALID_METHODS)
    if method == "bfs":
        edges_to_add = nx.edges_bfs(G)
    elif method == "dfs":
        edges_to_add = nx.edges_dfs(G)
    elif method == "sorted":
        edges_to_add = sorted(list(G.edges()))
    elif method == "default":
        edges_to_add = list(G.edges())

    while edges_to_add:
        edge = edges_to_add.pop()
        new_graph.add_edge(edge)
        is_planar = nx.check_planarity(new_graph, counterexample=False)
        if not is_planar:
            # Bad luck on this edge, we can't add it.
            new_graph.remove_edge(edge)
            continue

    return new_graph

def find_maximal_planar_subgraph(G, method="removal"):

    _METHOD_DICT = {"removal": lambda G: _maximal_planar_subgraph_removal(G),
                    "bfs": lambda G: _maximal_planar_subgraph_addition(G, method="bfs"),
                    "dfs": lambda G: _maximal_planar_subgraph_addition(G, method="dfs")
                    "sorted": lambda G: _maximal_planar_subgraph_addition(G, method="sorted")
                    "default": lambda G: _maximal_planar_subgraph_addition(G, method="default")}

    if method not in _METHOD_DICT.keys():
        raise NotImplementedError("Method {method} not supported. Pick one of ", _METHOD_DICT.keys())