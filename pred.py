# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 14:13:36 2020

@author: Matt-HJ-Bailey
"""

import networkx as nx
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle as pkl

import scipy

from typing import Dict, Optional, NewType, Any, Iterable, Tuple
import copy

Node = Any
Position = Iterable[float]


def node_list_to_edges(node_list: Iterable[Node], is_ring: bool = True):
    """
    Takes a list of connected nodes, such that node[i] is connected
    to node[i - 1] and node[i + 1] and turn it into a set of edges.

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


def _project_onto_line(
    node_pos: Position, line_u: Position, line_v: Position
) -> Tuple[Position, bool]:
    """
    Project a point onto a line.

    Parameters
    ----------
    node_pos
        The vector from the origin to a given point
    line_u
        The vector from the origin to the first point of a line
    line_v
        The vector from the origin to the second point of a line.
    Returns
    -------
    projected_pos
        The projected position of the node onto the line
    within_line
        Whether the projected position is within the line segment
    """

    line_vec = line_v - line_u
    line_length_sq = np.sum(line_vec ** 2)

    line_frac = np.dot(node_pos - line_u, line_vec) / line_length_sq

    projected_pos = line_u + line_frac * line_vec

    # Note that being exactly on a node at either edge of the line counts
    # as being on the line.
    within_line = bool(0 <= line_frac <= 1.0)
    return projected_pos, within_line


def calculate_segment(angle: float, num_segments: int) -> int:
    """
    Calculate which segment around a circle a given angle corresponds to.

    Parameters
    ----------
    angle
        The angle between a vector and the +ve x axis to check.
    num_segments
        The number of segments to split this circle into

    Returns
    -------
    segment_idx
        The index of this segment, counting anticlockwise from +x
    """
    # First, map the angle to the range [0, 2pi)
    if angle < 0:
        angle = (2 * np.pi) + angle
    if angle > (2 * np.pi):
        angle = angle % (2 * np.pi)

    segment_size = 2 * np.pi / num_segments
    segment_idx = int(np.floor(angle / segment_size))
    return segment_idx


def calculate_neighbours(positions: np.array, cutoff: Optional[float] = None):
    """
    Calculate all the neighbours of points within a cutoff.

    Parameters
    ----------
    positions
        An NxD array of positions
    cutoff
        The maximum euclidean distance to consider pairs to be neighbours
        Defaults to np.inf
    Returns
    -------
    distances
        The pairwise distance matrix between all pairs of positions
    neighbours
        The list of neighbour pairs
    """
    if cutoff is None:
        cutoff = np.inf
    distances = scipy.spatial.distance.pdist(positions)
    distances = scipy.spatial.distance.squareform(distances)

    within_cutoff = np.argwhere(distances < cutoff)
    pairwise_cutoffs = within_cutoff[:, 0] < within_cutoff[:, 1]
    return distances, within_cutoff[pairwise_cutoffs]


def calculate_faces(G: nx.Graph):
    is_planar, embedding = nx.check_planarity(G, counterexample=True)
    if not is_planar:
        raise RuntimeError("Expected a planar graph.")

    seen_edges = set()
    faces = set()
    for v, w in embedding.edges():
        if (v, w) in seen_edges:
            continue
        face = embedding.traverse_face(v, w, mark_half_edges=seen_edges)
        faces.add(tuple(face))
    return faces


class PREDLayout:
    """
    Implementation of PRED, a graph drawing algorithm that preserves edge crossings.

    A Force-Directed Algorithm that Preserves Edge Crossing Properties
    Francois Bertault
    """

    DEFAULT_ATTRACTION_EXP = 1.0
    DEFAULT_REPULSION_EXP = 2.0
    DEFAULT_EDGE_EXP = 2.0

    def __init__(
        self,
        G: nx.Graph,
        start_pos: Optional[Dict[Node, Position]] = None,
        desired_length: float = 1.0,
        max_distance_to_virtual: float = 1.0,
        num_zones: int = 8,
        gravity_force_scale: float = 0.0,
    ) -> None:
        """
        Initialise the layout algorithm.

        Parameters
        ----------
        G
            The graph to optimise
        start_pos
            A dictionary mapping nodes to positions. If this is None,
            defaults to a planar layout of G as decided by networkx.
        desired_length
            The length of edges we want in the output, defaults to 1.0
        max_distance_to_virtual
            The maximum distance of a node to its virtual image before we cut it off.
        num_zones
            The number of segments around a node to limit forces in
        gravity_force_scale
            How strongly nodes are attracted to the barycentre of the graph. Defaults to 0.0
        Returns
        -------
        PREDLayout
            An instance of a layout optimiser.
        """

        # Networkx graphs are mutable, so let's not cause trouble.
        self.G = copy.deepcopy(G)
        self.G = nx.convert_node_labels_to_integers(
            G, ordering="sorted", label_attribute="label"
        )
        is_planar = nx.check_planarity(G)
        if not is_planar:
            raise RuntimeError("This graph is not planar. Cannot plot.")
        if start_pos is None:
            start_pos = nx.planar_layout(G)

        self.delta = desired_length
        self.gamma = max_distance_to_virtual
        self.num_zones = num_zones

        self.gravity_force_scale = gravity_force_scale
        # Set up arrays for positions, velocities and accelerations.
        self.positions = np.empty([len(G), 2], dtype=float)
        self.velocities = np.zeros_like(self.positions)
        self.accelerations = np.zeros_like(self.positions)

        self.max_sep = 3.0 * self.delta

        for idx, node in enumerate(G.nodes(data=True)):
            # If we relabelled the nodes to integers, we need
            # the old label for the start position.
            node, data = node
            if "label" in data:
                node = data["label"]
            self.positions[idx, :] = start_pos[node]

        # Precomputation region here.
        self._bounding_edges = self._calculate_bounding_edges()

        # These should get changed every step as part of the update
        # loop, and are cached here.
        self._distances, self._node_pairs = calculate_neighbours(self.positions)
        self._virtual_nodes = self._precompute_virtual_nodes()

    def _precompute_virtual_nodes(
        self,
    ) -> Dict[Tuple[Node, Node, Node], Tuple[Position, bool, float]]:
        """
        Precompute a set of virtual nodes, which are projections of real nodes onto lines.

        Returns
        -------
        virtual_nodes
            The data structure returned takes a key of (node, edge_u, edge_v)
            and returns a tuple of (virtual_position, within_line, distance_to_virtual)
        """

        virtual_nodes = {}
        for node in self.G.nodes():
            node_pos = self.positions[node]
            for u, v in self._bounding_edges[node]:
                assert node not in (u, v), "Nodes cannot interact with their own edges"
                line_u = self.positions[u]
                line_v = self.positions[v]

                proj_pos, within_line = _project_onto_line(node_pos, line_u, line_v)
                distance_to_proj = np.linalg.norm(node_pos - proj_pos)
                if np.isclose(distance_to_proj, 0.0):
                    raise RuntimeError(f"{node} lies along the edge ({u}, {v})")
                virtual_nodes[node, u, v] = proj_pos, within_line, distance_to_proj
        return virtual_nodes

    def _calculate_bounding_edges(self):
        """
        Calculate a set of 'bounding edges' for each node.

        The set of bounding edges is the edges that can be reached without
        crossing any other edges.

        Returns
        -------
        bounding_edges
        """
        faces = calculate_faces(self.G)
        bounding_edges = {}
        for node in self.G.nodes():
            faces_with_node = [face for face in faces if node in face]
            faces_edges = set()
            for face in faces_with_node:
                edge_list = node_list_to_edges(face)
                faces_edges.update(edge_list)
            bounding_edges[node] = []
            for u, v in faces_edges:
                if node not in (u, v):
                    bounding_edges[node].append(tuple([u, v]))
        return bounding_edges

    def _positions_to_label_dict(self) -> Dict[Node, Position]:
        """
        Convert a numpy position array to a networkx label dictionary.

        Returns
        -------
        pos
            A dictionary with keys being node labels and values being positions
        """
        # Normalise to be centred at 0, 0
        self.positions -= np.mean(self.positions, axis=0)
        pos = {}
        for node, data in self.G.nodes(data=True):
            if "label" in data:
                label = data["label"]
            else:
                label = node
            pos[label] = self.positions[node]
        return pos

    def _calculate_node_attractions(self, exponent: float = None) -> np.array:
        """
        Calculate the forces due to node attractions.

        Nodes are only attracted to their neighbouring nodes.
        This is given by the formula
        F(u, v) = d(u, v)/ delta * (x(v) - x(u))

        Parameters
        ----------
        exponent
            The power to raise the attractive force to. Defaults to 1.0
        Returns
        -------
        forces
            A numpy array of the forces due to node attractions.
        """
        if exponent is None:
            exponent = self.DEFAULT_ATTRACTION_EXP

        forces = np.zeros_like(self.positions)
        for u, v in self.G.edges():
            separation = self.positions[v] - self.positions[u]
            distance = self._distances[u, v]
            force = (distance / self.delta) ** exponent * separation
            forces[u] += force
            forces[v] -= force
        return forces

    def _calculate_node_repulsions(self, exponent: float = None) -> np.array:
        """
        Calculate the forces due to node repulsions.

        Nodes are repelled by all other nodes.
        This is given by the formula
        F(u, v) = delta^2/d(u, v)^2 * (x(v) - x(u))

        Parameters
        ----------
        exponent
            The power to raise the repulsive force to. Defaults to 2.0
        Returns
        -------
        forces
            A numpy array of the forces due to node attractions.
        """
        if exponent is None:
            exponent = self.DEFAULT_REPULSION_EXP
        forces = np.zeros_like(self.positions)
        for u, v in self._node_pairs:
            separation = self.positions[v] - self.positions[u]
            distance = self._distances[u, v]
            force = -((self.delta / distance) ** exponent) * separation
            forces[u] += force
            forces[v] -= force
        return forces

    def _calculate_edge_repulsions(
        self, gamma: Optional[float] = None, exponent: float = None
    ) -> np.array:
        """
        Calculate the forces between nodes and edges.

        To do this, project a node v onto an edge (a, b) to get
        a virtual node.
        Nodes are not repelled by their own edges.
        If that virtual node is not on the edge, the force is 0.
        If that virtual node is on the edge, and the distance from
        the virtual node to the real node is smaller than gamma,
        calculate the force.

        Parameters
        ----------
        gamma
            The cutoff for the distance between the virtual node and the real node.
            Defaults to self.delta, the desired edge distance.
        exponent
            The power to raise the distance term to. Defaults to 2.0.
        Returns
        -------
        forces
            A set of repulsive forces on the nodes.
        """
        # TODO: Precompute a set of bounding edges, as we know we don't
        # change edge intersections in this algorithm.
        if exponent is None:
            exponent = self.DEFAULT_EDGE_EXP
        if gamma is None:
            gamma = self.gamma

        forces = np.zeros_like(self.positions)
        for (v, a, b) in self._virtual_nodes.keys():
            proj_pos, within_line, distance_to_proj = self._virtual_nodes[v, a, b]

            # Skip cases where the virtual position isn't within the edge
            if not within_line:
                continue

            # Skip cases where the virtual position is too far from the real node
            if distance_to_proj > gamma:
                continue

            separation = -proj_pos - self.positions[v]

            forces[v] += (
                -(((gamma - distance_to_proj) ** exponent) / distance_to_proj)
                * separation
            )
        return forces

    def _calculate_max_forces(self) -> np.array:
        """
        Calculate the maximum forcess in each allowed segment.

        See section 2.2 in paper.
        Returns
        -------
        max_forces
            A len(G) x num_zones numpy array, with each entry being the maximum force in a given direction
            for a given node.
        """
        # Initialise the maximum forces on each node across zones.
        def s_to_r(j: int) -> int:
            if j < 0:
                j = self.num_zones - j
            return j % self.num_zones

        max_forces = np.empty([len(self.G), self.num_zones], dtype=float)
        max_forces[:, :] = np.inf

        for (v, a, b) in self._virtual_nodes.keys():
            proj_pos, within_line, distance_to_proj = self._virtual_nodes[v, a, b]
            if within_line:
                node_to_pos_vec = proj_pos - self.positions[v]
                angle = np.arctan2(node_to_pos_vec[1], node_to_pos_vec[0])
                segment = calculate_segment(angle, self.num_zones)

                v_indices = np.array(
                    [s_to_r(s) for s in range(segment - 2, segment + 2 + 1)]
                )
                max_forces[v, v_indices] = np.minimum(
                    max_forces[v, v_indices], distance_to_proj / 3.0
                )

                a_b_indices = np.array(
                    [s_to_r(s) for s in range(segment + 2, segment + 6 + 1)]
                )
                max_forces[a, a_b_indices] = np.minimum(
                    max_forces[a, a_b_indices], distance_to_proj / 3.0
                )
                max_forces[b, a_b_indices] = np.minimum(
                    max_forces[b, a_b_indices], distance_to_proj / 3.0
                )
            else:
                dist_to_a_3 = self._distances[v, a] / 3.0
                dist_to_b_3 = self._distances[v, b] / 3.0
                max_forces[v, :] = np.minimum(
                    max_forces[v, :], min(dist_to_a_3, dist_to_b_3)
                )
                max_forces[a, :] = np.minimum(max_forces[a, :], dist_to_a_3)
                max_forces[b, :] = np.minimum(max_forces[b, :], dist_to_b_3)

        return max_forces

    def _calculate_gravity(self, scaling_factor: float = 1.0):
        """
        Attract all nodes to the centre of mass of the graph.

        See
        Force-Directed Graph Drawing Using Social Gravity and Scaling
        Michael J. Bannister; David Eppstein; Michael T. Goodrich; Lowell Trott
        arXiv:1209.0748v1

        Parameters
        ----------
        scaling_factor
            How strong the gravitational force is.
            "a maximum value of 2.5 seems to give pleasing results"
        Returns
        -------
        forces
            the forces due to gravity
        """

        # If no gravity, don't both calculating the barycentre.
        if scaling_factor == 0:
            return np.zeros_like(self.positions)
        barycentre = np.mean(self.positions, axis=0)
        forces = -scaling_factor * (barycentre - self.positions)
        return forces

    def _calculate_forces(
        self,
        attraction_exp: Optional[float] = None,
        repulsion_exp: Optional[float] = None,
        edge_exp: Optional[float] = None,
    ) -> np.array:
        """
        Calculate all the forces on all the nodes.

        The force on a node is given by
        F_repulsion + F_attraction + F_edges

        Parameters
        ----------
        exponent
            The power to raise the attraction term to. Defaults to 1.0.
        exponent
            The power to raise the repulsion distance term to. Defaults to 2.0.
        exponent
            The power to raise the edge distance term to. Defaults to 2.0.
        Returns
        -------
        forces
            A numpy array of the combined forces on each node.
        """
        forces = np.zeros_like(self.positions)
        forces += self._calculate_node_attractions(attraction_exp)
        forces += self._calculate_node_repulsions(repulsion_exp)
        forces += self._calculate_edge_repulsions(edge_exp)

        forces += self._calculate_gravity(self.gravity_force_scale)

        # Apply the segment wise maximum forces.
        max_forces = self._calculate_max_forces()
        for node in self.G.nodes():
            force_vec = forces[node]
            # Note that atan2 takes arguments (y, x)
            angle_with_x = np.arctan2(force_vec[1], force_vec[0])
            segment_idx = calculate_segment(angle_with_x, self.num_zones)
            force_magnitude = np.linalg.norm(force_vec)
            if force_magnitude > max_forces[node, segment_idx]:
                forces[node] *= max_forces[node, segment_idx] / force_magnitude
        return forces

    def update(
        self,
        step_size: float = 0.05,
        attraction_exp: Optional[float] = None,
        repulsion_exp: Optional[float] = None,
        edge_exp: Optional[float] = None,
        max_step: Optional[float] = None,
    ) -> Dict[Node, Position]:
        """
        Run one iteration of the forces and update the positions.

        Parameter
        ---------
        step_size
            The multiplier of the forces to make a step over
        max_distance
            The maximum distance in any direction that node can move.
            Useful for annealing.
        Returns
        ------
        positions
            The positions after one update.
        """
        if max_step is None:
            max_step = np.inf

        self._distances, self._node_pairs = calculate_neighbours(self.positions)
        self._virtual_nodes = self._precompute_virtual_nodes()

        forces = self._calculate_forces(
            attraction_exp=attraction_exp,
            repulsion_exp=repulsion_exp,
            edge_exp=edge_exp,
        )
        delta_pos = np.clip(forces * step_size, -max_step, max_step)
        self.positions += delta_pos
        return self._positions_to_label_dict()

    def anneal(
        self,
        num_steps: int = 100,
        step_size: float = 1.0,
        attraction_exp_minmax: Optional[Tuple[float, float]] = None,
        repulsion_exp_minmax: Optional[Tuple[float, float]] = None,
        edge_exp_minmax: Optional[Tuple[float, float]] = None,
        max_step_minmax: Optional[Tuple[float, float]] = None,
    ):
        """
        Optimise the layout over a number of steps.

        See ImPrEd paper, Section 4.4 "Force System Cooling"

        Parameters
        ----------
        num_steps
            The number of steps to optimize over.
        step_size
            A factor to multiply each step by, defaults to 1.0.
        attraction_exp_minmax
            Change the attraction exponent linearly over num_steps, from
            max(attraction_exp_minmax) to min(attraction_exp_minmax).
            Paper uses (0.4, 1.0)
        repulsion_exp_minmax
            Change the attraction exponent linearly over num_steps, from
            max(repulsion_exp_minmax) to min(repulsion_exp_minmax).
            Paper uses (2.0, 4.0)
        max_step_minmax
            Change the maximum amount a node can move by linearly
            from max(max_step_minmax) to min(max_step_minmax)
            Paper uses (3.0*delta, 0.0)
        """
        last_stress = np.inf
        if attraction_exp_minmax is None:
            attraction_exps = np.array([None] * num_steps)
        else:
            attraction_exps = np.linspace(
                max(attraction_exp_minmax), min(attraction_exp_minmax), num_steps
            )

        if repulsion_exp_minmax is None:
            repulsion_exps = np.array([None] * num_steps)
        else:
            repulsion_exps = np.linspace(
                max(repulsion_exp_minmax), min(repulsion_exp_minmax), num_steps
            )
        if edge_exp_minmax is None:
            edge_exps = np.array([None] * num_steps)
        else:
            edge_exps = np.linspace(
                max(edge_exp_minmax), min(edge_exp_minmax), num_steps
            )
        if max_step_minmax is None:
            max_steps = np.array([np.inf] * num_steps)
        else:
            max_steps = np.linspace(
                max(max_step_minmax), min(max_step_minmax), num_steps
            )

        for step in range(num_steps):
            print(
                attraction_exps[step],
                repulsion_exps[step],
                edge_exps[step],
                max_steps[step],
            )
            self.update(
                step_size=step_size,
                attraction_exp=attraction_exps[step],
                repulsion_exp=repulsion_exps[step],
                edge_exp=edge_exps[step],
                max_step=max_steps[step],
            )
        return self._positions_to_label_dict()


def update(ax, graph, layout, num):
    ax.clear()
    pos = layout.update(step_size=1)
    nx.draw_networkx(test_graph, pos, ax=ax)
    ax.scatter(*np.mean(layout.positions, axis=0))
    ax.set_title(f"Iteration {num}")
    ax.set_xticks([])
    ax.set_yticks([])


if __name__ == "__main__":
    test_graph = nx.ladder_graph(10)
    test_graph = pkl.load(open("./edge_graph_1.pkl", "rb"))
    layout = PREDLayout(
        test_graph, max_distance_to_virtual=1.0, gravity_force_scale=0.0
    )
    # pos = layout.anneal(repulsion_exp_minmax=(2, 2),
    #                     attraction_exp_minmax=(1.0, 1),
    #                     edge_exp_minmax=(2, 2),
    #                     max_step_minmax=(3, 0.0))

    for i in range(1):
        layout.update()
    fig, ax = plt.subplots()
    nx.draw(test_graph, pos=nx.planar_layout(test_graph), ax=ax)
    # animation = mpl.animation.FuncAnimation(fig, lambda num: update(ax, test_graph, layout, num), frames=100, interval=16)
    plt.show()
