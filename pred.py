# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 14:13:36 2020

@author: Matt-HJ-Bailey
"""

import copy
import pickle as pkl
import warnings
from typing import Dict, List, Optional, Tuple, Union, Set
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from numpy.random import MT19937, Generator

from joblib import Parallel, delayed, parallel_backend

from graph_utils import (Node, Position, calculate_faces,
                         count_intersections_involving, node_list_to_edges)
from line_utils import (_project_onto_line, calculate_neighbours,
                        calculate_segment)

Edge = Union[Tuple[Node, Node]]


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
        # Relabel the node keys to be integers so we can use them
        # with numpy without trouble.
        # This requires a bit of book-keeping later!
        self.G = nx.convert_node_labels_to_integers(
            G, ordering="sorted", label_attribute="label"
        )
        is_planar = nx.check_planarity(self.G)
        if not is_planar:
            raise RuntimeError("This graph is not planar. Cannot plot.")
        if start_pos is None:
            start_pos = nx.planar_layout(G)

        self.delta = desired_length
        self.gamma = max_distance_to_virtual
        self.num_zones = num_zones

        self.gravity_force_scale = gravity_force_scale
        # Set up arrays for positions, velocities and accelerations.
        self.positions = np.empty([len(self.G), 2], dtype=float)
        self.velocities = np.zeros_like(self.positions)
        self.accelerations = np.zeros_like(self.positions)

        self.max_sep = 3.0 * self.delta

        for node, data in self.G.nodes(data=True):
            # If we relabelled the nodes to integers, we need
            # the old label for the start position.
            if "label" in data:
                node_idx = data["label"]
            else:
                node_idx = node
            self.positions[node, :] = start_pos[node_idx]
        # Precomputation region here.
        self._bounding_edges = self._calculate_bounding_edges()

        # These should get changed every step as part of the update
        # loop, and are cached here.
        self._distances, self._node_pairs = calculate_neighbours(self.positions)
        self._virtual_nodes = self._precompute_virtual_nodes()

        # This is just ridiculously expensive, as it's O(V * E)
        # for each node.
        self._offset_collinear_points(0.01)

    def _offset_single_node(self, node: Node, offset_size: float):
        """
        Move one node slightly to a new point, without creating new intersections.

        Brute forces the problem and may not find a solution.

        Parameters
        ----------
        node
            The node to move about
        offset_size
            The size of step to move the node about.
        Returns
        -------
        None
        Raises
        -----
        RuntimError
            If we can't move the node to a new position.
        """
        node_intersections = count_intersections_involving(self.G, self.positions, node)
        old_pos = copy.deepcopy(self.positions[node, :])

        num_offsets = 32
        rg = Generator(MT19937())
        _offsets = rg.uniform(-offset_size, offset_size, 2 * num_offsets).reshape(
            [num_offsets, 2]
        )
        for offset in _offsets:
            new_position = old_pos + offset
            self.positions[node, :] = new_position
            new_node_intersections = count_intersections_involving(
                self.G, self.positions, node
            )
            if new_node_intersections == node_intersections:
                # Job done, we've offset this
                return
            self.positions[node, :] = old_pos
        raise RuntimeError(
            f"Could not offset {node} without causing new intersections."
        )

    def _offset_collinear_points(self, offset_size: float = np.finfo(np.float32).eps):
        """
        Offset any nodes that lie on another edge.

        If a node v lies on the edge (a, b) in the current embedding,
        we need to offset it a bit as the distance to a virtual node is
        then zero, which causes trouble.

        To do this, just shoogle it about by a tiny amount and check we
        haven't introduced any new intersections.

        Parameters
        ----------
        offset_size
            The size of offset to try. Defaults to the float32 machine epsilon.

        """
        # Reuse the virtual node calculation. We'll re-do it later!
        num_collinear: int = sum(
            [
                True
                for _, within_line, distance_to_proj in self._virtual_nodes.values()
                if np.isclose(distance_to_proj, 0.0) and within_line
            ]
        )
        while num_collinear:
            node_changed = None
            for (node, u, v), (
                _,
                within_line,
                distance_to_proj,
            ) in self._virtual_nodes.items():
                if np.isclose(distance_to_proj, 0.0) and within_line:
                    self._offset_single_node(node, offset_size)
                    node_changed = node
                    break

            # Sadly, we have to redo all this effort.
            if node_changed is not None:
                for node, u, v in self._virtual_nodes.keys():
                    if node_changed in (node, u, v):
                        node_pos = self.positions[node]
                        assert node not in (
                            u,
                            v,
                        ), "Nodes cannot interact with their own edges"
                        line_u = self.positions[u]
                        line_v = self.positions[v]

                        proj_pos, within_line, distance_to_proj = _project_onto_line(
                            node_pos, line_u, line_v
                        )
                        self._virtual_nodes[node, u, v] = (
                            proj_pos,
                            within_line,
                            distance_to_proj,
                        )

            num_collinear = sum(
                [
                    True
                    for _, within_line, distance_to_proj in self._virtual_nodes.values()
                    if np.isclose(distance_to_proj, 0.0) and within_line
                ]
            )

    def _precompute_virtual_nodes_involving(self, node: Node, workdict):
        node_pos = self.positions[node]
        for u, v in self._bounding_edges[node]:
            assert node not in (u, v), "Nodes cannot interact with their own edges"
            line_u = self.positions[u]
            line_v = self.positions[v]

            proj_pos, within_line, distance_to_proj = _project_onto_line(
                    node_pos, line_u, line_v
            )
            workdict[node, u, v] = proj_pos, within_line, distance_to_proj

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
        virtual_nodes: Dict[Tuple[Node, Node, Node], Tuple[Position, bool, float]] = {}
        with parallel_backend("threading", n_jobs=-1):
            Parallel()(delayed(self._precompute_virtual_nodes_involving)
                       (node, virtual_nodes) for node in self.G.nodes())
        return virtual_nodes

    def _calculate_bounding_edges(self) -> Dict[Node, List[Edge]]:
        """
        Calculate a set of 'bounding edges' for each node.

        The set of bounding edges is the edges that can be reached without
        crossing any other edges.

        Returns
        -------
        bounding_edges
            A dictionary, with keys as node labels and values being a list
            of all edges reachable by this node.
        """
        faces = calculate_faces(self.G)
        bounding_edges: Dict[Node, List[Edge]] = {}
        for node in self.G.nodes():
            faces_with_node = [face for face in faces if node in face]
            faces_edges: Set[List[Edge]] = set()
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
        # self.positions -= np.mean(self.positions, axis=0)
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
        if exponent is None:
            exponent = self.DEFAULT_EDGE_EXP
        if gamma is None:
            gamma = self.gamma

        # If two edges get too close to one another, we get this
        # problem of collinearity in which the edge repulsions don't work.
        self._offset_collinear_points()
        forces = np.zeros_like(self.positions)
        for (v, a, b) in self._virtual_nodes.keys():
            proj_pos, within_line, distance_to_proj = self._virtual_nodes[v, a, b]

            # Skip cases where the virtual position isn't within the edge
            if not within_line:
                continue

            # Skip cases where the virtual position is too far from the real node
            if distance_to_proj > gamma:
                continue

            separation = proj_pos - self.positions[v]
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
        max_forces = np.empty([len(self.G), self.num_zones], dtype=float)
        max_forces[:, :] = np.inf

        # Cache the function lookup locally
        minimum = np.minimum
        for (v, a, b), (proj_pos, within_line, distance_to_proj) in self._virtual_nodes.items():
            if within_line:
                distance_to_proj_3 = distance_to_proj / 3.0
                node_to_pos_vec = proj_pos - self.positions[v]
                angle = np.arctan2(node_to_pos_vec[1], node_to_pos_vec[0])
                segment = calculate_segment(angle, self.num_zones)
                v_indices = np.array(
                    [s % self.num_zones for s in range(segment - 2, segment + 2 + 1)]
                )
                max_forces[v, v_indices] = minimum(
                    max_forces[v, v_indices], distance_to_proj_3
                )

                a_b_indices = np.array(
                    [s % self.num_zones for s in range(segment + 2, segment + 6 + 1)]
                )
                max_forces[a, a_b_indices] = minimum(
                    max_forces[a, a_b_indices], distance_to_proj_3
                )
                max_forces[b, a_b_indices] = minimum(
                    max_forces[b, a_b_indices], distance_to_proj_3
                )
            else:
                dist_to_a_3 = self._distances[v, a] / 3.0
                dist_to_b_3 = self._distances[v, b] / 3.0
                max_forces[v, :] = minimum(
                    max_forces[v, :], min(dist_to_a_3, dist_to_b_3)
                )
                max_forces[a, :] = minimum(max_forces[a, :], dist_to_a_3)
                max_forces[b, :] = minimum(max_forces[b, :], dist_to_b_3)

        return max_forces

    def _calculate_max_forces_impred(self) -> np.array:
        """
        Calculate the maximum forces according to the ImPrEd algorithm.

        WARNING: This function is extremely slow, and may be incorrect.
        """
        warnings.warn("max_forces_impred is extremely slow, and may be incorrect",
                      UserWarning)

        def s_to_r(j: int) -> int:
            if j < 0:
                j = self.num_zones + j
            return j % self.num_zones

        max_forces = np.empty([len(self.G), self.num_zones], dtype=float)
        max_forces[:, :] = np.inf

        for (v, a, b) in self._virtual_nodes.keys():
            proj_pos, within_line, distance_to_proj = self._virtual_nodes[v, a, b]

            # If the projection is within the edge, draw the
            # 'collision line' halfway along the node -> virtual node
            # boundary.
            # If not, draw the 'collision line' halfway between
            # the node and the closest part of the edge.
            if within_line:
                vec_to_proj = self.positions[v] - proj_pos
                halfway_between = (self.positions[v] + proj_pos) / 2
            else:
                v_to_a = self.positions[a] - self.positions[v]
                v_to_b = self.positions[b] - self.positions[v]
                len_v_to_a = np.linalg.norm(v_to_a)
                len_v_to_b = np.linalg.norm(v_to_b)
                if len_v_to_a < len_v_to_b:
                    vec_to_proj = v_to_a
                    halfway_between = (self.positions[v] + self.positions[a]) / 2
                else:
                    vec_to_proj = v_to_b
                    halfway_between = (self.positions[v] + self.positions[b]) / 2

            perpendicular = np.array([-vec_to_proj[1], vec_to_proj[0]])

            hw_line_a = halfway_between + perpendicular
            hw_line_b = halfway_between - perpendicular
            for node in [v, a, b]:
                # Note that the 'collision line' has infinite length, so we can
                # throw out the 'within line' check.
                hw_proj, _, collision_dist = _project_onto_line(
                    self.positions[node], hw_line_a, hw_line_b
                )
                collision_vec = hw_proj - self.positions[node]

                angle = np.arctan2(collision_vec[1], collision_vec[0])
                seg_i = calculate_segment(angle, self.num_zones)
                for seg_j in range(self.num_zones):
                    if (seg_i - seg_j) % self.num_zones == 0:
                        sigma = 1.0
                    elif (seg_i - seg_j) % self.num_zones <= 0.25 * self.num_zones:
                        angle_delta = (seg_j + 1) * 2 * np.pi / self.num_zones
                        sigma = 1.0 / np.cos(angle - angle_delta)
                    elif (seg_i - seg_j) % self.num_zones >= 0.75 * self.num_zones:
                        angle_delta = seg_j * 2 * np.pi / self.num_zones
                        sigma = 1.0 / np.cos(angle - angle_delta)
                    else:
                        sigma = np.inf
                    if sigma < 0 or sigma > 3:
                        sigma = np.inf
                    max_forces[node, seg_j] = min(
                        max_forces[node, seg_j], sigma * collision_dist
                    )
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
        # max_forces = self._calculate_max_forces_impred()
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
        step_size: float = 1.0,
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
        assert (
            0 <= step_size <= 1
        ), f"Step size must be in the range [0, 1) but got {step_size}"
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
        if attraction_exp_minmax is None:
            attraction_exps = np.array([None] * num_steps)
        else:
            attraction_exps = np.linspace(
                max(attraction_exp_minmax),
                min(attraction_exp_minmax),
                num_steps,
                dtype=float,
            )

        if repulsion_exp_minmax is None:
            repulsion_exps = np.array([None] * num_steps)
        else:
            repulsion_exps = np.linspace(
                max(repulsion_exp_minmax),
                min(repulsion_exp_minmax),
                num_steps,
                dtype=float,
            )
        if edge_exp_minmax is None:
            edge_exps = np.array([None] * num_steps)
        else:
            edge_exps = np.linspace(
                max(edge_exp_minmax), min(edge_exp_minmax), num_steps, dtype=float
            )
        if max_step_minmax is None:
            max_steps = np.array([np.inf] * num_steps)
        else:
            max_steps = np.linspace(
                max(max_step_minmax), min(max_step_minmax), num_steps, dtype=float
            )

        start_time = datetime.now()
        for step in range(num_steps):
            if step % (num_steps // 100) == 0 and step != 0:
                cur_time = datetime.now()
                time_per_step = (cur_time - start_time) / step
                time_remaining = (num_steps - step) * time_per_step
                print(f"Step {step} / {num_steps}; {100 * step / num_steps:.1f} %; Time Remaining {time_remaining}")
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
    pos = layout.update(step_size=1.0)
    nx.draw_networkx(test_graph, pos, ax=ax)
    ax.scatter(*np.mean(layout.positions, axis=0))
    ax.set_title(f"Iteration {num}")
    ax.set_xticks([])
    ax.set_yticks([])


if __name__ == "__main__":
    test_graph = nx.ladder_graph(50)
    # with open("./edge_graph_9.pkl", "rb") as fi:
    #     test_graph = pkl.load(fi)
    layout = PREDLayout(
        test_graph,
        max_distance_to_virtual=1.0,
        gravity_force_scale=0.0,
        start_pos=nx.planar_layout(test_graph),
    )

    # for step in range(500):
    #     if step % 50 == 0:
    #         print(f"Step {step} / 500")
    #     try:
    #         layout.update(step_size=1.00)
    #     except RuntimeError:
    #         pass
    # # try:
    #     layout.anneal(attraction_exp_minmax=(0.4, 1),
    #                   repulsion_exp_minmax=(2.0, 4.0),
    #                   edge_exp_minmax=(2.0, 4.0),
    #                   max_step_minmax=(0.0, 3.0),
    #                   num_steps=5*len(test_graph))
    # except RuntimeError:
    #     pass
    # nx.draw(test_graph, layout._positions_to_label_dict())
    fig, ax = plt.subplots()
    nx.draw(test_graph, layout._positions_to_label_dict())
    animation = mpl.animation.FuncAnimation(fig, lambda num: update(ax, test_graph, layout, num), frames=100, interval=16)
    plt.show()
