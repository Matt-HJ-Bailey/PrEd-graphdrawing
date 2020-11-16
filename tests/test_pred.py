# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 15:06:12 2020

@author: Matt
"""

import numpy as np
import networkx as nx

from line_utils import _project_onto_line, calculate_segment, do_lines_intersect, line_orientation, OrientationResult

from graph_utils import calculate_neighbours, calculate_faces, count_intersections


class TestProjection:
    def test_simple_projection(self):
        line_u = np.array([0.0, 0.0])
        line_v = np.array([0.0, 1.0])

        node = np.array([0.5, 0.5])

        projected_pos, is_on_line = _project_onto_line(node, line_u, line_v)

        assert np.all(np.isclose(projected_pos, np.array([0.0, 0.5])))
        assert is_on_line == True

    def test_offset_projection(self):
        line_u = np.array([0.0, 1.0])
        line_v = np.array([0.0, 2.0])

        node = np.array([0.5, 0.5])

        projected_pos, is_on_line = _project_onto_line(node, line_u, line_v)

        assert np.all(np.isclose(projected_pos, np.array([0.0, 0.5])))
        assert is_on_line == False

    def test_long_projection(self):
        line_u = np.array([0.0, 0.0])
        line_v = np.array([0.0, 100.0])

        node = np.array([0.5, 0.5])

        projected_pos, is_on_line = _project_onto_line(node, line_u, line_v)

        assert np.all(np.isclose(projected_pos, np.array([0.0, 0.5])))
        assert is_on_line == True

    def test_long_offset_projection(self):
        line_u = np.array([0.0, 100.0])
        line_v = np.array([0.0, 200.0])

        node = np.array([0.5, 0.5])

        projected_pos, is_on_line = _project_onto_line(node, line_u, line_v)

        assert np.all(np.isclose(projected_pos, np.array([0.0, 0.5])))
        assert is_on_line == False

    def test_negative_projection(self):
        line_u = np.array([0.0, 0.0])
        line_v = np.array([0.0, 1.0])

        node = np.array([-0.5, -0.5])

        projected_pos, is_on_line = _project_onto_line(node, line_u, line_v)

        assert np.all(np.isclose(projected_pos, np.array([0.0, -0.5])))
        assert is_on_line == False

    def test_negative_projection_both(self):
        line_u = np.array([0.0, 0.0])
        line_v = np.array([0.0, -1.0])

        node = np.array([-0.5, -0.5])

        projected_pos, is_on_line = _project_onto_line(node, line_u, line_v)

        assert np.all(np.isclose(projected_pos, np.array([0.0, -0.5])))
        assert is_on_line == True


class TestCalculateSegment:
    def test_segment_midpoints(self):
        num_segments = 8
        for i in range(num_segments):
            angle = (i + 0.5) * 2 * np.pi / num_segments
            assert calculate_segment(angle, num_segments) == i

    def test_segment_bottoms(self):
        num_segments = 8
        for i in range(num_segments):
            angle = (i) * 2 * np.pi / num_segments
            assert calculate_segment(angle, num_segments) == i

    def test_segment_tops(self):
        num_segments = 8
        for i in range(num_segments):
            angle = (i + 1) * 2 * np.pi / num_segments
            assert calculate_segment(angle, num_segments) == i + 1


class TestCalculateNeighbours:
    """
    Test node neighbour detection functions.
    """

    def test_no_cutoff(self):
        positions = np.array([[0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        distances, neighbours = calculate_neighbours(positions)
        assert np.all(
            np.isclose(distances[:, 0], np.array([0.0, 1.0, np.sqrt(2), 1.0]))
        )

    def test_num_neighbours(self):
        """
        Make sure we predict the correct number of pairs, i.e. N(N-1)/ 2
        """
        for i in range(2, 10):
            positions = np.array([[j, j] for j in range(i)])
            distances, neighbours = calculate_neighbours(positions)
            assert len(neighbours) == positions.shape[0] * (positions.shape[0] - 1) // 2


class TestFindFaces:
    """"""

    def test_wheel_graph(self):
        """
        The faces of a wheel graph should all be triangles.
        """
        G = nx.wheel_graph(10)
        faces = calculate_faces(G)
        assert len(faces) == 10
        face_sizes = sorted([len(face) for face in faces])
        assert all([size == 3 for size in face_sizes[:-1]])
        assert face_sizes[-1] == 9

class TestFindIntersections:
    """
    Make sure we correctly find intersections between graph edges.
    """
    def test_line_intersection(self):
        p1, q1 = np.array([0, 0]), np.array([2, 0])
        p2, q2 = np.array([1, 1]), np.array([1, -1])
        assert line_orientation(p1, p2, q1) == OrientationResult.CLOCKWISE
        assert line_orientation(p1, p2, q2) == OrientationResult.CLOCKWISE
        assert line_orientation(q1, q2, p1) == OrientationResult.CLOCKWISE
        assert line_orientation(q1, q2, p2) == OrientationResult.CLOCKWISE
        assert do_lines_intersect(p1, q1, p2, q2)

    def test_line_collinear_overlap(self):
        p1, q1 = np.array([0, 0]), np.array([2, 0])
        p2, q2 = np.array([1, 0]), np.array([3, 0])

        assert line_orientation(p1, p2, q1) == OrientationResult.COLLINEAR
        assert line_orientation(p1, p2, q2) == OrientationResult.COLLINEAR
        assert line_orientation(q1, q2, q1) == OrientationResult.COLLINEAR
        assert line_orientation(q1, q2, q2) == OrientationResult.COLLINEAR

        assert do_lines_intersect(p1, q1, p2, q2)

    def test_line_collinear_no_overlap(self):
        p1, q1 = np.array([0, 0]), np.array([2, 0])
        p2, q2 = np.array([5, 0]), np.array([10, 0])

        assert line_orientation(p1, p2, q1) == OrientationResult.COLLINEAR
        assert line_orientation(p1, p2, q2) == OrientationResult.COLLINEAR
        assert line_orientation(q1, q2, q1) == OrientationResult.COLLINEAR
        assert line_orientation(q1, q2, q2) == OrientationResult.COLLINEAR

        assert not do_lines_intersect(p1, q1, p2, q2)

    def test_ladder_graph(self):
        G = nx.ladder_graph(20)
        pos = nx.planar_layout(G)
        num_intersections = count_intersections(G, pos)
        assert num_intersections == 0

    def test_wheel_graph(self):
        G = nx.wheel_graph(20)
        pos = nx.planar_layout(G)
        num_intersections = count_intersections(G, pos)
        assert num_intersections == 0

    def test_petersen_graph(self):
        G = nx.petersen_graph()
        petersen_pos = {0: [0, -2],
                    1: [-2, 0],
                    2: [-2, 2],
                    3: [2, 2],
                    4: [2, 0],
                    5: [0, -1],
                    6: [-1, 0],
                    7: [-1, 1],
                    8: [1, 1],
                    9: [1, 0]}
        num_intersections = count_intersections(G, petersen_pos)
        assert num_intersections == 5