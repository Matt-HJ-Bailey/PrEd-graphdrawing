# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 15:06:12 2020

@author: Matt
"""

import numpy as np
import networkx as nx
from pred import _project_onto_line, calculate_segment, calculate_neighbours, calculate_faces

class TestProjection:
    def test_simple_projection(self):
        line_u = np.array([0.0, 0.0])
        line_v = np.array([0.0, 1.0])
        
        node = np.array([0.5, 0.5])
    
        projected_pos, is_on_line = _project_onto_line(node, line_u, line_v)
        
        assert np.all(np.isclose(projected_pos,
                                 np.array([0.0, 0.5])))
        assert is_on_line == True
        
    def test_offset_projection(self):
        line_u = np.array([0.0, 1.0])
        line_v = np.array([0.0, 2.0])
        
        node = np.array([0.5, 0.5])
    
        projected_pos, is_on_line = _project_onto_line(node, line_u, line_v)
        
        assert np.all(np.isclose(projected_pos,
                                 np.array([0.0, 0.5])))
        assert is_on_line == False
        
    def test_long_projection(self):
        line_u = np.array([0.0, 0.0])
        line_v = np.array([0.0, 100.0])
        
        node = np.array([0.5, 0.5])
    
        projected_pos, is_on_line = _project_onto_line(node, line_u, line_v)
        
        assert np.all(np.isclose(projected_pos,
                                 np.array([0.0, 0.5])))
        assert is_on_line == True
        
    def test_long_offset_projection(self):
        line_u = np.array([0.0, 100.0])
        line_v = np.array([0.0, 200.0])
        
        node = np.array([0.5, 0.5])
    
        projected_pos, is_on_line = _project_onto_line(node, line_u, line_v)
        
        assert np.all(np.isclose(projected_pos,
                                 np.array([0.0, 0.5])))
        assert is_on_line == False
        
    def test_negative_projection(self):
        line_u = np.array([0.0, 0.0])
        line_v = np.array([0.0, 1.0])
        
        node = np.array([-0.5, -0.5])
    
        projected_pos, is_on_line = _project_onto_line(node, line_u, line_v)
        
        assert np.all(np.isclose(projected_pos,
                                 np.array([0.0, -0.5])))
        assert is_on_line == False
        
    def test_negative_projection_both(self):
        line_u = np.array([0.0, 0.0])
        line_v = np.array([0.0, -1.0])
        
        node = np.array([-0.5, -0.5])
    
        projected_pos, is_on_line = _project_onto_line(node, line_u, line_v)
        
        assert np.all(np.isclose(projected_pos,
                                 np.array([0.0, -0.5])))
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
        assert np.all(np.isclose(distances[:, 0],
                                 np.array([0.0, 1.0, np.sqrt(2), 1.0])))
        
    def test_num_neighbours(self):
        """
        Make sure we predict the correct number of pairs, i.e. N(N-1)/ 2
        """
        for i in range(2, 10):
            positions= np.array([[j, j] for j in range(i)])
            distances, neighbours = calculate_neighbours(positions)
            assert len(neighbours) == positions.shape[0] * (positions.shape[0] - 1) // 2
            
class TestFindFaces:
    """
    """
    
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
        