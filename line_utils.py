# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 09:59:45 2020

@author: Matt
"""

from enum import Enum
from typing import Optional, Tuple, Union, Collection

import numpy as np
import scipy
from numba import jit

Position = Union[Collection[float]]

@jit(nopython=True)
def _project_onto_line(
    node_pos: Position, line_u: Position, line_v: Position
) -> Tuple[Position, bool, float]:
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
    line_length_sq = np.dot(line_vec, line_vec)

    line_frac = np.dot(node_pos - line_u, line_vec) / line_length_sq

    projected_pos = line_u + line_frac * line_vec

    # Note that being exactly on a node at either edge of the line counts
    # as being on the line.
    within_line = bool(0 <= line_frac <= 1.0)

    distance = np.linalg.norm(node_pos - projected_pos)
    return projected_pos, within_line, distance


def calculate_segment(angle: float, num_segments: int) -> int:
    """
    Calculate which segment around a circle a given angle corresponds to.

    Note this is a different meaning of the word 'segment'
    to the line segments.
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


def is_on_segment(p: Position, q: Position, r: Position):
    """
    Test if a point q lies on line segment p-r .

    This requires pq and pr to be collinear with one another.

    Parameters
    ----------
    p, q
        Start and end of a line segment.
    Returns
    -------
    is_on_segment
        True if q lies on pr, False otherwise.
    """
    if (
        q[0] <= max(p[0], r[0])
        and q[0] >= min(p[0], r[0])
        and q[1] <= max(p[1], r[1])
        and q[1] >= min(p[1], r[1])
    ):
        return True
    return False


class OrientationResult(Enum):
    """Represent the result of an orientation test on three points in a plane."""

    COLLINEAR = 0
    CLOCKWISE = 1
    ANTICLOCKWISE = 2


def line_orientation(p: Position, q: Position, r: Position) -> OrientationResult:
    """
    Find the orientation of the ordered triplet (p, q, r).

    This can be clockwise if we go (a, c, b) or anticlockwise if we go(a, b, c).
    These three points could also be collinear.

    Parameters
    ----------
    p, q, r
        Three 2D coordinates in the Cartesian Plane

    Returns
    -------
    OrientationResult
        An enum representing whether we go clockwise, anticlockwise, or the
        points are collinear.
    """
    val = ((q[1] - p[1]) * (r[0] - q[0])) - ((q[0] - p[0]) * (r[1] - q[1]))

    if np.isclose(val, 0.0):
        return OrientationResult.COLLINEAR

    if val > 0:
        return OrientationResult.CLOCKWISE
    return OrientationResult.ANTICLOCKWISE


def do_lines_intersect(p1: Position, q1: Position, p2: Position, q2: Position) -> bool:
    """
    Test if two line segments intersect with one another.

    Transcribed from
    https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/

    Parameters
    ----------
    p1, q1
        The coordinates of the start and end of line 1
    p2, q2
        The coordinates of the start and end of line 2
    Returns
    -------
    do_lines_intersect
        True if the lines intersect, False otherwise.
    """
    o1 = line_orientation(p1, q1, p2)
    o2 = line_orientation(p1, q1, q2)
    o3 = line_orientation(p2, q2, p1)
    o4 = line_orientation(p2, q2, q1)

    # This is the general case in which the lines are not collinear.
    if o1 != o2 and o3 != o4:
        return True

    # However, if the segments are collinear we must see if they
    # overlap.
    if o1 == OrientationResult.COLLINEAR and is_on_segment(p1, p2, q1):
        return True

    if o2 == OrientationResult.COLLINEAR and is_on_segment(p1, q2, q1):
        return True

    if o3 == OrientationResult.COLLINEAR and is_on_segment(p2, p1, q2):
        return True

    if o4 == OrientationResult.COLLINEAR and is_on_segment(p2, q1, q2):
        return True
    return False


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
