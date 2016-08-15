"""
parameters.py
=============
Copyright 2013-2016 Sebastian Kamann

This file is part of PampelMuse.

PampelMuse is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.

PampelMuse is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with PampelMuse. If not, see <http://www.gnu.org/licenses/>.


+++
Provides
--------
function pampelmuse.utils.distance.pnt2line
function pampelmuse.utils.distance.point_in_poly
function pampelmuse.utils.distance.dist2path

Purpose
-------
This module provides several utility functions needed to check if a
point is inside a given field of view and what is its distance to the
edge of the field o view.

Latest SVN revision
-------------------
365, 2016/08/15
"""
import logging
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import numpy as np


__author__ = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__revision__ = 365


def pnt2line(pnt, start, end):
    """
    This function calculates the shortest 2D distance of one or more points to
    a line that is defined by a start and an end point. It is a modified
    version of the code presented at this webpage:
    http://stackoverflow.com/questions/23937076/distance-to-convexhull/23941569
    I revised it so that more than one input point can be used.

    Parameters
    ----------
    pnt : nd_array, shape (N, 2)
        The array containing x- & y-coordinates for the N points for which
        distances should be determined
    start : nd_array, shape(2,)
        The x- & y-coordinates of the starting point of the line.
    end : nd_array, shape(2,)
        The x- & y-coordinates of the end point of the line.

    Returns
    -------
    distance : nd_array, shape (N,)
        The euclidean distance of each input point to the list
    nearest : nd_array, shape (N, 2)
        The x- & y-coordinates of the point on the line that is closest
        to each input point.
    """
    # constructs vectors pointing from starting point of line to its end point  and to each input point
    line_vec = end - start  # has shape (2,)
    pnt_vec = pnt - start  # has shape (N, 2)

    # normalize all vectors with constant factor so that line vector has a length of one
    line_len = np.linalg.norm(line_vec)  # returns a float
    line_unitvec = line_vec/line_len  # has shape (2,)
    pnt_vec_scaled = pnt_vec/line_len  # has shape (N, 2)

    # calculate dot product of normalized line vector with each normalized vector pointing to an input point
    # t is the fractional length of the line vector to its point of closest approach with an input point
    t = np.dot(line_unitvec, pnt_vec_scaled.T) # has shape (N,)
    t[t < 0] = 0.
    t[t > 1] = 1.

    # calculate coordinates of the points of closest approach on the line and the distances of these points to the
    # respective input points
    nearest = line_vec.reshape(2, -1)*t  # has shape (2, N)
    distance = np.sqrt(np.sum((nearest.T - pnt_vec)**2, axis=-1))
    nearest = nearest.T + start  # has shape (N, 2)
    return distance, nearest


def point_in_poly(x, y, poly):
    """
    This function checks if a point, defined by its x- and y-coordinates, is
    inside a polygon. It is based on this blog entry:
    http://geospatialpython.com/2011/01/point-in-polygon.html

    Parameters
    ----------
    x : float
        The x-coordinate of the point.
    y : float
        The y-coordinate of the point.
    poly : array_like, shape (n, 2)
        A polygon of length n, defined by its x- and y-coordinates

    Returns
    -------
    inside : boolean
        True if the point is located inside the polygon, False otherwise.
    """
    n = len(poly)
    inside = False

    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y - p1y)*(p2x - p1x)/(p2y - p1y) + p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def dist2path(pnt, path):
    """
    This function calculates the distance of one or more points to a closed
    path in x & y. It is an adapted version of the code presented here:
    http://stackoverflow.com/questions/23937076/distance-to-convexhull/23941569

    Parameters
    ----------
    pnt : nd_array, shape (N, 2)
        The array containing x- & y-coordinates for the N points for which
        distances should be determined
    path : nd_array, shape(M, 2)
        The x- & y-coordinates of the points that define the path to which
         the distances should be determined.

    Returns
    -------
    pt_dist : nd_array, shape (N,)
        The closest distance of each input point to the path. Points outside
        the path have negative distances.
    """
    # for each segment of the path, calculate the distance of all input points and collect them in a common list
    all_distances = []

    if ((path[0][0] - path[-1][0])**2 + (path[0][1] - path[-1][1])**2) > 1e-6:
        logging.warning('Provided path is not closed.')

    for i in range(len(path) - 1):
        start = path[i]
        end = path[i+1]
        dist_list = pnt2line(pnt, start, end)[0]
        all_distances.append(dist_list)

    # for each input point, find segment of path that gives smallest distance
    pt_dist = np.min(all_distances, axis=0)

    # for each point, check if it is inside or outside the hull
    for p_idx in range(len(pnt)):
        x, y = pnt[p_idx]

        inside = point_in_poly(x, y, path)
        if not inside:
            pt_dist[p_idx] *= -1

    return pt_dist


if __name__ == "__main__":
    """
    For testing purposes only.
    """

    # Original points, hull and test points
    points = np.random.rand(30, 2)   # 30 random points in 2-D
    hull = ConvexHull(points)
    newpnt = np.random.rand(30, 2)   # 30 random points in 2-D

    pt_distances = dist2path(newpnt, hull.points[hull.vertices])

    # Plot original points, hull and new points
    plt.plot(points[:, 0], points[:, 1], 'ro')
    plt.plot(points[hull.vertices,0], points[hull.vertices, 1], 'r--', lw=2)
    plt.plot(newpnt[:, 0], newpnt[:, 1], 'go')
    for idx in range(30):
        point = newpnt[idx, :]
        dist = pt_distances[idx]
        distLabel = "%.2f" % dist
        plt.annotate(distLabel, xy=point)

    plt.show()
