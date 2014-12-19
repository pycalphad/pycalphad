"""
This module defines a class representing an arbitrary Simplex in arbitrary
dimensional space.
"""

from __future__ import division

__author__ = "Shyue Ping Ong"
__version__ = "2.0"
__maintainer__ = "Shyue Ping Ong"
__email__ = "shyuep@gmail.com"
__date__ = "May 15, 2012"

import itertools

import numpy as np


class Simplex(object):
    """
    A generalized simplex object. See http://en.wikipedia.org/wiki/Simplex.
    """

    def __init__(self, coords):
        """
        Initializes a Simplex from coordinates.

        Args:
            coords ([[float]]): Coords of the vertices of the simplex. E.g.,
                [[1, 2, 3], [2, 4, 5], [6, 7, 8]].
        """
        self._coords = np.array(coords)
        self.space_dim, self.simplex_dim = self._coords.shape
        self.origin = self._coords[-1]
        if self.space_dim == self.simplex_dim + 1:
            # precompute attributes for calculating bary_coords
            self.T = self._coords[:-1] - self.origin
            self.T_inv = np.linalg.inv(self.T)

    def bary_coords(self, point):
        try:
            c = np.dot((point - self.origin), self.T_inv)
            return np.concatenate([c, [1 - np.sum(c)]])
        except AttributeError:
            raise ValueError('Simplex is not full-dimensional')

    def in_simplex(self, point, tolerance=1e-8):
        """
        Checks if a point is in the simplex using the standard barycentric
        coordinate system algorithm.

        Taking an arbitrary vertex as an origin, we compute the basis for the
        simplex from this origin by subtracting all other vertices from the
        origin. We then project the point into this coordinate system and
        determine the linear decomposition coefficients in this coordinate
        system.  If the coeffs satisfy that all coeffs >= 0, the composition
        is in the facet.

        Args:
            point ([float]): Point to test
            tolerance (float): Tolerance to test if point is in simplex.
        """
        return (self.bary_coords(point) >= -tolerance).all()

    def __eq__(self, other):
        for p in itertools.permutations(self._coords):
            if np.allclose(p, other.coords):
                return True
        return False

    def __hash__(self):
        return len(self._coords)

    def __repr__(self):
        output = ["{}-simplex in {}D space".format(self.simplex_dim,
                                                   self.space_dim),
                  "Vertices:"]
        for coord in self._coords:
            output.append("\t({})".format(", ".join(map(str, coord))))
        return "\n".join(output)

    def __str__(self):
        return self.__repr__()

    @property
    def coords(self):
        """
        Returns a copy of the vertex coordinates in the simplex.
        """
        return self._coords.copy()
