"""
background.py
=============
Copyright 2013-2017 Sebastian Kamann

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
class BackgroundGrid

Purpose
-------
This module implements a handling of an unresolved background. The background
may originate from unresolved stars, gaseous emission, or the night sky. Also,
the background can vary across the field of view. To this aim, the class
'BackgroundGrid' models the background as a homogeneoous grid. A background
patch is centred on each gridpoint and the weight of each pixel in the patch
scales with the inverse distance to the gridpoint.

Added
-----
Revision 221, 2014/12/02

Latest Git revision
-------------------
2017/12/04
"""
import logging
import numpy as np
from scipy.spatial import distance
from scipy.sparse import coo_matrix

from ..psf import Canvas


__author__ = "Sebastian Kamann (s.kamann@ljmu.ac.uk)"
__revision__ = 20171204


class BackgroundGrid(Canvas):
    """
    The purpose of this class is to define a background that is spatially
    variable. To this aim, it defines a background grid. For each gridpoint,
    a weighting function w(r) of the form

        w(r) ~ EXP(-0.5*(r/Rs)^n)

    is defined. r measures the distance to the gridpoint, Rs is the scale
    radius which is equal to half the distance between to adjacent gridpoints.
    In this way, a smooth but variable background is created across the field
    of view.
    """
    def __init__(self, centroids, n=2, maxrad=None, **kwargs):
        """
        Initialize an instance of the Background class.

        Parameters
        ----------
        centroids : ndarray
            The central coordinates of the individual background patches, as an
            array of shape (<no. of patches>, 2). Note that the centroids must
            be on a homogeneous grid currently.
        n : int, optional
            The power law index that controls the weighting of the spaxels with
            distance. It should currently be set to 2 (the default).
        maxrad : float, optional
            The radius out to which a background patch expands, spaxels at
            larger radii are assigned zero weight. By default, it will be set
            to 3 x rscale.
        """
        # call parent-class to access handling of profiles on (possibly irregularly sampled) spatial grid.
        super(BackgroundGrid, self).__init__(need_spaxels=False, **kwargs)

        # check if input is valid
        self.centroids = np.asarray(centroids)
        assert self.centroids.ndim == 2 and self.centroids.shape[-1] == 2,\
            "Centroids must be a 2dim. array of shape (N,2)."

        # make sure centroids are on a homogeneous grid: calculate distance to nearest neighbour, must be the same for
        # each grid point
        if self.centroids.shape[0] > 1:
            distances = distance.cdist(self.centroids, self.centroids)
            distances.sort(axis=0)
            assert np.unique(distances[1]).size == 1, "Centroids must form homogeneous grid."

            self.rscale = distances[1][0] / 2.
            # scale radius is set to half the distance between two adjacent centroids.
        else:
            self.rscale = np.inf

        self.n = n  # the power-law index used to assign weights

        # To make the computation more efficient, each background patch is only calculated out to a given radius
        if maxrad is None:
            self.maxrad = 3 * self.rscale
        else:
            self.maxrad = maxrad

        # initialize arrays that will contain information about background patches
        self._r = None
        self._f = None
        self._used = None
        self.background_properties = ["_r", "_f", "_used"]

        # initialise arrays containing information about slice-dependent background
        self.n_ifu = 0
        self.n_slice = 0
        self.ifus = None
        self.slices = None

    @classmethod
    def from_shape(cls, shape, rscale=None, **kwargs):
        """
        This method initializes an instance of the Background class using an
        array shape and a scale radius. The centroids of the background
        patches will be defined on a homogeneous grid with constant distance
        2 x scale radius between adjacent centroids.

        Parameters
        ----------
        shape : tuple
            The spatial size of the array to work on, provided as (dy, dx)
        rscale : float, optional
            The effective radius of a background patch in units of spaxels.
            The distance between neighbouring patches will be 2 x rscale.
        kwargs : optional
            All remaining arguments are passed on to the call of the
            Background.__init__ method.
        """
        if rscale is None:
            rscale = np.inf

        if not np.isfinite(rscale):
            return cls(np.array([[0, 0], ], dtype=np.float32))

        # create the grid with the centroids of the background patches
        # first get centroids along x and y
        xshape, yshape = shape
        _xcentroids = np.arange(0, xshape, 2 * rscale)
        _ycentroids = np.arange(0, yshape, 2 * rscale)
        # then make sure that distances to edges of FoV are approximately the same
        _xcentroids += 0.5 * (xshape - 1 - _xcentroids.max())
        _ycentroids += 0.5 * (yshape - 1 - _ycentroids.max())
        # combine x and y centroids to array of shape (2, gridsize)
        xcentroids = np.resize(_xcentroids, _xcentroids.size * _ycentroids.size)
        xcentroids.sort()
        ycentroids = np.resize(_ycentroids, _xcentroids.size * _ycentroids.size)

        return cls(np.vstack((xcentroids, ycentroids)).T, **kwargs)

    @classmethod
    def constant(cls):
        """
        This method initializes a constant background across an arbitrary field
        of view.
        """
        return cls(np.array([[0, 0], ], dtype=np.float32))

    def reset(self, omit=None):
        """
        The method resets the current arrays containing the properties of the
        background profiles/patches and sets them for recalculation.
        
        Parameters
        ----------
        omit : list, optional
            A list with profile properties that
        """
        for prop in self.background_properties:
            if omit is None or prop not in omit:
                setattr(self, prop, None)

    @property
    def gridsize(self):
        """
        Returns number of sky patches across field of view.
        """
        return self.centroids.shape[0]

    @property
    def r(self):
        """
        Returns distance of each pixel to each gridpoint. The returned array
        has a size of (BackgroundGrid.gridsize, BackgroundGrid.n_pixel)
        """
        if self._r is None:  # only calculate if not already available
            if self._transform is None:
                logging.critical("Spaxel locations must be defined before calculating distances.")
            assert self._transform is not None, "Spaxel locations not known."
            self._r = distance.cdist(self.centroids, self.transform)  # resulting array has size (gridsize, n_pixel)

            # if per-slice background is used, set their centroid distances to infinity for all pixels
            self._r[self.gridsize - self.n_ifu * self.n_slice:] = np.inf
        return self._r

    @property
    def f(self):
        """
        Returns the weight of each background patch in each pixel. The array
        has a size of (BackgroundGrid.gridsize, BackgrounGrid.n_pixel)
        """
        if self._f is None:  # only calculate if not already available
            self._f = np.where(np.isfinite(self.r), np.exp(-0.5 * (self.r / self.rscale) ** self.n), 0)
            self._f[~self.used] = 0
            self._f /= self._f.sum(axis=0)  # normalize weights

            # handle slice/ifu-dependent components if any
            if self.n_ifu > 0 or self.n_slice > 0:
                if self.ifus is not None and self.slices is not None:
                    n = self.n_ifu * self.n_slice
                    self._f[-n:] = (self.centroids[-n:, [1]] == self.slices) & (self.centroids[-n:, [0]] == self.ifus)
                else:
                    logging.error('Cannot properly set background values because IFU/slice numbers are missing.')

        return self._f

    @property
    def used(self):
        """
        An array indicating for each background patch which pixels contribute.

        Note that there are two possibilities that lead to pixels being
        excluded from one or more patches:
        a) They are located at distances >'maxrad' to a grid point
        b) They are included in a user-provided bad-pixel mask

        Returns
        -------
        used : nd_array
            Boolean mask indicating for each background component which pixels
            contribute.The shape of the array is (gridsize, n_pixel)
        """
        if self._used is None:  # only calculate if not already available
            self._used = (self.r < self.maxrad)

            # handle slice/ifu-dependent components if any
            if self.n_ifu > 0 or self.n_slice > 0:
                if self.ifus is not None and self.slices is not None:
                    n = self.n_ifu * self.n_slice
                    self._used[-n:] = (self.centroids[-n:, [1]] == self.slices) & (
                        self.centroids[-n:, [0]] == self.ifus)
                else:
                    logging.error('Cannot properly set background pixels because IFU/slice numbers are missing.')

        return self._used

    @used.setter
    def used(self, value):
        """
        Setter method for the BackgroundGrid.used property.

        A dedicated setter method is required to make sure that pixels
        excluded because of their distance to a grid point are not
        included again when the user provides a new mask.

        Parameters
        ----------
        value : array_like
            The array

        Returns
        -------

        """
        try:
            value = np.asarray(value, dtype=np.bool)
        except ValueError:
            logging.critical("Invalid input for background pixel mask: {0}".format(value))
        if value.size != self.n_pixel:
            logging.critical("Provided background pixel mask has invalid size ({0} instead of {1}).".format(
                value.size, self.n_pixel))

        self._used = value & (self.r < self.maxrad)

        # handle slice/ifu-dependent components if any
        if self.n_ifu and self.n_slice:
            if self.ifus is not None and self.slices is not None:
                n = self.n_ifu * self.n_slice
                self._used[-n:] = (self.centroids[-n:, [1]] == self.slices) & (self.centroids[-n:, [0]] == self.ifus)
            else:
                logging.error('Cannot properly set background pixels because IFU/slice numbers are missing.')

    def add_slices(self, n_ifu, n_slice):
        self.n_ifu = int(n_ifu)
        self.n_slice = int(n_slice)

        # use y-coordinates as IFU numbers and x-coordinates as slice numbers
        _x, _y = np.mgrid[1:self.n_slice + 1, 1:self.n_ifu + 1]
        _centroids = np.vstack((_x.flatten(), _y.flatten())).T
        self.centroids = np.concatenate((self.centroids, _centroids), axis=0)

        self.reset()

    def get_flux(self, i):
        """
        Returns the indices of the used pixels and the corresponding weights
        for an individual background patch.

        Parameters
        ----------
        i : integer
            The index of the background patch for which pixel and weights
            should be returned.
        
        Returns
        -------
        data : tuple
            A tuple of length 2 containing the indices of the used pixel
            and the corresponding fluxes.
        """
        return self.pixel[self.used[i]], self.f[i, self.used[i]]

    def get_sparse_matrix(self, offset=0, purge_empty=False):
        """
        The method prepares a sparse matrix from the background components
        that can be used in optimization process.

        The number of columns in the matrix can be increased so that it can
        be easily combined with a second sparse matrix containing the
        contributions of the stellar sources.

        Parameters
        ----------
        offset : int, optional
            The offset applied to all background columns. Typically the
            number of stellar components in the source matrix.
        purge_empty : bool, optional
           Flag indicating if background components that do not contribute to
           should be removed from the matrix.

        Returns
        -------
        background_matrix : scipy.linalg.sparse.coo_matrix
            The sparse matrix containing the background components, one per
            column. The matrix contain one row for each available spaxel and
            one column for each requested background component (omitting
            empty one if purge_empty was set). The column indices are offset
            as requested by the offset parameter.
        included_components : nd_array
            The indices of the background components that are returned (i.e.
            omitting the empty ones if purge_empty was set).
        """
        i, j = np.nonzero(self.f)
        values = self.f[(i, j)]

        # check if any empty components should be removed
        if purge_empty and self.gridsize > len(np.unique(i)):
            empty_components = np.flatnonzero(np.isin(np.arange(self.gridsize), i, invert=True))
            for component in empty_components[::-1]:
                i[i > component] -= 1
        else:
            empty_components = []

        # check if numpy.isin is available, otherwise fall back to np.in1d
        isin = np.isin if np.__version__ >= "1.13.0" else np.in1d
        included_components = np.flatnonzero(isin(np.arange(self.gridsize), empty_components, invert=True))
        included_components += offset

        # IMPORTANT: to be consistent with the matrix of sources, we need to swap i and j because the source matrix
        # has shape (n_spaxel, n_components)
        n_rows = self.transform.shape[0]
        n_columns = self.gridsize + offset - len(empty_components)

        return coo_matrix((values, (j, i + offset)), shape=(n_rows, n_columns)), included_components
