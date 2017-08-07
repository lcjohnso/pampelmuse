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

Latest GIT revision
-------------------
2017/08/07
"""
import logging
import numpy as np
from scipy.spatial import distance
from scipy.sparse import coo_matrix

from ..psf import Canvas


__author__ = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__revision__ = 20170807


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

    @classmethod
    def from_shape(cls, shape, rscale, **kwargs):
        """
        This method initializes an instance of the Background class using an
        array shape and a scale radius. The centroids of the background
        patches will be defined on a homogeneous grid with constant distance
        2 x scale radius between adjacent centroids.

        Parameters
        ----------
        shape : tuple
            The spatial size of the array to work on, provided as (dy, dx)
        rscale : float
            The effective radius of a background patch in units of spaxels.
            The distance between neighbouring patches will be 2 x rscale.
        kwargs : optional
            All remaining arguments are passed on to the call of the
            Background.__init__ method.
        """
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
        # get number of patches across field of view
        return self.centroids.shape[0]

    @property
    def r(self):
        # get distance of each spaxel to each gridpoint, returned array has size (gridsize, nspaxel)
        if self._r is None:  # only calculate if not already available
            if self._transform is None:
                logging.critical("Spaxel locations must be defined before calculating distances.")
            assert self._transform is not None, "Spaxel locations not known."
            self._r = distance.cdist(self.centroids, self.transform)  # resulting array has size (gridsize, nspaxel)
        return self._r

    @property
    def f(self):
        # get weight of each spaxel in each background patch, returned array has size (gridsize, nspaxel)
        if self._f is None:  # only calculate if not already available
            self._f = np.exp(-0.5 * (self.r / self.rscale) ** self.n)
            self._f[~self.used] = 0
            self._f /= self._f.sum(axis=0)  # normalize weights
        return self._f

    # 'used' is an array that indicates for each background patch which spaxels contribute
    # There are two possibilities that lead to spaxels spaxels being excluded from one or more patches:
    #  a) They are located at distances >'maxrad' to a gridpoint
    #  b) They are included in a user-provided bad-pixel spectrumMask
    # To make sure that spaxels excluded based on criterion (a) are not included again when the user provides a mask,
    # the setter method is modified.
    def get_used(self):
        if self._used is None:  # only calculate if not already available
            self._used = (self.r <= self.maxrad)
        return self._used

    def set_used(self, value):
        self._used = None  # remove possibly existing spectrumMask
        try:
            value = np.asarray(value, dtype=np.bool)
        except:
            logging.critical("Invalid input for background pixel mask: {0}".format(value))
        if value.size != self.used.shape[-1]:
            logging.critical(
                "Provided background spaxels spectrumMask has invalid size ({0} instead of {1}).".format(value.size,
                                                                                                 self.used.shape[-1]))
        self._used = self.used & value  # note call to 'self.used' leads to recalculation of included pixels based
        # on distance wrt. 'maxrad'

    used = property(get_used, set_used, None, "The mask indicating which spaxels are used to measure the background.")

    def get_flux(self, i):
        """
        Returns the indices of the used pixels and the corresponding weights
        for an individual background patch.

        Parameters
        ----------
        i : integer
            The index of the background patch for which spaxels and weights
            should be returned
        
        Returns
        -------
        data : tuple
            A tuple of length 2 containing the indices of the used spaxels 
            and the corresponding fluxes.
        """
        return self.spaxel[self.used[i]], self.f[i, self.used[i]]

    def get_sparse_matrix(self, offset=0):

        i, j = np.nonzero(self.f)
        values = self.f[(i, j)]

        # check if any background components do not contribute
        missing = np.in1d(np.arange(self.centroids.shape[0]), np.unique(i), assume_unique=True, invert=True)
        if missing.any():
            missing_components = np.arange(self.centroids.shape[0])[missing]
            logging.warning('Background contains components outside FoV: {0}'.format(
                ', '.join([str(comp) for comp in missing_components])))

        # IMPORTANT: to be consistent with the matrix of sources, we need to swap i and j because the source matrix
        # has shape (n_spaxel, n_components)
        return coo_matrix((values, (j, i + offset)), shape=(self.transform.shape[0], self.gridsize + offset))


# following code outdated. - SK 14/12/01
# def __call__(self, u, v):
#        """
#        The __call__ method of the Background class is intended to be provided
#        to the call of the scipy.distance.cdist routine. The idea is that this
#        routine is used to calculate weights instead of distances between two
#        points u and v.
#
#        Parameters
#        ----------
#        u : array_like
#            The coordinates of the first datapoint, can be N dimensional.
#        v : array_like
#            The coordinates of the second datapoint. Note than len(v)==len(u)
#            is required.
#
#        Returns
#        -------
#        weight : float
#            The weight of point v with respect to point u.
#        """
#        if self.rscale <= 0:
#            return 1
#        else:
#            return np.exp(-0.5*((u-v)**self.n).sum()/self.rscale**self.n) if np.sqrt(((u-v)**2).sum())<self.rcut else 0
#
#
#    def get_weights(self, data):
#        """
#        This method performs a call to scipy.distance.cdist using the funtion
#        defined by Background.__call__ as 'metric'.
#
#        Parameters
#        ----------
#        data : array_like, shape (m, 2)
#            The coordinates for which the weights in each background patch
#            should be calculated. `m' is the number of sources.
#
#        Returns
#        -------
#        weights : nd_array, shape (gridsize, m)
#            The weight of each provided datapoint on each background patch.
#        """
#        return distance.cdist(self.centroids, data, self)
#
#    def calculate_weights(self, data):
#        """
#        Basically a wrapper around the Background.get_weights method, but
#        instead of returning the weights, they are stored in a class
#        attribute. In contrast to the Background.get_weights method, the
#        weights are also normalized by this method.
#        """
#        self.weights  = self.get_weights(data)
#        self.weights /= self.weights.sum(axis=0)
#
#    def prepare_sparse(self, spectrumMask=None):
#        """
#        This method prepares a sparse matrix using the weights as values and
#        the spaxel and patch indices as column and row indices. Note that the
#        weights must have been calculated with the Background.calculate_weights
#        method before.
#
#        Parameters
#        ----------
#        spectrumMask : nd_array, optional
#            A spectrumMask indicating invalid spaxels/coordinates. Must have a length
#            equal to the number of spaxels for which weights were calculated.
#            The idea is that a single call to Background.calculate_weights -
#            which can be quite time-consuming for large arrays - can be
#            followed by many calls to this method even if some spaxels are
#            flagged as invalid. If None, it is assumed that all spaxels are
#            valid.
#
#        Returns
#        -------
#        A : sparse coo-matrix
#            The sparse matrix containing the weights.
#        """
#        assert hasattr(self, "weights"), "Need to calculate weights first."
#
#        if spectrumMask==None:
#            used = np.ones(self.weights.shape[1], dtype=np.bool)
#        else:
#            assert len(spectrumMask)==self.weights.shape[1], "Invalid spectrumMask size."
#            used = (spectrumMask==False)
#
#        coeffs   = []
#        indices0 = []
#        indices1 = []
#
#        for i in xrange(self.gridsize):
#            slc = used&(self.weights[i]>0)
#            coeffs.append(self.weights[i][slc])
#            indices0.append(np.cumsum(used)[slc]-1)
#            indices1.append(i*np.ones(slc.sum(), dtype=np.int32))
#
#        return sparse.coo_matrix((np.hstack(coeffs), (np.hstack(indices0), np.hstack(indices1))),
#                                 shape=(used.sum(), self.gridsize))
#
#
# class BackgroundPatch(Canvas):
#    
#    def __init__(self, xc, yc, rscale, maxrad, **kwargs):
#        # initialize parent class
#        super(BackgroundPatch, self).__init__(**kwargs)
#
#        # initialize basic properties
#        self.xc = xc
#        self.yc = yc
#        self.rscale = rscale
#        self.maxrad = maxrad
#            
#        self._x = None
#        self._y = None
#        self._r = None
#        self._used = None
#        self._f = None
#        
#        # All the arrays must eventually be recalculated if certain parameters are changed.
#        self.profile_properties = ["_used", "_x", "_y", "_r", "_f"]
#        
#    def reset(self, omit=[]):
#        """
#        When parameters of the profile are updated, all arrays that are affected
#        by the parameter must eventually be recalculated. To assure this, they
#        can be set to None using this method.
#        
#        Parameters
#        ----------
#        omit : list, optional
#            A list containing the arrays that should not be reset because they are
#            still valid with the updated profile parameters.
#        """
#        for prop in self.profile_properties:
#            if not prop in omit:
#                setattr(self, prop, None)
#
#    @property
#    def r(self):
#        """
#        Returns
#        -------
#        r : nd_array
#            A one dimensional array containing the distance of each spaxel
#            used in the calculation to the profile centroid.
#        """
#        if self._r == None:
#            self._r = np.sqrt((self.x-self.xc)**2 + (self.y-self.yc)**2)
#            self._r[self._r==0] = 1e-5
#        return self._r
#        
#    def get_f(self, **kwargs):
#        """
#        Method that calculates the actual intensity profile. Note that because
#        this class is not designed to compute an actual intensity profile, the
#        method is mainly a place-holder for the `get_f`-methods of the
#        individual sub-classes.
#
#        Parameters:
#        -----------
#
#        Returns:
#        --------
#        result : zero value integer
#            The return value is only used so that the method is fully
#            compatible with the `make_array`-methods of the individual
#            sub-classes.
#        """
#        if self._f == None:
#            self._f = 10**(-0.4*self.mag)/(2.*np.pi*self.std**2*(1.-self.e))*np.exp(-self.r**2/(2.*self.std**2)) \
#                + super(Gaussian, self).get_f(integrate=False)
#        else:
#            return self._f
#
#    f = property(get_f, None, None, "A place-holder for the intensity distribution of the PSF.")

