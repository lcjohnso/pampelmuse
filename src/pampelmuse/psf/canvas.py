"""
profiles.py
===========
Copyright 2013-2018 Sebastian Kamann

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
class pampelmuse.core.psf.profiles.Canvas

Purpose
-------
This module implements the Canvas-class which is needed to calculate a spatial
profile (such as a PSF or a background patch) on an irregularly shaped pixel
grid.

Latest Git revision
-------------------
2018/04/20
"""
import logging
import numpy as np


__author__ = "Sebastian Kamann (s.kamann@ljmu.ac.uk)"
__revision__ = 20180420


class Canvas(object):
    """
    This class does provides a frame for working on pixel arrays. The arrays
    may be irregularly sampled and must be initialized via directly providing
    the y- and x-position of each pixel.
    """

    def __init__(self, transform=None, **kwargs):
        """
        Initialize a new instance of the 'Canvas'-class.

        Parameters
        ----------
        transform : ndarray, optional
            This parameter may be used to provide the transformation from
            pixels to spatial positions as an array of shape (n, 2) for n
            pixels.

        Returns
        -------
        canvas : The newly created instance.
        """
        # initialize basic properties
        self._transform = transform

        # super(Canvas, self).__init__(**kwargs)

    @classmethod
    def from_shape(cls, shape, **kwargs):
        """
        This method can be used to initialize a new instance of the Canvas
        class from a size tuple, containing the grid dimensions along x and
        y. In this case, a homogeneous spatial sampling is assumed with a
        pixel size of 1 along x and y.

        Parameters
        ----------
        shape : tuple
            The shape of the canvas to create.
        kwargs
            Any additional keyword arguments are passed on to the call of
            Canvas.__init__.

        Returns
        -------
        canvas : The newly created instance.
        """
        assert len(shape) == 2, 'Parameter "shape" must be a tuple of length 2.'

        n_pixel = np.prod(shape)
        transform = np.vstack((np.arange(n_pixel, dtype=np.float32) // shape[1],
                               np.mod(np.arange(n_pixel, dtype=np.float32), shape[1]))).T
        return cls(transform=transform, **kwargs)

    @property
    def transform(self):
        """
        Returns the current transformation from pixels to x- & y-coordinates.
        """
        return self._transform

    @transform.setter
    def transform(self, value):
        """
        Specify the transformation from pixels to x-& y-coordinates.

        Parameters
        ----------
        value : nd_array
            An array specifying the x- and y-coordinate of each pixel, having
            size (n, 2) for n pixels. Note that following the numpy counting,
            the first number gives the size in y-direction.
        """
        if value.ndim != 2:
            logging.critical("'transform' must be 2dim., is {0}dim.".format(value.ndim))
        assert value.ndim == 2, "'transform' must be 2dim."

        if value.shape[1] != 2:
            logging.critical("'transform' must contain 2 spatial coordinates, is {0}.".format(value.shape[1]))
        assert value.shape[1] == 2, "'transform' must contain 2 spatial coordinates."

        # set basic class properties
        self._transform = value

        # reset all properties calculates so far in this instance that inherit from the transformation
        self.reset()

    def reset(self, **kwargs):
        """
        The is nothing to reset in this class. However, classes that inherit
        from 'Canvas' calculate arrays that must be recalculated if the size
        or the pixel positions of the array change.
        """
        return

    @property
    def pixel(self):
        """
        Return the indices of the individual pixels
        """
        return np.arange(self.transform.shape[0], dtype=np.int32)

    @property
    def n_pixel(self):
        """
        Returns the number of pixels on the canvas.
        """
        return self.transform.shape[0]

    @property
    def x(self):
        """
        Returns the x-coordinate of each pixel.
        """
        return self.transform[:, 1]

    @property
    def y(self):
        """
        Returns the y-coordinate of each pixel.
        """
        return self.transform[:, 0]

    @property
    def shape(self):

        return int(self.y.max() - self. y.min()), int(self.x.max() - self. x.min())
