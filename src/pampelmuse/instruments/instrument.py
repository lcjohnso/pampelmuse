"""
instrument.py
=============
Copyright 2013-2024 Sebastian Kamann

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
This module provides the base class for all instrument classes defined in
PampelMuse. It should not be used to handle any real IFS data. Instead, it
defines generic methods that are inherited by all its subclasses.

Latest Git revision
-------------------
2024/07/09
"""
import logging
import warnings

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import path
from scipy.spatial import distance, ConvexHull
from astropy.nddata import NDData, VarianceUncertainty, NDUncertainty
from astropy.io import fits
from astropy.wcs import WCS
from ..filters import get_filter
from ..utils.distance import dist2path, point_in_poly


__author__ = "Sebastian Kamann (s.kamann@ljmu.ac.uk)"
__revision__ = 20240709


logger = logging.getLogger(__name__)


class Instrument(object):
    """
    The Instrument class serves as super-class for all instrument specific
    sub-classes defined in PampelMuse. Note that it only defines the framework
    used by the other classes and cannot be used to load any IFU data itself.
    """
    name = ""

    # define place holders for FITS HDUs containing IFS data
    primary_header = None
    datahdu = None
    varhdu = None
    maskhdu = None
    residualshdu = None

    nvoxel = None
    nlayer = None

    _wave = None  # The wavelength values of the individual layers in Angstrom

    _angle = None  # The rotation angle of the IFS on the sky, given in degrees(!)
    _scale = None  # The spaxel scale of the IFS on the sky, given in arcseconds^-1
    _shape = (np.nan, np.nan)  # The shape of the spaxel array

    _edgepath = None  # will be a matplotlib.path around the edge of the FoV

    # define very basic WCS instance
    wcs = WCS(naxis=3)
    wcs.wcs.cd = np.diagflat([1, 1, 1e-10])
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN', 'AWAV']
    wcs.wcs.cunit = ['deg', 'deg', 'm']

    _sky_fibers = []  # The indices of the sky fibers
    _cal_fibers = []  # The indices of the calibration fibers

    def __init__(self, transform=None, edge=None, bad=None, angle=None, scale=None, **kwargs):
        """
        Initialize an instance of the Instrument-class

        Parameters
        ----------
        transform : nd_array, optional
            Can be an array of shape (2, n_spaxel) that gives the y- and x-
            coordinates of the spaxels. Note that following the numpy counting,
            the first axis is the y-axis. If transform=None, it can be
            specified after initialization.
        edge : nd_array, optional
            Can be an array of integers that contains the indices of the
            spaxels that define the edge of the field of view. NOTE that the
            indices should be provided in such an order that the spaxels follow
            a loop around the field of view.
            If edge=None, it can be specified after initialization.
        bad : array_like, optional
            A list containing the indices of bad or dead spaxels.
        angle : float, optional
            The rotation angle of the IFS on the sky, given in degrees.
        scale : float, optional
            The spaxel scale of the IFS on the sky, given in arcseconds.
        """
        super(Instrument, self).__init__(**kwargs)

        self._transform = transform
        self._edge = edge

        # Handle bad spaxels (e.g., dead fibers)
        if bad is None:
            self.bad = []
        else:
            self.bad = list(bad)
        for spaxel in self.bad:
            logging.info("Excluding spaxel #{0}.".format(spaxel))

        if angle is not None:
            self.angle = angle

        if scale is not None:
            self.scale = scale

    @property
    def transform(self):
        """
        The getter method of the 'transform'-property

        Returns
        -------
        transform : nd_array
            The current transformation from spaxels to coordinates on the
            (x,y)-plane as an array of shape (N_spaxels, 2)
        """
        return self._transform

    @transform.setter
    def transform(self, value):
        """
        Set the transformation from spaxels to coordinates on the (x,y)-plane.
        Serves as the setter-method for the 'transform'-property

        Parameters
        ----------
        value : nd_array or None
            Unless 'value' is None, an array of shape (2, n_spaxel) that gives
            the y- and x-coordinates of all spaxels. Note that following the
            numpy counting, the first axis is the y-axis.
        """
        if value is None:
            self._transform = None
            return  # No transformation specified

        # Perform sanity checks on provided data
        if value.ndim != 2:
            logging.critical("'transform' must be 2dim., is {0}dim.".format(value.ndim))
        assert value.ndim == 2, "'transform' must be 2dim."

        if value.shape[1] != 2:
            logging.critical("'transform' must contain 2 spatial coordinates, has {0}.".format(value.shape[1]))
        assert value.shape[1] == 2, "'transform' must contain 2 spatial coordinates."

        self._transform = np.asarray(value, dtype=np.float32)

    @transform.deleter
    def transform(self):
        """
        Delete the current spaxel transformation
        """
        self._transform = None
        if self.edge is not None:  # selection of spaxels along edge must also be reset
            del self.edge

    @property
    def shape(self):
        """
        Returns the shape of the IFS spatial array. If the IFS has a fixed
        spatial shape (i.e. for fiber-feb instruments), it is returned.
        Otherwise, the shape is estimated from the spaxel-to-xy
        transformation.
        """
        if np.isfinite(self._shape).all():
            return self._shape
        elif self._transform is None:
            return None
        else:
            return int(round(self.transform[:, 0].max())), int(round(self.transform[:, 1].max()))

    @property
    def edge(self):
        """
        The getter-method of the 'edgepix'-property. Note that if the spaxels
        that define edge of FoV have not yet been provided but the 'transform'-
        property  is set, the code tries to infer the spaxels on the edge from
        the provided transformation from spaxels to x- & y-coordinates.
        """
        if self._edge is None and self.transform is not None:
            chull = ConvexHull(self.transform)
            self._edge = chull.vertices
        return self._edge

    @edge.setter
    def edge(self, value):
        """
        Set the spaxels that define the edge of the field of view. Serves as
        the setter-method for the 'edge'-property.

        Parameters
        ----------
        value : nd_array or None
            If 'value' is not None, a 1dim. array of integers containing the
            indices of the spaxels along the edge of the field of view.
            IMPORTANT: The order of the indices matters, it should be such
            that the indices follow a continuous path along the edge of the
            field of view.
        """
        if value is None:
            return  # edge not defined.

        # Perform sanity checks on provided data
        if self.transform is None:
            logging.error("Cannot set 'edge' unless 'transform' is specified.")
        assert self.transform is not None, "No transformation from spaxels to x- & y-coordinates available."

        if value.ndim != 1:
            logging.critical("'edge' must be 1dim., but is {0}dim.".format(value.ndim))
        assert value.ndim == 1, "'edge' must be 1dim."

        self._edge = value

        # need to recalculate the path around the edge
        self._edgepath = None

    @edge.deleter
    def edge(self):
        """
        Delete the current selection of spaxels on the edge of the FoV. This
        implies that the path along the edge must also be reset.
        """
        self._edge = None
        self._edgepath = None

    @property
    def edgepath(self):
        """
        Create a matplotlib.path.Path instance that follows the edges of
        the field of view. This instance can be used for plotting purposes
        and to check whether a point is inside or outside the FoV.

        Returns
        -------
        edgepath:  a matplotlib.path.Path instance
            The edges of the field of view.
        """
        # can only create Path if transformation and edge spaxels have been defined
        if self.edge is None:
            return None

        if self._edgepath is None:
            # estimate center of field of view
            center = np.median(self.transform, axis=0)

            # apply offset of +/-0.5 to spaxels along the edges of the FoV
            edgepos = np.asarray(self.transform[np.r_[self.edge, self.edge[0]]], dtype=np.float32)
            offsets = 0.5 * np.ones(edgepos.shape, dtype=np.float32)
            offsets[edgepos < center] *= -1
            edgepos += offsets

            # initialize path instance:
            t = matplotlib.transforms.Affine2D(np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]))
            self._edgepath = path.Path(edgepos).transformed(t)
        return self._edgepath

    @property
    def angle(self):
        """
        The getter method of the Instrument.rotation_angle property.
        """
        return self._angle

    @angle.setter
    def angle(self, value):
        """
        The setter method of the Instrument.angle property. Sets the rotation
        angle of the IFS on the sky. Note that the angle must be provided in
        degrees.

        Parameters
        ----------
        value : float
            The rotation angle in degrees.
        """
        try:
            self._angle = float(value)
            logging.info("Rotation angle of IFS on the sky is {0:.1f} degrees.".format(self._angle))
        except ValueError:
            logging.error("Illegal value for rotation angle: {0}".format(value))

        if self._scale is not None:  # update WCS instance
            self._make_cd()

    @property
    def scale(self):
        """
        The getter method of the Instrument.spaxel_scale property.
        """
        return self._scale

    @scale.setter
    def scale(self, value):
        """
        The setter method of the Instrument.scale property. Sets the spaxel
        scale of the IFS on the sky.

        Parameters
        ----------
        value : float
            The spaxel_scale, given per arcsecond.
        """
        try:
            self._scale = float(value)
            logging.info("Sampling of IFS is {0:.2f} spaxel/arcsec".format(self._scale))
        except ValueError:
            logging.error("Illegal value for spaxel scale: {0}".format(value))

        if self._scale is not None:  # update WCS instance
            self._make_cd()

    @property
    def nspaxel(self):
        """
        Returns
        -------
        nspaxel : int
            The number of spaxels of the current instance of 'Instrument'.
        """
        if self._transform is not None:
            return self.transform.shape[0]
        else:
            return None

    @property
    def x(self):
        """
        Returns
        -------
        x : nd_array
            The x-coordinates of the spaxels of the current instance of
            'Instrument'.
        """
        return self.transform[:, 1]

    @property
    def y(self):
        """
        Returns
        -------
        y : nd_array
            The y-coordinates of the spaxels of the current instance of
            'Instrument'.
        """
        return self.transform[:, 0]

    @property
    def sky_fibers(self):
        """
        The getter-method of the 'sky_fibers'-method.

        Returns
        -------
        sky_fibers : array_like
            A list containing the indices of the sky fibers.
        """
        return self._sky_fibers

    @sky_fibers.setter
    def sky_fibers(self, value):
        """
        Set the indices of the fibers used to observe the night sky
        simultaneously with the science data. This method serves a setter for
        the 'sky_fibers' property.

        Parameters
        ----------
        value : array_like
            A list containing the indices of the sky fibers
        """
        self._sky_fibers = value

    @property
    def cal_fibers(self):
        """
        The getter-method of the 'cal_fibers'-method.

        Returns
        -------
        cal_fibers : array_like
            A list containing the indices of the calibration fibers.
        """
        return self._cal_fibers

    @cal_fibers.setter
    def cal_fibers(self, value):
        """
        Set the indices of the fibers used to calibrate the data simultaneously
        with the observations of the science data. This method serves a setter
        for the 'cal_fibers' property.

        Parameters
        ----------
        value : array_like
            A list containing the indices of the calibration fibers.
        """
        self._cal_fibers = value

    @property
    def aux_fibers(self):
        """
        Returns
        -------
        aux_fibers : array_like
            A list containing the indices of all auxiliary fibers used either
            as sky or calibration fibers.
        """
        return np.asarray(np.sort(np.hstack((self.sky_fibers, self.cal_fibers))), dtype=np.int16)

    @property
    def naux(self):
        """
        Returns
        -------
        naux : int
            The number of auxiliary (= sky or calibration) fibers per datacube.
        """
        return len(self.aux_fibers)

    @property
    def nfiber(self):
        """
        Returns
        -------
        nfiber : int
            The total number of fibers (science + sky + calib.) of the instrument.
        """
        return self.naux + self.nspaxel

    @property
    def isscience(self):
        """
        Returns
        -------
        isscience : array_like
            A boolean array that is True if a science fibre is present at this
            index and False if an auxiliary fibre is located there.
        """
        _isscience = np.ones((self.nspaxel + self.naux,), dtype=bool)
        _isscience[self.aux_fibers] = False

        return _isscience

    @property
    def wave(self):
        """
        This property contains the wavelength in meters for every layer of the
        IFS data.
        """
        if self._wave is None:
            if self.nlayer is None or self.wcs.naxis < 3:
                return None
            _, _, self._wave = self.wcs.wcs_pix2world(1, 1, np.arange(self.nlayer, dtype=np.float64), 0)
            # if self.wcs.wcs.has_cd():
            #     self._wave *= self.wcs.wcs.cd[2, 2]
            # else:
            #     self.wave *= self.wcs.wcs.cdelt[2]
            # self._wave += self.wcs.wcs.crval[2]

        return self._wave

    @wave.setter
    def wave(self, value):
        """
        Set the wavelength value for each layer of the IFS data.

        Note that the wavelength scale is always assumed to be in meters,
        because astropy does automatically convert to meters when creating a
        WCS instance from the header of a FITS file.

        Parameters
        ----------
        value : array_like
            The wavelength values in meters.  The length of the array must
            match the number of layers in the IFS data.
        """
        self._wave = np.asarray(value, dtype=np.float64)
        self.nlayer = self._wave.size

    def _make_cd(self):
        """
        This method adds a CD matrix to the WCS instance defined inside the
        Instrument instance. To do so, it uses the information of the pixel
        scale and the rotation angle of the instrument (if available).
        """
        cdnew = np.diagflat([-self.scale/3600., self.scale/3600., self.wcs.wcs.cd[2, 2]])
        if self.angle is not None and self.angle != 0:
            c, s = np.cos(np.radians(self.angle)), np.sin(np.radians(self.angle))
            rot = np.matrix([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
            self.wcs.wcs.cd = np.dot(rot, cdnew)
        else:
            self.wcs.wcs.cd = cdnew

    def distance_to_spaxels(self, x, y, metric="euclidean"):
        """
        Calculate the distance of one or more points located at positions
        (x_i,y_i) to all spaxels of the current Instrument-instance.

        Parameters
        ----------
        x : float or array_like
            The x-position(s) of the source(s), in units of spaxels.
        y : float or array_like
            The y-position(s) of the source(s), in units of spaxels.
        metric : string
            The metric used to calculate the distances. For the different
            options and their meanings consult the documentation of
            scipy.distance.cdist. The default is 'euclidean'.

        Returns
        -------
        distances : float or array_like
            The distance(s) calculated using the provided metric and in units
            of spaxels.
        """
        return distance.cdist(self.transform, np.vstack((y, x)).T, metric=metric)

    def infov(self, xy):
        """
        Check if the provided (x,y)-positions are inside the field of view of
        the current Instrument-instance. The method uses the 'contains_point'-
        method of matplotlib.path.Path for this purpose. The this reason, the
        provided indices of 'edge' should form a polygon along the edge of the
        field of view.

        Parameters
        ----------
        xy : array_like, shape (N,2)
            The x-position(s) of the N source(s) on the (x,y) plane, in units
            of spaxels.

        Returns
        -------
        infov : array_like or bool
            A boolean flag for each source indicating whether or not it is
            inside the field of view.
        """
        return np.array([point_in_poly(x, y, self.edgepath.vertices) for x, y in xy], dtype=bool)

    def distance_to_edge(self, xy, metric=None):
        """
        Calculate the distance of one or more point in (x,y)-space to the
        edge of the field of view of the current Instrument instance.

        Parameters
        ----------
        xy : array_like, shape (N, 2)
            The position(s) of the N source(s) on the (x,y) plane, in units
            of spaxels.
        metric : string
            DEPRECATED

        Returns
        -------
        distance_to_edge : array_like or float
            The distance of each provided source to the edge of the field
            of view. Note that sources outside the field of view will have
            negative distances.
        """
        if metric is not None:
            logging.warning('The "metric" option to Instrument.distance_to_edge is deprecated.')

        if self.edgepath is None:
            logging.error('Cannot determine distances to edge of field of view because edge not defined.')

        return dist2path(xy, self.edgepath.vertices)

    def is_valid(self, hdu):
        """
        This method inspects if a provided FITS HDU contains data that is
        compatible with the current Instrument instance.

        Parameters
        ----------
        hdu : a FITS HDU containing some data
            The FITS HDU that should be checked.

        Returns
        -------
        validity_flag : boolean
            A flag indicating if the data in the HDU can be used with the
            current Instrument instance.

        """
        if not isinstance(hdu, fits.hdu.image._ImageBaseHDU):
            logging.error('Expected FITS image HDU, got {0}.'.format(type(hdu)))
            return False
        elif self.nvoxel is not None:
            if np.prod(hdu.shape) != self.nvoxel:
                logging.error('Size of new HDU differs from size of existing HDU(s).')
                return False
        return True

    def _set_datahdu(self, hdu):
        """
        This method sets the data HDU for the Instrument instance, i.e. it
        loads an HDU containing the observed data into the current instance.

        IMPORTANT: This method should not be used directly, instead, use the
        Instrument.open() method for loading data.

        Parameters
        ----------
        hdu : a FITS HDU containing the observed data.
        """
        if self.is_valid(hdu):
            self.datahdu = hdu
            self.nvoxel = np.prod(self.datahdu.shape)
            logging.info('Loaded IFS data from HDU <{0}> of file {1}'.format(
                self.datahdu.name, self.datahdu.fileinfo()['file'].name))
        else:
            logging.error('Data HDU <{0}> of file {1} has invalid format.'.format(
                    hdu.name, hdu.fileinfo()['file'].name))

    def _set_varhdu(self, hdu):
        """
        This method sets the variance HDU for the Instrument instance, i.e. it
        loads an HDU containing the variances for the observed data into the
        current instance. Note that it should be loaded AFTER the data itself.

        IMPORTANT: This method should not be used directly, instead, use the
        Instrument.open() method for loading data.

        Parameters
        ----------
        hdu : a FITS HDU containing the observed data.
        """
        if self.datahdu is not None and self.is_valid(hdu):  # HDU containing data must be loaded first
            self.varhdu = hdu
            logging.info('Loaded variances from HDU <{0}> of file {1}'.format(
                self.varhdu.name, self.varhdu.fileinfo()['file'].name))
        else:
            logging.error('Variance HDU <{0}> of file {1} has invalid format.'.format(
                    hdu.name, hdu.fileinfo()['file'].name))

    def _set_maskhdu(self, hdu):
        """
        This method sets the mask HDU for the Instrument instance, i.e. it
        loads an HDU containing a bad-pixel mask for the observed data into
        the current instance. Note that it should be loaded AFTER the data
        itself.

        IMPORTANT: This method should not be used directly, instead, use the
        Instrument.open() method for loading data.

        Parameters
        ----------
        hdu : a FITS HDU containing the bad-pixel mask.
        """
        if self.datahdu is not None and self.is_valid(hdu):  # HDU containing data must be loaded first
            self.maskhdu = hdu
            logging.info('Loaded bad-pixel mask from HDU <{0}> of file {1}'.format(
                self.maskhdu.name, self.maskhdu.fileinfo()['file'].name))
        else:
            logging.error('Mask HDU <{0}> of file {1} has invalid format.'.format(
                    hdu.name, hdu.fileinfo()['file'].name))

    def _set_residualshdu(self, hdu):
        """
        This method sets the residuals HDU for the Instrument instance, i.e.
        it loads an HDU containing residuals from a previous fit to the
        observed data into the current instance. Note that it should be loaded
        AFTER the data  itself.

        IMPORTANT: This method should not be used directly, instead, use the
        Instrument.open() method for loading data.

        Parameters
        ----------
        hdu : a FITS HDU containing the fit residuals.
        """
        if self.datahdu is not None and self.is_valid(hdu):  # HDU containing data must be loaded first
            self.residualshdu = hdu
            logging.info('Loaded residuals from HDU <{0}> of file {1}'.format(
                self.residualshdu.name, self.residualshdu.fileinfo()['file'].name))
        else:
            logging.error('Residuals HDU <{0}> of file {1} has invalid format.'.format(
                    hdu.name, hdu.fileinfo()['file'].name))

    def open(self, filename, extno=0, isvar=False, ismask=False, isresiduals=False, **kwargs):
        """
        This method is designed to fill the Instrument() instance with some
        observed data. To this aim, provide the name of the FITS file that
        contains the data. Note that only one HDU can be handled at a time.
        To load the various components of a data set (data, variances,
        bad-pixel mask), use multiple calls to open().

        Parameters
        ----------
        filename : string
            The name of the FITS file containing the data.
        extno : int, optional
            The extension of the file containing the data that should be
            opened.
        isvar : boolean, optional
            A flag indicating if the extension contains the variances instead
            of the data itself.
        ismask : boolean, optional
            A flag indicating if the extension contains a bad-pixel mask
            instead of the data itself.
        isresiduals : boolean, optional
            A flag indicating if the extension contains the residuals from a
             previous fit to the data instead of the data itself.
        kwargs
            Any additional keyword argument is passed on to the call of
            fits.open() that opens the FITS file.
        """
        fitsfile = fits.open(filename, **kwargs)
        if ismask:
            self._set_maskhdu(fitsfile[extno])
        elif isvar:
            self._set_varhdu(fitsfile[extno])
        elif isresiduals:
            self._set_residualshdu(fitsfile[extno])
        else:
            self._set_datahdu(fitsfile[extno])

    def get_spectrum(self, i, **kwargs):
        """
        This method returns a single spectrum (i.e. the content of a single
        spaxel) from the data.
        IMPORTANT: This method should be redefined by every subclass that
        inherits from the Instrument class.

        Parameters
        ----------
        i : int
            The index of the spaxel for which the spectrum should be returned
        kwargs
            Used for consistency.

        Returns
        -------
        spectrum : ndarray
            The spectrum in the requested spaxel.
        variances : ndarray
            The variances of the spectrum in the requested spaxel.
        """
        return np.array([]), np.array([])

    def _verify_layer_range(self, i_min: int = None, i_max: int = None,
                            weights: np.ndarray = None) -> (int, int, np.ndarray):
        """

        Parameters
        ----------
        i_min
        i_max
        weights

        Returns
        -------

        """
        if i_min is None and weights is None:
            raise ValueError('Must provide either `i_min` or `weights`.')
        if i_min is not None and i_max is None:
            i_max = i_min + 1
        if weights is None:
            weights = np.ones(i_max - i_min)
        else:
            if i_min is not None and len(weights) != (i_max - i_min):
                raise ValueError('Len(weights) does not match the number of layers in `i_max` - `i_min`.')
            elif i_min is None and len(weights) != self.nlayer:
                raise ValueError('Len(weights) must match total number of layers unless `i_min` provided.')
            elif i_min is None:
                i_min = 0
                i_max = self.nlayer
        return i_min, i_max, weights

    def weights_from_filtercurve(self, passband: str, layer_range: tuple = None,
                                 min_weight: float = 0.1) -> (int, int, np.ndarray):
        """
        Assign weights to the individual layers of the iFS data based on the
        throughput curve of a photometric filter.

        Parameters
        ----------
        passband
            The name of the filter over which to integrate. Note that it must
            be available in the set of filters available to PampelMuse. Check
            the PampelMuse manual for details.
        layer_range
            The min. and max layer of the IFS data that should be considered.
            This is helpful in cases when the first/last layer of the data is
            corrupted.
        min_weight
            The minimumm weight of a layer considered. This is helpful in
            cases where the passbands have long tails witj low throughput.

        Returns
        -------
        int
            The index of the first layer overlapping with the filter curve.
        int
            The index of the last layer overlapping with the filter curve.
        np.ndarray
            The weights assigned to the individual layers based on the
            throughput of the passband.
        """
        if self.wave is None:
            raise AttributeError("Missing wavelength information for IFS data.")

        # open filter curve, check if it overlaps with IFs data
        fct = get_filter(passband)

        # note that IFS data are in m while filter curves are currently in Angstrom
        w_min = 1e-10*fct.get_knots().min()  # min. of defined wavelength range
        i_min = np.argmax(self.wave > w_min)

        w_max = 1e-10*fct.get_knots().max()  # max. of defined wavelength range
        i_max = self.wave.size - 1 - np.argmax(self.wave[::-1] < w_max)
        logger.debug('Wavelength range of filter {}: {:.3g}-{:.3g}'.format(passband, w_min, w_max))
        logger.debug('Wavelength range of IFS data: {:.3g}-{:.3g}'.format(self.wave.min(), self.wave.max()))

        if layer_range is not None:
            i_min = max([i_min, layer_range[0]])
            i_max = min([i_max, layer_range[1]])

        if min_weight > 0:
            while fct(1e10*self.wave[i_min]) < min_weight:
                i_min += 1
            while fct(1e10*self.wave[i_max]) < min_weight:
                i_max -= 1

        if i_min >= i_max or self.wave.max() < w_min or self.wave.min() > w_max:
            logging.error("No overlap between provided wavelength range and passband {0}.".format(passband))
            return i_min, i_max, np.zeros(max(i_max - i_min, 0))
        else:
            weights = fct(1e10*self.wave[i_min:i_max])
            assert not np.isnan(weights).any()

        return i_min, i_max, weights

    def get_layer(self, i_min: int = None, i_max: int = None, weights: np.ndarray = None,
                  use_variances: int = False, return_residuals: bool = False, **kwargs) -> pd.DataFrame:
        """
        This method returns a single layer (i.e. the content of all spaxels
        with a fixed wavelength) from the data as a Pandas DataFrame.

        IMPORTANT: This method should be redefined by every subclass that
        inherits from the Instrument class.

        In case a range of layers is requested, the layers are combined using
        internally before the data for the combined layer are returned. It is
        possible to either assign different weights to each layer or use the
        variances (when available) in the combination process.

        Parameters
        ----------
        i_min
            The index indicating the start of the layer range to be extracted.
        i_max
            The index indicating the end of the layer range to be extracted.
        weights
            The weights assigned to the individual layers when combining them.
            Only used when `use_variances` is set to False.
        use_variances
            Flag indicating whether to use the variances when combining
            layers. If True, inverse-variance-weighting is used when combining
            the individual layers and the `variance` values provided in the
            output are obtained through error propagation. Otherwise,
            (weighted, if the `weights` parameter is used) averages of both
            input values and variances are returned.
        return_residuals
            Flag indicating whether to return the residuals instead of the
            data.
        kwargs
            Used for consistency.

        Returns
        -------
        pd.DataFrame
            The DataFrame containing the requested data in four columns: `x`,
            `y`, `value`, and `variance`. If more than one layer from the IFS
            data were requested, they are combined internally, such that each
            spaxel has a unique value.
        """
        _ = self._verify_layer_range(i_min, i_max, weights)

        return pd.DataFrame(columns=['x', 'y', 'value', 'variance'], dtype=np.float64)

    def get_image(self, i_min: int = None, i_max: int = None, weights: np.ndarray = None,
                  use_variances: bool = False, return_residuals: bool = False, **kwargs) -> NDData:
        """
        This method returns the data from a single layer or a range of layers
        as a 2dim image.

        In case more than one layer is requested, the data from the different
        layers are combined internally prior to creating the 2dim array. It is
        possible to either assign different weights to the individual layers,
        or to use the variances (if available) during the combination.

        Parameters
        ----------
        i_min
            The index indicating the start of the layer range to be extracted.
        i_max
            The index indicating the end of the layer range to be extracted.
        weights
            The weights assigned to the individual layers when combining them.
        use_variances
            Flag indicating whether to use the variances when combining
            layers.
        return_residuals
            Flag indicating whether to return the residuals instead of the
            data.
        kwargs
            Used for consistency.

        Returns
        -------
        NDData
            The requested data as an astropy NDData object. The variances of
            the data can be accessed via the `uncertainty` attribute.

        Raises
        -------
        NotImplementedError
            If the call to `get_layer()` returns an empty DataFrame.
        """
        layer_data = self.get_layer(i_min, i_max, weights, use_variances, **kwargs)

        if not len(layer_data):
            raise NotImplementedError

        data = NDData(self.make_image(layer=layer_data['value']), **kwargs)
        if use_variances:
            data.uncertainty = VarianceUncertainty(self.make_image(layer=layer_data['variance']), **kwargs)

        return data

    def integrate(self, imin: int, imax: int, apply_variances: bool = False, weights: np.ndarray = None,
                  nbin: int = 1, **kwargs) -> (np.ndarray, np.ndarray):
        """
        DEPRECATED: This method is deprecated and will be removed in a future release.

        This method is designed to obtain an image by integrating over a range
        of individual layers.

        Parameters
        ----------
        imin
            The index of the first layer to be used in the integration.
        imax
            The index of the last layer to be used in the integration.
        apply_variances
            A flag indicating if the layers should be weighted according to
            their variances in the combination.
        weights
            An additional weight for each value (e.g. the throughput of a
            filter curve at the wavelength of the layer). Note that this
            weights are multiplied to the variances if apply_variances is
            set to True.
        nbin
            A binning that may be used to speed up the integration process.
            If set to a value larger than one, only every 'nbin'th layer is
             considered.
        kwargs
            Any additional keyword argument is passed on to the call of
            Instrument.get_layer() used to obtain the data for each layer.

        Returns
        -------
        ndarray
            The array containing the integrated data.
        ndarray
            An estimate of the variance in each data pixel. Note that if no
            variances or weights are used in the integration, this array will
            contain 1/(no. of integrated pixels) for each output pixel.
        """
        warnings.warn("The `integrate` method is deprecated and will be removed in a future release.",
                      DeprecationWarning)

        # make sure variances are returned with data if variance weighting is requested.
        if apply_variances:
            kwargs['return_var'] = True

        if weights is not None:
            if len(weights) != (imax - imin):
                raise IOError('Length of weights array must be equal to number of layers.')
        else:
            weights = np.ones(imax - imin, dtype=np.int16)

        # initialize output data from first layer
        data, var = self.get_image(i=imin, **kwargs)
        # if variances are not used, give each valid pixel a value of one.
        if var is None:
            var = np.where(np.isfinite(data), 1., np.nan)
        var /= weights[0]

        data /= var
        wsum = 1./var

        for i in range(imin + 1, imax):

            # if requested, only every Nth layer is used to speed up process
            if (i - imin) % nbin > 0:
                continue

            newdata, newvar = self.get_image(i=i, **kwargs)[:2]
            if newvar is None:
                newvar = np.where(np.isfinite(newdata), 1., np.nan)
            newvar /= weights[i - imin]

            data = np.nansum([data, newdata/newvar], axis=0)
            wsum = np.nansum([wsum, 1./newvar], axis=0)

            logging.debug('Completed layer #{0} [{1}/{2}]'.format(i, i - imin, imax - imin))

        wsum[wsum == 0] = np.nan
        return data/wsum, 1./wsum

    def slice_for_layer(self, i):
        """
        This function provides a slice that can be applied to the full data
        array of any HDU to extract an individual layer from it.

        Parameters
        ----------
        i : int
            The index of the layer.

        Returns
        -------
        slice : a slice object
            The slice that returns the layer if applied to the data.
        """
        return Ellipsis

    def make_image(self, data: np.ndarray = None, layer: pd.Series = None, **kwargs) -> np.ndarray:
        """
        This method is designed to create a 2D image from a provided
        1D array.

        Parameters
        ----------
        data
            A 1dim array to be transformed to a 2dim. image. Note that the
            length of the data must match the number of spaxels in the
            instance.
        layer
            The data to be transformed, provided as a pandas Series instead.
             Contrary to data`, spaxels may be missing in `layer`. Values in
             layer are assigned to output spaxels based on the index of the
             data series.
        kwargs
            Used for consistency.

        Returns
        -------
        np.ndarray
            The transformed 2dim. data.
        """
        if data is None and layer is None:
            raise IOError("Either 'data' or 'layer' must be provided.")
        elif data is not None:
            if layer is not None:
                warnings.warn("Ignoring provided 'layer', as 'data' is provided.")
            elif data.size != self.nspaxel:
                logger.error('Unable to create image: Size of data ({0} differs from no. of spaxels ({1}).'.format(
                    data.size, self.nspaxel))

        out_data = np.zeros(self.shape, dtype=np.float32)
        out_data[:] = np.nan

        y = self.y.astype('<i4')
        x = self.x.astype('<i4')
        if data is not None:
            out_data[y, x] = data
        else:
            y = self.y.astype('<i4')[layer.index]
            x = self.x.astype('<i4')[layer.index]
            out_data[y, x] = layer.values
        return out_data

    def search_configuration(self, hdu, **kwargs):
        """
        This method is designed to update the configuration of the current
        instance using specific header keywords from a given HDU.

        Parameters
        ----------
        hdu : a FITS HDU
            The HDU containing the header to be checked for the keywords.
        kwargs
            Used for consistency.
        """
        # see if WCS coordinates of pointing can be derived from header
        if 'RA' in hdu.header and 'DEC' in hdu.header and (self.wcs.wcs.crval[:2] == 0).all():
            try:  # note that currently only floats can be used
                self.wcs.wcs.crval[:2] = [hdu.header['RA'], hdu.header['DEC']]
            except ValueError:
                pass
