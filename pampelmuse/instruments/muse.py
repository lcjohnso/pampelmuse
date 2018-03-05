"""
muse.py
=======
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
This module defines two classes that are designed to handle MUSE data. The
class 'Muse' enables the analysis of data cubes saa they are produced by the
muse_scipost and muse_exp_combine recipes of the MUSE pipeline. The class
'MusePixtable' enables the analysis of pixtables as they are produced by the
muse_scipost recipe.

Latest GIT revision
-------------------
2018/03/05
"""
import logging
import warnings
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull
from .instrument import Instrument
from .generic_ifs import GenericIFS


__author__ = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__revision__ = 20180305


class Muse(GenericIFS):
    """
    This class is designed to handle data observed with the MUSE instrument
    (Bacon et al. 2010, SPIE, 7735, 7), mounted at UT4 of the Very Large
    Telescope. Note that this class works on cubes, support for MUSE pixtables
    is provided via the MusePixtable class.
    """
    name = "MUSE"

    def __init__(self, **kwargs):
        """
        Initialize an instance of the MUSE class.

        Parameters
        ----------
        kwargs are passed to the initialisation os the GenericIFS super-class.

        Returns
        -------
        The newly created instance.
        """
        super(Muse, self).__init__(**kwargs)

    @Instrument.edge.setter
    def edge(self, value):
        """
        To set the edge for a MUSE data cube, the convex hull around the 2D
        array of valid pixels is determined.

        Parameters
        ----------
        value : nd_array
            A 1dim. array containing a boolean flag for each spaxel that
            indicates whether the spaxel contains valid data.

        Returns
        -------
        edge : nd_array, int
            The indices of the spaxels that define the edge of the field of
            view.
        """
        if self.transform is None:
            logging.error("No transformation from spaxels to x & y available.")
            return

        if len(value) != self.nspaxel:
            logging.error("Size of boolean array [{0}] must match number of spaxels [{1}].".format(
                    len(value), self.nspaxel))
            return

        chull = ConvexHull(self.transform[np.asarray(value, dtype=np.bool)])
        self._edge = chull.vertices

    def find_edge(self, valid):
        """
        DEPRECATED

        The data structure of MUSE is such that the field of view is not a
        fixed property of the instrument, but that each pointing comes with
        a distinct number and arrangement of spaxels. Even for a single
        exposure, the field of view will usually vary from one layer to the
        next. For this reason, the edge of the field of view is determined
        by providing a boolean array that contains a flag for each spaxel
        that indicates whether it contains valid data or NaN. The routine
        uses this information to determine which spaxels define the edge of
        the field of view.

        Parameters
        ----------
        valid : nd_array
            A 1dim. array containing a boolean flag for each spaxel that
            indicates whether the spaxel contains valid data.
        """
        warnings.warn('The method Muse.find_edge is deprecated', DeprecationWarning)

        logging.info("Determining edge of MUSE field of view...")

        if self.transform is None:
            logging.error("No transformation from spaxels to x & y available.")
            return

        if len(valid) != self.nspaxel:
            logging.error("Size of boolean array [{0}] must match number of spaxels [{1}].".format(
                    len(valid), self.nspaxel))
            return

        # first scan through the spaxels along the y and the x axis, find the spaxel with the lowest and the highest
        # x resp. y value. Their indices are combined in the list 'ind'.
        ind = []
        for yi in range(int(self.y.min()), int(self.y.max())):
            select = np.flatnonzero(valid & (self.y >= yi) & (self.y < (yi + 1)))
            if not select.any():
                continue
            j1 = self.x[select].argmin()
            j2 = self.x[select].argmax()
            ind.extend([select[j1], select[j2]])
        for xi in range(int(self.x.min()), int(self.x.max())):
            select = np.flatnonzero(valid & (self.x >= xi) & (self.x < (xi + 1)))
            if not select.any():
                continue
            j1 = self.y[select].argmin()
            if j1 not in ind:
                ind.append(select[j1])
            j2 = self.y[select].argmax()
            if j2 not in ind:
                ind.append(select[j2])

        # for each spaxel included in 'ind', calculate the angle wrt. the centre of gravity, then sort the spaxels by
        # increasing angle so that the polygon follows the edge of the field of view.
        center = self.transform[ind].mean(axis=0)
        angles = np.arctan2(self.x[ind] - center[0], self.y[ind] - center[1])

        self.edge = np.asarray(ind, dtype=np.int32)[angles.argsort()]


class MusePixtable(Instrument):
    """
    This class is designed to handle data observed with the MUSE instrument
    (Bacon et al. 2010, SPIE, 7735, 7), mounted at UT4 of the Very Large
    Telescope. Note that this class works on pixtables, support for MUSE data
    cubes is provided via the Muse class.
    """
    name = "MUSE_Pixtable"
    issorted = True

    _layer_limits = None  # place-holders for indices defining individual layers
    _limit_indices = None

    xhdu = None  # place-holders for HDUs containing x- & y-positions (& optionally the IFU and slice number).
    yhdu = None
    ifuhdu = None
    slicehdu = None

    dqmax = 4
    deg2rad = np.pi/180.

    rawwave = None
    rawwave_for_current_layer = None
    wstep = 1.25e-10  # Dispersion in m/pixel

    ifus_for_current_layer = None
    slices_for_current_layer = None

    def __init__(self, **kwargs):
        """
        Initialize an instance of the MUSE class.

        Parameters
        ----------
        kwargs are passed to the initialisation os the GenericIFS super-class.

        Returns
        -------
        The newly created instance.
        """
        super(MusePixtable, self).__init__(**kwargs)

    def _set_xhdu(self, hdu):
        """
        This method sets the HDU with the x-coordinates for the MusePixtable
        instance, i.e. it loads an HDU containing the coordinates for the
        observed data into the current instance. Note that it should be loaded
        AFTER the data itself.

        IMPORTANT: This method should not be used directly, instead, use the
        MusePixtable.open() method for loading data.

        Parameters
        ----------
        hdu : a FITS ImageHDU containing the coordinates.
        """
        if self.datahdu is None:  # HDU containing data must be loaded first
            logging.error('Can only load HDU containing x-positions after data HDU has been loaded.')
        elif not self.is_valid(hdu):
            logging.error('x-positions HDU [{0}] of file {1} has invalid format.'.format(
                    hdu.name, hdu.fileinfo()['file'].name))
        else:
            self.xhdu = hdu
            logging.info('Loaded x-coordinates from HDU <{0}> of file {1}'.format(
                self.xhdu.name, self.xhdu.fileinfo()['file'].name))

    def _set_yhdu(self, hdu):
        """
        This method sets the HDU with the y-coordinates for the MusePixtable
        instance, i.e. it loads an HDU containing the coordinates for the
        observed data into the current instance. Note that it should be loaded
        AFTER the data itself.

        IMPORTANT: This method should not be used directly, instead, use the
        MusePixtable.open() method for loading data.

        Parameters
        ----------
        hdu : a FITS ImageHDU containing the coordinates.
        """
        if self.datahdu is None:  # HDU containing data must be loaded first
            logging.error('Can only load HDU containing y-positions after data HDU has been loaded.')
        elif not self.is_valid(hdu):
            logging.error('y-positions HDU [{0}] of file {1} has invalid format.'.format(
                    hdu.name, hdu.fileinfo()['file'].name))
        else:
            self.yhdu = hdu
            logging.info('Loaded y-coordinates from HDU <{0}> of file {1}'.format(
                self.yhdu.name, self.yhdu.fileinfo()['file'].name))

    def _set_ifuhdu(self, hdu):
        """
        This method sets the HDU with the IFU number of every pixel for the
        MusePixtable instance, i.e. it loads an HDU containing the IFU numbers
        for the observed data into the current instance. Note that it should
        be loaded AFTER the data itself.

        IMPORTANT: This method should not be used directly, instead, use the
        MusePixtable.open() method for loading data.

        Parameters
        ----------
        hdu : a FITS ImageHDU containing the IFU numbers.
        """
        if self.datahdu is None:  # HDU containing data must be loaded first
            logging.error('Can only load HDU containing IFU numbers after data HDU has been loaded.')
        elif not self.is_valid(hdu):
            logging.error('IFU-number HDU [{0}] of file {1} has invalid format.'.format(
                    hdu.name, hdu.fileinfo()['file'].name))
        else:
            self.ifuhdu = hdu
            logging.info('Loaded IFU numbers from HDU <{0}> of file {1}'.format(
                self.ifuhdu.name, self.ifuhdu.fileinfo()['file'].name))

    def _set_slicehdu(self, hdu):
        """
        This method sets the HDU with the slice number of every pixel for the
        MusePixtable instance, i.e. it loads an HDU containing the slice
        numbers for the observed data into the current instance. Note that it
        should be loaded AFTER the data itself.

        IMPORTANT: This method should not be used directly, instead, use the
        MusePixtable.open() method for loading data.

        Parameters
        ----------
        hdu : a FITS ImageHDU containing the slice numbers.
        """
        if self.datahdu is None:  # HDU containing data must be loaded first
            logging.error('Can only load HDU containing slice numbers after data HDU has been loaded.')
        elif not self.is_valid(hdu):
            logging.error('Slice-number HDU [{0}] of file {1} has invalid format.'.format(
                    hdu.name, hdu.fileinfo()['file'].name))
        else:
            self.slicehdu = hdu
            logging.info('Loaded slice numbers from HDU <{0}> of file {1}'.format(
                self.slicehdu.name, self.slicehdu.fileinfo()['file'].name))

    def open(self, filename, isresiduals=False, **kwargs):
        """
        This method opens a FITS file containing a MUSE pixtable. Note that
        in contrast to the other instrument classes, not a single HDU is
        opened, but that the code expects the following HDUs to be available
        in the file. Instead, the code expects the following HDU to be present
        and will assign them automatically to the right properties: 'data',
        'stat', 'dq', 'xpos', and 'ypos'.

        IMPORTANT: The code checks if the pixtable values are sorted by
        increasing wavelength because only this format is currently supported.
        To sort a given pixtable produced by the MUSE pipeline, consider using
        the PampelMuse routine PTSORT.

        Parameters
        ----------
        filename : string
            The name of the FITS file containing the pixtable.
        isresiduals : boolean, optional
            A flag indicating if the FITS file contains the residuals from a
            previous analysis. In that case only the first HDU is considered.
            Note that the HDU containing the original data must have been
            loaded before.
        kwargs
            Any additional keyword arguments are passed on to the call of the
            parent class.
        """
        # The residuals are stored in a separate file. Therefore the behaviour depends on whether residuals or a
        # full pixtable (containing data, positions, ...) is loaded
        if isresiduals:
            return super(MusePixtable, self).open(filename, isresiduals=True, **kwargs)

        pt = fits.open(filename, memmap=False)

        # initialize WCS instance
        self.wcs = WCS(pt[0].header)
        self.wcs.wcs.crval = [pt[0].header['RA'], pt[0].header['DEC']]
        self.wcs.wcs.crpix = [self.wcs.wcs.crpix[0] - pt[0].header['ESO DRS MUSE PIXTABLE PREDAR LIMITS XLO'],
                              self.wcs.wcs.crpix[1] - pt[0].header['ESO DRS MUSE PIXTABLE PREDAR LIMITS YLO']]

        # Note that for the MUSE pixtables two wavelength ranges are defined; MusePixtable.rawwave contains one value
        # per pixel while MusePixtable.wave contains one value per layer. As the pixel tables are provided in units of
        # Angstrom, we need to convert to meters
        logging.info('Reading wavelength array ...')
        self.rawwave = 1e-10*pt['lambda'].data[:, 0]
        logging.info('... done.')

        # check if wavelength array is strictly increasing
        logging.info('Checking consistency of pixtable ...')
        if (np.diff(self.rawwave) >= 0).all():
            self.issorted = True
        else:
            # self.issorted = False
            logging.error('Pixtable is not sorted by increasing wavelength.')
            logging.error('Currently support exists only for sorted pixtables.')
            logging.error('To sort this pixtable, you may run "PampelMuse PTSORT {0}".'.format(filename))
            raise IOError('Pixtable data not sorted by wavelength.')
        logging.info('... done.')

        # fill other HDU instances with data
        self._set_datahdu(pt['data'])
        self._set_varhdu(pt['stat'])
        self._set_maskhdu(pt['dq'])
        self._set_xhdu(pt['xpos'])
        self._set_yhdu(pt['ypos'])
        if 'ifu' in pt:
            self._set_ifuhdu(pt['ifu'])
        if 'slice' in pt:
            self._set_slicehdu(pt['slice'])

        # define default layers for this cube. This may be changed by the user later on
        layers = np.linspace(self.rawwave.min(), self.rawwave.max(),
                             int((self.rawwave.max() - self.rawwave.min())/self.wstep), dtype=np.float32)
        self.layer_limits = np.vstack((layers[:-1], layers[1:])).T

        # a pixel table has no fixed transformation from spaxels to spatial coordinates, it changes for every layer
        # that is defined. Get coordinate transformation for central layer to have useful idea of edges of FoV.
        self.set_transform_for_layer(len(layers)//2)

    def _convert_coordinates(self, xpos, ypos):
        """
        This method is mainly a python version of the c-routine written by
        P. Weilbacher for the MUSE pipeline (muse_wcs_position_celestial in
        muse_wcs.c). It converts the native spherical coordinates found in
        the pixel table to RA and DEC coordinates using a TAN projection.

        Parameters
        ----------
        xpos : array_like
            The x-coordinates of the pixels as native spherical coordinates.
        ypos : array_like
            The y-coordinates of the pixels as native spherical coordinates.

        Returns
        -------
        ra : array_like
            The RA coordinates of the pixels in a TAN projection.
        dec : array_like
            The Dec coordinates of the pixels in a TAN projection.
        """
        dp = self.wcs.wcs.crval[1]*self.deg2rad

        phi = xpos
        theta = ypos + np.pi/2.

        ra = np.arctan2(np.cos(theta)*np.sin(phi),
                        np.sin(theta)*np.cos(dp) + np.cos(theta)*np.sin(dp)*np.cos(phi))
        ra /= self.deg2rad

        dec = np.arcsin(np.sin(theta)*np.sin(dp) - np.cos(theta)*np.cos(dp)*np.cos(phi))
        dec /= self.deg2rad

        return ra + self.wcs.wcs.crval[0], dec

    @property
    def layer_limits(self):
        """
        Returns the wavelengths that define the edges between individual
        layers in the data as an array of shape (MusePixtable.nlayer, 2).
        For each layer, the min. and max. wavelength must be provided.
        """
        return self._layer_limits

    @layer_limits.setter
    def layer_limits(self, value):
        """
        This method sets the edges between individual layers of the cube.
        Because each data point in a pixtable has a different wavelength, the
        principle of having layers with a fixed wavelength cannot be applied
        directly. Instead, the user can define a wavelength range for each
        layer. To do so, the min. and  max. wavelength for each layer should
        be provided as a (MusePixtable.nlayer, 2) array. All data points with
        a wavelength within this range will be combined to a pseudo-layer.

        Parameters
        ----------
        value : array_like
            The min. and max. wavelength of each layer as an array of shape
            (MusePixtable.nlayer, 2)
        """
        if self.datahdu is None:
            return

        value = np.asarray(value, dtype=np.float32)
        assert value.ndim == 2, 'List of layer limits for pixtable must be 2dim., not {0}dim.'.format(value.ndim)
        assert value.shape[1] == 2, 'Last axis of list with layer limits must have length 2.'

        self._layer_limits = value

        # The following call to searchsorted will return the indices of the data where a layer starts/ends,
        # again as an array of shape (MusePixtable.nlayer, 2)
        self._limit_indices = np.searchsorted(self.rawwave, self._layer_limits)

        # we define a mean wavelength for each layer to be consistent with the other instrument classes.
        self.wave = np.mean(value, axis=1)

    @property
    def layer_indices(self):
        """
        Returns the indices that define the start and end of each pseudo-layer
        as an array of shape (MusePixtable.nlayer, 2)
        """
        return self._limit_indices

    def get_layer(self, i, return_residuals=False, return_grid=False, return_var=False, **kwargs):
        """
        This method returns a single layer (i.e. the content of all spaxels
        with a given wavelength) from the data.

        Parameters
        ----------
        i : int
            The index of the layer for which the content should be returned
        return_residuals : boolean, optional
            A flag indicating if the residuals from a previous analysis should
            be returned instead of the data.
        return_grid : boolean, optional
            A flag indicating if the layer should be returned as 1D or 2D
            array. The default is to return a 1D array.
        return_var : boolean, optional
            A flag indicating if the variances should be returned along with
            the data
        kwargs
            The kwargs may be used to control the parameters of the resampling
            routine MusePixtable.make_image in cases where return_grid is set.

        Returns
        -------
        values : ndarray
            The data values (or residuals) in the requested layer. Note that
            if a bad-pixel mask is provided, bad pixels are filtered out from
            the data before it is returned. This is different from the
            behaviour for the other instruments where those pixels are set to
            NaN. The reason is that for a pixtable, the spaxel to (x, y)
            transformation must also be set for each pseudo-layer and the NaN
            pixels are also excluded from this transformation.
        variances : ndarray or None
            If return_var was set, this array will contain the variances of
            the data values in the requested layer. It has the same shape as
            the data array.
        ifu : ndarray or None
            If return_slice was set abd the IFU and slice numbers of the
            individual pixels are available, an array with the same size as
            values will be returned containing the IFU numbers. Otherwise
            None is returned.
        """
        imin, imax = self.layer_indices[i]

        self.set_transform_for_layer(i)

        if self.maskhdu is not None:
            mask = (self.maskhdu.section[imin:imax + 1][:, 0] <= self.dqmax)
        else:
            mask = np.ones(imax - imin, dtype=np.bool)

        if not return_residuals or self.residualshdu is None:
            data = self.datahdu.section[imin:imax + 1][mask, 0]
        else:
            data = self.residualshdu.section[imin:imax + 1][mask, 0]

        var = self.varhdu.section[imin:imax + 1][mask, 0] if return_var else None

        # store information about the wavelength of each pixel returned (& IFU, slice if HDUs available)
        self.rawwave_for_current_layer = self.rawwave[imin:imax + 1][mask]
        if self.ifuhdu:
            self.ifus_for_current_layer = self.ifuhdu.section[imin:imax + 1][mask, 0]
        if self.slicehdu:
            self.slices_for_current_layer = self.slicehdu.section[imin:imax + 1][mask, 0]

        if not return_grid:
            return data, var
        else:
            return self.make_image(data, **kwargs), self.make_image(var, **kwargs) if return_var else None

    def get_spectrum(self, **kwargs):
        """
        This method is designed to extract a single spectrum from a pixtable.
        However, as the concept of spaxels cannot directly be applied to a
        pixtable and extracting the nearest data points to a given location
        would require loading the whole pixtable in memory, currently only
        empty arrays are returned.

        Parameters
        ----------
        kwargs

        Returns
        -------
        data : ndarray
            An empty array of length MusePixtable.nlayer.
        variances : ndarray
            An empty array of length MusePixtable.nlayer.
        """
        logging.error('Grabbing spectra is currently not supported for MUSE pixtables.')
        return np.empty(self.wave.shape), np.empty(self.wave.shape)

    def set_transform_for_layer(self, i):
        """
        This method loads the pixel to (x,  y) transformation for a given
        layer. Note that this has to be done every time a new layer is loaded
        via MusePixtable.get_layer() because each layer will have a different
        transformation.

        Parameters
        ----------
        i : int
            The index of the current layer.
        """
        imin, imax = self.layer_indices[i]
        if self.maskhdu is not None:
            mask = (self.maskhdu.section[imin:imax + 1][:, 0] <= self.dqmax)
        else:
            mask = np.ones(imax - imin, dtype=np.bool)

        # make sure any valid pixels exist
        if not mask.any():
            x = y = []
        else:
            ra, dec = self._convert_coordinates(self.xhdu.section[imin:imax+1][mask, 0],
                                                self.yhdu.section[imin:imax+1][mask, 0])
            x, y = self.wcs.wcs_world2pix(ra, dec, 1)
        self.transform = np.vstack((y, x)).T

    def integrate(self, imin, imax, apply_variances=False, weights=None, nbin=1, **kwargs):
        """
        This method is designed to obtain an image by integrating over a range
        of individual layers. Note that because each layer has a different size
        and different pixel positions, integration is only possible after
        resampling the data to a constant (2dim.) grid.

        Parameters
        ----------
        imin : int
            The index of the first layer to be used in the integration.
        imax : int
            The index of the last layer to be used in the integration.
        apply_variances : boolean, optional
            A flag indicating if the layers should be weighted according to
            their variances in the combination.
        weights : array_like, optional
            An additional weight for each value (e.g. the throughput of a
            filter curve at the wavelength of the layer). Note that this
            weights are multiplied to the variances if apply_variances is
            set to True.
        nbin : int, optional
            A binning that may be used to speed up the integration process.
            If set to a value larger than one, only every 'nbin'th layer is
             considered.
        kwargs
            Any additional keyword argument is passed on to the call of
            MusePixtable.get_layer() used to obtain the data for each layer.

        Returns
        -------
        data : ndarray
            The array containing the integrated data.
        variances : ndarray
            An estimate of the variance in each data pixel. Note that if no
            variances or weights are used in the integration, this array will
            contain 1/(no. of integrated pixels) for each output pixel.
        """
        kwargs['return_grid'] = True
        if 'grid' not in kwargs:
            kwargs['grid'] = np.mgrid[0:300:300j, 0:300:300j]  # need to make sure all layers have same size

        return super(MusePixtable, self).integrate(imin, imax, apply_variances=apply_variances, weights=weights,
                                                   nbin=nbin, **kwargs)

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
        imin, imax = self.layer_indices[i]
        if self.maskhdu is None:
            return slice(imin, imax+1), np.zeros(imax-imin, dtype=np.int32)
        else:
            mask = (self.maskhdu.section[imin:imax + 1][:, 0] <= self.dqmax)
            return imin + np.flatnonzero(mask), np.zeros(mask.sum(), dtype=np.int32)

    def make_image(self, data, grid=None, method='nearest', **kwargs):
        """
        This method is designed to create a 2D image from a provided
        1D array.

        Parameters
        ----------
        data : array_like
            The data to be transformed to a 2dim. image. Note that for the
            code to work, the length of the data must match the current
            number of spaxels in the instance of MusePixtable.
        grid : ndarray, optional
            The x- and y-coordinates of the output pixels. The array should
            be of the shape that is provided bz numpy.mgrid.
        method : string, optional
            The method used in the resampling process. Default is to use
            the nearest pixel which is the fastest but least accurate method.
        kwargs
            Used for consistency.

        Returns
        -------
        outdata : ndarray
            The transformed 2dim. data.
        """
        # perform consistency checks on provided data
        if data.size != self.nspaxel:
            logging.error('Unable to create image: Size of data ({0} differs from no. of spaxels ({1}).'.format(
                data.size, self.nspaxel))
            return None
        elif self.nspaxel == 0:
            if grid is None:
                logging.error('Cannot create image from empty pixel grid.')
                return None
            else:
                return np.nan*np.ones_like(grid[0], dtype=np.float32)

        if grid is None:
            grid = np.mgrid[int(self.y.min()):int(self.y.max()+1):int(self.y.max() - self.y.min() + 1)*1j,
                            int(self.x.min()):int(self.x.max()+1):int(self.x.max() - self.x.min() + 1)*1j]

        return griddata((self.y, self.x), data, (grid[0], grid[1]), method=method)
