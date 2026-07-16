"""
muse.py
=======
Copyright 2013-2026 Sebastian Kamann

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
2026/07/16
"""
import logging
import warnings
import numpy as np
import pandas as pd
from typing import Tuple
from astropy import units as u
from astropy.io import fits
from astropy.nddata import NDData, VarianceUncertainty
from astropy.wcs import WCS
from scipy.interpolate import griddata
from .instrument import Instrument
from .generic_ifs import GenericIFS

try:
    import mpdaf
except ImportError:
    mpdaf = None


__author__ = "Sebastian Kamann (s.kamann@ljmu.ac.uk)"
__revision__ = 20260716


logger = logging.getLogger(__name__)


def _require_mpdaf():
    """
    Helper function to check if the MPDAF package is available.
    """
    if mpdaf is None:
        raise ImportError('MPDAF is needed to process MUSE pixtables. '
                          'Please install it via pip or conda, or follow the instructions at '
                          'https://mpdaf.readthedocs.io/en/latest/installation.html')


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


class MusePixtable(Instrument):
    """
    This class is designed to handle data observed with the MUSE instrument
    (Bacon et al. 2010, SPIE, 7735, 7), mounted at UT4 of the Very Large
    Telescope. Note that this class works on pixtables, support for MUSE data
    cubes is provided via the Muse class.
    """
    name = "MUSE_Pixtable"
    is_sorted = True

    _layer_limits = None  # place-holders for indices defining individual layers
    _limit_indices = None

    x_hdu = None  # place-holders for HDUs containing x- & y-positions (& optionally the IFU and slice number).
    y_hdu = None
    ifu_hdu = None
    slice_hdu = None
    origin_hdu = None

    dqmax = 4
    deg2rad = np.pi/180.

    raw_wave = None
    raw_wave_for_current_layer = None
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
        _require_mpdaf()

        super(MusePixtable, self).__init__(**kwargs)

        self.mpdaf_object = None

    def _set_x_hdu(self, hdu):
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
            self.x_hdu = hdu
            logging.info('Loaded x-coordinates from HDU <{0}> of file {1}'.format(
                self.x_hdu.name, self.x_hdu.fileinfo()['file'].name))

    def _set_y_hdu(self, hdu):
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
            self.y_hdu = hdu
            logging.info('Loaded y-coordinates from HDU <{0}> of file {1}'.format(
                self.y_hdu.name, self.y_hdu.fileinfo()['file'].name))

    def _set_origin_hdu(self, hdu):
        """
        This method sets the HDU with the 'origin' values for the MusePixtable
        instance, i.e. it loads an HDU containing the 'origin'-values for the
        observed data into the current instance. Note that it should be loaded
        AFTER the data itself.

        IMPORTANT: This method should not be used directly, instead, use the
        MusePixtable.open() method for loading data.

        Parameters
        ----------
        hdu : a FITS ImageHDU containing the 'origin' values.
        """
        if self.datahdu is None:  # HDU containing data must be loaded first
            logging.error('Can only load HDU containing "origin" values after data HDU has been loaded.')
        elif not self.is_valid(hdu):
            logging.error('HDU containing "origin" values [{0}] of file {1} has invalid format.'.format(
                    hdu.name, hdu.fileinfo()['file'].name))
        else:
            self.origin_hdu = hdu
            logging.info('Loaded origin values from HDU <{0}> of file {1}'.format(
                self.origin_hdu.name, self.origin_hdu.fileinfo()['file'].name))

    def _set_ifu_hdu(self, hdu):
        """
        DEPRECATED: Use _set_origin_hdu() instead.

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
        warnings.warn('Using an "ifu" extension is deprecated, use "origin" instead.', DeprecationWarning)

        if self.datahdu is None:  # HDU containing data must be loaded first
            logging.error('Can only load HDU containing IFU numbers after data HDU has been loaded.')
        elif not self.is_valid(hdu):
            logging.error('IFU-number HDU [{0}] of file {1} has invalid format.'.format(
                    hdu.name, hdu.fileinfo()['file'].name))
        else:
            self.ifu_hdu = hdu
            logging.info('Loaded IFU numbers from HDU <{0}> of file {1}'.format(
                self.ifu_hdu.name, self.ifu_hdu.fileinfo()['file'].name))

    def _set_slice_hdu(self, hdu):
        """
        DEPRECATED: Use _set_origin_hdu()

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
        warnings.warn('Using a "slic" extension is deprecated, use "origin" instead.', DeprecationWarning)

        if self.datahdu is None:  # HDU containing data must be loaded first
            logging.error('Can only load HDU containing slice numbers after data HDU has been loaded.')
        elif not self.is_valid(hdu):
            logging.error('Slice-number HDU [{0}] of file {1} has invalid format.'.format(
                    hdu.name, hdu.fileinfo()['file'].name))
        else:
            self.slice_hdu = hdu
            logging.info('Loaded slice numbers from HDU <{0}> of file {1}'.format(
                self.slice_hdu.name, self.slice_hdu.fileinfo()['file'].name))

    def open(self, filename, is_residuals=False, **kwargs):
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
        is_residuals : boolean, optional
            A flag indicating if the FITS file contains the residuals from a
            previous analysis. In that case only the first HDU is considered.
            Note that the HDU containing the original data must have been
            loaded before.
        kwargs
            Any additional keyword arguments are passed on to the call of the
            parent class.
        """
        from mpdaf.drs import PixTable

        # The residuals are stored in a separate file. Therefore, the behaviour depends on whether residuals or a
        # full pixtable (containing data, positions, ...) is loaded
        if is_residuals:
            return super(MusePixtable, self).open(filename, isresiduals=True, **kwargs)

        pt = fits.open(filename, memmap=False)
        self.mpdaf_object = PixTable(filename)

        # initialize WCS instance
        self.wcs = WCS(pt[0].header)
        self.wcs.wcs.crval = [pt[0].header['RA'], pt[0].header['DEC']]
        self.wcs.wcs.crpix = [self.wcs.wcs.crpix[0] - pt[0].header['ESO DRS MUSE PIXTABLE PREDAR LIMITS XLO'],
                              self.wcs.wcs.crpix[1] - pt[0].header['ESO DRS MUSE PIXTABLE PREDAR LIMITS YLO']]

        # Note that for the MUSE pixtables two wavelength ranges are defined; MusePixtable.raw_wave contains one value
        # per pixel while MusePixtable.wave contains one value per layer. As the pixel tables are provided in units of
        # Angstrom, we need to convert to meters
        logging.info('Reading wavelength array ...')
        self.raw_wave = self.mpdaf_object.get_lambda(unit=u.m)
        logging.info('... done.')

        # check if wavelength array is strictly increasing
        logging.info('Checking consistency of pixtable ...')
        if (np.diff(self.raw_wave) >= 0).all():
            self.is_sorted = True
        else:
            self.is_sorted = False
            logging.error('Pixtable is not sorted by increasing wavelength.')
            logging.error('Currently support exists only for sorted pixtables.')
            logging.error('To sort this pixtable, you may run "PampelMuse PTSORT {0}".'.format(filename))
            raise IOError('Pixtable data not sorted by wavelength.')
        logging.info('... done.')

        # fill other HDU instances with data
        self._set_datahdu(pt['data'])
        self._set_varhdu(pt['stat'])
        self._set_maskhdu(pt['dq'])
        self._set_x_hdu(pt['xpos'])
        self._set_y_hdu(pt['ypos'])
        if 'origin' in pt:
            self._set_origin_hdu(pt['origin'])
        elif 'ifu' in pt and 'slice' in pt:
            self._set_ifu_hdu(pt['ifu'])
            self._set_slice_hdu(pt['slice'])

        # define default layers for this cube. This may be changed by the user later on
        min_wave, median_wave, max_wave = np.percentile(self.raw_wave, [0, 50, 100])
        n_step_blue = int((median_wave - min_wave)/self.wstep)
        n_step_red = int((max_wave - median_wave)/self.wstep)
        layers = np.arange(-n_step_blue, n_step_red+1, dtype=np.float64)*self.wstep
        layers += median_wave
        self.layer_limits = np.vstack((layers - self.wstep, layers + self.wstep)).T

        # a pixel table has no fixed transformation from voxels to spatial coordinates, it changes for every layer
        # that is defined. Get coordinate transformation for central layer to have useful idea of edges of FoV.
        i_min, i_max = self.layer_indices[len(layers) // 2]
        x, y = self.get_coordinates(j_min=i_min, j_max=i_max)
        self.transform = np.vstack((y, x)).T

    def _convert_coordinates(self, xpos, ypos):
        """
        DEPRECATED: Use the get_pos_sky method of mpdaf.drs.PixTable instead.

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
        warnings.warn(
            'MusePixtable._convert_coordinates() is deprecated.  Use mpdaf.drs.PixTable.get_pos_sky() instead.',
            DeprecationWarning)

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
        self._limit_indices = np.searchsorted(self.raw_wave, self._layer_limits)

        # we define a mean wavelength for each layer to be consistent with the other instrument classes.
        self.wave = np.mean(value, axis=1)

    @property
    def layer_indices(self):
        """
        Returns the indices that define the start and end of each pseudo-layer
        as an array of shape (MusePixtable.nlayer, 2)
        """
        return self._limit_indices

    def get_layer(self, i_min: int = None, i_max: int = None, weights: np.ndarray = None, use_variances: bool = False,
                  return_residuals: bool = False, **kwargs) -> pd.DataFrame:
        """
        Overwrites the get_layer() method of the Instrument class.
        """
        i_min, i_max, weights = self._verify_layer_range(i_min, i_max, weights)
        if use_variances and weights is not None and (weights != 1).any():
            logger.warning('Weights are ignored when using variance weighting.')
            weights = None

        j_min = self.layer_indices[i_min, 0]
        j_max = self.layer_indices[i_max, 1]

        # estimate memory usage
        mem_gb = 7*64*(j_max-j_min)/(8*1e9)
        if mem_gb > 1:
            warnings.warn("This step will consume about {:.1f} GB of memory".format(mem_gb), UserWarning)

        if self.maskhdu is not None:
            mask = (self.maskhdu.section[j_min:j_max + 1][:, 0] <= self.dqmax)
        else:
            mask = np.ones(j_max - j_min, dtype=bool)

        if not return_residuals or self.residualshdu is None:
            _data = self.datahdu.section[j_min:j_max + 1][mask, 0]
        else:
            _data = self.residualshdu.section[j_min:j_max + 1][mask, 0]

        if self.varhdu is not None:
            _var = self.varhdu.section[j_min:j_max + 1][mask, 0]
        else:
            _var = None

        # return empty array if layer does not contain any valid data.
        if not len(_data):
            return pd.DataFrame(columns=['x', 'y', 'value', 'variance', 'ifu', 'slice'], dtype=np.float64)

        # get the wavelength of each pixel returned (& IFU, slice if HDUs available)
        self.raw_wave_for_current_layer = self.raw_wave[j_min:j_max + 1][mask]

        # Decode 'origin' values to get CCD x-coordinates, IFU and slice numbers
        origin = self.origin_hdu.section[j_min:j_max + 1][mask, 0]
        pos = self.mpdaf_object.origin2xpix(origin)
        ifu_number = self.mpdaf_object.origin2ifu(origin)
        slice_number = self.mpdaf_object.origin2slice(origin)

        # get map of pixels that should be combined
        # use np.unique(..., return_inverse=True) to remove any blank indices in array
        pixel_map = np.unique(pos + pos.max()*ifu_number, return_inverse=True)[1]

        # perform resampling/drizzling to central wavelength of new layer
        if weights is None:
            drizzled = self.drizzle_1d(
                wave_in=self.raw_wave_for_current_layer, flux_in=_data,  wave_out=self.wave[i_min:i_max].mean(),
                delta_wave_out=self.wstep*(i_max-i_min), variances=_var, pixel_map=pixel_map, return_weights=True)
            if _var is None:
                data, _weights = drizzled
                var = None
            else:
                data, var, _weights = drizzled
        else:
            _weights = np.interp(self.raw_wave_for_current_layer, self.wave[i_min:i_max], weights)
            data = self.combine_voxel(_data, pixel_map=pixel_map, weights=_weights)
            if _var is not None:
                var = self.combine_voxel(_var, pixel_map=pixel_map, weights=_weights)
            else:
                var = None

        # calculate coordinates
        x, y = self.get_coordinates(j_min=j_min, j_max=j_max, valid=mask, pixel_map=pixel_map, weights=_weights)

        # collect columns for DataFrame
        columns = {'value': data, 'x': x, 'y': y}
        if var is not None:
            columns['variance'] = var

        # The value of the IFU/slice is the same for voxels that are combined. Hence, we adopt the values
        # of the first voxel.
        first = np.unique(pixel_map, return_index=True)[1]
        columns['ifu'] = ifu_number[first]
        columns['slice'] = slice_number[first]
        return pd.DataFrame(data=columns)

    def get_image(self, i_min: int = None, i_max: int = None, weights: np.ndarray = None, use_variances: bool = False,
                  return_residuals: bool = False, **kwargs) -> NDData:
        """
        Overwrites the get_image() method of the Instrument class.
        """
        _data = self.get_layer(i_min=i_min, i_max=i_max, weights=weights, use_variances=use_variances,
                              return_residuals=return_residuals)
        # ignore IFU/slice information if 2D grid should be returned
        if not len(_data):
            return NDData(data=[])

        # need to update transformation first
        self.transform = np.vstack((_data['y'], _data['x'])).T

        data = NDData(data=self.make_image(data=_data['value'].values, **kwargs))
        if 'variance' in _data.columns:
            data.uncertainty = VarianceUncertainty(self.make_image(data=_data['variance'].values, **kwargs))

        return data

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

    def get_coordinates(self, j_min: int = None, j_max: int = None, indices: np.ndarray = None,
                        valid: np.ndarray = None, pixel_map: np.ndarray = None,
                        weights: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Determine the transformation from voxels to (x, y) coordinates for a
        given data slice.

        If a continuous set of voxels is used, it is sufficient to provide the
        start and stop indices of the set. Otherwise, the full set of indices
        must be provided.

        Parameters
        ----------
        j_max
            The starting index of a continuous set of voxels.
        j_min
            The ending index of a continuous set of voxels.
        indices
            The set of indices to be used. Only used if i_min and i_max are
            not set.
        valid
            A boolean array that indicates which voxels are masked. If
            provided, its length must match the number of voxels.
        pixel_map
            In case the coordinates of the voxels should be averaged, a
            mapping from input voxels to output pixels can be provided.
        weights
            An array of weights used when averaging the input voxels. Must
            have the same length as `pixel_map`.

        Returns
        -------
        numpy.ndarray
            The x-coordinates of the provided voxels.
        numpy.ndarray
            The y-coordinates of the provided voxels.
        """
        if j_min is not None and j_max is not None:
            _x = self.x_hdu.section[j_min:j_max + 1][:, 0]
            _y = self.y_hdu.section[j_min:j_max + 1][:, 0]
        elif indices is not None:
            _x = self.mpdaf_object.get_xpos(indices)
            _y = self.mpdaf_object.get_ypos(indices)
        else:
            raise IOError("Either 'i_max' and 'i_min' or 'indices' must be provided.")

        if valid is not None:
            if len(valid) != len(_x):
                raise ValueError("Length of 'valid' array does not match number of voxels.")
            _x = _x[valid]
            _y = _y[valid]

        # make sure any valid pixels exist
        if len(_x) == 0:
            x = y = []
        else:
            # ra, dec = self._convert_coordinates(_x, _y)
            ra, dec = self.mpdaf_object.get_pos_sky(_x, _y)
            x, y = self.wcs.wcs_world2pix(ra, dec, 1)

        if pixel_map is not None:
            x = self.combine_voxel(x, pixel_map=pixel_map, weights=weights)
            y = self.combine_voxel(y, pixel_map=pixel_map, weights=weights)

        return x, y

    @staticmethod
    def combine_voxel(input_array: np.ndarray, pixel_map: np.ndarray, weights: np.ndarray = None) -> np.ndarray:
        """
        Average the values in a set of voxels using a given mapping between
        input voxels and output pixels and optional weights for the input
        voxels.

        Parameters
        ----------
        input_array
            The array of input values to be combined.
        pixel_map
            The mapping of input voxels to output pixels.
        weights
            The weight of each input voxel in the combination process.

        Returns
        -------
        numpy.ndarray
            The combined values. Note that the length of the output array is
            the same as the maximum value of `pixel_map`, i.e. output pixel
            missing in `pixel_map` are not removed, but assigned numpy.nan as
            a value.
        """
        # perform sanity checks
        if len(pixel_map) != len(input_array):
            raise IOError("Length of 'pixel_map' must match number of voxels.")
        if weights is not None:
            weights = np.ones_like(pixel_map)
        elif len(weights) != len(pixel_map):
            raise IOError("If provided, length of 'weights' must match length of 'pixel_map'.")

        norm = np.bincount(pixel_map, weights=weights)
        output_array = np.bincount(pixel_map, weights=weights * input_array)
        output_array /= np.where(norm > 0, norm, np.nan)

        return output_array

    def integrate(self, imin, imax, apply_variances=False, weights=None, nbin=1, **kwargs):
        """
        DEPRECATED: This method is deprecated and will be removed in a future release.

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
        j_min, j_max = self.layer_indices[i]
        if self.maskhdu is None:
            return slice(j_min, j_max+1), np.zeros(j_max-j_min, dtype=np.int32)
        else:
            mask = (self.maskhdu.section[j_min:j_max + 1][:, 0] <= self.dqmax)
            return j_min + np.flatnonzero(mask), np.zeros(mask.sum(), dtype=np.int32)

    def make_image(self, data: np.ndarray = None, layer: pd.Series = None, grid: np.ndarray = None,
                   method: str = 'nearest', **kwargs) -> np.ndarray:
        """
        This method is designed to create a 2D image from a provided 1dim set
        of data.

        Parameters
        ----------
        data
            A numpy array containing the data to be transformed to a 2dim
            image. Note that the length of the data must match the current
            number of spaxels in the instance of MusePixtable.
        layer
            A pandas series containing the data to be transformed into a
            2dim image. Contrary to `data`, `layer` supports missing values
            because the index is used to match input and output voxels.
        grid
            The x- and y-coordinates of the output pixels. The array should
            be of the shape that is provided bz numpy.mgrid.
        method
            The method used in the resampling process. Default is to use
            the nearest pixel which is the fastest but least accurate method.
        kwargs
            Used for consistency.

        Returns
        -------
        numpy.ndarray
            The transformed 2dim. data.
        """
        if data is None and layer is None:
            return super().make_image(data=data, layer=layer)
        elif data is not None:
            if layer is not None:
                warnings.warn("Ignoring `layer` argument to `make_image()` because `data` provided.")
                # perform consistency checks on provided data
            if data.size != self.nspaxel:
                raise IndexError('Unable to create image: Size of data ({0} differs from no. of spaxels ({1}).'.format(
                    data.size, self.nspaxel))

        if grid is None:
            if self.nspaxel == 0:
                raise IOError('Cannot create image from empty pixel grid.')
            grid = np.mgrid[int(self.y.min()):int(self.y.max()+1):int(self.y.max() - self.y.min() + 1)*1j,
                            int(self.x.min()):int(self.x.max()+1):int(self.x.max() - self.x.min() + 1)*1j]
        if self.nspaxel == 0:
            return np.nan*np.ones_like(grid[0], dtype=np.float32)
        elif data is not None:
            return griddata((self.y, self.x), data, (grid[0], grid[1]), method=method)
        else:
            return griddata((self.y[layer.index], self.x[layer.index]), layer.values, (grid[0], grid[1]), method=method)

    def drizzle_1d(self, wave_in: np.ndarray, flux_in: np.ndarray, wave_out: float, delta_wave_in: float = None,
                   delta_wave_out: float = None, variances: np.ndarray = None, pixel_map: np.ndarray = None,
                   return_weights: bool = False) -> tuple:
        """
        Resample the voxels in a given wavelength range to a single
        wavelength.

        Parameters
        ----------
        wave_in
            The wavelength values of the input data.
        flux_in
            The flux values of the input data. Must have same shape as
            `wave_in`.
        wave_out
            The unique wavelength value of the output data.
        delta_wave_in
            The (average) size of the input voxels along the spectral axis. If
            not provided, it will be estimated from the input data.
        delta_wave_out
            The size of the output voxels along the spectral axis. If not
            provided, the same size as `delta_wave_in` will be used.
        variances
            The variance valuess associated with the input data. If provided,
            must have same shape as `wave_in`.
        pixel_map
            A mapping between input and output voxels. If provided, must have
            same shape as `wave_in`. For each unique value in `pixel_map`, an
            output value is determined. If None, all input voxels will be
            mapped to a single output voxel.
        return_weights
            If True, an additional array containing the weights of the input
            voxels used to determine the output values will be returned.

        Returns
        -------
        tuple
            The first entry of the return tuple is the array containing the
            resampled fluxes. If `variances` where provided with the data,
            the resampled variances are provided as well. Finally, if
            `return_weights` is True, the voxel weights are returned.
        """
        # unless given, determine (average) input pixel size
        if delta_wave_in is None:
            if pixel_map is None:
                delta_wave_in = (wave_in.max() - wave_in.min()) / (len(wave_in) - 1.)
            else:
                n = np.bincount(pixel_map)
                w_max = np.zeros_like(n, dtype=np.float64)
                w_min = np.inf * np.ones_like(w_max)
                np.maximum.at(w_max, pixel_map, wave_in)
                np.minimum.at(w_min, pixel_map, wave_in)
                valid = n > 1
                sampling_per_row = (w_max - w_min)[valid]/(n[valid].astype(np.float64) - 1.)
                delta_wave_in = np.nanmedian(sampling_per_row)
        if delta_wave_out is None:
            delta_wave_out = delta_wave_in

        delta = abs(wave_in - wave_out)

        weights = np.zeros_like(wave_in)
        processed = np.zeros_like(wave_in, dtype=bool)

        # case 1: output pixel encompasses input pixel
        slc = delta_wave_out >= (delta_wave_in + 2 * delta)
        weights[slc] = 1.
        processed[slc] = True
        # case 2: input pixel encompasses output pixel
        slc = delta_wave_in >= (delta_wave_out + 2 * delta)
        weights[slc] = delta_wave_out / delta_wave_in
        processed[slc] = True
        # case 3: input pixel partially overlaps with output pixel
        slc = ~processed & (delta < (0.5 * (delta_wave_in + delta_wave_out)))
        weights[slc] = 0.5 * (delta_wave_in + delta_wave_out) - delta[slc]
        # case 4: no overlap, can be ignored

        # calculate drizzled flux and variances
        variances_out = None
        if pixel_map is None:
            flux_out = np.atleast_1d(np.average(flux_in, weights=weights))
            if variances is not None:
                variances_out = np.atleast_1d(np.average(variances, weights=weights ** 2))
        else:
            flux_out = self.combine_voxel(flux_in, pixel_map=pixel_map, weights=weights)
            if variances is not None:
                variances_out = self.combine_voxel(variances, pixel_map=pixel_map, weights=weights ** 2)
        ret = (flux_out,)
        if variances_out is not None:
            ret += (variances_out,)
        if return_weights:
            ret += (weights,)
        return ret
