"""
generic_ifs.py
==============
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
Purpose
-------
This module provides the class GenericIFS which is designed to handle data
coming from an arbitrary integral field spectrograph. To this aim, the data
must be provided as 3dim. cubes, with the zeroth axis (in python counting,
i.e. NAXIS3 in FITS counting) being the spectral axis.

Latest Git revision
-------------------
2024/07/09
"""
import logging
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, distance
from astropy.nddata import NDData, VarianceUncertainty
from astropy.wcs import WCS
from .instrument import Instrument


__author__ = "Sebastian Kamann (s.kamann@ljmu.ac.uk)"
__revision__ = 20240709


logger = logging.getLogger(__name__)


class GenericIFS(Instrument):
    """
    This class provides the functionality to work with a generic IFS.
    An IFS is considered generic if it has square spaxels, arranged on a
    continuous spatial grid with filling factor one.
    """
    name = "Generic IFS"

    def __init__(self, **kwargs):
        """
        Initialize an instance of the GenericIFS class.

        Parameters
        ----------
        kwargs are passed to the initialisation os the Instrument super-class.

        Returns
        -------
        The newly created instance.
        """
        super(GenericIFS, self).__init__(**kwargs)

    @Instrument.transform.getter
    def transform(self):
        """
        Returns the transformation from spaxels to spatial (x, y) coordinates.
        The reason for redefining it here is that for 3dim. data sets, the
        transformation can be inferred from the size of the cube and must not
        be known a priori.
        """
        # for this type of IFS, the transformation can also be inferred if the shape is known
        if self._transform is None and self.shape is not None:
            self._transform = np.vstack((np.arange(self.nspaxel, dtype=np.float32) // self.shape[1],
                                         np.mod(np.arange(self.nspaxel, dtype=np.float32), self.shape[1]))).T

        return self._transform

    @property
    def nspaxel(self):
        """
        Returns
        -------
        nspaxel : int
            The number of spaxels of the current instance of 'Instrument'.
        """
        return self.shape[0]*self.shape[1]

    @Instrument.edge.getter
    def edge(self):
        """
        Returns the spaxels that define the closest path around edge of the
        field of view.

        The reason for redefining the method here is that for 3dim. data, the
        edge can be directly inferred from minimum and maximum coordinates of
        the spatial dimensions and must not be known a priori.
        """
        if self._edge is None and self.transform is not None:
            self._edge = self.find_edge(np.zeros(self.shape, dtype=bool))

        return self._edge

    def find_edge(self, mask):
        """
        Sets the edge of the field of view to the convex hull around all valid
        pixels in the provided mask.

        The reason for introducing this method is that the field of view of
        some instruments, such as MUSE, is not strictly rectangular, so that
        a number of pixels in the rectangular cube are always NaN. Such pixels
        must be set to True in the (2D) mask that is provided.

        Parameters
        ----------
        mask : array, 2D
            A boolean array containing a flag for each spaxel indicating if
            the spaxel is part of the field of view that contains no signal.

        Returns
        -------
        edge : nd_array, int
            The indices of the spaxels that define the edge of the field of
            view.
        """
        py, px = np.mgrid[:mask.shape[0], :mask.shape[1]]
        valid_spaxels = np.c_[py[~mask], px[~mask]]

        chull = ConvexHull(valid_spaxels)
        edge_spaxels = valid_spaxels[chull.vertices]

        # need to find indices of (nearest) spaxels in original transformation
        return distance.cdist(self.transform, edge_spaxels).argmin(axis=0)

    def is_valid(self, hdu):

        if not super(GenericIFS, self).is_valid(hdu):
            return False
        if hdu.header['NAXIS'] != 3:  # any data loaded for this type of IFS must be 3dim., i.e. a cube
            logging.error('Expected 3dim. data cube, got {0}dim.'.format(hdu.header['NAXIS']))
            return False
        return True

    def _set_datahdu(self, hdu):
        """
        This method sets the data HDU for the Instrument instance, i.e. it
        loads an HDU containing the observed data into the current instance.
        For the GenericIFS class, it also initializes a WCS system for the
        instance and sets the spatial shape and the number of layers.

        IMPORTANT: This method should not be used directly, instead, use the
        Instrument.open() method for loading data.

        Parameters
        ----------
        hdu : a FITS HDU containing the observed data.
        """
        super(GenericIFS, self)._set_datahdu(hdu)

        self.wcs = WCS(header=self.datahdu.header)

        # sometimes the CD matrix in the header is only 2D (i.e. contains only RA and DEC) and the CDELT3 key is
        # present, but not used by astropy.
        if self.wcs.wcs.has_cd() and self.wcs.wcs.cd.size == 9:
            if 'CDELT3' in self.datahdu.header and 'CD3_3' not in self.datahdu.header:
                self.wcs.wcs.cd[2, 2] *= self.datahdu.header['CDELT3']
                # use *= to include possible earlier conversion from e.g. Angstrom to meters

        self._shape = (self.datahdu.header['NAXIS2'], self.datahdu.header['NAXIS1'])
        self.nlayer = self.datahdu.header['NAXIS3']

    def get_spectrum(self, i, return_var=False):
        """
        Extract an individual spectrum from the available data.

        Parameters
        ----------
        i : int
            The index of the fiber/spaxel for which the spectrum should be
            extracted. Note that instead of the x- & y-position of the spaxel,
            its index in the 'transform' array must be provided.
        return_var : bool,  optional
            If this flag is set, the corresponding variances are returned
            together with the spectrum.

        Returns
        -------
        data : nd_array
            The spectrum in the requested spaxel. Note that if a bad-pixel
            mask is available, the masked pixels are still included in the
            data but set to NaN.
        variance : nd_array or None
            If the return_var flag is set, return the variance spectrum
            together with the data.
        """
        if self.transform is None:
            logging.error('Cannot provide spectrum unless a FITS cube has been opened.')
            return
        elif i < -self.nspaxel or i >= self.nspaxel:
            logging.error("Spaxel index must be <{0} but is {1}.".format(self.nspaxel, i))
            return None
        j, k = self._transform[i].astype('<i4')

        mask = self.maskhdu.data[:, j, k] if self.maskhdu is not None else np.zeros(self.wave.shape, dtype=bool)

        return (np.where(mask, np.nan, self.datahdu.data[:, j, k]),
                np.where(mask, np.nan, self.varhdu.data[:, j, k]) if return_var and self.varhdu is not None else None)

    def get_image(self, i_min: int = None, i_max: int = None, weights: np.ndarray = None, use_variances: bool = False,
                  return_residuals: bool = False, **kwargs) -> NDData:
        """
        This method overwrites the `get_image()` method of the Instrument-class.
        """
        i_min, i_max, weights = self._verify_layer_range(i_min, i_max, weights)
        if use_variances and weights is not None and (weights != 1).any():
            logger.warning("Weights are ignored when using variance weighting.")
            weights = None

        if not return_residuals:
            _data = np.ma.masked_invalid(self.datahdu.section[i_min:i_max], copy=False)
        else:
            _data = np.ma.masked_invalid(self.residualshdu.section[i_min:i_max], copy=False)

        if self.varhdu is not None:
            _var = np.ma.masked_invalid(self.varhdu.section[i_min:i_max], copy=False)
        else:
            _var =  None
            if use_variances:
                logger.error("Variance-weighting requested but no variance HDU available.")

        if self.maskhdu is not None:
            _mask = self.maskhdu.section[i_min:i_max].astype(bool)
            _data[_mask] = np.ma.masked
        else:
            _mask = None

        if not use_variances:
            data = NDData(np.ma.average(_data, weights=weights, axis=0))
            if _var is not None:
                data.uncertainty = VarianceUncertainty(np.ma.average(_var, weights=weights, axis=0))
        else:
            data = NDData(np.ma.average(_data, weights=1./_var if _var is not None else None, axis=0))
            if _var is not None:
                data.uncertainty = VarianceUncertainty(np.ma.sqrt(1./np.ma.sum(1./_var**2, axis=0)))

        return data

    def get_layer(self, i_min: int = None, i_max: int = None, weights: np.ndarray = None, use_variances: bool = False,
                  return_residuals: bool = False, **kwargs) -> pd.DataFrame:
        """
        Overwrites the `get_layer()` method of the Instrument-class.
        """
        data = self.get_image(i_min=i_min, i_max=i_max, weights=weights, use_variances=use_variances,
                              return_residuals=return_residuals, **kwargs)

        valid = ~data.mask
        out = pd.DataFrame(data={'x': self.transform[valid.flatten(), 1],
                                 'y': self.transform[valid.flatten(), 0],
                                 'value': data.data[valid]},
                           index=np.flatnonzero(valid.flatten()))
        if data.uncertainty is not None:
            out['variance'] = data.uncertainty.array[valid]

        return out

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
        return i, self.transform[:, 0].astype('<i4'), self.transform[:, 1].astype('<i4')
