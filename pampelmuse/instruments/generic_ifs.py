"""
generic_ifs.py
==============
Copyright 2013-2021 Sebastian Kamann

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
2021/04/20
"""
import logging
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, distance
from astropy.wcs import WCS
from .instrument import Instrument


__author__ = "Sebastian Kamann (s.kamann@ljmu.ac.uk)"
__revision__ = 20210420


class GenericIFS(Instrument):
    """
    This class is provides the functionality to work with a generic IFS.
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

    def get_image(self, i: int, return_residuals: bool = False, return_var: bool = False,
                  **kwargs) -> (np.ndarray, np.ndarray):
        """
        Extract the individual image with index i from the available data.

        Parameters
        ----------
        i
            The index of the monochromatic layer that should be extracted.
        return_residuals
            If True, return the fit residuals instead of the data.
        return_var
            If True return the variance image together with the data.
        kwargs

        Returns
        -------
        np.ndarray
            The image in the requested layer. Note that if a bad-pixel mask
            is available, the masked pixels are still included in the image,
            but set to NaN.
        np.ndarray
            The variance values associated with the image, if requested.
            Otherwise, None is returned.
        """
        if self.maskhdu is not None:
            valid = (self.maskhdu.section[i] == 0)
        else:
            valid = None

        if not return_residuals:
            data = self.datahdu.section[i]
        else:
            data = self.residualshdu.section[i]
        if valid is not None:
            data[~valid] = np.nan

        if return_var and self.varhdu is not None:
            var = self.varhdu.section[i]
            if valid is not None:
                var[~valid] = np.nan
        else:
            var = None

        return data, var

    def get_layer(self, i: int, return_residuals: bool = False, return_var: bool = False, **kwargs) -> pd.DataFrame:
        """
        Extract the monochromatic layer with index i from the provided HDU and
        return it as a dataframe.

        Parameters
        ----------
        i
            The index of the monochromatic layer that should be extracted.
        return_residuals
            If True, return the fit residuals (if available) instead of the
            data.
        return_var
            If True, the variances are returned together with the data or
            residuals.

        Returns
        -------
        pd.DataFrame
            The dataframe contains the voxel coordinates as columns `x` and
            `y`, the data or residual values as column `value`, and optionally
            the variance values as column `variance`.
        """
        data, var = self.get_image(i=i, return_residuals=return_residuals, return_var=return_var)

        masked = np.isnan(data) | np.isnan(var)
        out = pd.DataFrame(data={'x': self.transform[~masked, 1], 'y': self.transform[~masked, 0],
                                 'value': data[~masked]},
                           index=np.flatnonzero(~masked))
        if var is not None:
            out['variance'] = var[~masked]

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
