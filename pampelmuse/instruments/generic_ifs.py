"""
generic_ifs.py
==============
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
Purpose
-------
This module provides the class GenericIFS which is designed to handle data
coming from an arbitrary integral field spectrograph. To this aim, the data
must be provided as 3dim. cubes, with the zeroth axis (in python counting,
i.e. NAXIS3 in FITS counting) being the spectral axis.

Latest SVN revision
-------------------
372, 2016/08/30
"""
import logging
import numpy as np
from astropy.wcs import WCS
from .instrument import Instrument


__author__ = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__revision__ = 372


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

    @Instrument.edge.getter
    def edge(self):
        """
        Returns the spaxels along the edge of the field of view. The reason for
        redefining it here is that for 3dim. data, the edge can be directly
        inferred from the first and last indices of the spatial dimensions and
        must not be known a priori.
        """
        # for this type of IFS, the spaxels that define the edge can be inferred directly from the transformation
        # this assumes a rectangular field of view and fibres arranged row after row (e.g. PMAS/LArr).
        if self._edge is None and self.transform is not None:
            self._edge = np.hstack((np.flatnonzero((self.y == self.y.min()) & (self.x > self.x.min())),
                                    np.flatnonzero((self.x == self.x.max()) & (self.y > self.y.min())),
                                    np.flatnonzero((self.y == self.y.max()) & (self.x < self.x.max()))[::-1],
                                    np.flatnonzero((self.x == self.x.min()) & (self.y < self.y.max()))[::-1]))
        return self._edge

    @property
    def nspaxel(self):
        """
        Returns
        -------
        nspaxel : int
            The number of spaxels of the current instance of 'Instrument'.
        """
        return self.shape[0]*self.shape[1]

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
            If the return_var flag is set, the variance spectrum is returned
            together with the data.
        """
        if self.transform is None:
            logging.error('Cannot provide spectrum unless a FITS cube has been opened.')
            return
        elif i < -self.nspaxel or i >= self.nspaxel:
            logging.error("Spaxel index must be <{0} but is {1}.".format(self.nspaxel, i))
            return None
        j, k = self._transform[i]

        mask = self.maskhdu.data[:, j, k] if self.maskhdu is not None else np.zeros(self.wave.shape, dtype=np.bool)

        return (np.where(mask, np.nan, self.datahdu.data[:, j, k]),
                np.where(mask, np.nan, self.varhdu.data[:, j, k]) if return_var and self.varhdu is not None else None)

    def get_layer(self, i, return_residuals=False, return_grid=False, return_var=False, **kwargs):
        """
        Extract the monochromatic layer with index i from the provided HDU.
        To keep the memory usage low, the spectrum is extracted using Section
        -class provided by fits.

        Parameters
        ----------
        i : int
            The index of the monochromatic layer that should be extracted.
        return_residuals : boolean, optional
            Set this flag to return the fit residuals (if available) instead
            of the data.
        return_grid : boolean, optional
            Set this flag to return the layer as a 2dim. grid. By default,
            a 1dim. (i.e. flattened) array is returned.
        return_var : bool,  optional
            If this flag is set, the corresponding variances are returned
            together with the layer.

        Returns
        -------
        data : nd_array
            The extracted monochromatic layer. Note that if a bad-pixel mask
            is available, the masked pixels are still included in the array
            but set to NaN.
        variance : nd_array or None
            If the return_var parameter was set, this array contains the
            variances of the data array. It has the same shape as the data
            array.
        """
        if self.maskhdu is not None:
            valid = (self.maskhdu.section[i] > 0)
        else:
            valid = None

        if not return_residuals:
            data = self.datahdu.section[i] if valid is None else np.where(valid, self.datahdu.section[i], np.nan)
        else:
            data = self.residualshdu.section[i] if valid is None else np.where(
                    valid, self.residualshdu.section[i], np.nan)
        if return_var:
            var = self.varhdu.section[i] if valid is None else np.where(valid, self.varhdu.section[i], np.nan)
        else:
            var = None

        if return_grid:
            return data, var
        else:
            return data.flatten(), var.flatten() if var is not None else None

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
