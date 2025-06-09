"""
pmas.py
========
Copyright 2013-2025 Sebastian Kamann

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
This module provides the Pmas class that is designed to facilitate the usgae
of data from the PMAS spectrograph in the analysis.

Latest Git revision
-------------------
2025/06/09
"""
import logging
import numpy as np
import pandas as pd
from astropy.nddata import NDData
from .instrument import Instrument


__author__ = "Sebastian Kamann (s.kamann@ljmu.ac.uk)"
__revision__ = 20240709


logger = logging.getLogger(__name__)


class Pmas(Instrument):
    """
    This class is designed to handle data observed with the PMAS/LArr
    instrument (Roth et al. 2005, PASP, 117, 620), mounted at the 3.5m
    telescope at Calar Alto observatory.
    """
    name = "PMAS"
    _shape = (16, 16)

    def __init__(self, **kwargs):
        """
        Initialize an instance of the PMAS class.

        Parameters
        ----------
        kwargs are passed to the initialisation os the Instrument super-class.
        Note that unless provided, the parameter 'transform' is initialized
        to its default value taken from the PMAS manual.

        Returns
        -------
        The newly created instance.
        """
        # initialize transformation from (x,y)-coordinates to spaxels
        transform = np.vstack((15 - np.mod(np.arange(256, dtype=np.int16), 16), np.arange(256, dtype=np.int16) // 16)).T

        if 'angle' not in kwargs:
            kwargs['angle'] = 0.  # PMAS cannot be rotated on the sky. The rotation angle is fixed to 0 degrees.
        if 'scale' not in kwargs:
            kwargs['scale'] = 0.5  # By default, the finer sampling of PMAS (0.5" insetad of 1.0") is assumed

        super(Pmas, self).__init__(transform=transform, **kwargs)
        self.wcs.wcs.crpix[:2] = [self.shape[1]/2., self.shape[0]/2.]

    @property
    def nlayer(self):
        """
        Returns the number of layers of the current data set.
        """
        return self.datahdu.shape[1] if self.datahdu is not None else None

    def is_valid(self, hdu):
        """
        This method inspects if a provided FITS HDU contains data that is
        compatible with the current Pmas instance.

        Parameters
        ----------
        hdu : a FITS HDU containing some data
            The FITS HDU that should be checked.

        Returns
        -------
        validity_flag : boolean
            A flag indicating if the data in the HDU can be used with the
            current Pmas instance.
        """
        if not super(Pmas, self).is_valid(hdu):
            return False

        elif hdu.header['NAXIS'] != 2:
            logging.error('HDUs containing PMAS data must be 2dimensional.')
            return False
        elif hdu.header['NAXIS2'] != self.nfiber:
            logging.error('HDUs containing PMAS data must have {0} fibers (NAXIS2).'.format(self.nspaxel))
            return False
        return True

    def get_spectrum(self, i, return_var=False, **kwargs):
        """
        This method returns a single spectrum (i.e. the content of a single
        spaxel) from the data.

        Parameters
        ----------
        i : int
            The index of the spaxel for which the spectrum should be returned
        return_var : boolean, optional
            A flag indicating if the variances should be returned along with
            the data.
        kwargs
            Any additional keyword argument is passed on to the call of the
            parent class.

        Returns
        -------
        spectrum : ndarray
            The spectrum in the requested spaxel. Note that if a bad-pixel
            mask is available, flagged pixels are set to NaN but still
            included in the data.
        variances : ndarray or None
            If the return_var parameter was set, this array will contain the
             variances of the spectrum in the requested spaxel.
        """
        if i < -self.nspaxel or i >= self.nspaxel:
            logging.error("Spaxel index must be <{0} but is {1}.".format(self.nspaxel, i))
            return None, None

        mask = self.maskhdu.section[i] if self.maskhdu is not None else np.zeros(self.wave.shape, dtype=np.bool)
        return (np.where(mask, np.nan, self.datahdu.section[i]),
                np.where(mask, np.nan, self.varhdu.section[i]) if return_var and self.varhdu is not None else None)

    def get_layer(self, i_min: int = None, i_max: int = None, weights: np.ndarray = None, use_variances: bool = False,
                  return_residuals: bool = False, **kwargs) -> pd.DataFrame:
        """
        This method overwrites the get_layer() method of the Instrument
        super-class.
        """
        if use_variances and weights is not None:
            logger.warning('The weights parameter is ignored when use_variances=True.')
        i_min, i_max, weights = self._verify_layer_range(i_min, i_max, weights)

        if self.maskhdu is not None:
            _mask = self.maskhdu.data[:, i_min:i_max] > 0
        else:
            _mask = None

        if not return_residuals or self.residualshdu is None:
            _data = np.ma.masked_invalid(self.datahdu.data[:, i_min:i_max])
        else:
            _data = np.ma.masked_invalid(self.residualshdu.data[:, i_min:i_max])
        if _mask is not None:
            _data[_mask] = np.ma.masked

        if self.varhdu is not None:
            _var = np.ma.masked_invalid(self.varhdu.data[:, i_min:i_max])
            if _mask is not None:
                _var[_mask] = np.ma.masked
        else:
            _var = None

        if use_variances:
            data = np.ma.average(_data, weights=1./_var if _var is not None else None, axis=1)
            if _var is not None:
                var = 1./np.sqrt(np.sum(1./_var, axis=1))
            else:
                var = None
        else:
            data = np.ma.average(_data, weights=weights, axis=1)
            if _var is not None:
                var = np.ma.average(_var, weights=weights, axis=1)
            else:
                var = None
        out = pd.DataFrame({'x': self.x, 'y': self.y, 'value': data})
        if var is not None:
            out['var'] = var
        return out.dropna()

    def get_image(self, i_min: int = None, i_max: int = None, weights: np.ndarray = None,
                  use_variances: bool = False, return_residuals: bool = False, **kwargs) -> NDData:
        """
        Overwrites the get_image() method of the Instrument super-class.
        """
        layer = self.get_layer(i_min, i_max, weights, use_variances, return_residuals, **kwargs)

        data = NDData(self.make_image(layer=layer['value']))
        if 'variance' in layer.columns:
            data.uncertainty = self.make_image(layer=layer['uncertainty'])
        return data

    def search_configuration(self, hdu, **kwargs):
        """
        This method is designed to update the configuration of the current
        Pmas instance using specific header keywords from a given HDU.

        Parameters
        ----------
        hdu : a FITS HDU
            The HDU containing the header to be checked for the keywords.
        kwargs
            Used for consistency.
        """
        super(Pmas, self).search_configuration(hdu)

        if 'CRVAL1' in hdu.header:
            self.wcs.wcs.crval[-1] = hdu.header['CRVAL1']
        if 'CDELT1' in hdu.header:
            self.wcs.wcs.cd[-1, -1] = hdu.header['CDELT1']
        if 'CRPIX1' in hdu.header:
            self.wcs.wcs.crpix[-1] = hdu.header['CRPIX1']
        if 'CUNIT1' in hdu.header:
            self.wcs.wcs.cunit[-1] = hdu.header['CUNIT1']
        if 'CTYPE1' in hdu.header:
            self.wcs.wcs.ctype[-1] = hdu.header['CTYPE1']

        # make sure wavelength units are converted to meters
        self.wcs.fix()

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
        return Ellipsis, i
