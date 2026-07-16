"""
instrument.py
=============
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
This module provides the class to handle data observed with the MEGARA
instrument at the GTC telescope on La Palma.

Latest Git revision
-------------------
2026/07/16
"""
from __future__ import annotations
import logging
import astropy.io.fits
import numpy as np
import pandas as pd
from typing import TYPE_CHECKING
from typing import Tuple
from astropy import units as u
from astropy.nddata import NDData, VarianceUncertainty, NDUncertainty
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

# Needed because the Megara.__init__() method does a check that the fiber_conf parameters is an instance of the FocalPlaneConf class.
if TYPE_CHECKING:
    from megaradrp.instrument.focalplane import FocalPlaneConf

try:
    import megaradrp
except ImportError:
    megaradrp = None


from .instrument import Instrument


__author__ = "Sebastian Kamann (s.kamann@ljmu.ac.uk)"
__added__ = 20240701
__revision__ = 20260716


logger = logging.getLogger(__name__)


def _require_megaradrp():
    """
    Helper function to check if the megaradrp package is available.
    """
    if megaradrp is None:
        raise ImportError('The MEGARA instrument class requires the megaradrp package to be installed. '
                          'Please install it via pip or conda, or follow the instructions at '
                          'https://megaradrp.readthedocs.io/en/latest/installation.html')



class Megara(Instrument):

    name = "MEGARA"

    HEX_TO_RECT = (3./4.)**(1./4.)
    FIBER_DIAMETER = 0.62

    def __init__(self, fiber_conf: FocalPlaneConf | None = None, **kwargs):
        """
        Class to handle data from the MEGARA spectrograph mounted onn GTC.

        General information about the spectrograph is available at:
        https://www.gtc.iac.es/instruments/megara/

        A data reduction cookbook as PDF file is available at:
        https://www.gtc.iac.es/instruments/megara/media/MEGARA_cookbook_1I.pdf

        The class makes use of the megaradrp package developed by the
        instrument consortium. It is available via Github and documented at
        ReadTheDocs:
        https://github.com/guaix-ucm/megaradrp/
        https://megaradrp.readthedocs.io/en/latest/installation.html

        In order to initialize the fiber configuration used during the
        observations, the FocalPlaneConf class proovided in the megaradrp
        package is used. It is strongly recommended that the configuration
        is provided while initializing a new instance of the Megara class,
        e.g., via the from_fiber_setup_hdu() class method.

        Parameters
        ----------
        fiber_conf
            The fiber configuration of the spectrograph used during the
            observations.
        kwargs
            Keyword arguments passed to the Instrument constructor.
        """
        _require_megaradrp()
        import megaradrp.instrument.constants as megara_constants
        from megaradrp.instrument.focalplane import FocalPlaneConf
        self.GTC_PLATESCALE = u.plate_scale(megara_constants.GTC_FC_A_PLATESCALE)
    
        # Size scale of the spaxel grid in arcseconds
        self.HEX_SCALE = megara_constants.SPAXEL_SCALE.to(u.arcsec, self.GTC_PLATESCALE).value

        super(Megara, self).__init__(**kwargs)

        if fiber_conf is None:
            self.fiber_conf = FocalPlaneConf()
        else:
            self.fiber_conf = fiber_conf

    @classmethod
    def from_fiber_setup_hdu(cls, hdu: astropy.io.fits.FitsHDU, **kwargs):
        """
        Initialize a new instance of the Megara class using a FITS HDU
        containing the fiber configuration.

        Note that a dedicated HDU containing the information necessary to
        recover the fiber configuration is provided with the RSS files
        produced by the instrument pipeline.

        Parameters
        ----------
        hdu
            The FITS HDU containing the fiber configuration.
        kwargs
            Keyword arguments passed to the Megara constructor.
        """
        # initialize fiber configuration from header of provided HDU
        _require_megaradrp()
        import megaradrp.instrument.constants as megara_constants
        from megaradrp.instrument.focalplane import FocalPlaneConf
        from megaradrp.processing import hexgrid
        from megaradrp.processing.cube import calc_matrix_from_fiberconf

        fp = FocalPlaneConf.from_header(hdu.header)

        # store information on fiber used for sky subtraction.
        sky_fibers = np.asarray(fp.sky_fibers())

        # initialize transformation from fibers to (x, y) coordinates
        transform, (x_ref, y_ref) = calc_matrix_from_fiberconf(fpconf=fp)

        # calc_matrix_from_fiberconf() returns the fibre positions in units of the HEX_SCALE of MEGARA, which is
        # defined as 2x the inradius of the hexagonal fibers (~0.537") - not to be confused with the fiber diameter
        # of 0.62", which is 2x the circumradius. To create rectangular pixels with the same area as the hexagonal
        # fibers, the sampling needs to be increased by (3/4)^(1/4) along x and y.
        transform = transform/Megara.HEX_TO_RECT

        # when resampling onto a rectangular grid, the function create_cube_from_array() from the megaradrp package
        # is used. It internally calculates the dimensions of the grid, and does not necessarily place the origin at
        # (0, 0). This would lead to shifts between the positions of the fibers and the rectangular pixels. To avoid
        # this issue, we need to account for the origin offset.
        # Note that the offstes need to be determined using the regular HEX_SCALE to be consistent with the internal
        # calulation of create_cube_from_array()
        (i_min, i_max), (j_min, j_max) = hexgrid.hexgrid_extremes(transform*Megara.HEX_TO_RECT, Megara.HEX_TO_RECT)
        transform[0] -= i_min
        transform[1] -= j_min

        # need to swap axes and x/y-order of `transform` to match internal format of PampelMuse
        megara = cls(fiber_conf=fp, transform=transform[::-1].T, **kwargs)
        megara._sky_fibers = sky_fibers

        _wcs = WCS(hdu.header)
        # The WCS information in the header does the conversion from mm to RA, Dec. Instead, we need to convert
        # from (rectangular) spaxel to (RA, Dec)
        _wcs.wcs.cdelt *= megara_constants.SPAXEL_SCALE.to(u.mm).value
        _wcs.wcs.cdelt *= Megara.HEX_TO_RECT
        _wcs.wcs.crpix -= [x_ref, y_ref]
        indices = [i + 1 for i in range(_wcs.wcs.naxis)]
        indices.insert(_wcs.wcs.naxis, 0)
        megara.wcs = _wcs.sub(indices)
        megara.wcs.wcs.ctype[_wcs.wcs.naxis] = 'AWAV'

        return megara

    def search_configuration(self, hdu: astropy.io.fits.FitsHDU, **kwargs):
        """
        Search the header of a FITS-HDU for keywords containing properties of
        the observed data.

        Parameters
        ----------
        hdu
           The FITS-HDU for which the header is to be searched.
        kwargs
            This method does not accept any keyword arguments.
        """
        # Get wavelength calibration data
        wave_wcs = WCS(hdu.header).dropaxis(1)
        self.wcs.wcs.ctype[2] = wave_wcs.wcs.ctype[0]
        self.wcs.wcs.crval[2] = wave_wcs.wcs.crval[0]
        self.wcs.wcs.crpix[2] = wave_wcs.wcs.crpix[0]
        self.wcs.wcs.cdelt[2] = wave_wcs.wcs.cdelt[0]

        self.wave = self.wcs.spectral.all_pix2world(np.arange(hdu.header['NAXIS1']), 0)[0]

    def _fiber_from_index(self, i: int) -> int:
        """
        Helper function to convert an index of a spaxel into a fiber ID.

        Parameters
        ----------
        i
            The index of the spaxel.

        Returns
        -------
        The fiber ID.
        """

        if i < 0 or i >= self.nspaxel:
            raise IndexError('Fiber index out of range')

        n_sky_prior = np.sum(self.sky_fibers < (i + 1))
        return i + n_sky_prior

    def get_spectrum(self, i: int, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the 1D spectrum for a spaxel.

        Parameters
        ----------
        i
            The index of the spaxel.
        kwargs
            This method does not accept any keyword arguments.

        Returns
        -------
        The 1D spectrum.
        The variance of the spectrum, if available.
        """
        fibid = self._fiber_from_index(i)

        spec = self.datahdu.data[fibid]
        if self.varhdu is not None:
            var = self.varhdu.data[fibid]
        else:
            var = None

        return spec, var

    def get_layer(self, i_min: int = None, i_max: int = None, weights: np.ndarray = None,
                  use_variances: bool = False, return_residuals: bool = False, **kwargs) -> pd.DataFrame:
        """
        This method overwrites the `get_layer` method of the basic
        Instrument()-class.
        """
        if use_variances and weights is not None:
            logger.warning('Using variances instead of weights.')
            weights = None

        i_min, i_max, weights = self._verify_layer_range(i_min, i_max, weights)

        fibers = []
        for fiber in self.fiber_conf.connected_fibers():
            fibers.append(fiber.fibid - 1)

        if return_residuals:
            all_layers = self.residualshdu.data[fiber, i_min:i_max]
        else:
            all_layers = self.datahdu.data[fibers, i_min:i_max]

        if self.varhdu is not None:
            all_variances = self.varhdu.data[fibers, i_min:i_max]
        else:
            all_variances = None
            if use_variances:
                logger.error('Parameter `use_variances` is True, but no variance data available.')

        if use_variances:
            data = np.average(all_layers, weights=1./all_variances if all_variances is not None else None, axis=1)
            if all_variances is not None:
                var = np.sqrt(1./np.sum(1./all_variances**2, axis=1))
            else:
                var = None
        else:
            data = np.average(all_layers, weights=weights, axis=1)
            if all_variances is not None:
                var = np.average(all_variances, weights=weights, axis=1)
            else:
                var = None

        out = pd.DataFrame({'x': self.transform[:, 1], 'y': self.transform[:, 0], 'value': data})
        if var is not None:
            out['variance'] = var

        return  out

    def get_image(self, i_min: int = None, i_max: int = None, weights: np.ndarray = None,
                  use_variances: bool = False, return_residuals: bool = False, **kwargs) -> NDData:
        """
        This method overwrites the `get_image` method of the basic
        Instrument()-class.
        """
        from megaradrp.processing.cube import create_cube_from_array

        layer = self.get_layer(i_min, i_max, weights, use_variances, return_residuals, **kwargs)

        # sky fibers have been removed from the `layer` array. Need to add them back in before resampling to 2D
        # array via `create_cube_from_array` from megaradrp package.
        data_1d = np.zeros(self.nfiber, dtype=np.float64)
        data_1d[self.isscience] = layer['value']

        # get average pixel scale in arcsec
        _p = proj_plane_pixel_scales(self.wcs.celestial)
        target_scale_arcsec = 3600.*np.mean(_p)

        data_2d = create_cube_from_array(data_1d, fiberconf=self.fiber_conf, target_scale_arcsec=target_scale_arcsec)[0]
        out = NDData(data_2d)

        if 'variance' in layer.columns:
            var_1d = np.zeros_like(data_1d)
            var_1d[self.isscience] = layer['variance']
            var_2d = create_cube_from_array(var_1d, fiberconf=self.fiber_conf,
                                            target_scale_arcsec=target_scale_arcsec)[0]
            out.uncertainty = VarianceUncertainty(var_2d)

        return out
