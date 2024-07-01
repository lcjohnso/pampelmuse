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
This module provides the class to handle data observed with the MEGARA
instrument at the GTC telescope on La Palma.

Latest Git revision
-------------------
2024/07/01
"""
import astropy.io.fits
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
import megaradrp.instrument.constants as megara_constants
from megaradrp.instrument.focalplane import FocalPlaneConf
from megaradrp.processing.cube import create_cube_from_array

from .instrument import Instrument


__author__ = "Sebastian Kamann (s.kamann@ljmu.ac.uk)"
__added__ = "20240701"
__revision__ = __added__


class Megara(Instrument):

    name = "MEGARA"

    HEX_TO_RECT = (3./4.)**(1./4.)
    FIBER_DIAMETER = 0.62

    def __init__(self, fiber_conf: FocalPlaneConf = None, **kwargs):
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
        # initialize fiber configuration from header of provided HDu
        fp = FocalPlaneConf.from_header(hdu.header)

        # store information on fiber used for sky subtraction.
        sky_fibers = np.asarray(fp.sky_fibers())

        # initialize transformation from fibers to (x, y) coordinates
        transform = np.zeros((fp.nfibers - len(sky_fibers), 2), dtype=np.float64)
        for _, fiber in fp.fibers.items():
            if fiber.fibid in sky_fibers:
                continue
            n_sky_prior = np.sum(sky_fibers < fiber.fibid)
            idx = fiber.fibid - 1 - n_sky_prior
            transform[idx, 0] = fiber.y
            transform[idx, 1] = fiber.x

        # From MEGARA documentation: Fiber in lower left corner is 614
        ref_fiber = fp.fibers[614]
        transform -= [ref_fiber.y, ref_fiber.x]

        # By default, the (x, y) positions of the fibers are provided in mm on focal plane. For creating the internal
        # image grid, the sampling should be about 1 pixel/fiber.
        # The SPAXEL_SCALE of MEGARA is defined as 2x the inradius of the hexagonal fibers - not to be confused with
        # the fiber diameter of 0.62", which is 2x the circumradius. To create rectangular pixels with the same area
        # as the hexagonal fibers, the sampling needs to be increased by (3/4)^(1/4) along x and y.
        if fp.funit == 'mm':
            transform = transform/megara_constants.SPAXEL_SCALE.to(u.mm).value
        else:
            raise NotImplementedError
        transform = transform/Megara.HEX_TO_RECT

        megara = cls(fiber_conf=fp, transform=transform, **kwargs)
        megara._sky_fibers = sky_fibers

        _wcs = WCS(hdu.header)
        # The WCS information in the header does the conversion from mm to RA, Dec. Instead, we need to convert
        # from (rectangular) spaxel to (RA, Dec)
        _wcs.wcs.cdelt *= megara_constants.SPAXEL_SCALE.to(u.mm).value
        _wcs.wcs.cdelt *= Megara.HEX_TO_RECT
        _wcs.wcs.crpix -= [ref_fiber.x, ref_fiber.y]
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

    def get_image(self, i: int, return_residuals: bool = False, return_var: bool = False,
                  **kwargs) -> (np.ndarray, np.ndarray):
        """
        Return the data of an individual layer as a 2dim-array.

        Given the hexagonal shape of the MEGARA fibers, resampling onto a
        2dim. rectangular grid is required, as PampelMuse does not support
        `hexbin` plots. To this aim, the `create_cube_from_array` function
        defined in the megaradrp-package is used.

        Parameters
        ----------
        i
            The index of the layer to be returned.
        return_residuals
            Flag indicating if the residuals should be returned instead of
            the data.
        return_var
        kwargs
            This method does not accept any keyword arguments.

        Returns
        -------
        """

        if not return_residuals:
            _data = self.datahdu.data[:, i]
        else:
            _data = self.residualshdu.data[:, i]

        # get average pixel scale in arcsec
        _p = proj_plane_pixel_scales(self.wcs.celestial)
        target_scale_arcsec = 3600.*np.mean(_p)

        data = create_cube_from_array(_data, fiberconf=self.fiber_conf, target_scale_arcsec=target_scale_arcsec)[0]

        if return_var and self.varhdu is not None:
            _var = self.varhdu.data[:, i]
            var = create_cube_from_array(_var, fiberconf=self.fiber_conf, target_scale_arcsec=target_scale_arcsec)[0]
            # if valid is not None:
            #     var[~valid] = np.nan
        else:
            var = None

        return data, var

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

    def get_spectrum(self, i: int, **kwargs) -> (np.ndarray, np.ndarray):
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

    def get_layer(self, i: int, **kwargs) -> pd.DataFrame:
        """
        Return a single layer of the data as a Pandas DataFrame.


        Parameters
        ----------
        i
            The index of the layer to be returned.
        kwargs
            This method does not accept any keyword arguments.

        Returns
        -------
        The layer data as a Pandas DataFrame.
        """
        _data = self.datahdu.data[self.fiber_conf.connected_fibers(), i]

        if self.varhdu is not None:
            _var = self.varhdu.data[self.fiber_conf.connected_fibers(), i]
        else:
            _var = None

        _out = {'x': self.transform[:, 1], 'y': self.transform[:, 0], 'value': _data}
        if _var is not None:
            _out['var'] = _var
        return pd.DataFrame(_out)
