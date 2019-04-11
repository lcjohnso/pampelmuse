"""
variables.py
============
Copyright 2013-2019 Sebastian Kamann

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
The class pampelmuse.core.psf.variables.Variables

Purpose
-------
The module handles the parameters of analytical PSF profile that is used by
PampelMuse. To this aim, it provides the class 'Variables' as an interface
to load, update, provide, and store the PSF parameters. 'Variables' inherits
from the class pampelmuse.core.parameters.Parameters that defines a more
general framework for dealing with parameters within PampelMuse.

Latest Git revision
-------------------
2019/04/11
"""
import logging
import numpy as np
import pandas as pd
from astropy.io import fits
from ..core.parameters import Parameters


__author__ = "Sebastian Kamann (s.kamann@ljmu.ac.uk)"
__revision__ = 20190411


logger = logging.getLogger(__name__)


class Variables(Parameters):
    """
    The class Variables serves as an interface to the parameters that define
    the PSF profile.

    The core of the class is a pandas DataFrame array that contains the value
    of each PSF parameter as a function of wavelength - and potentially an
    polynomial fit to it. To this aim, the Variables-class sub-classes the
    Parameters-class defined in pampelmuse.core.parameters.

    The analytical PSF that are currently supported are 'moffat' for a Moffat-
    shaped PSF and 'gauss' for a Gaussian-shaped PSF.
    """

    # define a dictionary containing the parameter names for each PSF profile supported by PampelMuse
    PROFILES = {"moffat": ["beta", "fwhm", "e", "theta"],
                "gauss": ["fwhm", "e", "theta"],
                "double_gauss": ["f12", "fwhm1", "fwhm2", "e", "theta"],
                "moffat_gauss": ["f12", "beta", "fwhm1", "fwhm2", "e", "theta"],
                "double_moffat": ["f12", "beta1", "fwhm1", "beta2", "fwhm2", "e", "theta"]}

    #  define dictionary providing tex-style representations of each parameter name for plotting
    LABELS = {"fwhm": r"$\mathrm{FWHM}\ [\mathrm{spaxel}]$",
              "beta": r"$\beta$",
              "e": r"$e$",
              "theta": r"$\theta\ [\mathrm{rad}]$",
              "beta1": r"$\beta_1$",
              "beta2": r"$\beta_2$",
              "fwhm1": r"$\mathrm{FWHM}_1\ [\mathrm{spaxel}]$",
              "fwhm2": r"$\mathrm{FWHM}_2\ [\mathrm{spaxel}]$",
              "f12": r"$f_\mathrm{12}$"}

    def __init__(self, profile='moffat', data=None, free=None, lut=None, **kwargs):
        """
        Initialize an instance of pampelmuse.core.psf.attributes.Parameters
        
        Parameters
        ----------
        profile : string, optional
            The name of the profile to be initialized. Currently "moffat",
            "gauss", and "double_gauss" are supported.
        data : pandas.DataFrame, optional
            The data used to initialize a new instance should be provided as a
            pandas DataFrame containing one parameter per column. By default,
            all parameters are set to NaN.
        free : array_like,
            A boolean array containing a flag for each parameter in 'data'
            indicating if the parameter is allowed to vary in a fit. By
             default, all parameters are free.
        lut : pandas.DataFrame, optional
            A look-up table containing correction factors for the fluxes
            determined from the analytical PSF profiles as a function of
            radius. The data frame should contain one column per radius and
            have the dispersion information available as Index.
        kwargs
            Any additional keyword arguments are passed on to the call of the
            initialization method of the super-class.
        """
        # check if requested PSF profile is supported
        self.profile = profile
        if self.profile in self.PROFILES:
            logger.info('Initializing PSF parameters for type "{0}" ...'.format(self.profile))
        else:
            logger.error('Invalid PSF type "{0}" provided.'.format(self.profile))
            logger.error('Supported PSF types: "{0}"'.format('", "'.join([p for p in self.PROFILES.keys()])))
            return

        # if no data provided, initialise all parameters to NaN
        if data is None:
            data = pd.DataFrame([[np.nan]*len(self.PROFILES[self.profile])], columns=self.PROFILES[self.profile])

        # call initialization method of super-class.
        super(Variables, self).__init__(data=data, free=free, **kwargs)

        # look-up-table
        if lut and not isinstance(lut, pd.DataFrame):
            logger.error('If provided, parameter "lut" must be pandas DataFrame, not {0}.'.format(type(data)))
        self.lut = lut

        if not self.lut.index.equals(self.data.index):
            logger.warning('Index data of PSF parameters and look-up table disagree.')

        # check if all required parameters are available
        for name in self.PROFILES[profile]:
            if name not in self.names:
                logger.error("Missing required PSF parameter '{0}'.".format(name))

    @classmethod
    def from_hdu(cls, hdu, profile=None, lut_hdu=None):

        # if no profile is provided, the header keyword PSFTYPE must be available in the FITS header
        if profile is None and 'PSFTYPE' not in hdu.header:
            logger.error('Cannot infer type of PSF from FITS header.')
        # if PSF type provided and in  header, latter has priority
        elif "PSFTYPE" in hdu.header:
            if profile is not None and profile != hdu.header['PSFTYPE']:
                logger.warning('Provided PSF type {0} does not match PSF type {1} in HDU'.format(
                    profile, hdu.header['PSFTYPE']))
            profile = hdu.header['PSFTYPE']

        # open look-up table
        if lut_hdu is not None:
            index_lut = np.arange(lut_hdu.header['NAXIS2'])
            index_lut *= lut_hdu.header['CDELT2']
            index_lut += lut_hdu.header['CRVAL2']

            columns_lut = np.arange(lut_hdu.header['NAXIS1'])
            columns_lut *= lut_hdu.header['CDELT1']
            columns_lut += lut_hdu.header['CRVAL1']

            lut = pd.DataFrame(lut_hdu.data, index=index_lut, columns=columns_lut)
        else:
            lut = None

        # call class method of super-class
        return super(Variables, cls).from_hdu(hdu=hdu, profile=profile, lut=lut)

    def info(self, names=None):
        """
        Print out some information about the current status of the PSF
        parameters, i.e. whether they have been defined, what is their (mean)
        value, are they free parameters, and have the been fitted with a
        smooth function.

        names : list, optional
            The parameter can be used to restrict the output to a subset of
            parameters or to print them in a given order. By default, all
            parameters are printed in the order of the Variables.PROFILES
            attribute.
        """
        # print out some information about the PSF profile in use
        logger.info("Summary of current status of PSF parameters:")
        logger.info(" Type: '{0}'".format(self.profile))
        logger.info(" Look-up table available? {0}".format(self.lut is not None))

        # if no order provided, use the one defined in PROFILES attribute
        if names is None:
            names = self.PROFILES[self.profile]

        # call info method of super-class. Provide parameter names to get order right.
        super(Variables, self).info(names=names)

    def interpolate(self):

        super(Variables, self).interpolate()

        if self.lut is not None:
            self.lut.interpolate(inplace=True)

    def resize(self, index):

        super(Variables, self).resize(index=index)

        if self.lut is not None:
            if not self.lut.index.isin(index).all():
                logger.warning('Changing dispersion axis as requested will alter existing values of look-up table.')
            self.lut = self.lut.reindex(index=index, method='nearest')
            assert(np.isfinite(self.lut.data).all())  # TODO: check if end-of-range NaNs are replaced.

    def make_hdu(self, include_fit=True, header_keys=None):
        """
        Write the current values of the PSF parameter to an fits.BinTableHDU
        instance.

        The method is basically a wrapper of the 'make_hdu' instance
        defined in the super-class and adds some header keywords that provide
        information about the PSF in use.
        
        Parameters
        ----------
        include_fit : boolean, optional
            Whether or not the fits to the parameters should  be included
            in the table.
        header_keys : dict, optional
            Any additional keywords that should be included in the FITS
            header of the  table.
        """
        if header_keys is None:
            header_keys = {}
        header_keys['EXTNAME'] = 'PSFPARS'
        header_keys['PSFTYPE'] = self.profile

        logger.info("Saving PSF parameters in BinTableHDU '{0}'...".format(header_keys['EXTNAME']))

        return super(Variables, self).make_hdu(include_fit=include_fit, header_keys=header_keys)

    def make_lut_hdu(self, header_keys=None):

        if self.lut is None:
            return None

        if header_keys is None:
            header_keys = {}
        header_keys['EXTNAME'] = 'PSFLUT'

        logger.info("Saving PSF look-up table in Image HDU '{0}' ...".format(header_keys['EXTNAME']))

        header_keys['CDELT1'] = np.diff(self.lut.columns).mean()
        header_keys['CRVAL1'] = self.lut.columns[0]
        header_keys['CDELT2'] = np.diff(self.lut.index).mean()
        header_keys['CRVAL2'] = self.lut.index[0]

        return fits.ImageHDU(self.lut.values, header=header_keys)
