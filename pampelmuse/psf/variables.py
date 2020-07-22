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
2019/04/23
"""
import logging
import numpy as np
import pandas as pd
from astropy.io import fits
from ..core.parameters import Parameters


__author__ = "Sebastian Kamann (s.kamann@ljmu.ac.uk)"
__revision__ = 20190423


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
                "double_moffat": ["f12", "beta1", "fwhm1", "beta2", "fwhm2", "e", "theta"],
                "maoppy": ["r0", "bck", "amp", "alpha", "beta", "e", "theta"]}

    #  define dictionary providing tex-style representations of each parameter name for plotting
    LABELS = {"fwhm": r"$\mathrm{FWHM}\,[\mathrm{spaxel}]$",
              "beta": r"$\beta$",
              "e": r"$e$",
              "theta": r"$\theta\,[\mathrm{rad}]$",
              "beta1": r"$\beta_1$",
              "beta2": r"$\beta_2$",
              "fwhm1": r"$\mathrm{FWHM}_1\,[\mathrm{spaxel}]$",
              "fwhm2": r"$\mathrm{FWHM}_2\,[\mathrm{spaxel}]$",
              "f12": r"$f_\mathrm{12}$",
              "r0": r"$r_{\rm 0}\,[\mathrm{m}]$",
              "bck": r"$C\,[\mathrm{rad}^2\mathrm{m}^2]$",
              "amp": r"$A\,[\mathrm{rad}^2]$",
              "alpha": r"$\alpha\,[\mathrm{m}^{-1}]$"}

    def __init__(self, profile='moffat', data=None, free=None, lut=None, **kwargs):
        """
        Initialize an instance of pampelmuse.psf.variables.Variables
        
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
            if self.profile == 'maoppy':
                logger.info('Please cite Fetick et al. (2019, A&A, 628, A99) as reference for the MAOPPY PSF.')
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
        if lut is not None:
            self.lut = self.verify_data_frame(lut, swaplevel=True)
            if self.lut is None:
                logger.error('Provided look-up table has invalid format and will be ignored.')
        else:
            self.lut = None

        if self.lut is not None and not self.lut.index.equals(self.data.index):
            logger.warning('Index data of PSF parameters and look-up table disagree.')

        # check if all required parameters are available
        for name in self.PROFILES[profile]:
            if name not in self.names:
                logger.error("Missing required PSF parameter '{0}'.".format(name))

    @classmethod
    def from_hdu(cls, hdu, profile=None, lut_hdu=None):
        """

        Parameters
        ----------
        hdu
        profile
        lut_hdu : instance of astropy.io.fits.ImageHDU, optional
            An additional HDU containing the look-up table for the PSF.

        Returns
        -------
        variables : instance of pampelmuse.psf.variables.Variables
            The provided data is used to initialize a new instance of the
            Variables class.
        """

        # if no profile is provided, the header keyword PSFTYPE must be available in the FITS header
        if profile is None and 'PSFTYPE' not in hdu.header:
            logger.error('Cannot infer type of PSF from FITS header.')
        # if PSF type provided and in  header, latter has priority
        elif "PSFTYPE" in hdu.header:
            if profile is not None and profile != hdu.header['PSFTYPE']:
                logger.warning('Provided PSF type {0} does not match PSF type {1} in HDU'.format(
                    profile, hdu.header['PSFTYPE']))
            profile = hdu.header['PSFTYPE']

        # open look-up table (if any)
        if lut_hdu is not None:
            if not isinstance(lut_hdu, fits.ImageHDU):
                logger.error('If provided, "lut_hdu" must be instance of astropy.io.fits.ImageHDU.')
                lut = None
            else:
                # pandas requires native byte order while FITS data is big-endian.
                data = lut_hdu.data.byteswap().newbyteorder()

                # recover radii at which look-up table is defined
                radii = np.arange(lut_hdu.header['NAXIS1'], dtype=np.float64)
                for i in range(1, lut_hdu.header['NAXIS1'] + 1):
                    if 'RLUT{0:04d}'.format(i) in lut_hdu.header:
                        radii[i-1] = lut_hdu.header['RLUT{0:04d}'.format(i)]
                    else:
                        logger.warning('Missing header keyword "RLUT{0:04d}" in look-up table HDU.'.format(i))
                if lut_hdu.header['NAXIS'] == 2:
                    lut = pd.DataFrame(data, columns=pd.MultiIndex.from_product([['value', ], radii]))
                    lut = lut.reindex(columns=pd.MultiIndex.from_product([['value', 'fit'], radii]))
                else:
                    lut_value = pd.DataFrame(data[0], columns=radii)
                    lut_fit = pd.DataFrame(data[1], columns=radii)
                    lut = pd.concat([lut_value, lut_fit], keys=('value', 'fit'), axis=1)

            # _lut = Table.read(lut_hdu)
            # lut = pd.DataFrame(_lut['value'].data.T.byteswap().newbyteorder(), columns=_lut['radius'])

            # if 'polyfit' in _lut.colnames:
            #     lut_fit = pd.DataFrame(_lut['polyfit'].data.T.byteswap().newbyteorder(), columns=_lut['radius'])
            #     lut = pd.concat([lut, lut_fit], keys=('value', 'fit'), axis=1).swaplevel(
            #         0, 1, axis=1).sort_index(axis=1)
        else:
            lut = None

        # call class method of super-class
        return super(Variables, cls).from_hdu(hdu=hdu, profile=profile, lut=lut)

    @property
    def lut_fitted(self):
        """
        Returns boolean flag indicating if the look-up  table has been fitted
        using polynomials.
        """
        if self.lut is None:
            return False
        return (~self.lut['fit'].isnull()).any().any()

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
            if len(index) == len(self.lut.index):
                self.lut.index = index
            else:
                if not self.lut.index.isin(index).all():
                    logger.warning('Changing dispersion axis as requested will alter existing values of look-up table.')
                self.lut = self.lut.reindex(index=index, method='nearest')

    def fit_lut(self, order, mask=None):
        """
        Perform polynomial fits to the look-up table data.

        The data are fitted as a function of wavelength in a column-by-column
        manner, i.e. each radial bin is fitted independently.

        Parameters
        ----------
        order: int
            The order of the polynomials that should be fitted to the data.
        mask : pandas.Series, optional
            An optional bad-pixel mask. If specified, it must be a boolean
            pandas Series that uses the same indices as the data. Bad pixels
            should be set to True, good ones to False.
        """
        if self.lut is None:  # nothing to do
            return

        # create pseudo-mask if None provided
        if mask is None:
            mask = pd.Series(0, index=self.lut.index)

        for i in self.lut.columns.levels[1]:
            mask_i = mask | self.lut[('value', i)].isnull()
            _coeffs = np.polyfit(self.lut.index[~mask_i], self.lut.loc[~mask_i, ('value', i)], deg=order)
            f = np.poly1d(_coeffs)
            self.lut[('fit', i)] = f(self.lut.index)

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

    def make_lut_hdu(self, include_fit=True, header_keys=None):

        if self.lut is None:
            return None

        header = fits.Header()
        if header_keys is not None:
            for key, value in header_keys.items():
                header[key] = value
        header['EXTNAME'] = 'PSFLUT'

        logger.info("Saving PSF look-up table in BinTableHDU '{0}' ...".format(header['EXTNAME']))

        if include_fit:
            data = np.stack([self.lut.loc[:, 'value'], self.lut.loc[:, 'fit']], axis=0)
        else:
            data = self.lut['value']

        for i, r in enumerate(self.lut.columns.levels[1], start=1):
            header['RLUT{0:04d}'.format(i)] = r

        return fits.ImageHDU(data, header=header)
