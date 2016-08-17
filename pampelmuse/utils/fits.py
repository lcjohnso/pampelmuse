"""
fits.py
=======
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
Provides
--------
function pampelmuse.utils.fit.open_ifs_data
function pampelmuse.utils.fits.open_prm
function pampelmuse.utils.fits.save_prm

Purpose
-------
The provided functions facilitate working with FITS data.

Latest SVN revision
-------------------
371, 2016/08/17
"""
import os
import logging
import numpy as np
from astropy.io import fits
from ..core import Sources
from ..core.coordinates import Transformation
from ..instruments import *
from ..instruments import available_instruments
from ..psf import Variables


__author__ = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__revision__ = 371


def open_ifs_data(prefix, **kwargs):
    """
    This function is designed to open FITS files containing IFS data and
    use it to initialize one of classes provided in pampelmuse.instruments
    so that in can be used in the analysis.

    Parameters
    ----------
    prefix : string
        The common prefix of the FITS files to be opened. The code will look
        for several files that may contain the IFS data, variances, a bad-
        pixel spectrumMask, or fit residuals:
        * <prefix>.fits, <prefix>.dat.fits - data (extension DATA or first
          extension containing data)
        * <prefix>.fits, <prefix>.err.fits, <prefix>.var.fits - variances
          (extension STAT, ERROR, or SIGMA)
        * <prefix>.fits, <prefix>.msk.fits - a bad pixel-spectrumMask (extension MASK)
        * <prefix>.res.fits - residuals from a previous fit (first extension
          containing data)
    kwargs
        Any additional arguments are passed on to the initialisation of the
        pampelmuse.instrument class.

    Returns
    -------
    instrument : an instance of a subclass of pampelmuse.instrument.Instrument
        The instrument instance that can be used throughout the analysis.
    """

    if os.path.exists("{0}.dat.fits".format(prefix)):
        datafile = "{0}.dat.fits".format(prefix)
    elif os.path.exists("{0}.fits".format(prefix)):
        datafile = "{0}.fits".format(prefix)
    else:
        raise IOError("Cannot find IFS data: {0}.dat.fits or {0}.fits must be present".format(prefix))
    fitsdata = fits.open(datafile, memmap=False)

    # Try to find out which instrument was used:
    instrument = GenericIFS()
    for hdu in fitsdata:
        if 'INSTRUME' in hdu.header:
            logging.info('IFS data is coming from instrument {0}.'.format(hdu.header['INSTRUME']))
            if hdu.header['INSTRUME'] not in available_instruments:
                logging.warning('PampelMuse was not tested on "{0}"-data.'.format(hdu.header['INSTRUME']))
            elif hdu.header['INSTRUME'] == 'MUSE':  # check if pixtable was provided
                if 'ESO PRO CATG' in hdu.header and hdu.header['ESO PRO CATG'] == 'PIXTABLE_REDUCED':
                    logging.info('IFS data is a MUSE pixtable.')
                    instrument = MusePixtable(**kwargs)
                else:
                    logging.info('IFS data is a MUSE data cube.')
                    instrument = Muse(**kwargs)
            elif hdu.header['INSTRUME'] == 'GIRAFFE':
                try:
                    instrument = Argus.from_fiber_setup_hdu(fitsdata['FIBER_SETUP'])
                except KeyError:
                    instrument = Argus(**kwargs)
            else:
                instrument = available_instruments[hdu.header['INSTRUME']](**kwargs)
            instrument.search_configuration(hdu)
            break

    if isinstance(instrument, GenericIFS) and not isinstance(instrument, Muse):
        logging.warning('Could not determine instrument used to obtain IFS data.')
        logging.warning('IFS data must be provided as a 3dim. cube.')

    # for later usage, store primary header
    instrument.primary_header = fitsdata[0].header

    # assumption is that data is stored in first HDU that contains data
    for i, hdu in enumerate(fitsdata):
        if hdu.header['NAXIS'] > 0:
            instrument.open(datafile, extno=i, **kwargs)
            break

    # open HDU with residuals if it exists
    if os.path.exists("{0}.res.fits".format(prefix)):
        logging.info('Found file with residuals: {0}.res.fits'.format(prefix))
        instrument.open("{0}.res.fits".format(prefix), extno=0, isresiduals=True, **kwargs)

    if isinstance(instrument, MusePixtable):  # if a MUSE pixtable is loaded, all information should be there now
        return instrument

    # open HDU with variances
    if os.path.exists("{0}.err.fits".format(prefix)):
        instrument.open("{0}.err.fits".format(prefix), extno=0, isvar=True, **kwargs)
    elif os.path.exists("{0}.var.fits".format(prefix)):
        instrument.open("{0}.var.fits".format(prefix), extno=0, isvar=True, **kwargs)
    else:
        for hduname in ['STAT', 'ERROR', 'SIGMA']:
            if hduname in fitsdata:
                instrument.open(datafile, extno=fitsdata.index_of(hduname), isvar=True, **kwargs)
    if instrument.varhdu is None:
        logging.error('Did not find variances for IFS data.')

    # open (optional) HDU with spectrumMask
    if os.path.exists("{0}.msk.fits".format(prefix)):
        instrument.open("{0}.msk.fits".format(prefix), extno=0, ismask=True, **kwargs)
    else:
        if 'MASK' in fitsdata:
            instrument.open(datafile, extno=fitsdata.index_of('MASK'), ismask=True, **kwargs)
    if instrument.maskhdu is None:
        logging.warning('Did not find bad-pixel mask for IFS data.')

    return instrument


def open_prm(filename, ndisp=-1):
    """
    This function opens a PampelMuse prm-file and extracts the contained
    information about the source catalogue, the spectra, the coordinates, and
    the PSF. Returned are an instance of pampelmuse.core.Sources containing
    the information about the source catalog, the fluxes and positions, and
    an instance of pampelmuse.psf.Variables containing the information about
    the PSF parameters.

    Parameters
    ----------
    filename : string
        The name of the prm-file that should be opened.
    ndisp : int, optional
        To make sure that the dispersion information is correct, this
        parameter may be set to a value >0 in which case the code will
        check if the prm-file contains n_disp layers.

    Returns
    -------
    sources : an instance of pampelmuse.core.Sources
        This instance contains the extracted information about the source
        catalogue, the fluxes, and positions of the sources.
    psf_attributes : an instance of pampelmuse.psf.Variables
        This instance contains the extracted information about the PSF
        parameters.
    """
    logging.info('Reading PampelMuse setup from {0}...'.format(filename))
    hdulist = fits.open(filename)

    # Collect information about sources/create Sources-instance
    logging.info("Obtaining information on sources in the field...")

    # Source catalogue
    sources = Sources(data=hdulist["SOURCES"])

    # Spectra
    try:
        i = hdulist.index_of("SPECTRA")
        try:
            j = hdulist.index_of("WAVE")
            wave_hdu = hdulist[j]
        except KeyError:
            wave_hdu = None
        try:
            k = hdulist.index_of("SIGMASPC")
            sigma_hdu = hdulist[k]
        except KeyError:
            sigma_hdu = None
        sources.open_spectra_hdu(hdulist[i], sigma_hdu=sigma_hdu, wave_hdu=wave_hdu)
    except KeyError:
        logging.critical("No spectra (HDU 'SPECTRA') have been provided.")

    if 0 < ndisp != sources.ndisp:
        logging.error("PRM-file and IFS data have incompatible dispersion dimensions.")

    # Positions
    logging.info("Obtaining information about source coordinates...")

    try:
        i = hdulist.index_of("POSITIONS")
        sources.open_coordinates_hdu(hdulist[i])
    except KeyError:
        pass

    try:
        i = hdulist.index_of("POSPARS")
        sources.transformation = Transformation(hdu=hdulist[i])
    except KeyError:
        logging.warning("No coord. transf. (HDU 'POSPARS') detected.")
        sources.ifs_coordinates_free[sources.status < 4] = True
    sources.info()

    # Initializing PSF properties
    logging.info("Obtaining PSF properties...")
    try:
        i = hdulist.index_of("PSFPARS")
        psf_attributes = Variables(hdu=hdulist[i])
    except KeyError:  # otherwise, assume Gaussian, round with FWHM=3.0pixel
        logging.warning("No HDU 'PSFPARS' found. Using default PSF.")

        psf_attributes = Variables(**{"profile": "gauss", "fwhm": 3.0})
        psf_attributes.free = np.atleast_1d([1, 0, 0])

    psf_attributes.info()

    return sources, psf_attributes


def save_prm(sources, psf_attributes, filename, header=None):
    """
    This function saves the current status of an analysis to a PampelMuse
    prm-file.

    Parameters
    ----------
    sources : an instance of pampelmuse.core.Sources
        This instance should contain all information about the source
        catalogue, the fluxes, and positions of the sources.
    psf_attributes: an instance of pampelmuse.psf.Variables
        This instance should contain available information about the PSF
        parameters.
    filename : string
        The name used to store the prm-file.
    header : an instance of astropy.fits.Header
        A header containing additional keywords that should be stored in
        the prm-file.

    Returns
    -------
    Nothing, only the prm-file is stored to disk.
    """

    logging.info("Saving results to file {0}...".format(filename))

    hdulist = fits.HDUList(hdus=[fits.PrimaryHDU(header=header), ])

    # writing source catalog
    hdulist.append(sources.make_catalogue_hdu())

    # storing spectra
    spectra_hdu, wave_hdu = sources.make_spectra_hdu()
    hdulist.append(spectra_hdu)
    if wave_hdu is not None:
        hdulist.append(wave_hdu)

    # Store IFU coordinates (only needed if either they are free fit parameters or polynomial fits exist
    if sources.ifs_coordinates_free.any():
        hdulist.append(sources.make_ifs_coordinates_hdu(fit=False))
    if sources.ifs_coordinates_fitted.any():
        hdulist.append(sources.make_ifs_coordinates_hdu(fit=True))

    # store coordinate transformation (only if a valid one exists)
    if sources.transformation.valid:
        hdulist.append(sources.transformation.make_hdu())

    # store PSF (only if a valid one exists)
    if psf_attributes.valid:
        hdulist.append(psf_attributes.make_hdu())

    # Save the file
    if os.path.exists(filename):
        logging.info('Overwriting existing file {0}...'.format(filename))
        os.remove(filename)

    hdulist.writeto(filename, clobber=True)
