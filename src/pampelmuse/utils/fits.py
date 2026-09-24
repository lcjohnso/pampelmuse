"""
fits.py
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
function pampelmuse.utils.fit.open_ifs_data
function pampelmuse.utils.fits.open_prm
function pampelmuse.utils.fits.save_prm

Purpose
-------
The provided functions facilitate working with FITS data.

Latest Git revision
-------------------
2026/09/24
"""
import os
import logging
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from ..core import Sources
from ..core.coordinates import Transformation
from .. import instruments 
from ..psf import Variables


__author__ = "Sebastian Kamann (s.kamann@ljmu.ac.uk)"
__revision__ = 20260924


logger = logging.getLogger(__name__)


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
    instrument_class = instruments.GenericIFS
    for hdu in fitsdata:
        if 'INSTRUME' in hdu.header:
            name = hdu.header['INSTRUME']
            logger.info('IFS data is coming from instrument {0}.'.format(name))
            if name == 'MUSE':
                # check if pixtable was provided
                if 'ESO PRO CATG' in hdu.header and hdu.header['ESO PRO CATG'] == 'PIXTABLE_REDUCED':
                    name = "PIXTABLE"
            if name not in instruments.known_instruments():
                logger.warning(f'PampelMuse was not tested on "{name}" data.')
            elif name not in instruments.available_instruments():
                logger.error(f'PampelMuse installation is missing dependencies for processing "{name}" data.')
            instrument_class = instruments.available_instruments()[name]

    if instrument_class == instruments.available_instruments().get('GIRAFFE'):
        try:
            instrument = instrument_class.from_fiber_setup_hdu(fitsdata['FIBER_SETUP'])
        except KeyError:
            logger.warning('Cannot identify GIRAFFE fiber setup. Using default.')
            instrument = instrument_class(**kwargs)
    elif instrument_class == instruments.available_instruments().get('MEGARA'):
        try:
            instrument = instrument_class.from_fiber_setup_hdu(fitsdata['FIBERS'], **kwargs)
        except KeyError:
            raise NotImplementedError('Cannot identify MEGARA fiber setup. Please provide a valid FIBERS HDU.')
    else:
        instrument = instrument_class(**kwargs)
    instrument.search_configuration(hdu)
            
    if isinstance(instrument, instruments.GenericIFS) and not isinstance(instrument, instruments.available_instruments()['MUSE']):
        logger.warning('No instrument-specific class available, IFS data must be 3dim.')
    elif instrument.name is not None:
        logger.info(f'Input data identified as type "{instrument.name}".')

    # for later usage, store primary header
    instrument.primary_header = fitsdata[0].header

    logging.info('Loading data ...')
    # assumption is that data is stored in first HDU that contains data
    for i, hdu in enumerate(fitsdata):
        if hdu.header['NAXIS'] > 0:
            instrument.open(datafile, extno=i, **kwargs)
            break

    # open HDU with residuals if it exists
    if os.path.exists("{0}.res.fits".format(prefix)):
        logger.info('Found file with residuals: {0}.res.fits'.format(prefix))
        instrument.open("{0}.res.fits".format(prefix), extno=0, isresiduals=True, **kwargs)

    # if a MUSE pixtable is loaded, all information should be there now
    if isinstance(instrument, instruments.available_instruments().get('PIXTABLE', type(None))):
        return instrument

    # open HDU with variances
    if os.path.exists("{0}.err.fits".format(prefix)):
        instrument.open("{0}.err.fits".format(prefix), extno=0, isvar=True, **kwargs)
    elif os.path.exists("{0}.var.fits".format(prefix)):
        instrument.open("{0}.var.fits".format(prefix), extno=0, isvar=True, **kwargs)
    else:
        for hduname in ['STAT', 'ERROR', 'SIGMA', 'VARIANCE', 'ERR']:
            if hduname in fitsdata:
                instrument.open(datafile, extno=fitsdata.index_of(hduname), isvar=True, **kwargs)
            elif hduname.lower() in fitsdata:
                instrument.open(datafile, extno=fitsdata.index_of(hduname.lower()), isvar=True, **kwargs)
    if instrument.varhdu is None:
        logger.error('Did not find variances for IFS data.')

    # open (optional) HDU with mask
    if os.path.exists("{0}.msk.fits".format(prefix)):
        instrument.open("{0}.msk.fits".format(prefix), extno=0, ismask=True, **kwargs)
    else:
        for hduname in ['MASK', 'data_quality']:
            if hduname in fitsdata:
                instrument.open(datafile, extno=fitsdata.index_of(hduname), ismask=True, **kwargs)
    if instrument.maskhdu is None:
        logger.warning('Did not find bad-pixel mask for IFS data.')

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
    catalog = Table.read(filename, hdu=1)
    sources = Sources(catalog=catalog.to_pandas())

    # Spectra
    try:
        i = hdulist.index_of("SPECTRA")
    except KeyError:
        logger.critical("No spectra (HDU 'SPECTRA') have been provided.")
    else:
        try:
            j = hdulist.index_of("WAVE")
        except KeyError:
            wave_hdu = None
        else:
            wave_hdu = hdulist[j]
        sources.open_spectra_hdu(hdulist[i], data_wave_hdu=wave_hdu)

    if 0 < ndisp != sources.n_dispersion:
        logger.error("PRM-file and IFS data have incompatible dispersion dimensions.")

    # Positions
    logger.info("Obtaining information about source coordinates...")

    # note that if individual source positions are fitted, there may be two extensions named POSITIONS, containing
    # the coordinates and the polynomial fits to them, respectively. To handle this case, we search for HDUs using
    # the EXTNAME and EXTVER keywords. The latter is defaulted to 1 if missing.
    try:
        i = hdulist.index_of(("POSITIONS", 1))
        sources.open_coordinates_hdu(hdulist[i])
    except KeyError:
        pass

    try:
        i = hdulist.index_of(("POSITIONS", 2))
        sources.open_coordinates_hdu(hdulist[i])
    except KeyError:
        pass

    try:
        i = hdulist.index_of("POSPARS")
        sources.transformation = Transformation.from_hdu(hdulist[i])
    except KeyError:
        logger.warning("No coord. transf. (HDU 'POSPARS') detected.")
        sources.ifs_coordinates_free[sources.status < 4] = True
    sources.info()

    # Initializing PSF properties
    logging.info("Obtaining PSF properties...")
    try:
        i = hdulist.index_of("PSFPARS")
    except KeyError:  # otherwise, assume Gaussian, round with FWHM=3.0pixel
        logger.warning("No HDU 'PSFPARS' found. Using default PSF.")
        psf_attributes = Variables(profile="gauss",
                                   data=pd.DataFrame([(3.0, 0.0, 0.0)], columns=["fwhm", "e", "theta"]))
        psf_attributes.free = pd.Series([0, 0, 0], index=psf_attributes.names)
    else:
        lut = None
        try:
            j = hdulist.index_of('PSFLUT')
            lut = hdulist[j]
        except KeyError:
            pass
        finally:
            psf_attributes = Variables.from_hdu(hdu=hdulist[i], lut_hdu=lut)
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

    logger.info("Saving results to file {0}...".format(filename))

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
        new_positions_hdu = sources.make_ifs_coordinates_hdu(fit=True)
        # make sure that EXTVER key is set properly if >1 POSITIONS extension exist
        if 'POSITIONS' in [hdu.name for hdu in hdulist]:
            new_positions_hdu.ver = 2
        hdulist.append(new_positions_hdu)

    # store coordinate transformation (only if a valid one exists)
    if sources.transformation.valid:
        hdulist.append(sources.transformation.make_hdu())

    # store PSF (only if a valid one exists)
    if psf_attributes.valid:
        psf_hdu = psf_attributes.make_hdu()
        hdulist.append(psf_hdu)
        if psf_attributes.lut is not None:  # check if look-up table is in use
            lut_hdu = psf_attributes.make_lut_hdu()
            hdulist.append(lut_hdu)
    else:
        logger.error("PSF profile invalid. Not included in PRM-file.")

    # Save the file
    if os.path.exists(filename):
        logger.info('Overwriting existing file {0}...'.format(filename))
        os.remove(filename)

    hdulist.writeto(filename, overwrite=True)


def read_config_from_header(header, existing_configuration=None):
    """
    Read an existing configuration of PampelMuse from a FITS header
    
    Parameters
    ----------
    header : fits.Header
        The header instance containing the configuration.
    existing_configuration : dict, optional
        An existing configuration that should be updated using the values from
        the header. If provided, the code will ignore header keys that have no
        counterpart in the existing configuration and change the values of the
        provided configuration to the values found in the header.

    Returns
    -------
    config : dict
        The configuration fetched from the header.
    """
    # prepare dictionary
    config = {}

    # loop over header keys:
    for header_key, value in header.items():
        # each valid configuration key is structured as 'HIERARCH PAMPELMUSE <SECTION> <KEY> [<NO>] = <VALUE>'
        keywords = header_key.split()
        if keywords[0] == 'PAMPELMUSE' and len(keywords) >= 3:
            if len(keywords) == 3:
                section, key = keywords[1:]
                part = None
            else:
                section, key, part = keywords[1:]

            # add section to dictionary if it does not exist
            if section not in config.keys():
                config[section] = {}

            # add value=key pair to section
            # empty strings are converted to None
            if value == '' or isinstance(value, fits.card.Undefined):
                config[section][key] = None
            elif part is None:
                config[section][key] = value
            # if length of header key is 4, it is part of a list of values [e.g. catalog.saturated or polynom.mask]
            elif int(part) == 1:
                config[section][key] = [value, ]
            # use insert to make sure values are in order
            else:
                try:
                    config[section][key].insert(int(part) - 1, value)
                except ValueError:
                    raise IOError('Header contains invalid configuration: {0}'.format(header_key))

    # check if existing configuration should be updated
    if existing_configuration is None:
        # if not, just return header values
        return config
    else:
        # loop over configuration sections
        for section, items in config.items():
            # loop over keys in section
            for key, value in items.items():
                # check if key exists and differs from default value
                if section in existing_configuration.keys() and key in existing_configuration[section].keys():
                    if existing_configuration[section][key] != value:
                        # update
                        existing_configuration[section][key] = value
        return existing_configuration


def make_header_from_config(config, **kwargs):
    """
    Create a FITS header from the current configuration of PampelMuse.

    Parameters
    ----------
    config : dict
        Dictionary containing the configuration of PampelMuse
    kwargs : optional
        The additional keyword arguments can be used to pass any keywords that
        are not included in the configuration but should be written to the
        FITS header nevertheless

    Returns
    -------
    header : fits.Header instance
        The header including the configuration of PampelMuse
    """
    # prepare empty header
    header = fits.Header()

    # loop over configuration
    # the structure of the keywords is HIERARCH PAMPELMUSE <SECTION> <KEYWORD>
    for key, item in config.items():
        for sub_key, value in item.items():
            # parameters 'mask' and 'saturated' can be lists and need to be handled separately:
            if isinstance(value, (list, tuple)):
                for i, value_i in enumerate(value, start=1):
                    card = fits.Card('HIERARCH PAMPELMUSE {0} {1} {2}'.format(key, sub_key, i), value_i)
                    header.append(card=card)
            else:
                card = fits.Card('HIERARCH PAMPELMUSE {0} {1}'.format(key, sub_key), value)
                try:
                    card.verify()
                except ValueError:
                    logger.warning('Configuration key {0} = {1} is too long for header.'.format(
                        card.keyword, card.value))
                else:
                    header.append(card=card)

    # add additional keywords (if any)
    for key, value in kwargs:
        header['HIERARCH PAMPELMUSE {0}'.format(key)] = value

    # return header
    return header


def get_history(filename):
    """
    Get HISTORY keywords from the primary HDU header of a FITS file as a list.

    Parameters
    ----------
    filename : str
        The name of the FITS-file.

    Returns
    -------
    history : list
        The list of HISTORY entries. If no HISTORY keys exist, an empty list
        is returned.
    """
    header = fits.getheader(filename, ext=0)
    if 'HISTORY' in header:
        history = list(header['HISTORY'])
    else:
        history = []
    return history
