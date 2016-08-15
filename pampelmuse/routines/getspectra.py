"""
getspectra.py
=============
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
function pampelmuse.routines.getspectra

Purpose
-------
The function getspectra is called when Pampelmuse is started with the
GETSPECTRA command. Its purpose is to read all the spectra that have
been extracted from a datacube and store them as individual FITS files.
The structure of each created FITS file is such that the primary exten-
sion contains the spectrum and the second extension (named 'SIGMA') con-
tains the uncertainty for the measured spectra as estimated by the CUBE-
FIT routine of PampelMuse.

Added
-----
2014/06/26

Latest SVN revision
-------------------
365 # 2016/08/15
"""
import datetime
import logging
import os
import numpy as np
from astropy.io import fits
from pampelmuse.utils.statistics import der_snr
from ..utils.fits import open_prm

__author__ = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__revision__ = 365


def add_parser(parser):
    p = parser.add_parser("GETSPECTRA", help="""
    Store all spectra extracted from a datacube as individual FITS files.""")
    p.add_argument("specprefix", type=str, help="""
    The common prefix of all generated FITS filenames, typically the name of
    the target that has been observed.""")
    p.add_argument("--folder", type=str, default="spectra",  help="""
    The folder in which the spectra will be stored. Note that the folder will
    be created if it does not exist.""")
    p.set_defaults(func=getspectra)


def getspectra(pampelmuse):
    logging.info("Executing requested routine GETSPECTRA [Revision {0:d}].".format(__revision__))

    if not os.path.exists(pampelmuse.getspectra.folder):
        logging.info("Creating folder '{0}' to store spectra...".format(pampelmuse.getspectra.folder))
        os.mkdir(pampelmuse.getspectra.folder)

    # Collect information about sources/create Sources-instance:
    logging.info("Obtaining information on sources in the field...")
    sources, psf_attributes = open_prm(filename=pampelmuse.filename, ndisp=pampelmuse.ifs_data.wave.size)

    # initialize an empty header that will contain the header cards that are the same for all spectra collected from
    # an observation
    header = fits.Header()
        
    # add dispersion information to header
    if sources.data_wave is None:
        header["CRPIX1"] = sources.crpix - pampelmuse.start
        header["CRVAL1"] = sources.crval
        header["CDELT1"] = sources.cdelt
        header["CTYPE1"] = sources.ctype
        header["CUNIT1"] = sources.cunit
        header["WSTART"] = (sources.crval - (sources.crpix - pampelmuse.start - 1)*sources.cdelt)
        header["WSTEP"] = sources.cdelt
    else:
        header['WAVE'] = 'WAVE'  # if dedicated HDU with wavelength information exists

    for key in ["OBJECT", "TELESCOP", "INSTRUME", "DATE-OBS", "MJD-OBS", "JUL-DATE", "EXPTIME", "RA", "DEC", "EPOCH"]:
        if pampelmuse.ifs_data.primary_header is not None and key in pampelmuse.ifs_data.primary_header:
            header[key] = pampelmuse.ifs_data.primary_header[key]
        elif key in pampelmuse.ifs_data.datahdu.header:
            header[key] = pampelmuse.ifs_data.datahdu.header[key]
        else:
            logging.warning('Missing header keyword "{0}".'.format(key))

    # add JUL-DATE keyword if only MJD-OBS is provided
    if "MJD-OBS" in header and "JUL-DATE" not in header:
        header["JUL-DATE"] = header["MJD-OBS"] + 2400000.5
    
    if "JUL-DATE" in header:
        juldate = header["JUL-DATE"]
    else:
        juldate = 0
    
    # add keywords describing observing conditions
    for ambikey in ["AIRM START", "AIRM END", "AMBI FWHM START", "AMBI FWHM END", "AMBI PRES START", "AMBI PRESS END", 
                    "AMBI RHUM", "AMBI TAU0", "AMBI TEMP", "AMBI WINDDIR", "AMBI WINDSP"]:
        inkey = "ESO TEL {0}".format(ambikey)
        if pampelmuse.ifs_data.primary_header is not None and inkey in pampelmuse.ifs_data.primary_header:
            header["HIERARCH OBSERVATION {0}".format(ambikey)] = pampelmuse.ifs_data.primary_header[inkey]
        elif inkey in pampelmuse.ifs_data.datahdu.header:
            header["HIERARCH OBSERVATION {0}".format(ambikey)] = pampelmuse.ifs_data.datahdu.header[inkey]
        else:
            pass

    # add possibly relevant PampelMuse keywords to the FITS header
    for key in ["version", "revision", "filename", "utc", "passband"]:
        if hasattr(pampelmuse, key):
            if isinstance(getattr(pampelmuse, key), str) and len(getattr(pampelmuse, key)) > (55-len(key)):
                # check that entry not longer than 80 characters
                header["hierarch PAMPELMUSE {0}".format(key.upper())] = getattr(pampelmuse, key)[:55-len(key)]
            else:
                header["hierarch PAMPELMUSE {0}".format(key.upper())] = getattr(pampelmuse, key)

    # add estimate for seeing on detector using the fitted FWHM of the PSF profile
    if psf_attributes is not None:
        i = list(psf_attributes.names).index("fwhm")
        header["hierarch PAMPELMUSE SEEING"] = (psf_attributes.data[~psf_attributes.mask[:, i, 0], i, 0].mean(),
                                                "Fitted PSF FWHM in spaxels")

    # load spectra for all components
    sources.load_spectra()

    # loop over all components, i.e. all spectra
    for i in xrange(sources.ncmp):  # collect spectrum and properties of every requested spectrum in observation
        j = sources.indexmap[i][0]  # The index of the primary (brightest) component contributing to the spectrum

        # sanity check: sources with status 1 (=near brighter source) or -1 (=not used) should have no spectrum
        if sources.status[j] in [1, -1]:
            logging.error("Found spectrum for source #{0} with status {1}".format(sources.ID[j], sources.status[j]))
            continue

        flag = 0
        # each spectrum is assigned a flag which is the sum of the following numbers:
        #  1 - More than one star contribute to the spectrum
        #  2 - The estimated S/N of the spectrum is <10
        #  4 - The mean flux of the spectrum is <0
        #  8 - The centroid of the star is outside the field of view

        # load spectrum for current component into memory
        primary_hdu = fits.PrimaryHDU(sources.data[pampelmuse.start:pampelmuse.stop, i, 0], header=header)

        sigma_hdu = fits.ImageHDU(sources.data[pampelmuse.start:pampelmuse.stop, i, 1])
        sigma_hdu.header['EXTNAME'] = 'SIGMA'
        # copy dispersion information
        for key in ['CRPIX1', 'CRVAL1', 'CDELT1', 'CTYPE1', 'CUNIT1', 'WSTART', 'WSTEP', 'WAVE']:
            if key in header:
                sigma_hdu.header[key] = header[key]

        if sources.data_wave is not None:
            wave_hdu = fits.ImageHDU(sources.data_wave[pampelmuse.start:pampelmuse.stop, i])
            wave_hdu.header['EXTNAME'] = 'WAVE'
        else:
            wave_hdu = None

        if sources.status[j] in [2, 3]:
            
            source_id = sources.ID[j]
            logging.info("Processing source #{0} [{1}/{2}]".format(source_id, i+1, sources.ncmp))
            
            primary_hdu.header["HIERARCH STAR ID"] = source_id

            # update RA & Dec values if provided in source catalog
            if not np.isnan(sources.ra[j]) and not np.isnan(sources.dec[j]):
                primary_hdu.header["RA"] = sources.ra[j]
                primary_hdu.header["DEC"] = sources.dec[j]
                primary_hdu.header["HIERARCH STAR RA"] = sources.ra[j]
                primary_hdu.header["HIERARCH STAR DEC"] = sources.dec[j]

            # add magnitude to header
            if not np.isnan(sources.mag[j]):
                primary_hdu.header["HIERARCH STAR MAG"] = sources.mag[j]

            #  Check if more than one source contribute to the spectrum
            cmt = "Whether multiple stars contribute"
            if len(sources.indexmap[i]) > 1:
                flag += 1
                for l in xrange(1, len(sources.indexmap[i])):
                    # if so, add RA, DEC, and magnitude information for all secondary stars
                    m = sources.indexmap[i][l]
                    if not np.isnan(sources.ra[m]) and not np.isnan(sources.dec[m]):
                        primary_hdu.header["HIERARCH STAR{0} RA".format(l+1)] = sources.ra[m]
                        primary_hdu.header["HIERARCH STAR{0} DEC".format(l+1)] = sources.dec[m]
                    if not np.isnan(sources.mag[m]):
                        primary_hdu.header["HIERARCH STAR{0} MAG".format(l+1)] = sources.mag[m]
                primary_hdu.header["HIERARCH SPECTRUM MULTISTAR"] = (True, cmt)
                primary_hdu.header["OBJNAME"] = "{0} {1} {2}+".format(
                    pampelmuse.getspectra.specprefix.upper(), pampelmuse.ifs_data.name, source_id)
            else:
                primary_hdu.header["HIERARCH SPECTRUM MULTISTAR"] = (False, cmt)
                primary_hdu.header["OBJNAME"] = "{0} {1} {2}".format(
                    pampelmuse.getspectra.specprefix.upper(), pampelmuse.ifs_data.name, source_id)

            # load IFS coordinates for the star and check if it is inside the field of view; omit NaN values
            sources.load_ifs_coordinates(ids=source_id)
            xy_coord = sources.ifs_coordinates[:, 0, :, 0]
            xy_j = np.mean(xy_coord[(~np.isnan(xy_coord)).all(axis=1)], axis=0)

            # add x- and y-position of PSF centroid in cube
            primary_hdu.header["HIERARCH SPECTRUM XCUBE"] = (xy_j[0], "[spaxel] x-position in cube")
            primary_hdu.header["HIERARCH SPECTRUM YCUBE"] = (xy_j[1], "[spaxel] y-position in cube")

            infield = pampelmuse.ifs_data.infov([xy_j])[0]
            if not infield:
                flag += 8
            primary_hdu.header["HIERARCH SPECTRUM INFIELD"] = (infield, "Whether centroid inside FOV")

            edge_distance = pampelmuse.ifs_data.distance_to_edge([xy_j])[0]
            primary_hdu.header["HIERARCH SPECTRUM EDGEDIST"] = (edge_distance, "[spaxel] Distance to FOV edge")

        elif sources.status[j] == 4:
            # for sky component, add x- and y-position of patch centroid in cube
            primary_hdu.header["HIERARCH SPECTRUM XCUBE"] = (sources.x[j], "[spaxel] x-centroid in cube")
            primary_hdu.header["HIERARCH SPECTRUM YCUBE"] = (sources.y[j], "[spaxel] y-centroid in cube")

        # calculate S/N of the spectrum
        snr = der_snr(sources.data[pampelmuse.start:pampelmuse.stop, i, 0])
        if snr < 10:
            flag += 2
        if not np.isnan(snr):
            primary_hdu.header["HIERARCH SPECTRUM SNRATIO"] = (snr, "Estimated signal-to-noise")
        else:
            logging.error("S/N estimation returned NaN - Spectrum of source #{0} corrupted.".format(source_id))

        if np.mean(sources.data[:, 0, 0]) < 0:
            flag += 4

        primary_hdu.header["HIERARCH SPECTRUM QLTFLAG"] = (flag, "Quality flag of source")

        if sources.status[j] == 0:
            name = "unresolved"
            primary_hdu.header["HIERARCH SPECTRUM TYPE"] = name
            primary_hdu.header["OBJNAME"] = "{0} {1} UNRES".format(
                pampelmuse.getspectra.specprefix.upper(), pampelmuse.ifs_data.name)
        elif sources.status[j] == 4:
            name = "sky{0:08d}".format(abs(sources.ID[j]))
            primary_hdu.header["HIERARCH SPECTRUM TYPE"] = "sky"
            primary_hdu.header["OBJNAME"] = "{0} {1} SKY{2:d}".format(
                pampelmuse.getspectra.specprefix.upper(), pampelmuse.ifs_data.name, abs(sources.ID[j]))
        else:
            name = "id{0:09d}".format(source_id)
            primary_hdu.header["HIERARCH SPECTRUM TYPE"] = "star"

        outname = "{0}{1}jd{2:07d}p{3:04d}f{4:03d}".format(pampelmuse.getspectra.specprefix, name, int(juldate),
                                                           int(10000*(juldate-int(juldate))), int(flag))
        
        utc = datetime.datetime.utcnow().strftime("%y-%m-%d %H:%M:%S")
        primary_hdu.header["DATE-GEN"] = (utc, "UTC this file was generated.")
        
        outfile = fits.HDUList(hdus=[primary_hdu, sigma_hdu])
        if wave_hdu is not None:
            outfile.append(wave_hdu)
        outfile.writeto(os.path.join(pampelmuse.getspectra.folder, "{0}.fits".format(outname)), clobber=True)
        
        logging.info("Wrote spectrum '{0}' [{1}/{2}]".format(
            os.path.join(pampelmuse.getspectra.folder, "{0}.fits".format(outname)), i+1, sources.ncmp))
