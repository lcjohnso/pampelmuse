"""
singlesrc.py
============

Provides
--------
function pampelmuse.routines.singlesrc.singlesrc

Purpose
-------
The module contains the function 'singlesrc' that is used to prepare a data
cube for the analysis for which no photometric or astrometric information is
available. It prepares a PRM-file directly from a set of provided IFS coordi-
nates and statuses. For each of the sources with status=3, aperture spectro-
photometry is performed and the initial guesses for the IFS coordinates are im-
proved using the first order moments determined in the calculation.
 
Latest SVN revision
-------------------
253, 2015/04/22
"""
__author__ = "Sebastian Kamann (skamann@astro.physik.uni-gooettingen.de)"
__revision__ = 253

import datetime
import logging
import os
import re
import numpy as np
from astropy.io import fits

# required PampelMuse routines
from pampelmuse import ConfigurationError
from pampelmuse.core import AperPhoto, BackgroundGrid, Sources
from pampelmuse.core.psf import *


def singlesrc(pampelmuse):

    logging.info("Executing requested routine SINGLESRC [Revision {0:d}].".format(__revision__))
    
    # get x, y, and status from user-provided strings
    split_input = re.compile(r"(\-?[0-9\.]+),(\-?[0-9\.]+),([1-3])")
    
    sourcelist = []
    i = 1
    for entry in pampelmuse.singlesrc.source:
        source = split_input.match(entry)
        if source is None:
            logging.critical("Cannot grab source information from string '{0}'.".format(entry))
            logging.critical("Source information must be provided as '<x>,<y>,<status>'.")
            assert source is not None, "Invalid input for source: {0}".format(entry)
        sourcelist.append((i, source.group(1), source.group(2), source.group(3), i))
        logging.info("Added source at ({0},{1}) with status {2} and ID {3}.".format(
            source.group(1),  source.group(2), source.group(3), i))
        i += 1
        
    if pampelmuse.globsky:  # add background component(s) if requested
        if pampelmuse.skysize <= 0:
            background = BackgroundGrid.constant()
        else:
            background = BackgroundGrid.from_shape(shape=pampelmuse.ifs_data.instrument.shape,
                                                   rscale=pampelmuse.skysize/2.)

        n = 0
        for i, [y, x] in enumerate(background.centroids):
            # ignore sky components outside FoV
            if pampelmuse.ifs_data.instrument.distance_to_edge([[x, y]])[0] < -pampelmuse.skysize \
                    and pampelmuse.skysize > 0:
                continue

            sourcelist.append((-i-1, x, y, 4, -i-1))
            n += 1

        logging.info("Added {0} sky component(s).".format(n))

    # initialize Sources instance
    logging.info("Initializing the catalog of sources...")
    input_catalog = np.array(sourcelist, dtype=[("id", "<i4"), ("x", "<f4"), ("y", "<f4"), ("status", "<i2"),
                                                ("primary", "<i4")])
    sources = Sources(input_catalog)

    # initialize the PSF (attributes and profiles)
    logging.info("Initializing the PSF...")
    
    psf_variables = Variables(profile=pampelmuse.psf.profile)

    if pampelmuse.psf.profile == "moffat":
        PSF = Moffat
    elif pampelmuse.psf.profile == "gauss":
        PSF = Gaussian

    i = 0
    while hasattr(pampelmuse.psf, "parname{0}".format(i)):
        parname_i = getattr(pampelmuse.psf, "parname{0}".format(i))
        parvalue_i = getattr(pampelmuse.psf, "parvalue{0}".format(i))
        
        # include value and fit request in PSF variables
        j = psf_variables.names.index(parname_i)
        psf_variables.data[0, j, 0] = parvalue_i
        psf_variables.free[j] = getattr(pampelmuse.psf, "freepar{0}".format(i))

        # include value in PSF profiles
        getattr(PSF, "set_{0}".format(parname_i))(parvalue_i)
        
        i += 1
        
    psf_variables.info()
    
    PSF.set_transform(pampelmuse.ifs_data.instrument.transform)
    
    psf_instances = []
    for i in xrange(sources.nsrc):
        if sources.ID[i] <= 0:
            psf_instances.append(None)
        else:
            psf_instances.append(PSF(uc=sources.x[i], vc=sources.y[i], mag=0, src_id=sources.ID[i], maxrad=10.))

    # include dispersion information in Sources catalogue
    sources.ndisp = pampelmuse.stop - pampelmuse.start + 1
    sources.crval = pampelmuse.ifs_data.crval + pampelmuse.start*pampelmuse.ifs_data.cdelt
    sources.cdelt = pampelmuse.ifs_data.cdelt
    sources.ctype = pampelmuse.ifs_data.ctype
    sources.cunit = pampelmuse.ifs_data.cunit
    sources.load_spectra()  # create dummy data: every spectrum is empty
    sources.load_ifs_coordinates()  # initialize IFS coordinates
    sources.ifs_coordinates_free[sources.ID >= 0] = True

    # carry out aperture spectrophotometry for PSF sources
    if (sources.status == 3).any():
        logging.info("Performing aperture spectrophotometry on PSF stars...")

        make_aper = AperPhoto(pampelmuse.ifs_data, [psf_instances[i] for i in np.flatnonzero(sources.status == 3)])
        results = make_aper(r_aper=pampelmuse.aperrad, layer_range=(pampelmuse.start, pampelmuse.stop),
                            background="local", ncpu=pampelmuse.cpucount, nbin=pampelmuse.layerbin)

        dispmask = np.ones(pampelmuse.ifs_data.ndisp, dtype=np.bool)  # only work on layers used by aperture photometry
        dispmask[make_aper.layers_to_analyse] = False

        for i in xrange(results.size):
            # add flux to Sources instance
            j = np.where(sources.ID == results["id"][i])[0][0]
            sources.data[:, sources.specrow[j], 0] = results["flux"][i]

            # add IFS coordinates to Sources instance
            k = sources.ifs_coordinates_loaded.cumsum()[j] - 1
            sources.ifs_coordinates[:, k, 0, 0] = results["x"][i]
            sources.ifs_coordinates[:, k, 1, 0] = results["y"][i]

        sources.data[dispmask] = np.nan
        sources.ifs_coordinates[dispmask] = np.nan

        # Use results from aperture spectrophotometry to refine initial PSF parameters
        # TODO: Make 'update_psf' a parameter in the PampelMuse configuration
        update_psf = True if psf_variables.free.any() else False
        if update_psf:
            stars = (results["id"] >= 0)

            logging.info("Estimated PSF properties from measured moments:")
            x2 = np.ma.array(results[["x2"]][stars].view(np.float32).reshape(stars.sum(), -1))
            x2.mask = np.isnan(x2)
            y2 = np.ma.array(results[["y2"]][stars].view(np.float32).reshape(stars.sum(), -1))
            y2.mask = np.isnan(y2)
            xy = np.ma.array(results[["xy"]][stars].view(np.float32).reshape(stars.sum(), -1))
            xy.mask = np.isnan(xy)

            # get A, B:
            with np.errstate(invalid="ignore"):
                A = np.sqrt((x2 + y2)/2. + np.sqrt(((x2 - y2)/2.)**2 + xy**2))
                A.mask[A <= 0] = True
                B = np.sqrt((x2 + y2)/2. - np.sqrt(((x2 - y2)/2.)**2 + xy**2))
                B.mask[B <= 0] = True

            # FWHM
            fwhm = np.sqrt(8.*np.log(2))*np.mean(A, axis=0)
            # TODO: should FWHM be updated as well?
            logging.info("Mean FWHM: {0:.2f}pix.".format(fwhm.mean()))

            # Ellipticity
            e = 1. - np.mean(B/A, axis=0)
            logging.info("Mean ellipticity: {0:.2f}".format(e.mean()))
            ii = psf_variables.names.index("e")
            if psf_variables.free[ii]:
                # need to check if dispersion information already included in PSF variables, otherwise, use numpy.resize
                #  that fills enlarged array with repeated copies of available data
                if psf_variables.data.shape[0] == 1:
                    psf_variables.data = np.resize(psf_variables.data, e.shape+psf_variables.data.shape[1:])
                psf_variables.data[:, ii, 0] = e.data

            # Position angle
            with np.errstate(invalid="ignore"):
                _theta = 0.5*np.arctan(2*xy/(x2-y2))
                _theta -= (-1)**(_theta < 0)*np.pi/4.
                _theta += (-1)**(xy > 0)*np.pi/4.
                theta = _theta.mean(axis=0)
            logging.info("Mean position angle: {0:.2f}rad.".format(theta.mean()))
            ii = psf_variables.names.index("theta")
            if psf_variables.free[ii]:
                if psf_variables.data.shape[0] == 1:
                    psf_variables.data = np.resize(psf_variables.data, theta.shape+psf_variables.data.shape[1:])
                psf_variables.data[:, ii, 0] = theta.data

            if psf_variables.data.shape[0] > 1:
                psf_variables.data[dispmask] = np.nan

    elif psf_variables.free.any():
        raise ConfigurationError("PSF variability must be set to zero if no source with status=3 provided.")

    # write results to file
    logging.info("Saving sources setup in file {0}.".format(pampelmuse.filename))

    # make sure individual sources use individual PSF fits
    pampelmuse.smallifu = False

    # add information about routine to namespace
    pampelmuse.singlesrc.utc = datetime.datetime.utcnow().strftime("%y-%m-%d %H:%M:%S")
    pampelmuse.singlesrc.revision = __revision__

    primary_hdu = pampelmuse.makeHeader()  # create header from namespace
    source_hdu = sources.make_catalogue_hdu()
    spectra_hdu = sources.make_spectra_hdu()
    positions_hdu = sources.make_ifs_coordinates_hdu()
    psfpars_hdu = psf_variables.make_hdu(include_fit=False)

    hdu_list = fits.HDUList([primary_hdu, source_hdu, spectra_hdu, positions_hdu, psfpars_hdu])

    if os.path.exists(pampelmuse.filename):
        logging.warning("Overwriting existing PRM-file.")
        os.remove(pampelmuse.filename)
    hdu_list.writeto(pampelmuse.filename)
