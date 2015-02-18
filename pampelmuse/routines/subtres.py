"""
subtres.py
==========

Provides
--------
function pampelmuse.routines.subtres.subtres

Purpose
-------
The method implements the function 'subtres' that is designed to subtract th
contribution of the resolved sources from the IFS data. To this aim, a valid
PampelMuse PRM-file must exist along with the data.  This function is called
if PampelMuse is started with the 'SUBTRES' option.

Latest SVN revision
-------------------
243, 2015/02/18
"""
from scipy.interpolate import InterpolatedUnivariateSpline

__author__ = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__revision__ = 243
import datetime
import itertools
import logging
import os
from multiprocessing import Pool
import numpy as np
from astropy.io import fits
from scipy import sparse

# required PampelMuse-packages
from pampelmuse.core import BackgroundGrid, Sources
from pampelmuse.core.coordinates import Transformation
from pampelmuse.core.psf import *
from pampelmuse.core.fit_array import get_psf_data, init_worker


def subtres(pampelmuse):

    logging.info("Executing requested routine SUBTRES [Revision {0:d}].".format(__revision__))

    # Collect information about sources/create Sources-instance
    logging.info("Obtaining information on sources in the field...")
    
    # Source catalogue
    sources = Sources(data=pampelmuse.hdulist["SOURCES"])

    # Check if PRM-file contains results of successful fit
    if hasattr(pampelmuse, "cubefit") and hasattr(pampelmuse.cubefit, "status"):
        fitStatus = pampelmuse.cubefit.status
    else:
        fitStatus = "Unknown"

    if fitStatus not in ["SUCCESS", "Unknown"]:
        logging.error("PRM-file does not contain results of successful fit.")

    # Collect source spectra
    logging.info("Obtaining source spectra...")
    try:
        i = pampelmuse.hdulist.index_of("SPECTRA")
        sources.open_spectra_hdu(pampelmuse.hdulist[i])
    except KeyError:
        logging.critical("Cannot find spectra (HDU `SPECTRA') in provided PRM-file.")

    # Check which type of background is used
    if pampelmuse.subsky:
        if sources.is_background.sum() == 1:
            background = BackgroundGrid.constant()
        elif sources.is_background.sum() > 1:
            background = BackgroundGrid(np.vstack((sources.y[sources.is_background], sources.x[sources.is_background])).T)
        else:
            background = None
        if background is not None:
            background.set_transform(pampelmuse.ifs_data.instrument.transform)

    # Collect source positions
    logging.info("Obtaining information about source coordinates...")

    try:
        i = pampelmuse.hdulist.index_of("POSITIONS")
        sources.open_coordinates_hdu(pampelmuse.hdulist[i])
        try:
            i = pampelmuse.hdulist.index_of("INPOSITIONS")
            sources.open_coordinates_hdu(pampelmuse.hdulist[i])
        except KeyError:
            pass
    except KeyError:
        pass

    try:
        i = pampelmuse.hdulist.index_of("POSPARS")
        sources.transformation = Transformation(hdu=pampelmuse.hdulist[i])
        
    except KeyError:
        logging.warning("No coordinate transformation (HDU 'POSPARS') detected.")

    # Initialize PSF variables
    logging.info("Obtaining PSF properties...")
    try:
        i = pampelmuse.hdulist.index_of("PSFPARS")
        psf_attributes = Variables(hdu=pampelmuse.hdulist[i])
    except KeyError:  # otherwise, assume Gaussian, round with FWHM=3.0pixel
        logging.warning("No HDU 'PSFPARS' found. Using default PSF.")
        
        kwargs = {"profile": pampelmuse.pfsftype}
        free = []
        i = 0
        while hasattr(pampelmuse.psf, "parname{0}".format(i)):
            kwargs[getattr(pampelmuse.psf, "parname{0}".format(i))] = getattr(pampelmuse.psf, "parvalue{0}".format(i))
            free.append(getattr(pampelmuse.psf, "freepar{0}".format(i)))
            i += 1
        psf_attributes = Variables(**kwargs)
        psf_attributes.free = np.atleast_1d(free)

    psf_attributes.info()

    # Initialize PSF profiles of the sources:
    if psf_attributes.profile == "gauss":
        PSF = Gaussian
    elif psf_attributes.profile == "moffat":
        PSF = Moffat
    else:
        logging.critical("Invalid PSF type: '{0}'".format(psf_attributes.psftype))
        raise IOError("PSF type not understood.")
    PSF.set_transform(pampelmuse.ifs_data.instrument.transform)

    psf_instances = []
    included = np.zeros((sources.nsrc, ), dtype=np.bool)
    for i in xrange(sources.nsrc):
        if sources.status[i] == 4:  # handle background component, add index of background patch if background should
            # be subtracted
            if pampelmuse.subsky:
                psf_instances.append(sources.is_background[:(i+1)].cumsum()[-1]-1)
                included[i] = True
            continue
        if sources.status[i] == 0 and not pampelmuse.subunres:  # check if unresolved component should be subtracted
            continue
        psf_instances.append(PSF(Xc=sources.x[i], Yc=sources.y[i], mag=sources.weights[i], src_id=sources.ID[i],
                                 maxrad=pampelmuse.psfrad, osfactor=5, osradius=2))
        included[i] = True
    specrows = sources.specrow[included]

    # get source coordinates over whole wavelength range
    logging.info("Loading IFS coordinates...")
    sources.load_ifs_coordinates(ids=sources.ID[included & (sources.status < 4)])

    # load full(!) cube into memory.
    logging.info("Loading datacube to hold the results...")
    data = np.copy(pampelmuse.ifs_data.data.data)

    if pampelmuse.ifs_data.ndim == 3:
        xindex = np.asarray(pampelmuse.ifs_data.instrument.x, dtype=np.int32)
        yindex = np.asarray(pampelmuse.ifs_data.instrument.y, dtype=np.int32)

    for jj in xrange(pampelmuse.ifs_data.nlayer):
        
        logging.info("Handling layer #{0}/{1} [{2:.1f}%].".format(jj, pampelmuse.ifs_data.nlayer,
                                                                  (100.*jj/pampelmuse.ifs_data.nlayer)))
        
        if jj < pampelmuse.start or jj > pampelmuse.stop:
            continue
        
        # load fluxes for current layer
        sources.load_spectra(layers=jj)

        k = 0
        for i, j in enumerate(included.nonzero()[0]):
            if not isinstance(psf_instances[i], Profile):
                continue
            if sources.ifs_coordinates_fitted[j]:
                psf_instances[i].set_xc(sources.ifs_coordinates[jj, k, 0, 1])
                psf_instances[i].set_yc(sources.ifs_coordinates[jj, k, 1, 1])
            else:
                psf_instances[i].set_xc(sources.ifs_coordinates[jj, k, 0, 0])
                psf_instances[i].set_yc(sources.ifs_coordinates[jj, k, 1, 0])
            k += 1
        
        # update PSF instances of the sources with the variables fitted for current layer
        i = 0 
        for prm in psf_attributes.names:
            if psf_attributes.fitted[i]:
                value = psf_attributes.data[jj, i, 1]
            else:
                if psf_attributes.mask[jj, i, 0]:
                    logging.warning("Access to NaN value in PSF data, interpolating...")
                    f = InterpolatedUnivariateSpline((~psf_attributes.mask[:, i, 0]).nonzero()[0],
                                                     psf_attributes.data[~psf_attributes.mask[:, i, 0], i, 0], k=2)
                    value = float(f(jj))  # newer versions of scipy return an array of zero size
                else:
                    value = psf_attributes.data[jj, i, 0]
            getattr(PSF, "set_{0}".format(prm))(value)
            i += 1
        
        # Initialize parallelization (if requested)
        if pampelmuse.cpucount > 1:
            pool = Pool(pampelmuse.cpucount, initializer=init_worker,
                        initargs=(background if pampelmuse.subsky else None,))
#             threads = []
#
#             # determine which sources should be calculated by which CPU
#             m = (len(psf_instances)//pampelmuse.cpucount) + 1
#
#             for n in xrange(pampelmuse.cpucount):
#                 imin = n*m
#                 imax = min((n+1)*m, len(psf_instances))
#
#                 threads.append(pool.apply_async(get_psf_data, args=(psf_instances[imin:imax])))
            _results = pool.map_async(get_psf_data, psf_instances)
            pool.close()
            pool.join()

            results = list(itertools.chain.from_iterable(_results.get()))
#            results = list(itertools.chain.from_iterable([threads[n].get() for n in xrange(pampelmuse.cpucount)]))
        else:
            init_worker(background if pampelmuse.subsky else None)
            results = get_psf_data(psf_instances)

        i0 = np.hstack((r[0] for r in results))  # the row indices

        i1 = np.hstack((len(results[i][0])*[specrows[i]] for i in xrange(included.sum())))  # the column indices

        f = np.hstack((r[1] for r in results))  # the fluxes

        Apsf = sparse.coo_matrix((f, (i0, i1)), shape=(pampelmuse.ifs_data.instrument.nspaxel, sources.ncmp)).tocsc()
        model_matrix = Apsf*sparse.dia_matrix((sources.data[0, :, 0], [0]), shape=(sources.ncmp, sources.ncmp))
        model = model_matrix.sum(axis=1).getA1()

        if pampelmuse.ifs_data.ndim == 3:
            # data[kk] -= pampelmuse.ifs_data.instrument.make_grid(model)
            # there seems to be a memory leak in underlying scipy.interpolate.griddata routine
            data[jj, yindex, xindex] -= model
        elif pampelmuse.ifs_data.instrument.spectral_axis == 0:
            data[jj, pampelmuse.ifs_data.instrument.isscience] -= model
        else:
            data[pampelmuse.ifs_data.instrument.isiscience, jj] -= model

    pampelmuse.hdulist.close()  # finished reading source catalog.

    pampelmuse.subtres.revision = __revision__
    pampelmuse.subtres.utc = datetime.datetime.utcnow().strftime("%y-%m-%d %H:%M:%S")

    # create HDU for output
    logging.info("Creating HDU from processed data...")
    # if pampelmuse.ifs_data.ndim == 3: ## make output data 3D if input was 3D
    #     data = pampelmuse.ifs_data.instrument.make_grid(data)
    outhdu = fits.PrimaryHDU(data, header=pampelmuse.ifs_data.data.header)
    
    # add PampelMuse configuration to HDU just created:
    outhdu = pampelmuse.makeHeader(hdu=outhdu)
  
    # delete any existing residuals file
    logging.info("Writing results to file {0}.res.fits...".format(pampelmuse.prefix))
    if os.path.exists("{0}.res.fits".format(pampelmuse.prefix)):
        os.remove("{0}.res.fits".format(pampelmuse.prefix))
        logging.warning("Existing residual cube has been overwritten.")

    # save output (including HDU with fiber setup if available)
    if pampelmuse.ifs_data.setup is not None:
        hdulist = fits.HDUList([outhdu, pampelmuse.ifs_data.setup])
        hdulist.writeto("{0}.res.fits".format(pampelmuse.prefix))
    else:
        outhdu.writeto("{0}.res.fits".format(pampelmuse.prefix))

