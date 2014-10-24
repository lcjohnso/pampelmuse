"""
initfit.py
==========

Provides
--------
class pampelmuse.routines.initfit.ConfigurationError
function pampelmuse.routines.initfit.calculate_noise
function pampelmuse.routines.initfit.estimate_snr
function pampelmuse.routines.initfit.initfit

Purpose
-------
The module contains the function 'initfit' that is called when the PampelMuse
subroutine INITFIT is called. The function is designed to create a sophisticated
list of sources to be analysed with the PampelMuse subroutine CUBEFIT. It is
called with a single parameter that is a namespace containing the configuration
of PampelMuse. INITFIT must be called with the mandatory argument 'sourcelist'
to provide an ascii-file containing the crowded field photometry. This data are
used to initialize the catalogue of sources and to create a mock image from the
catalogue at the spatial resolution of the IFS data. This image is displayed to-
gether with the IFS data using the PampelMuse GUI. The spatial position of at
least 2 sources from the catalogue must be interactively selected. This information
is used afterwards to locate the position of all sources from the cataogue in (or
outside) the IFS data, estimate their S/N, and perform the final selection of the
sources that are used in the analysis carried out by CUBEFIT. All the information
required for this analysis, including an intial estimate of the PSF and the coor-
dinate transformation, are saved in the PRM file that is written to disk.

The class 'ConfigurationError' is introduced to handle cases in which the confi-
guration of PampelMuse does not allow for an analysis of the provided IFS data.
For example, a ConfigurationError is raised if no sources that pass the criteria
for the selection of PSF sources are detected.

The function 'estimate_snr' is designed to estimate the S/N of all sources in a
catalogue and is called by the function 'initfit'.

Latest SVN Revision
-------------------
214, 2014/10/24
"""
__author__ = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__revision__ = 214

import datetime, itertools, os, sys, logging
from multiprocessing import cpu_count, Pool
import numpy as np 
from astropy.io import fits
from scipy import ndimage, signal, spatial
# note: PyQt4 imported below only if interaction enabled

# required PampelMuse routines
from pampelmuse.core import AperPhoto, Sources
from pampelmuse.core.psf import *
from pampelmuse.filters import ALIASES, get_filter
from pampelmuse.functions.io import read_asciifile
from pampelmuse.functions.statistics import DER_SNR
# note: pampelmuse.graphics.PampelMuseGui imported below only if interaction enabled


# class defines the exception that is raised when the parameters
# in the Pampel Muse-configuration are such that they do not allow
# for a proper analysis of the provided input
class ConfigurationError(EnvironmentError):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)



def estimate_snr(psf_instances, sigma):
        """
        Calculate the S/N ratio of all stars in the scene.

        Parameters
        ----------
        psf_instances : list
            A list containing a PSF-instance for each every source in the
            scene. This is required to determine the flux distribution
            across the FoV
        sigma : nd_array
            The average uncertainties in the individual spaxels.

        Returns
        -------
        snRatios : nd_array
            A 1dim. array containing the S/N-ratio for each source
        totFlux : float
            The integrated flux of all point sources over the individual
            spaxels.
        """
        snRatios = []
        totFlux  = 0.

        for psf in psf_instances:

            if psf==None:
                snRatios.append(0.)
                continue

            logging.debug("Estimating S/N  for source #{0}.".format(psf.src_id))

            if (sigma[psf.used]==0).any():
                logging.error("Sigma image contains invalid pixels with values <=0.")

            # measure S/N
            snRatios.append(np.sqrt(((psf.f/sigma[psf.used])**2).sum()))

            # add the flux contribution of this source in the FoV
            totFlux += psf.f.sum()

        return np.array(snRatios, dtype=np.float32), totFlux



def image_from_psf(psf_instances):
    nsrc = len(psf_instances)
    
    i=0
    for psf in psf_instances:
        logging.debug("Working on source #{0} [{1:0{2}d}/{3}]".format(psf.src_id, i+1, len(str(nsrc)), nsrc))
        if i==0:
            imsize = psf.size
            image  = np.zeros((np.product(imsize), ), dtype='float32')
        else:
            assert psf.size == imsize, "PSF instances have different array shapes."
        image[psf.used] += psf.f
        i+=1
    
    return image



def initfit(pampelmuse):

    logging.info("Executing requested routine INITFIT [Revision {0:d}].".format(__revision__))

    # get results from crowded-field photometry, initialize sources
    logging.info("Reading photometry results from file '{0}'.".format(pampelmuse.initfit.sourcelist))

    # Note that the routine will first check if the first (uncommented) line in the file is a header line from which it
    # can infer the column content. Only if that fails, it will use the association in 'usecols' and 'usenames'
    # the 'aliases' provide for each valid passband a list of other possible names the column may have
    cfp_results = read_asciifile(pampelmuse.initfit.sourcelist, usecols=(0,1,2,3,4,5), usenames="id,x,y,{0},ra,dec".format(pampelmuse.passband),
                                 optional="x,y,ra,dec", aliases=ALIASES)

    # make sure IDs are available
    if not "id" in cfp_results.keys():
        logging.critical("Did not find IDs in photometry file.")
        raise IOError, "Source IDs must be provided with phtotmetry"

    # make sure magnitudes are available
    if not pampelmuse.passband.lower() in cfp_results.keys():
        logging.critical("Did not find magnitudes in photometry file.")
        raise IOError, "Source magnitudes must be provided with photomtery."

    # make sure either RA & Dec or x & y are available; if both are available, RA & Dec are preferred over x & y
    usewcs=False # set to True if WCS coordinates are available
    if "ra" not in cfp_results.keys() or "dec" not in cfp_results.keys():
        if "x" not in cfp_results.keys() or "y" not in cfp_results.keys():
            logging.critical("Cannot find input source coordinates.")
            raise IOError, "Either x & y or RA & Dec must be provided with photometry."
    else:
        usewcs=True
        logging.info("Using RA & DEC coordinates as input.")
        
        ra = np.asarray(cfp_results["ra"], dtype=np.float64)
        dec = np.asarray(cfp_results["dec"], dtype=np.float64)
    
        # If the reference catalogue comes with RA and DEC coordinates, the routine will set up a WCS system for the
        # IFS data. Later this is used to convert the reference RA and DEC coordinates into IFS spaxel coordinates.
        # In the best case, the IFS data already contains all the necessary WCS information. If any information is
        # missing, the routine will do an educated guess:
        # (i) if it cannot infer the spaxel size, it will use the configuration parameter PIXSCALE
        # (ii) if the RA and DEC of the pointing are missing, they will be chosen so that the median RA and DEC values
        #      from the reference catalogue are in the centre of the FoV.
        # (iii) the rotation angle of the FoV will be set to zero degrees.
        
        if pampelmuse.ifs_data.instrument.spaxel_scale==None: # check spaxel scale
            pampelmuse.ifs_data.instrument.spaxel_scale = pampelmuse.pixscale
        
        pampelmuse.ifs_data.make_wcs() # initialize WCS

        if (pampelmuse.ifs_data.wcs.wcs.crval==0.).all(): # check pointing  RA and Dec
            logging.warning("Using median RA and DEC values of source catalogue to complete WCS for IFS.")
            pampelmuse.ifs_data.wcs.wcs.crval = [np.median(cfp_results["ra"]), np.median(cfp_results["dec"])]

        # get IFS spaxel coordinates from  reference RA and DEC
        x, y = pampelmuse.ifs_data.wcs.wcs_world2pix(ra, dec, 1)
        cfp_results["x"] = x.tolist()
        cfp_results["y"] = y.tolist()

    # Perform sanity checks on provided source list:
    valid = (np.isnan(np.asarray(cfp_results[pampelmuse.passband.lower()], dtype=np.float32))==False)
    if (valid==False).any(): logging.warning("Ignoring {0} source(s) with undefinded magnitudes.".format((valid==False).sum()))
    if len(np.unique(cfp_results["id"]))<len(cfp_results["id"]): logging.error("Catalog contains duplicated IDs.")

    # initialize the PSF (attributes and profiles) - mock image will have same PSF as initial guess for IFS data
    logging.info("Initializing the PSF...")
    
    psf_variables = Variables(profile=pampelmuse.psf.profile)

    if pampelmuse.psf.profile == "moffat": PSF = Moffat
    elif pampelmuse.psf.profile == "gauss": PSF = Gaussian

    i=0
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

    # In the next steps, the IFS data is convolved with a filter curve to create an image in the photometric filter in which the
    # magnitudes were provided.
    logging.info("Creating mock {0}-band image from IFS data...".format(pampelmuse.passband))
    
    # get photometric filter curve
    f = get_filter(pampelmuse.passband)
    fMin = f.get_knots().min() # min. of defined wavelength range
    fMax = f.get_knots().max() # max. of defined wavelength range

    # make loop to avoid extensive use of memory for large data-cubes
    data_image   = np.zeros(pampelmuse.ifs_data.instrument.transform.shape[0], dtype=np.float32)
    sigma_image  = np.zeros(data_image.shape, dtype=np.float32) # old method to determine noise was from SIGMA cube via integration over passband
    weight_image = np.zeros(data_image.shape, dtype=np.float32)
    logging.debug("...layers contain {0} spaxels.".format(data_image.size))

    for i in xrange(pampelmuse.start, pampelmuse.stop):
        logging.debug("...working on layer {0:0{1}d}/{2:{1}d}".format(i-pampelmuse.start+1, len(str(pampelmuse.stop-pampelmuse.start)), pampelmuse.stop-pampelmuse.start))
        wave_i = pampelmuse.ifs_data.crval+i*pampelmuse.ifs_data.cdelt
        if pampelmuse.ifs_data.cunit == "nm" or pampelmuse.ifs_data.ctype == "nm": wave_i *= 10
        elif pampelmuse.ifs_data.cunit == "um" or pampelmuse.ifs_data.ctype == "um": wave_i *= 1e4
        
        if wave_i > fMin and wave_i < fMax:
            wht_i = f(wave_i)
            assert not np.isnan(wht_i)
            
            data_i  = pampelmuse.ifs_data.data.layer(i)
            if pampelmuse.ifs_data.is_var:
                sigma_i = np.sqrt(pampelmuse.ifs_data.sigma.layer(i))
            else:
                sigma_i = pampelmuse.ifs_data.sigma.layer(i)

            mask_i = np.isnan(data_i)|np.isnan(sigma_i)|(sigma_i<=0)
            if pampelmuse.ifs_data.mask is not None:
                mask_i = mask_i|np.asarray(pampelmuse.ifs_data.mask.layer(i), dtype=np.bool)

            data_image[mask_i==0] += data_i[mask_i==0]*wht_i
            sigma_image[mask_i==0] += sigma_i[mask_i==0]*wht_i
            weight_image[mask_i==0] += wht_i

            del data_i, sigma_i, wht_i

    mask = (weight_image<=0)

    # if a MUSE data has been provided, try to determine edge of field of view as smallest polygon to includes all valid spaxels.
    if mask.any():
        if pampelmuse.ifs_data.instrument.name == "muse":
            pampelmuse.ifs_data.instrument.determine_edge((mask==0).flatten())
        
        sigma_image[mask] = np.inf
        weight_image[mask] = 1

    data_image  /= weight_image
    sigma_image /= weight_image

    # The source catalogue may cover a significantly larger area on the sky than the IFS data. To limit the memory usage and the
    # computation time in such cases, the code tries to cut out the relevant part. This is done by removing all sources with a
    # distance >CCMAXOFF to the edge of the IFS field of view.
    # Note: Only works when RA and DEC coordinates provided in reference catalogue
    if usewcs:
        
        # if CCMAXOFF<0, maxmimum offset is set to half size of IFS field of view
        if pampelmuse.ccmaxoff<0:
            pampelmuse.ccmaxoff = np.sum(pampelmuse.ifs_data.instrument.shape)>>2

        dist = pampelmuse.ifs_data.instrument.distance_to_edge(np.asarray([cfp_results["x"], cfp_results["y"]]).T)
        valid = valid&(dist>-(pampelmuse.ccmaxoff+10)) # sources outside FoV have negative values, include some margin

    # Initialize sources-class
    if pampelmuse.globsky: # add sky component if requested
        valid = np.r_[valid, True]
        cfp_results["id"].append("-1")
        cfp_results["x"].append("0")
        cfp_results["y"].append("0")
        cfp_results[pampelmuse.passband.lower()].append("0")
        if "ra" in cfp_results.keys(): cfp_results["ra"].append("0")
        if "dec" in cfp_results.keys():  cfp_results["dec"].append("0")

    sources     = Sources(ids=np.asarray(cfp_results["id"], dtype=np.int32)[valid])
    sources.x   = np.asarray(cfp_results["x"], dtype=np.float32)[valid]
    sources.y   = np.asarray(cfp_results["y"], dtype=np.float32)[valid]
    sources.mag = np.asarray(cfp_results[pampelmuse.passband.lower()], dtype=np.float32)[valid]
    if "ra" in cfp_results.keys(): sources.ra = np.asarray(cfp_results["ra"], dtype=np.float64)[valid]
    if "dec" in cfp_results.keys(): sources.dec = np.asarray(cfp_results["dec"], dtype=np.float64)[valid]

    # make sure that at least some sources are resolved according to the RESLIM configuration parameter. If not, raise a ConfigurationError
    if not (sources.mag[sources.ID>=0]<pampelmuse.reslim).any():
        logging.critical("All sources in input catalog are fainter than provided resolution limit (RESLIM={0}).".format(pampelmuse.reslim))
        raise ConfigurationError, "Photometry does not contain resolved sources. Check RESLIM parameter."

    # In the next step, a mock image is created from the photometric catalogue. For this purpose the sources are shifted in x and y
    # so that the source with the lowest x (y) coordinate is located in the first column (row). Otherwise a large part of the mock image
    # might be completely empty or sources could be outside the FoV.
    # Unless calculated from theRA and DEC coordinates, the pixel size of the photometric data is also changed to match the spaxelsize of
    # the IFS data.
    # These changes are undone after the source selection process is completed
    dx = int(sources.x[sources.ID>=0].min())
    dy = int(sources.y[sources.ID>=0].min())

    sources.x -= dx
    sources.y -= dy
    if not usewcs:
        sources.x /= pampelmuse.pixscale
        sources.y /= pampelmuse.pixscale
    
    logging.info("Creating mock image from photometry catalogue...")

    # prepare PSF for creation of mock image: set size to size of mock image
    size   = (int(sources.y.max()+1), int(sources.x.max()+1))
    PSF.set_size(size)
    
    # to get reasonable count rates in mock image, adjust zeropoint unless provided
    if pampelmuse.zeromag > 99.:
        magZ = pampelmuse.reslim + 7.5
    else:
        magZ = pampelmuse.zeromag
    
    # for parallelization, initialize cycle over CPUs so that whenever a new source should be included in the mock image,
    # the next CPU is picked until one source has been assigned to all CPUs, then start again with first CPU and repeat.
    cpu  = itertools.cycle(range(pampelmuse.cpucount))
    tocpu = -1*np.ones((sources.nsrc,), dtype=np.int16)
    
    # 1. loop: initialization of PSF instances and assignment to CPUs
    logging.info("...initializing PSF instances...")
    psf_instances = []
    for i in xrange(sources.nsrc):
        
        if sources.ID[i]<0: ## handle background components
            psf_instances.append(None)
            continue
        
        psf = PSF(Xc=sources.x[i], Yc=sources.y[i], mag=(sources.mag[i]-magZ), src_id=sources.ID[i], maxrad=10.)
        psf_instances.append(psf)

        if (psf.mag+magZ) <= pampelmuse.reslim: ## only add sources that are likely resolved
            tocpu[i] = cpu.next()

    # 2. loop: calculation of PSF profiles and inclusion in mock image (parallelized)
    logging.info("...adding sources, working on {0} CPUs...".format(pampelmuse.cpucount))
    for n in xrange(pampelmuse.cpucount):
        psf_instances_n = [psf_instances[i] for i in np.flatnonzero(tocpu==n)]
        if pampelmuse.cpucount > 1:
            if n==0:
                pool    = Pool(pampelmuse.cpucount)
                threads = []
            threads.append(pool.apply_async(image_from_psf, args=(psf_instances_n,)))
        else:
            _image = image_from_psf(psf_instances_n)
    if pampelmuse.cpucount > 1:        
        pool.close()
        pool.join()
        _image = np.sum([threads[n].get() for n in xrange(pampelmuse.cpucount)], axis=0)

    # resample image on regular (x, y) grid
    logging.info("...resampling image on regular 2D grid...")
    identification_image = np.zeros(size, dtype=np.float32)
    identification_image[np.asarray(PSF._transform[:,0], dtype=np.int32), np.asarray(PSF._transform[:,1], dtype=np.int32)] = _image

    if usewcs:
        if pampelmuse.ccmaxoff!=0:
            # To automatically obtain an initial guess for the coordinate transformation, the mock image is cross-correlated with
            # the whitelight image of the IFS data. The peak of the cross-correlation signal is located and translated into an
            # offset between mock and IFS image.
            logging.info("Cross-correlating mock image with IFS data to find likely location of IFS.")

            # Kernel is normalized to get approx. normalized cross-corr. signal
            cckernel = pampelmuse.ifs_data.instrument.make_grid((data_image - np.min(data_image[mask==0]))/np.std(data_image[mask==0]), fill_value=0)
            
            ccsignal = signal.fftconvolve(identification_image, cckernel[::-1,::-1], mode="full")
            
            # To avoid that the cc signal is dominated by individual bright stars, the signal is normalized using the flux in the mock image
            norm     = signal.fftconvolve(identification_image, np.ones(cckernel.shape, dtype=np.float32), mode="full")
            norm[norm<=0]=np.inf
            ccsignal /= norm
    
            # The maximum is searched in an area with radius CCMAXOFF with respect to the location suggested by the WCS information of the IFS data
            xsearchcenter = cckernel.shape[1]-1-dx
            ysearchcenter = cckernel.shape[0]-1-dy
            
            searchradius = int(pampelmuse.ccmaxoff)
    
            xslice = slice(max(xsearchcenter-searchradius, 3), min(xsearchcenter+searchradius, ccsignal.shape[1]-3))
            yslice = slice(max(ysearchcenter-searchradius, 3), min(ysearchcenter+searchradius, ccsignal.shape[0]-3))
            _ymax, _xmax = np.nonzero(ccsignal[yslice, xslice]==ccsignal[yslice, xslice].max())
            ymax = _ymax[0] + yslice.start
            xmax = _xmax[0] + xslice.start
    
            # Measure moments around peak pixel to get peak coordinates with subpixel accuracy
            momentradius=3
            peakdata = ccsignal[ymax-momentradius:ymax+momentradius+1,xmax-momentradius:xmax+momentradius+1]
            peakdata -= peakdata.min()
            Y, X = np.indices(peakdata.shape)
            X += xmax-momentradius
            Y += ymax-momentradius
            total = peakdata.sum()
            xc  = (X*peakdata).sum()/total
            yc  = (Y*peakdata).sum()/total
    
            # Get offsets to position suggested by WCS information
            ccoffx = xc - xsearchcenter
            ccoffy = yc - ysearchcenter
            logging.info("Pointing offset from cross-correlation dx={0:.1f}, dy={1:.1f}".format(ccoffx, ccoffy))

            # for testing purposes: plot cross-correlation signal and save as FITS file
##            import pylab, matplotlib
##            fig = pylab.figure(figsize=(7, 7))
##            ax  = fig.add_axes([0.05, 0.05, 0.9, 0.9])
##            ax.imshow(ccsignal, interpolation="nearest")
##            ax.plot(xsearchcenter, ysearchcenter, ls="None",  marker="x",  mew=2.5, c="g", ms=15)
##            rect = matplotlib.patches.Rectangle((xsearchcenter-searchradius, ysearchcenter-searchradius), 2*searchradius, 2*searchradius, edgecolor="r", facecolor="none")
##            ax.add_patch(rect)
##            ax.plot(xc, yc, ls="None",  marker="x",  mew=2.5, c="r",  ms=15)
##            pylab.show()
##            outhdu = fits.HDUList([fits.PrimaryHDU(ccsignal), fits.ImageHDU(cckernel), fits.ImageHDU(identification_image)])
##            outhdu.writeto("ccsignal.fits", clobber=True)

        else:
            ccoffx=0
            ccoffy=0

        # initialize coordinate transformation
        sources.transformation.data = np.array([1., 0., ccoffx, ccoffy], dtype=np.float32)

    if not pampelmuse.interact:
        if not sources.transformation.valid:
            logging.critical("Non-interactive modus only supported if sources have RA and DEC available.")
        identified = {}

    # start source identification GUI to identify bright(est) stars
    else:
        from PyQt4 import QtGui
        from pampelmuse.graphics import PampelMuseGui
        
        logging.info("Preparing GUI for source identification...")

        app  = QtGui.QApplication(sys.argv)
    
        identifier = PampelMuseGui(sources=sources, ifsData=pampelmuse.ifs_data, refImage=identification_image, magCut=pampelmuse.reslim-1.)  

        if usewcs: # overplot positions of sources on IFS as determined via cross-correlation
            identifier.addFovToReference(-dx+ccoffx+pampelmuse.ifs_data.instrument.shape[0]//2, -dy+ccoffy+pampelmuse.ifs_data.instrument.shape[1]//2)
        
            sizes = 10.*(10**(-0.2*(sources.mag-pampelmuse.reslim-1.)))
            sizes[sizes<=1] = 0
            select = (sources.status<4)&(sources.mag<=pampelmuse.reslim)&(pampelmuse.ifs_data.instrument.infov(np.vstack((sources.x+dx-ccoffx, sources.y+dy-ccoffy)).T))
            identifier.ifsPlot.addData(sources.x[select]+dx-ccoffx, sources.y[select]+dy-ccoffy, s=sizes[select], marker="o", edgecolor="g", facecolor="None")

        identifier.show()
        app.exec_()

        identified = identifier.identifiedSources

    # recover original coordinates in Sources instance and for PSF instances
    if not usewcs:
        sources.x *= pampelmuse.pixscale
        sources.y *= pampelmuse.pixscale
    sources.x += dx
    sources.y += dy

    PSF.set_transform(pampelmuse.ifs_data.instrument.transform)
    for i in xrange(sources.nsrc):
        if sources.ID[i]<0: continue
        psf_instances[i].Xc = sources.x[i]
        psf_instances[i].Yc = sources.y[i]

    if len(identified)>=2:

        # determine coordinate transformation using identified sources, when providing the IFS coordinates for the sources that were identified,
        # they need to be provided in the right order (the sources were not necessarily idenitified in the same order they appear in the catalogue)
        sorted_indices = np.argsort([list(sources.ID).index(entry) for entry in identified.keys()])
    
        sources.load_ifs_coordinates(ids=identified.keys())
        sources.ifs_coordinates[..., 0] = np.asarray(identified.values())[sorted_indices,:2]
        sources.fit_coordinate_transformation()
        
    elif not sources.transformation.valid:
        logging.critical("At least 2 sources must be identified.")
        
    if pampelmuse.posfit == 1: sources.transformation.free[2:] = True
    elif pampelmuse.posfit == 2: sources.transformation.free[:] = True

    # add coordinate transformation to PSF instances
    i=0
    for prm in sources.transformation.names:
        getattr(PSF, "set_{0}".format(prm))(sources.transformation.data[0,i,0])
        i+=1

    # Perform photometric calibration (unless provided) and S/N estimation:
    logging.info("Estimating S/N for all sources...")
    if pampelmuse.cpucount > 1: # Initialize parallelization (if requested)
        logging.info("...working on {0} CPUs...".format(pampelmuse.cpucount))
        pool    = Pool(pampelmuse.cpucount)
        threads = []
            
        # determine which sources should be calculated by which CPU
        m = (len(psf_instances)//pampelmuse.cpucount) + 1
            
        for n in xrange(pampelmuse.cpucount):
            imin = n*m
            imax = min((n+1)*m, sources.nsrc)
            threads.append(pool.apply_async(estimate_snr, args=(psf_instances[imin:imax], sigma_image)))
        
        pool.close()
        pool.join()

        results = [threads[n].get() for n in xrange(pampelmuse.cpucount)]
        signal_to_noise = np.hstack([r[0] for r in results])
        intFlux = np.sum([r[1] for r in results])
    else:
        signal_to_noise, intFlux = estimate_snr(psf_instances, sigma_image)

    logging.info("Estimating the photometric zeropoint...")
    # Calculate ratio between expected flux for current zeropoint and flux in image
    fratio = intFlux/data_image.sum()
    dmag = -2.5*np.log10(fratio)

    # determine zeropoint if requested and update S/N estimates 
    if pampelmuse.zeromag > 99.:
        signal_to_noise /= fratio
        magZ += dmag
        logging.info("Estimated photometric zeropoint ({0}) is {1:.2f}".format(pampelmuse.passband, float(magZ)))
    elif fratio > (2./3.) and fratio < 1.5:
        logging.info("Offset between provided and estimated photometric zeropoint ({0}) is {1:.2f}mag".format(pampelmuse.passband, abs(dmag)))
    else:
        logging.warning("Offset between provided and estimated photometric zeropoint ({0}) is {1:.2f}mag".format(pampelmuse.passband, abs(dmag)))

    pampelmuse.initfit.photzero = magZ

    # Source selection process, the status of a source can be one of the following:
    #  4 - global sky spectrum
    #  3 - resolved, own spectrum
    #  2 - resolved, inferred spectrum
    #  1 - resolved, but next to brighter source
    #  0 - unresolved
    # -1 - excluded
    logging.info("Selecting sources to be used in PSF fitting...")

    # collect FWHM of sources in single array
    fwhm = np.zeros((sources.nsrc,), dtype=np.float32)
    for i in xrange(len(psf_instances)): fwhm[i] = 0 if psf_instances[i]==None else psf.fwhm

    # get magnitude-sorted list of indices of the sources the instance of the Sources-class (brightest sources first)
    idx_magsorted = np.argsort(sources.mag)

    # load IFS coordinates for all stars and sort them by brightness
    sources.load_ifs_coordinates()
    ifs_coordinates = np.zeros((sources.nsrc, 2), dtype=np.float32)
    ifs_coordinates[sources.ID>=0] = sources.ifs_coordinates[0,...,0]

    # arrays to keep track of the status of each source and the ID of the brightest source in its component (that is identical to the source ID unless a
    # brighter source is so close that the two cannot be separated and are fitted as a single component)
    status  = -np.ones((sources.nsrc,), dtype=np.int16)
    primary = np.zeros((sources.nsrc,), dtype=np.int32)
    
    # If confusion limit should be refined, we initially set it to 99. and later set it to the magnitude of the faintest star that is included in the fit
    # (as a resolved source)
    reslim = 99. if pampelmuse.updaterl else pampelmuse.reslim
    
    # for larger fields of view, such as the MUSE WFM, the stellar density can be very different across its extent. To deal with this, the confusion limit
    # is estimated locally in an area of radius 10xFWHM around each star
    density = np.zeros(pampelmuse.ifs_data.instrument.nspaxel, dtype=np.int16)

    logging.info("       ID   mag    S/N  d_bright  d_edge  status")
    logging.info("                           [spx]   [spx]        ")
    for i in idx_magsorted:
    
        # handle background components
        if sources.ID[i]<0:
            status[i] = 4
            primary[i] = sources.ID[i]
            continue
    
        xi, yi = ifs_coordinates[i]
    
        # stop loop if current source falls below (fixed) confusion limit
        if sources.mag[i]>reslim: break

        # check if local confusion limit is reached
        jmin = pampelmuse.ifs_data.instrument.dist2spax(xi, yi).argmin()
        slc  = np.sqrt((pampelmuse.ifs_data.instrument.x-xi)**2+(pampelmuse.ifs_data.instrument.y-yi)**2)<(10.*fwhm[i])
        if slc.any():
            # no. of resolution elements is [no. of spaxels]/[spaxels per resolution element]
            if density[slc].sum()/(slc.sum()/(np.pi*(0.5*fwhm[i])**2))>pampelmuse.confdens:
                continue

        # get S/N estimate for the source
        snr = signal_to_noise[i]
        if snr < pampelmuse.snrcut: continue
        
        # penalize S/N if source is outside FoV:
        edgedist = pampelmuse.ifs_data.instrument.distance_to_edge(ifs_coordinates[[i]])[0]
        if edgedist<0:
            snr *= np.exp(-(edgedist/fwhm[i])**2)
            if snr < pampelmuse.snrcut: continue
        # otherwise, add source to array for calculation of stellar density
        else:
            density[jmin] += 1

        # calculate distance to next brighter source (excluding background component)
        if (status==2).any():
            dr = spatial.distance.cdist(ifs_coordinates[[i]], ifs_coordinates[status==2])[0]
            j_near = dr.argmin()
        
            # issue warning if two sources have basically the same IFS coordinates
            if dr[j_near]<1e-5: logging.warning("Source #{0} might be double.".format(sources.ID[i]))

            # penalize S/N if a brighter source is located inside the FWHM of the PSF
            if dr[j_near]<fwhm[i]:
                snr *= (1.+0.9*np.log10(dr[j_near]/fwhm[i]))

                if dr[j_near]<(pampelmuse.binrad*fwhm[i]) or snr < pampelmuse.snrcut:
                    status[i] = 1
                    primary[i] = sources.ID[status==2][j_near]
                    logging.info("{0:9d}  {1:4.1f}  {2:6.1f}  {3:8.2f}  {4:6.2f}       1".format(\
                        int(sources.ID[i]), float(sources.mag[i]), float(snr), float(dr[j_near]), float(edgedist)))
                    continue
                    
        status[i] = 2
        primary[i] = sources.ID[i]
        logging.info("{0:9d}  {1:4.1f}  {2:6.1f}            {3:6.2f}       2".format(int(sources.ID[i]), float(sources.mag[i]), float(snr), float(edgedist)))
    
    logging.info("Results of source selection: {0} resolved, {1} nearby sources.".format(int((status==2).sum()), int((status==1).sum())))
    
    if (status==2).sum()==0:
        logging.critical("Did not find any resolved sources. Aborting...")
        raise ConfigurationError, "No resolved sources found. Check parameter file."

    
    # if requested, update confusion limit. Note that the configuration that is stored in the header of the prm-file may contain two different
    # values for "RESLIM": the initial one provided by the user and the one just determined from the source list.
    if pampelmuse.updaterl:
        reslim = sources.mag[status==2].max()
        logging.info("Estimated confusion limit is m_{0}={1:.1f}".format(pampelmuse.passband, float(reslim)))
    pampelmuse.initfit.reslim = reslim
        
    # if requested, add a component of unresolved stars
    if pampelmuse.useunres:
        pampelmuse.initfit.unreslim  = pampelmuse.unresbin + pampelmuse.initfit.reslim
        
        logging.info("Combining sources in magnitude range from {0:.2f} to {1:.2f} to unresolved component...".format(\
                float(pampelmuse.initfit.reslim), float(pampelmuse.initfit.unreslim)))
        
        for i in idx_magsorted:
        
            if status[i]>0: continue
            elif sources.mag[i] > pampelmuse.initfit.unreslim: break
            
            edgedist = pampelmuse.ifs_data.instrument.distance_to_edge(ifs_coordinates[[i]])[0]
            if edgedist>-1:
                status[i] = 0
                primary[i] = -2
            
        logging.info("Unresolved component contains {0} sources.".format(int((status==0).sum())))
        
    sources.primary = primary
    sources.status  = status
        
    # include dispersion information in Sources catalogue
    sources.ndisp = pampelmuse.stop-pampelmuse.start+1
    sources.crval = pampelmuse.ifs_data.crval+pampelmuse.start*pampelmuse.ifs_data.cdelt
    sources.cdelt = pampelmuse.ifs_data.cdelt
    sources.ctype = pampelmuse.ifs_data.ctype
    sources.cunit = pampelmuse.ifs_data.cunit
    sources.load_spectra() ## create dummy data: every spectrum is empty
    sources.transformation.resize(sources.ndisp)

    if not pampelmuse.smallifu:

        logging.info("Searching for possible PSF stars...")
        # searching photometry results for usefuls PSF stars must fulfil following criteria:
        # (1) at least 3 magnitudes brighter than provided resolution limit (parameter `RESLIM')
        # (2) must have been identified as resolved (i.e. status==2) in the source selection 
        # (3) must have an estimated S/N>20 (this is checked after aperture photometry)
        # (4) at a distance of at least the PSF FWHM from the edge of the FoV 
        # (5) isolation criterium:
        #      - a) relative contribution from neighboring sources inside PSFRAD smaller than value of parameter APERNOIS)
        #      - b) no brighter source inside 1.5 x PSFRAD distance

        # The first isolation of a star is measured using a slightly modified version of the shoulder drop or topographic prominence
        # (for the exact definition of the topographic prominence, see http://en.wikipedia.org/wiki/Topographic_prominence).
        # It is a measure for how strongly a source dominates its surroundings. The measure used here is the flux ratio of the
        # source itself and the brightest source within a distance r that we set to the value of the PSFRAD parameter.

        logging.info("IDs of stars that fullfil search criteria:")
        logging.info("      ID    mag  shoulder_drop")
        for i in idx_magsorted:
            
            if sources.mag[i] > (reslim-3): break
            if sources.status[i] != 2: continue
            if pampelmuse.ifs_data.instrument.distance_to_edge(ifs_coordinates[[i]])[0] < fwhm[i]: continue

            # calculate shoulder drop
            dr = spatial.distance.cdist(ifs_coordinates[[i]], ifs_coordinates)[0]
            dr[i] = np.inf ## handle the source itself
            dr[sources.status==4] = np.inf ## handle background components
            
            if (dr<pampelmuse.psfrad).any():

                assert dr.size == sources.mag.size
                dmag = (sources.mag[(dr<pampelmuse.psfrad)]-sources.mag[i]).min()
                sd   = 10**(-0.4*dmag)
                if sd>pampelmuse.apernois: continue
            else:
                sd   = 0

            # check if brighter star inside 1.5xPSFRAD:
            if (dr<1.5*pampelmuse.psfrad).any() and (sources.mag[dr<1.5*pampelmuse.psfrad]<sources.mag[i]).any():
                continue

            sources.status[i] = 3
            logging.info("{0:8d}  {1:5.2f}  {2:13.2f}".format(int(sources.ID[i]), float(sources.mag[i]), float(sd)))

        if len(sources.psf_indices) == 0:
            logging.critical("No possible PSF sources found!")
            raise ConfigurationError, "No possible PSF sources found!"

        # carry out aperture spectrophotometry for PSF sources
        logging.info("Performing aperture spectrophotometry on possible PSF stars...")
 
        # initialize aperture routine
        make_aper = AperPhoto(pampelmuse.ifs_data, [psf_instances[i] for i in np.flatnonzero(sources.status==3)])
        
        # run it - note that the array that is returned may contain NaN values (if the moment calculation failed or lead to unphysical values)
        # to account for this, masked arrays are used in the following calculations
        results = make_aper(r_aper=pampelmuse.aperrad, layer_range=(pampelmuse.start, pampelmuse.stop), background="local", ncpu=pampelmuse.cpucount)
        
        # for testing purposes:
##        tablehdu = fits.BinTableHDU(results)
##        tablehdu.writeto("{0}.aperture.fits".format(pampelmuse.prefix), clobber=True)
        
        # write spectra and IFS coordinates into instance of Sources, estimate S/N and exclude sources with S/N<20
        sources.load_ifs_coordinates(ids=results["id"][results["id"]>=0])

        i=0 
        for j in np.flatnonzero(sources.status==3):

            # add flux of source to Sources instance
            sources.data[:, sources.specrow[j], 0] = results["flux"][i]

            if results["id"][i]>=0: ## skip background component - is not used anyway currently (background set to "local")
                # NaN pixels are set to zero since DER_SNR skips such pixels
                snr = DER_SNR(np.ma.array(results["flux"][i], mask=np.isnan(results["flux"][i]), fill_value=0.).filled())
                
                # check if source fulfils S/N criterium for PSF sources
                if snr<20:
                    sources.status[j]=2
                    logging.info("S/N of source #{0} is only {1:.1f} - not used as PSF source.".format(results["id"][i], snr))
                else:
                    logging.info("S/N of source #{0} is {1:.1f} - used as PSF source.".format(results["id"][i], snr))
                
                # add IFS coordinates to Sources instance
                sources.ifs_coordinates[:, i, 0, 0] = np.ma.array(results["x"][i], mask=np.isnan(results["x"][i]), fill_value=psf_instances[j].xc).filled() 
                sources.ifs_coordinates[:, i, 1, 0] = np.ma.array(results["y"][i], mask=np.isnan(results["y"][i]), fill_value=psf_instances[j].yc).filled()
            i+=1
        
        if len(sources.psf_indices) == 0:
            logging.critical("No PSF sources found!")
            raise ConfigurationError, "No PSF sources found!"
        elif len(sources.psf_indices) < 4:
            logging.warning("Found only {0} possible PSF source(s)".format(len(sources.psf_indices)))
        
        
        if results.size > 3:
            # determine wavelength-dependent coordinate transformation
            logging.info("Determining wavelength-dependent coordinate transformation...")
            sources.fit_coordinate_transformation()

            # determine FWHM, E and THETA
            stars = (results["id"]>=0)
            
            logging.info("Estimating PSF properties using measured moments...")
            x2 = np.ma.array(results[["x2"]][stars].view(np.float32).reshape(stars.sum(), -1))
            x2.mask = np.isnan(x2)
            y2 = np.ma.array(results[["y2"]][stars].view(np.float32).reshape(stars.sum(), -1))
            y2.mask = np.isnan(y2)
            xy = np.ma.array(results[["xy"]][stars].view(np.float32).reshape(stars.sum(), -1))
            xy.mask = np.isnan(xy)

            # get A, B:
            A = np.sqrt((x2+y2)/2.+np.sqrt(((x2-y2)/2.)**2+xy**2))
            A.mask[A<=0] = True
            B = np.sqrt((x2+y2)/2.-np.sqrt(((x2-y2)/2.)**2+xy**2))
            B.mask[B<=0] = True
            
            # FWHM
            fwhm = np.sqrt(8.*np.log(2))*np.mean(A, axis=0)
            # TODO: should FWMH be updated as well?
            logging.info("Mean FWHM: {0:.2f}pix.".format(fwhm.mean()))

            # Ellipticity
            e = 1. - np.mean(B/A, axis=0)
            logging.info("Mean ellipticity: {0:.2f}".format(e.mean()))
            ii = psf_variables.names.index("e")
            if psf_variables.free[ii]:
                # need to check if dispersion information already included in PSF variables, otherwise, use numpy.resize that fills enlarged
                # array with repeated copies of available data
                if psf_variables.data.shape[0] == 1: psf_variables.data =  np.resize(psf_variables.data, e.shape+psf_variables.data.shape[1:])
                psf_variables.data[:, ii, 0] = e.filled(e.mean())

            # Position angle
            _theta  = 0.5*np.arctan(2*xy/(x2-y2))
            _theta -= (-1)**(_theta<0)*np.pi/4.
            _theta += (-1)**(xy>0)*np.pi/4.
            theta = _theta.mean(axis=0)
            logging.info("Mean position angle: {0:.2f}rad.".format(theta.mean()))
            ii = psf_variables.names.index("theta")
            if psf_variables.free[ii]:
                if psf_variables.data.shape[0] == 1: psf_variables.data =  np.resize(psf_variables.data, theta.shape+psf_variables.data.shape[1:])
                psf_variables.data[:, ii, 0] = theta.filled(theta.mean())
    
    # write results to file
    logging.info("Saving source selection in file {0}.".format(pampelmuse.filename))

    # add information about routine to namespace
    pampelmuse.initfit.utc = datetime.datetime.utcnow().strftime("%y-%m-%d %H:%M:%S")
    pampelmuse.initfit.revision = __revision__

    primary_hdu = pampelmuse.makeHeader() ## create header from namespace
    if usewcs:
        wcsheader = pampelmuse.ifs_data.wcs.to_header()
        for key in wcsheader:
            primary_hdu.header["hierarch PAMPELMUSE WCS {0}".format(key)] = wcsheader[key] ## (wcsheader[key], wcsheader.comments[key]) # cards too long if comment is included
    source_hdu  = sources.make_catalogue_hdu()
    spectra_hdu = sources.make_spectra_hdu()
    pospars_hdu = sources.transformation.make_hdu(include_fit=False)
    psfpars_hdu = psf_variables.make_hdu(include_fit=False)

    hdu_list = fits.HDUList([primary_hdu, source_hdu, spectra_hdu, pospars_hdu, psfpars_hdu])

    if os.path.exists(pampelmuse.filename):
        logging.warning("Overwriting existing PRM-file.")
        os.remove(pampelmuse.filename)
    hdu_list.writeto(pampelmuse.filename)
