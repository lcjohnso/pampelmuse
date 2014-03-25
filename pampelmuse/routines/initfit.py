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
161, 2014/03/25
"""
__author__ = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__revision__ = 161

import datetime, os, sys, logging
import numpy as np, pyfits
from scipy import spatial, interpolate, ndimage
from PyQt4 import QtGui

## required PampelMuse routines
from pampelmuse.core import AperPhoto, Sources
from pampelmuse.core.psf import *
from pampelmuse.graphics import PampelMuseGui
from pampelmuse.functions.io import read_asciifile
from pampelmuse.functions.wcs import calc_offset

## class defines the exception that is raised when the parameters
## in the Pampel Muse-configuration are such that they do not allow
## for a proper analysis of the provided input
class ConfigurationError(EnvironmentError):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)



def estimate_snr(class_psf, sigma):
        """
        Calculate the S/N ratio of all stars in the scene.

        Parameters
        ----------
        psf : A pampelmuse.core.psf.profiles-class
            An instance of this a PSF-class should be present for every
            source in the scene. This is required to determine the flux
            distribution across the FoV
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

        for psf in class_psf:

            logging.debug("Estimating S/N  for source #{0}.".format(psf.src_id))

            if (sigma[psf.used]==0).any():
                logging.error("Sigma image contains invalid pixels with values <=0.")

            ## measure S/N
            snRatios.append(np.sqrt(((psf.f/sigma[psf.used])**2).sum()))

            ## add the flux contribution of this source in the FoV
            totFlux += psf.f.sum()

        return np.array(snRatios, dtype=np.float32), totFlux



def initfit(pampelmuse):

    logging.info("Executing requested routine INITFIT [Revision {0:d}].".format(__revision__))


    ## get results from crowded-field photometry, initialize sources
    logging.info("Reading photometry results from file '{0}'...".format(pampelmuse.initfit.sourcelist))

    cfp_results = read_asciifile(pampelmuse.initfit.sourcelist, usecols=(0,1,2,3,4,5), usenames="id,x,y,{0},ra,dec".format(pampelmuse.passband),
                                 aliases={"F606W":["Vvega", "V"], "F814W":["Ivega", "I"]})

    ## make sure either RA & Dec or x & y are available
    ## if both are available, RA & Dec are preferred over x & y. Pixel coordinates are obtained by calculating the offsets to the sources 
    ## with the lowest RA and Dec, respectively in units of arcseconds.
    if "ra" not in cfp_results.keys() or "dec" not in cfp_results.keys():
        if "x" not in cfp_results.keys() or "y" not in cfp_results.keys():
            logging.critical("Cannot find input source coordinates.")
            raise IOError, "Either x & y or RA & Dec must be provided with photometry."
    else:
        ra = np.asarray(cfp_results["ra"], dtype=np.float64)
        dec = np.asarray(cfp_results["dec"], dtype=np.float64)
    
        x, y = calc_offset(ra.max(), dec.min(), ra, dec)
        
        cfp_results["x"] = (-x).tolist()
        cfp_results["y"] = y.tolist()

    if pampelmuse.globsky: ## add sky component if requested
        cfp_results["id"].append("-1")
        cfp_results["x"].append("0")
        cfp_results["y"].append("0")
        cfp_results[pampelmuse.passband.lower()].append("0")
        if "ra" in cfp_results.keys(): cfp_results["ra"].append("0")
        if "dec" in cfp_results.keys():  cfp_results["dec"].append("0")

    ## Initialize sources-class
    sources     = Sources(ids=np.asarray(cfp_results["id"], dtype=np.int32))
    sources.x   = np.asarray(cfp_results["x"], dtype=np.float32)
    sources.y   = np.asarray(cfp_results["y"], dtype=np.float32)
    sources.mag = np.asarray(cfp_results[pampelmuse.passband.lower()], dtype=np.float32)
    if "ra" in cfp_results.keys(): sources.ra = np.asarray(cfp_results["ra"], dtype=np.float64)
    if "dec" in cfp_results.keys(): sources.dec = np.asarray(cfp_results["dec"], dtype=np.float64)

    ## for the purpose of creating a mock image from the provided photometric catalogue,
    ## the sources are shifted in x and y so that the source with the lowest x (y) coordinate
    ## is located in the first column (row). Otherwise a large part of the mock image might be
    ## compeletely empty. The pixel size of the photometric data is also changed to match the
    ## spaxelsize of the IFS data. These changes are undone after the source selection process
    ## is completed
    dx = sources.x[sources.ID>=0].min()
    dy = sources.y[sources.ID>=0].min()

    sources.x -= dx
    sources.x /= pampelmuse.pixscale
    sources.y -= dy
    sources.y /= pampelmuse.pixscale

    ## initialize the PSF (attributes and profiles)
    logging.info("Initializing the PSF...")
    
    psf_variables = Variables(profile=pampelmuse.psf.profile)

    if pampelmuse.psf.profile == "moffat": PSF = Moffat
    elif pampelmuse.psf.profile == "gauss": PSF = Gaussian

    i=0
    while hasattr(pampelmuse.psf, "parname{0}".format(i)):
        parname_i = getattr(pampelmuse.psf, "parname{0}".format(i))
        parvalue_i = getattr(pampelmuse.psf, "parvalue{0}".format(i))
        
        ## include value and fit request in PSF variables
        j = psf_variables.names.index(parname_i)
        psf_variables.data[0, j, 0] = parvalue_i
        psf_variables.free[j] = getattr(pampelmuse.psf, "freepar{0}".format(i))

        ## include value in PSF profiles
        getattr(PSF, "set_{0}".format(parname_i))(parvalue_i)

        i += 1
    
    psf_variables.info()
    
    ## collapse data-cube to image in the provided photometric filter
    logging.info("Creating mock {0}-band image from IFS data...".format(pampelmuse.passband))
    
    ## get location of ascii-file with photometric filter & open it
    this_dir = os.path.dirname(__file__)
    filterfile  = os.path.join(this_dir, os.pardir, os.pardir, "data", pampelmuse.passband.upper())
    filtercurve = np.genfromtxt(filterfile, usecols=(0,1), dtype=[("wl", np.float32), ("tp", np.float32)])

    f = interpolate.UnivariateSpline(filtercurve["wl"], filtercurve["tp"], k=3, s=0)
    fMin = filtercurve["wl"].min()
    fMax = filtercurve["wl"].max()

    ## make loop to avoid extensive use of memory for large data-cubes
    data_image   = np.zeros(pampelmuse.ifs_data.instrument.transform.shape[0], dtype=np.float32)
    sigma_image  = np.zeros(data_image.shape, dtype=np.float32) ## old method to determine noise was from SIGMA cube via integration over passband
    weight_image = np.zeros(data_image.shape, dtype=np.float32)
    logging.debug("Images contain {0} spaxels.".format(data_image.size))

    for i in xrange(pampelmuse.start, pampelmuse.stop):
        logging.debug("Working on layer {0:0{1}d}/{2:{1}d}".format(i-pampelmuse.start+1, len(str(pampelmuse.stop-pampelmuse.start)), pampelmuse.stop-pampelmuse.start))
        wave_i = pampelmuse.ifs_data.crval+i*pampelmuse.ifs_data.cdelt
        if pampelmuse.ifs_data.ctype == "nm": wave_i *= 10
        if wave_i > fMin and wave_i < fMax:
            wht_i = f(wave_i)
            
            data_i  = pampelmuse.ifs_data.data.layer(i)
            if pampelmuse.ifs_data.is_var:
                sigma_i = np.sqrt(pampelmuse.ifs_data.sigma.layer(i))
            else:
                sigma_i = pampelmuse.ifs_data.sigma.layer(i)

            if pampelmuse.ifs_data.mask is not None:
                mask_i = (data_i!=data_i)&pampelmuse.ifs_data.mask.layer(i)
            else:
                mask_i = (data_i!=data_i)

            data_image[mask_i==0] += data_i[mask_i==0]*wht_i
            sigma_image[mask_i==0] += sigma_i[mask_i==0]*wht_i
            weight_image[mask_i==0] += wht_i

    if (weight_image==0).any():
        data_image[weight_image==0] = 0
        sigma_image[weight_image==0] = 1
        weight_image[weight_image==0] = 1

    data_image  /= weight_image
    sigma_image /= weight_image
    sigma_image[sigma_image<1e-6] = 1e6 

    ## create pseudo observation from photometry results
    ## PSF is same as in IFU, with FWHM scaled by `pixScale'-Parameter
    ## Unless a zeropoint is provided (configuration parameter ZEROMAG),
    ## the zeropoint is such that a source with magnitude `reslim' will
    ## have a flux of 1000 counts and a peak-S/N of roughly 10
    logging.info("Creating mock image from photometry sources...")

    if pampelmuse.zeromag > 99.:
        magZ = pampelmuse.reslim + 7.5
    else:
        magZ = pampelmuse.zeromag

    # TODO: check if RA and Dec coordinates for IFS pointing are available;
    # if so, print pixel coordinates of field center in mock image

    ## prepare PSF for creation of mock image
    size   = (int(sources.y.max()+1), int(sources.x.max()+1))
    _image = np.zeros((size[0]*size[1], ), dtype='float32')

    PSF.set_size(size)
    for i in xrange(sources.nsrc-pampelmuse.globsky):
        logging.debug("Adding source {0:0{1}d}/{2:0{1}d} to mock image.".format(i+1, len(str(sources.nsrc)), sources.nsrc-pampelmuse.globsky))
        
        psf = PSF(Xc=sources.x[i], Yc=sources.y[i], mag=sources.mag[i]-magZ, src_id=sources.ID[i], maxrad=10.)

        if (psf.mag+magZ) <= pampelmuse.reslim: ## only add sources that are likely resolved
            _image[psf.used] += psf.f

    logging.debug("Adding noise to mock image.")
    _image += np.random.normal(loc=0., scale=10, size=_image.shape)

    ## resample image on regular (x, y) grid
    grid_y, grid_x = np.mgrid[0:size[0]-1:1j*size[0], 0:size[1]-1:1j*size[1]]
    identification_image = interpolate.griddata((PSF._transform[:,0], PSF._transform[:,1]), _image, (grid_y, grid_x), method="linear", fill_value=np.nan)

    ## start source identification GUI to identify bright(est) stars
    logging.info("Preparing GUI for source identification...")

    app  = QtGui.QApplication(sys.argv)
    identifier = PampelMuseGui(sources=sources, ifsData=pampelmuse.ifs_data, refImage=identification_image, magCut=pampelmuse.reslim-1.)

    ## try to include approximate position of IFS on reference image (at least RA and Dec of IFS data must be available)
    ra = pampelmuse.ifs_data.get_key("RA")
    dec = pampelmuse.ifs_data.get_key("Dec")
    
    if ra != None and dec != None:
        logging.info("Found central coordinates of {0}-field:".format(pampelmuse.ifs_data.instrument.name.upper()))
        logging.info("   RA={0}, DEC={1}".format(ra, dec))
        
        ## find source closest to this position, and use its x & y coordinates as central pixel coordinates
        ic = np.argmin((ra-sources.ra)**2+(dec-sources.dec)**2)
        xc = sources.x[ic]
        yc = sources.y[ic]

        ## for ARGUS, also look for rotation angle:
        if pampelmuse.ifs_data.instrument.name == "argus": angle = pampelmuse.ifs_data.get_key("ESO ADA POSANG")
        else: angle=0
        identifier.addFovToReference(xc, yc, angle=np.radians(angle))

    identifier.show()
    app.exec_()

    identified = identifier.identifiedSources

    ## recover original coordinates in Sources instance and for PSF instances
    sources.x *= pampelmuse.pixscale
    sources.x += dx
    sources.y *= pampelmuse.pixscale
    sources.y += dy

    PSF.set_transform(pampelmuse.ifs_data.instrument.transform)
    i=0
    for psf in PSF:
        psf.Xc = sources.x[i]
        psf.Yc = sources.y[i]
        i += 1

    if len(identified)<2:
        logging.critical("At least 2 sources must be identified.")

    ## determine coordinate transformation using identified sources, when providing the IFS coordinates for the sources that were identified,
    ## they need to be provided in the right order (the sources were not necessarily idenitified in the same order they appear in the catalogue)
    sorted_indices = np.argsort([list(sources.ID).index(entry) for entry in identified.keys()])
    
    sources.load_ifs_coordinates(ids=identified.keys())
    sources.ifs_coordinates[..., 0] = np.asarray(identified.values())[sorted_indices,:2]
    sources.fit_coordinate_transformation()
    if pampelmuse.posfit == 1: sources.transformation.free[2:] = True
    elif pampelmuse.posfit == 2: sources.transformation.free[:] = True

    ## add coordinate transformation to PSF instances
    i=0
    for prm in sources.transformation.names:
        getattr(PSF, "set_{0}".format(prm))(sources.transformation.data[0,i,0])
        i+=1

    ## Perform photometric calibration (unless provided) and S/N estimation:
    logging.info("Estimating S/N for all sources...")
    signal_to_noise, intFlux = estimate_snr(PSF, sigma_image)

    logging.info("Estimating the photometric zeropoint...")
    ## Calculate ratio between expected flux for current zeropoint and flux in image
    fratio = intFlux/data_image.sum()
    dmag = -2.5*np.log10(fratio)

    ## determine zeropoint if requested and update S/N estimates 
    if pampelmuse.zeromag > 99.:
        signal_to_noise /= fratio
        magZ += dmag
        logging.info("Estimated photometric zeropoint ({0}) is {1:.2f}".format(pampelmuse.passband, float(magZ)))
    elif fratio > (2./3.) and fratio < 1.5:
        logging.info("Offset between provided and estimated photometric zeropoint ({0}) is {1:.2f}mag".format(pampelmuse.passband, abs(dmag)))
    else:
        logging.warning("Offset between provided and estimated photometric zeropoint ({0}) is {1:.2f}mag".format(pampelmuse.passband, abs(dmag)))

    pampelmuse.initfit.photzero = magZ

    ## Source selection process, the status of a source can be one of the following:
    ##  4 - global sky spectrum
    ##  3 - resolved, own spectrum
    ##  2 - resolved, inferred spectrum
    ##  1 - resolved, but next to brighter source
    ##  0 - unresolved
    ## -1 - excluded
    logging.info("Selecting sources to be used in PSF fitting...")

    ## collect FWHM of sources in single array
    fwhm = np.asarray([psf.fwhm for psf in PSF])

    ## get magnitude-sorted list of indices of the sources the Sources-class (brightest sources first)
    idx_magsorted = np.argsort(sources.mag[sources.ID>=0])

    ## load IFS coordinates for all stars
    sources.load_ifs_coordinates()
    ifs_coordinates = sources.ifs_coordinates[0,...,0]

    ## arrays to keep track of the status of each source and the ID of the brightest source in
    ## its component (that is identical to the source ID unless a brighter source is so close
    ## that the two cannot be separated and are fitted as a single component)
    status  = -np.ones((sources.nsrc,), dtype=np.int16)
    primary = np.zeros((sources.nsrc,), dtype=np.int32)
    
    ## If confusion limit should be refined, we initially set it to 99. and later set it to the
    ## magnitude of the faintest star that is included in the fit (as a resolved source)
    reslim = 99. if pampelmuse.updaterl else pampelmuse.reslim

    logging.info("       ID   mag   S/N  d_bright   d_edge   status")
    logging.info("                       [spaxel]  [spaxel]        ")
    for i in idx_magsorted:
    
        ## stop loop if current source falls below (fixed) confusion limit
        if sources.mag[i]>reslim: break

        ## get S/N estimate for the source
        snr = signal_to_noise[i]
        if snr < pampelmuse.snrcut: continue
        
        ## penalize S/N if source is outside FoV:
        edgedist = pampelmuse.ifs_data.instrument.distance_to_edge(ifs_coordinates[[i]])[0]
        if edgedist<0:
            snr *= np.exp(-(edgedist/fwhm[i])**2)
            if snr < pampelmuse.snrcut: continue

        ## calculate distance to next brighter source
        if (status==2).any():
            dr = spatial.distance.cdist(ifs_coordinates[[i]], ifs_coordinates[status==2])[0]
            j_near = dr.argmin()
        
            ## issue warning if two sources have basically the same IFS coordinates
            if dr[j_near]<1e-5: logging.warning("Source #{0} might be double.".format(sources.ID[i]))

            ## penalize S/N if a brighter source is located inside the FWHM of the PSF
            if dr[j_near]<fwhm[i]:
                snr *= (1.+0.9*np.log10(dr[j_near]/fwhm[i]))

                if dr[j_near]<(pampelmuse.binrad*fwhm[i]) or snr < pampelmuse.snrcut:
                    status[i] = 1
                    primary[i] = sources.ID[status==2][j_near]
                    logging.info("   {0:6d}  {1:5.2f}  {2:5.1f}  {3:5.2f}  {4:5.2f}  1".format(\
                        int(sources.ID[i]), float(sources.mag[i]), float(snr), float(dr[j_near]), float(edgedist)))
                    continue
                    
        status[i] = 2
        primary[i] = sources.ID[i]
        logging.info("   {0:6d}  {1:5.2f}  {2:5.1f}         {3:5.2f}  2".format(int(sources.ID[i]), float(sources.mag[i]), float(snr), float(edgedist)))
    
    logging.info("Results of source selection: {0} resolved, {1} nearby sources.".format(int((status==2).sum()), int((status==1).sum())))
    
    ## set status of sky component if it is included
    if pampelmuse.globsky:
        status[-1] = 4
        primary[-1] = -1
    
    ## if requested, update confusion limit. Note that the configuration that is stored in the header of the prm-file may contain two different
    ## values for "RESLIM": the initial one provided by the user and the one just determined from the source list.
    if pampelmuse.updaterl:
        reslim = sources.mag[status==2].max()
        logging.info("Estimated confusion limit is m_{0}={1:.1f}".format(pampelmuse.passband, float(reslim)))
    pampelmuse.initfit.reslim = reslim
        
    ## if requested, add a component of unresolved stars
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
        
    ## include dispersion information in Sources catalogue
    sources.ndisp = pampelmuse.stop-pampelmuse.start+1
    sources.crval = pampelmuse.ifs_data.crval+pampelmuse.start*pampelmuse.ifs_data.cdelt
    sources.cdelt = pampelmuse.ifs_data.cdelt
    sources.ctype = pampelmuse.ifs_data.ctype
    sources.cunit = pampelmuse.ifs_data.cunit
    sources.load_spectra() ## create dummy data: every spectrum is empty
    sources.transformation.resize(sources.ndisp)

    if not pampelmuse.smallifu:

        logging.info("Searching for useful PSF stars...")
        ## searching photometry results for usefuls PSF stars must fulfil following criteria:
        ## (1) at least 3 magnitudes brighter than provided resolution limit (parameter `RESLIM')
        ## (2) must have been identified as resolved (i.e. status==2) in the source selection 
        ## (3) must have an estimated S/N>20
        ## (4) at a distance of at least the PSF FWHM from the edge of the FoV 
        ## (5) isolation criterium (relative contribution from neighboring sources smaller than value of `APERNOIS'-parameter)

        ## The isolation of a star is measured using a slightly modified version of the shoulder drop or topographic prominence
        ## (for the exact definition of the topographic prominence, see http://en.wikipedia.org/wiki/Topographic_prominence).
        ## It is a measure for how strongly a source dominates its surroundings. The measure used here is the flux ratio of the
        ## source itself and the brightest source within a distance r that we set to the value of the PSFRAD parameter.

        logging.info("IDs of stars that fullfil search criteria:")
        logging.info("       ID    mag  shoulder_drop")
        for i in idx_magsorted:
            
            if sources.mag[i] > (reslim-3): break
            if sources.status[i] != 2: continue
            if signal_to_noise[i] < 20: continue
            if pampelmuse.ifs_data.instrument.distance_to_edge(ifs_coordinates[[i]])[0] < fwhm[i]: continue

            ## calculate shoulder drop
            dr   = spatial.distance.cdist(ifs_coordinates[[i]], ifs_coordinates[idx_magsorted!=i])[0]
            dmag = (sources.mag[dr<pampelmuse.psfrad]-sources.mag[i]).min()
            sd   = 10**(-0.4*dmag)
            if sd>pampelmuse.apernois: continue
            
            sources.status[i] = 3
            logging.info("   {0:6d}  {1:5.2f}  {2:14.2f}".format(int(sources.ID[i]), float(sources.mag[i]), float(sd)))

        if len(sources.psf_indices) == 0:
            logging.error("No useful PSF sources found!")
            raise ConfigurationError, "No useful PSF sources found!"
        elif len(sources.psf_indices) < 4:
            logging.warning("Found only {0} useful PSF source(s)".format(len(sources.psf_indices)))

        ## carry out aperture spectrophotometry for PSF sources
        logging.info("Performing aperture spectrophotometry on PSF stars...")
 
        ## initialize aperture routine
        make_aper = AperPhoto(pampelmuse.ifs_data, sources)
        
        ## run it
        results = make_aper.aper_photo(r_aper=pampelmuse.aperrad, layer_range=(pampelmuse.start, pampelmuse.stop), PSF=PSF, background="local")
        
        ## write IFS coordinates into instance of Sources
        sources.load_ifs_coordinates(ids=results["ID"][results["ID"]>=0])

        for i in xrange(len(results)):
            if results["ID"][i]>=0:
                j = np.flatnonzero(sources.ID[sources.ifs_coordinates_loaded]==results["ID"][i])[0]
                sources.ifs_coordinates[:, j, 0, 0] = results["x"][i]
                sources.ifs_coordinates[:, j, 1, 0] = results["y"][i]
        
            k = sources.specrow[np.where(sources.ID==results["ID"][i])[0][0]]
            sources.data[:, k, 0] = results["flux"][i]
        
        # TODO: check if following if clause works
        if results.size > 3:
            ## determine wavelength-dependent coordinate transformation
            logging.info("Determining wavelength-dependent coordinate transformation...")
            sources.fit_coordinate_transformation()

            ##  determine FWHM, E and THETA
            stars = (results["ID"]>=0)
            
            logging.info("Estimating PSF properties using measured moments...")
            x2 = results[["x2"]][stars].view(np.float32).reshape(stars.sum(), -1)
            y2 = results[["y2"]][stars].view(np.float32).reshape(stars.sum(), -1)
            xy = results[["xy"]][stars].view(np.float32).reshape(stars.sum(), -1)

            ## get A, B:
            A = np.sqrt((x2+y2)/2.+np.sqrt(((x2-y2)/2.)**2+xy**2))
            B = np.sqrt((x2+y2)/2.-np.sqrt(((x2-y2)/2.)**2+xy**2))
            
            ## FWHM
            fwhm = np.sqrt(8.*np.log(2))*np.mean(A, axis=0)
            # TODO: should FWMH be updated as well?
            logging.info("Mean FWHM: {0:.2f}pix.".format(fwhm.mean()))

            ## Ellipticity
            e = 1. - np.mean(np.where(A!=0, B/A, 1.), axis=0)
            logging.info("Mean ellipticity: {0:.2f}".format(e.mean()))
            ii = psf_variables.names.index("e")
            if psf_variables.free[ii]:
                ## need to check if dispersion information already included in PSF variables, otherwise, use numpy.resize that fills enlarged
                ## array with repeated copies of available data
                if psf_variables.data.shape[0] == 1: psf_variables.data =  np.resize(psf_variables.data, e.shape+psf_variables.data.shape[1:])
                psf_variables.data[:, ii, 0] = e

            ## Position angle
            theta  = np.where(x2!=y2, 0.5*np.arctan(2*xy/(x2-y2)), 0.)
            theta -= (-1)**(theta<0)*np.pi/4.
            theta += (-1)**(xy>0)*np.pi/4.
            logging.info("Mean position angle: {0:.2f}rad.".format(theta.mean()))
            ii = psf_variables.names.index("theta")
            if psf_variables.free[ii]:
                if psf_variables.data.shape[0] == 1: psf_variables.data =  np.resize(psf_variables.data, theta.shape+psf_variables.data.shape[1:])
                psf_variables.data[:, ii, 0] = theta.mean(axis=0)
    
    ## write results to file
    logging.info("Saving source selection in file {0}.".format(pampelmuse.filename))

    ## add information about routine to namespace
    pampelmuse.initfit.utc = datetime.datetime.utcnow().strftime("%y-%m-%d %H:%M:%S")
    pampelmuse.initfit.revision = __revision__

    primary_hdu = pampelmuse.makeHeader() ## create header from namespace
    source_hdu  = sources.make_catalogue_hdu()
    spectra_hdu = sources.make_spectra_hdu()
    pospars_hdu = sources.transformation.make_hdu(include_fit=False)
    psfpars_hdu = psf_variables.make_hdu(include_fit=False)

    hdu_list = pyfits.HDUList([primary_hdu, source_hdu, spectra_hdu, pospars_hdu, psfpars_hdu])

    if os.path.exists(pampelmuse.filename):
        logging.warning("Overwriting existing PRM-file.")
        os.remove(pampelmuse.filename)
    hdu_list.writeto(pampelmuse.filename)
