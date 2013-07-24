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

The function 'calculate_noise' is currently unused. It is designed to calculate
the S/N in a spaxel-by-spaxel manner following the prescription of Siebert et al.,
(2011), AJ, 141, 187.

The function 'estimate_snr' is designed to estimate the S/N of all sources in a
catalogue and is called by the function 'initfit'.

Latest SVN Revision
-------------------
103, 2013/07/24
"""
__author__ = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__revision__ = 103

import datetime, os, sys, logging
import numpy as np, pyfits
from scipy import interpolate, ndimage
from PyQt4 import QtGui

## required PampelMuse routines
from pampelmuse.core import AperPhoto, Sources
from pampelmuse.core.psf import *
from pampelmuse.graphics import PampelMuseGui
from pampelmuse.functions.io import read_asciifile


## class defines the exception that is raised when the parameters
## in the Pampel Muse-configuration are such that they do not allow
## for a proper analysis of the provided input
class ConfigurationError(EnvironmentError):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)


## Routine to  estimate the S/N in the spectra. Adapted from the RAVE DR3
## method described in Siebert et al. (2011)
def calculate_noise(spectrum, norm=1):

    mask       = np.ones(spectrum.shape, dtype=np.bool)
    clippedPix = 0

    while True:
        smooth    = ndimage.uniform_filter1d(spectrum[mask], size=3)
        residuals = spectrum[mask] - smooth
        sigma     = residuals.std()
        
        mask[np.flatnonzero(mask)[abs(residuals)>(2.*sigma)]] = False

        if (mask==False).sum() == clippedPix:
            break
        else:
            clippedPix = (mask==False).sum()

    sigma_i = np.zeros((mask.sum()-2,), dtype=np.float32)
    for j in xrange(sigma_i.size):
        sigma_i[j] = np.std(residuals[j:j+3])

    return (1.62*norm)*np.median(sigma_i)



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
        logging.info("Determining Signal-to-Noise ratio (S/N) of sources...")

        snRatios = []
        totFlux  = 0.

        for psf in class_psf:
            
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
                                 aliases={"F606W":["f606w", "Vvega", "V"], "F814W":["f814w", "Ivega", "I"], "ra":["RA", ], "dec":["Dec", "DEC"]})

    if pampelmuse.globsky: ## add sky component if requested
        cfp_results["id"].append("-1")
        cfp_results["x"].append("0")
        cfp_results["y"].append("0")
        cfp_results[pampelmuse.passband].append("0")
        if "ra" in cfp_results.keys(): cfp_results["ra"].append("0")
        if "dec" in cfp_results.keys():  cfp_results["dec"].append("0")

    ## Initialize sources-class
    sources     = Sources(ids=np.asarray(cfp_results["id"], dtype=np.int32))
    sources.x   = np.asarray(cfp_results["x"], dtype=np.float32)
    sources.y   = np.asarray(cfp_results["y"], dtype=np.float32)
    sources.mag = np.asarray(cfp_results[pampelmuse.passband], dtype=np.float32)
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
        setattr(PSF, "set_{0}".format(parname_i),  parvalue_i)
        
        i += 1
        
    psf_variables.info()

    
    ## collapse data-cube to image in the provided photometric filter
    logging.info("Creating mock {0}-band image from IFS data...".format(pampelmuse.passband))
    
    ## get location of ascii-file with photometric filter & open it
    this_dir = os.path.dirname(__file__)
    filterfile  = os.path.join(this_dir, os.pardir, os.pardir, "data", pampelmuse.passband)
    filtercurve = np.genfromtxt(filterfile, usecols=(0,1), dtype=[("wl", np.float32), ("tp", np.float32)])

    f = interpolate.UnivariateSpline(filtercurve["wl"], filtercurve["tp"], k=3, s=0)
    fMin = filtercurve["wl"].min()
    fMax = filtercurve["wl"].max()
    fWht = 0.

    ## estimate noise using the method described by Siebert et al. (2001). This is currently too slow
    # TODO: increase speed of calculation
#    sigma_image = np.zeros(pampelmuse.ifs_data.instrument.transform.shape[0], dtype=np.float32)
#    i=0
#    j=0
#    for isscience in pampelmuse.ifs_data.instrument.isscience:
#        if isscience:
#            sigma_image[j] = calculate_noise(pampelmuse.ifs_data.get_spectrum(i)[1000:-1000], norm=1.3)
#            j += 1
#        i += 1

    ## make loop to avoid extensive use of memory for large data-cubes
    data_image  = np.zeros(pampelmuse.ifs_data.instrument.transform.shape[0], dtype=np.float32)
    sigma_image = np.zeros(data_image.shape, dtype=np.float32) ## old method to determine noise was from SIGMA cube via integration over passband
    logging.debug("Images contain {0} spaxels.".format(data_image.size))

    for i in xrange(pampelmuse.start, pampelmuse.stop):
        logging.debug("Working on layer {0:0{1}d}/{2:{1}d}".format(i+1, len(str(pampelmuse.stop-pampelmuse.start)), pampelmuse.stop-pampelmuse.start))
        wave_i = pampelmuse.ifs_data.crval+i*pampelmuse.ifs_data.cdelt
        if pampelmuse.ifs_data.ctype == "nm": wave_i *= 10
        if wave_i > fMin and wave_i < fMax:
            wht_i = f(wave_i)
            mask_i = pampelmuse.ifs_data.get_mask_for_layer(i)
            data_i, sigma_i = pampelmuse.ifs_data.get_layer(i)

            data_image[mask_i==0] += data_i*wht_i
            sigma_image[mask_i==0] += sigma_i*wht_i
            fWht += wht_i

    data_image  /= fWht
    sigma_image /= fWht

    ## create pseudo observation from photometry results
    ## PSF is same as in IFU, with FWHM scaled by `pixScale'-Parameter
    ## Unless a zeropoint is provided (configuration parameter ZEROMAG),
    ## the zeropoint is such that a source with magnitude `reslim' will
    ## have a flux of 1000 counts and a peak-S/N of roughly 10
    logging.info("Creating mock image from photometry sources...")

    PSF.set_size((int(sources.y.max()+1), int(sources.x.max()+1)))

    if pampelmuse.zeromag > 99.:
        magZ = pampelmuse.reslim + 7.5
    else:
        magZ = pampelmuse.zeromag

    identification_image  = np.zeros(PSF.size, dtype='float32')
    for i in xrange(sources.nsrc):
        if sources.ID[i] <= 0: continue 

        psf = PSF(Xc=sources.x[i], Yc=sources.y[i], mag=sources.mag[i]-magZ, src_id=sources.ID[i], maxrad=10.)

        identification_image[np.asarray(psf.y[psf.used], dtype=np.int16), np.asarray(psf.x[psf.used], dtype=np.int16)] += psf.f

    identification_image += np.random.normal(loc=0., scale=10, size=PSF.size)

    ## start source identification GUI to identify bright(est) stars
    logging.info("Preparing GUI for source identification...")

    app  = QtGui.QApplication(sys.argv)
    identifier = PampelMuseGui(sources=sources, ifsData=pampelmuse.ifs_data, refImage=identification_image, magCut=pampelmuse.reslim-1.)

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

    ## add coordinate transformation to PSF
    i=0
    for prm in sources.transformation.names:
        getattr(PSF, "set_{0}".format(prm))(sources.transformation.data[0,i,0])
        i+=1

    ## Source selection process 
    logging.info("Selecting sources to be used in PSF fitting...")

    ## Gather some basic information about sources: position in FoV, distance to next brighter source,...
    ## load IFS coordinates for all (point) sources
    sources.load_ifs_coordinates(ids=sources.ID[sources.ID>=0])

    ## Position in Fov/distance to edge of FoV
    edge_distance = pampelmuse.ifs_data.instrument.distance_to_edge(sources.ifs_coordinates[...,0].mean(axis=0))

    ## distance to next brighter source
    distance_to_brighter, ID_brighter = sources.distance_to_brighter()
    if (distance_to_brighter < 1e-5).any:
        for i in (distance_to_brighter < 1e-5).nonzero()[0]:
            logging.warning("Source #{0} might be double.".format(sources.ID[i]))

    if not pampelmuse.smallifu:
        shoulder_drop = sources.shoulder_drop(r=pampelmuse.psfrad)


    ## Perform photometric calibration (unless provided) and S/N estimation:
    signal_to_noise, intFlux = estimate_snr(PSF, sigma_image)
    
    ## Calculate ratio between expected flux for current zeropoint and flux in image
    fratio = intFlux/data_image.sum()
    dmag = -2.5*np.log10(fratio)

    ## determine zeropoint if requested and update S/N estimates 
    if pampelmuse.zeromag > 99.:
        signal_to_noise /= fratio
        magZ += dmag
        logging.info("Estimated photometric zeropoint ({0}) is {1:.2f}".format(pampelmuse.passband, float(magZ)))
    elif fratio > (2./3.) and fratio < 1.5:
        logging.info("The difference between provided and estimated photometric zeropoint ({0}) is {1:.2f}mag".format(pampelmuse.passband, abs(dmag)))
    else:
        logging.warning("The difference between provided and estimated photometric zeropoint ({0}) is {1:.2f}mag".format(pampelmuse.passband, abs(dmag)))
    
    pampelmuse.initfit.photzero = magZ

    ## collect FWHM of sources in single array
    fwhm = np.asarray([psf.fwhm for psf in PSF])

    ##  modify S/N of sources close to brighter neighbor:
    close = distance_to_brighter<fwhm
    signal_to_noise[close] *= (1.+0.9*np.log10(distance_to_brighter[close]/fwhm[close]))

    ##  penalize S/N of sources outside FoV:
    outside = edge_distance<0
    signal_to_noise[outside] *= np.exp(-(edge_distance[outside]/fwhm[outside])**2)


    ## determine status of all sources in catalogue:
    ##  4 - global sky spectrum
    ##  3 - resolved, own spectrum
    ##  2 - resolved, inferred spectrum
    ##  1 - resolved, but next to brighter source
    ##  0 - unresolved
    ## -1 - excluded

    ## Determine confusion limit, update parameter if requested
    magPS = sources.mag[sources.ifs_coordinates_loaded]
    mcut = magPS[edge_distance>0].min()
    while True:
        mcut += 0.1
#         if (magPS[edge_distance>0]<mcut).sum() > pampelmuse.confdens*data_image.size/(fwhm.mean()*fwhm.mean()):
#             mcut -= 0.1
#             break
        if not (magPS[edge_distance>0]>mcut).any():
            logging.info("No stars below m_{0}={1:.1f} detected.".format(pampelmuse.passband, mcut))
            break
        if signal_to_noise[edge_distance>0][magPS[edge_distance>0]>mcut].max() < pampelmuse.snrcut:
            logging.info("No stars below m_{0}={1:.1f} pass S/N-cut.".format(pampelmuse.passband, mcut))
            break
    logging.info("Estimated confusion limit: m_{0}={1:.1f}".format(pampelmuse.passband, mcut))
    
    ## Note that the configuuration may contain two different values for "RESLIM": The initial one provided
    ## by the user and the one just determined from the source list.
    pampelmuse.initfit.reslim = mcut
    if pampelmuse.updaterl: reslim = mcut
    else: reslim = pampelmuse.reslim
 
    if pampelmuse.useunres: unreslim = reslim+pampelmuse.unresbin

    status  = -np.ones((sources.nsrc,), dtype=np.int16)
    primary = np.zeros((sources.nsrc,), dtype=np.int32)

    logging.info("Results of source selection:")
    logging.info("       ID   mag   S/N  d_bright  d_edge  status")
    for i in xrange(sources.nsrc):
        
        if sources.ID[i] < 0:
            status[i]  = 4
            primary[i] = sources.ID[i]
        elif sources.mag[i] <= reslim:
            if signal_to_noise[i] > pampelmuse.snrcut:
                if distance_to_brighter[i] < pampelmuse.binrad*fwhm[i]:
                    status[i]  = 1
                    primary[i] = ID_brighter[i]
                else:
                    status[i]  = 2
                    primary[i] = sources.ID[i]
            else:
                if edge_distance[i] < 0:
                    continue
#                 if distance_to_brighter[i] < 2*pampelmuse.binrad*fwhm[i]:
#                     status[i]  = 1
#                     primary[i] = ID_brighter[i]
                else:
                    status[i]  = 0
                    primary[i] = -2 ## sources.ID[i]

            logging.info("   {0:6d}  {1:5.2f}  {2:5.1f}  {3:5.2f}  {4:5.2f}  {5:6d}".format(\
                int(sources.ID[i]), float(sources.mag[i]), float(signal_to_noise[i]), float(distance_to_brighter[i]), float(edge_distance[i]), int(status[i])))
            
        elif pampelmuse.useunres:
            if sources.mag[i]<=unreslim and edge_distance[i]>-fwhm[i]:
                status[i]  = 0
                primary[i] = -2

    sources.primary = primary
    sources.status  = status

    ## include dispersion information in Sources catalogue
    sources.ndisp = pampelmuse.stop-pampelmuse.start+1
    sources.crval = pampelmuse.ifs_data.crval+pampelmuse.start*pampelmuse.ifs_data.cdelt
    sources.cdelt = pampelmuse.ifs_data.cdelt
    sources.ctype = pampelmuse.ifs_data.ctype
    sources.cunit = pampelmuse.ifs_data.cunit
    sources.load_spectra() ## create dummy data: every spectrum is empty


    if not pampelmuse.smallifu:
        logging.info("Searching for useful PSF stars...")
        ## searching photometry results for usefuls PSF stars
        ## must fulfil following criteria:
        ## (1) at least 3 magnitudes brighter than provided resolution limit
        ##      (parameter `RESLIM')
        ## (2) at a distance of of at least 1 pixel from the edge of the FoV 
        ## (3) isolation criterium (relative contribution from neighboring
        ##      sources smaller than value of `APERNOIS'-parameter)

        logging.info("IDs of stars that fullfil search criteria:")
        logging.info("       ID    mag  shoulder_drop")
        for i in xrange(sources.nsrc):
            
            if sources.status[i] == 2 and sources.mag[i] <= (reslim-3) and edge_distance[i] > fwhm[i] and shoulder_drop[i] < pampelmuse.apernois:

                sources.status[i] = 3
                logging.info("   {0:6d}  {1:5.2f}  {2:14.2f}".format(int(sources.ID[i]), float(sources.mag[i]), float(shoulder_drop[i])))

        if len(sources.psf_indices) == 0:
            logging.error("No useful PSF sources found!")
            raise ConfigurationError, "No useful PSF sources found!"
        elif len(sources.psf_indices) < 4:
            logging.warning("Found only {0} useful PSF source(s)".format(len(sources.psf_indices)))

        ## carry out aperture spectrophotometry for PSF sources
        logging.info("Performing aperture spectrophotometry on PSF stars...")

        ## load IFS coordinates for the PSF sources
        sources.load_ifs_coordinates(ids=sources.ID[sources.status==3])
 
        # TODO: update aperture photometry!!!
        # include start/stop-values, replace, dataHDU and maskHDU with
        # Cube-instances 
        make_aper = AperPhoto(data=dataHDU, centers=aperData, mask=maskHDU)
        
        ## !!! NEEDS TO  BE UPDATED !!!
        aperResults = make_aper.aper_photo(rAper=pampelmuse.aperrad, psf=psf, calcMoments=True, thresh=3., localBG=True)
        
        if aperResults.size > 3:
            ## determine wavelength-dependent coordinate transformation
            logging.info("Determining wavelength-dependent coord. transf.")
            stars     = (aperResults["ID"]>=0)
            ifuCoords = aperResults[["x","y"]][stars].view(np.float32).reshape(stars.sum(), 2, -1)

            sources.loadIFUCoords(data=np.rollaxis(ifuCoords, 1, 0), ids=aperResults["ID"][stars])
            sources.calcTrans()
            if posFit == 1:   sources.ctFree[2:] = True
            elif posFit == 2: sources.ctFree[:] = True

            ##  determine FWHM, E and THETA
            logging.info("Approximating PSF from measured moments.")
            x2 = aperResults[["x2"]][stars].view(np.float32).reshape(stars.sum(), -1)
            y2 = aperResults[["y2"]][stars].view(np.float32).reshape(stars.sum(), -1)
            xy = aperResults[["xy"]][stars].view(np.float32).reshape(stars.sum(), -1)

            psf_variables.delValues()
            psf_variables.resizeStructure(ndisp=x2.shape[1])
            psf_variables.makeValues()

            ## get A, B:
            A = np.sqrt((x2+y2)/2.+np.sqrt(((x2-y2)/2.)**2+xy**2))
            B = np.sqrt((x2+y2)/2.-np.sqrt(((x2-y2)/2.)**2+xy**2))
            
            ## FWHM
            fwhm = np.sqrt(8.*np.log(2))*np.mean(A, axis=0)
##             psf_variables.values[psf_variables.parameters.index("fwhm")] = fwhm
            logging.info("Mean FWHM: {0:.2f}pix.".format(fwhm.mean()))

            ## Ellipticity
            e    = 1. - np.mean(np.where(B!=0, A/B, 1.), axis=0)
            psf_variables.values[psf_variables.parameters.index("e")] = e
            logging.info("Mean ellipticity: {0:.2f}".format(e.mean()))
        
            ## Position angle
            theta  = np.where(x2!=y2, 0.5*np.arctan(2*xy/(x2-y2)), 0.)
            theta -= (-1)**(theta<0)*np.pi/4.
            theta += (-1)**(xy>0)*np.pi/4.
            psf_variables.values[psf_variables.parameters.index("theta")] = theta.mean(axis=0)
            logging.info("Mean position angle: {0:.2f}rad.".format(theta.mean()))

            ##  copy spectra to sources instance
            for i in xrange(aperResults.size):
                j = np.where(sources.ID==aperResults["ID"][i])[0][0]
                sources.data[sources.specrow[j]] = aperResults["spectrum"][i]
            
    
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
