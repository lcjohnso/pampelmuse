"""
INITFIT
=======
The routine is designed to create a sophisticated list of sources to be
analysed with the PampelMuse routine CUBEFIT. It is called automatically if
PampelMuse is started using the `INITFIT`-option. As input, it requires the
prefix of the data-cube that should be analysed and an ASCII-file containing
photometry of the field (in a single photometric bandpass). The output will
be a PampelMuse PRM-file containing all the required information for the 
next steps of the analysis.

CHANGELOG:
----------
2011/08/08, v0.0.0.0
2011/09/26, v0.1.0.0, several change to improve connection to `cubefit'-
                      routine, especially storage of sky spectrum.
2011/10/13, v0.1.1.0, updated importing of PampelMuse-packages
2011/10/27, v0.1.2.0, modified source selection:
                      i) no sources are selected with distance >fitrad to FoV
                      ii) all stars above RESLIM in FoV are selected, merged
                          with nearby sources if SNR below SNRCUT.
2011/11/29, v0.1.2.1, changed how photometric calibration is performed.
                      Instead of using bright individual sources, the 
                      combination of all sources inside the FoV is used. Also
                      if the SMALLIFU flag is set, aperture spectrophotometry
                      is skipped. And spectra of fainter sources are no longer
                      infered from the spectra of the brighter ones.
2011/12/01, v0.1.2.2, modified PSF usage - now the same PSF as given in the
                      configuration file is used.
2011/12/05, v0.1.2.3, the routine now also stores the PSF parameters in the
                      PRM-file. If `smallifu'-flag is disabled, the moments
                      obtained during aperture spectrophotometry are used
                      to update the (fitted)  PSF parameters.
                      Some debugging.
2012/01/30, v0.2.0.0, changed the code to create PRM-files the are consistent
                      with the updated structure, i.e. PSF and coordinate
                      parameters are stored in binary FITS-tables.
2012/03/07, v0.3.0.0, major updates to the code in order to implement the
                      usage of the new classes that faciliate I/O of fit
                      parameters: `Sources` and `PSF`. Some dependencies have
                      been removed in the process (SelectSources and
                      StoreParam). Still need to implement a few things.
2012/03/17, v0.4.0.0, this version makes full use of the new structure that
                      uses the `Sources`- and `PSF`-classes.
2012/03/23, v0.4.0.1, some minor changes.
2012/04/16, v0.5.0.0, updated source selection; inlcuded possibility for
                      automated source selection using expected confusion
                      limit.
2012/04/17, v0.5.1.0, newer versions of scipy (later than `0.8.0`(?)) make
                      scipy.interpolate.UnivariateSpline.__call__ return a
                      float instead of an nd_array when only one value is
                      requested. This lead to IndexErrors when running
                      INITFIT.
2012/06/01, v0.5.2.0, bugfix: earlier versions crashed if during the
                              determination of the confusion limit the maximum
                              magnitude depth of the photometry was reached.
2012/06/04, v0.5.3.0, bugfix: `POSFIT'-parameter was not used properly together
                              with `SMALLIFU=False`; positions were not fitted
2012/06/19, v0.6.0.0, changed code to make it compatible with update PSF
                      profile classes. Now, if the FWHM is determined from the
                      data, the `Profile-class property is updated instead of
                      the respective PSF parameter.
2012/07/03, v0.6.0.1, modified logging so that a warning is triggered if two
                      sources have basically the same coordinates.
2012/09/03, v0.6.1.1, bugfix: replaced 'fwhm' by 'prof.fwhm' in line 298.
2012/10/09, v0.6.1.2, minor change when determining resolution limit: using
                      last magnitude sampled above critical source density
                      instead of first sampled one below.
2012/11/09, v0.7.0.0, improved handling of instrument-specific features
                      (missing spaxels, non-rectangular FoV, ...) by the code.
"""
__author__ = "Sebastian Kamann (skamann@aip.de)"
__version__ = "0.7.0.0"
import datetime, os, sys, logging
import numpy as np, pyfits
from scipy import interpolate, sparse
from PyQt4 import QtGui

## required PampelMuse routines
from pampelmuse.graphics.ident_sources import IdentSources
from pampelmuse.core.aperture import AperPhoto
from pampelmuse.core.sources import Sources
from pampelmuse.core.psf import PSF
from pampelmuse.core.instruments import Pmas, Argus



## class defines the exception that is raised when the parameters
## in the PampelMuse-configuration are such that they do not allow
## for a proper analysis of the provided input
class ConfigurationError(EnvironmentError):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)




def initfit(arguments):

    ## arguments is a namespace that MUST contain the following entries:
    ## 1) prefix - the prefix common to all FITS-files to be used
    ## 2) sourcelist - the name of the ASCII-file  containing the coordinates
    ##                 and magnitudes of the sources
    ## 3) config - a dictionary containing the current PampelMuse configuration

    print ""
    print 21* " "+"***  InitFit, version {0}  ***".format(__version__)+21*" "
    print ""
  
  
###### List of required definitions: #################
    prefix   = arguments.prefix                     ##
    cfpfile  = arguments.sourcelist                 ##
###### General information ###########################
    ifu      = arguments.config["IFU"]              ##
###### Photometry information:                      ##
    passband = arguments.config["PASSBAND"]         ##
    pixScale = arguments.config["PIXSCALE"]         ## 
###### Source selection process:                    ##
    smallifu = arguments.config["SMALLIFU"]         ##
    aperRad  = arguments.config["APERRAD"]          ##
    apernois = arguments.config["APERNOIS"]         ##
    reslim   = arguments.config["RESLIM"]           ##
    updateRL = arguments.config["UPDATERL"]         ##
    confDens = arguments.config["CONFDENS"]         ##
    snrcut   = arguments.config["SNRCUT"]           ##
    binrad   = arguments.config["BINRAD"]           ##
    useunres = arguments.config["USEUNRES"]         ##
    unresbin = arguments.config["UNRESBIN"]         ##
    globsky  = arguments.config["GLOBSKY"]          ##
###### Source positions:                            ##
    posFit   = arguments.config["POSFIT"]           ##
###### The PSF                                      ##
    psfRad   = arguments.config["PSFRAD"]           ##
    psfType  = arguments.config["PSFTYPE"]          ##
    psfPars  = arguments.config["PSFPARS"]          ##
    psfFit   = arguments.config["PSFFIT"]           ##
    ##################################################
    DATAPATH = os.path.expandvars("$pampelmuse_path/data/")


    ## get results from crowded-field photometry, initialize sources
    logging.info("Reading photometry results from file `{0}'".format(cfpfile))

    cfpResults = np.genfromtxt(cfpfile, names=True, dtype=None, comments="#")

    if passband in cfpResults.dtype.names: magColName = passband
    elif "mag" in cfpResults.dtype.names: magColName = "mag"
    else: logging.error(\
        "Cannot find mags, missing field `{0}` or `mag`.".format(passband))

##     if useunres: select = cfpResults[magColName]<unreslim
##     else:        select = cfpResults[magColName]<reslim
## 
##     if globsky:
##         ids = np.r_[cfpResults['id'][select], -1]
##         x   = np.r_[cfpResults['x'][select], 0]
##         y   = np.r_[cfpResults['y'][select], 0]
##         mag = np.r_[cfpResults[magColName][select], 0]
##     else:
##         ids = cfpResults['id'][select]
##         x   = cfpResults['x'][select]
##         y   = cfpResults['y'][select]
##         mag = cfpResults[magColName][select]

    if globsky:
        ids = np.r_[cfpResults['id'], -1]
        x   = np.r_[cfpResults['x'], 0]
        y   = np.r_[cfpResults['y'], 0]
        mag = np.r_[cfpResults[magColName], 0]
    else:
        ids = cfpResults['id']
        x   = cfpResults['x']
        y   = cfpResults['y']
        mag = cfpResults[magColName]


    x -= x[ids>=0].min()
    y -= y[ids>=0].min()

    sources     = Sources(ids=ids)
    sources.x   = x
    sources.y   = y
    sources.mag = mag


    ## initialize PSF
    if psfType == "moffat": psfArgs = ["theta", "e", "Mfwhm", "beta"]
    elif psfType == "gauss": psfArgs = ["Gtheta", "e", "fwhm"]
    elif psfType == "lorentz": psfArgs = ["Ltheta", "e", "fwhm"]
    elif psfType == "gaumoff": psfArgs = ["Gfwhm", "beta", "Mfwhm", "Mfrac",
                                          "e", "fwhm"]

    psf = PSF() ## PSF used in analysis
    psf.makeNewPSF(profile=psfType)

    ## check which PSF parameters should be fitted and add parameters
    n = len(psfArgs)
    for arg in psfArgs:
        if psfFit >= 2**n:
            psfFit -= 2**n
            psf.addParameter(name=arg, value=psfPars[n-1], freepar=True)
        else:
            psf.addParameter(name=arg, value=psfPars[n-1], freepar=False)
        n -= 1

    psf.makeValues()
    psf.info()


    ## open FITS-files:
    logging.info("Reading IFS data...")

    ## Initialize instance of instrument-class
    if ifu == "pmas":
        logging.info("Selected instrument is PMAS.")
        sources.instrument = Pmas()
    elif ifu == "argus":
        logging.info("Selected instrument is FLAMES/ARGUS.")
        sources.instrument = Argus()
    else:
        logging.warning("No valid instrument specifier provided.")
        sources.instrument = None

    ## data
    dataHDU = pyfits.open("{0}.dat.fits".format(prefix))[0]
    cdelt = dataHDU.header['CDELT3']
    crval = dataHDU.header['CRVAL3']
    ndisp = dataHDU.header['NAXIS3']
    logging.info("   {0}.dat.fits - OK".format(prefix))

    ## sigma
    sigmaHDU = pyfits.open("{0}.err.fits".format(prefix))[0]
    logging.info("   {0}.err.fits - OK".format(prefix))

    ## mask
    try:
        maskHDU = pyfits.open("{0}.msk.fits".format(prefix))[0]
        logging.info("   {0}.msk.fits - OK".format(prefix))
    except IOError:
        maskHDU = None
        logging.info("   {0}.msk.fits - Not found".format(prefix))

    ## collapse data-cube to image in the provided photometric filter
    logging.info("Creating mock {0}-band image from datacube.".format(passband))
    filtercurve = np.genfromtxt(os.path.join(DATAPATH, passband), usecols=(0,1),
                                dtype=[("wl", np.float32), ("tp", np.float32)])

    f = interpolate.UnivariateSpline(\
        filtercurve["wl"], filtercurve["tp"], k=3, s=0)
    fMin = filtercurve["wl"].min()
    fMax = filtercurve["wl"].max()

    ## make loop to avoid extensive use of memory for large data-cubes
    dataIma  = np.zeros((dataHDU.header["NAXIS2"], dataHDU.header["NAXIS1"]),
                        dtype=np.float32)
    sigmaIma = np.zeros(dataIma.shape, dtype=np.float32)
    for i in xrange(ndisp):
        wave_i = crval+i*cdelt
        if wave_i > fMin and wave_i < fMax:
            dataIma  += dataHDU.section[i]*f(wave_i)## [0]
            sigmaIma += sigmaHDU.section[i]*f(wave_i)## [0]
    dataIma  /= filtercurve["tp"].sum()
    sigmaIma /= filtercurve["tp"].sum()


    ## create pseudo observation from photometry results
    ## PSF is same as in IFU, with FWHM scaled by `pixScale'-Parameter
    ## The zeropoint is such that a source with magnitude `reslim' will
    ## have a flux of 1000 counts and a peak-S/N of roughly 10
    logging.info("Creating mock image from photometry sources.")

    MZERO     = reslim + 7.5
    size      = (int((sources.y.max()/pixScale)+1),
                 int((sources.x.max()/pixScale)+1))
    identIma  = np.zeros(size, dtype='float32')
    prof      = psf(index=0)
    for i in xrange(sources.nsrc):
        if sources.ID[i] < 0: continue 
        identIma += prof(size=size, x_c=sources.x[i]/pixScale,
                         y_c=sources.y[i]/pixScale, mag=(sources.mag[i]-MZERO),
                         maxrad=10, return_sparse=True, as_sparse_matrix=True)\
                         .todense()
    identIma += np.random.normal(loc=0., scale=10, size=size)

    ## start source identification GUI to identify bright(est) stars
    logging.info("Preparing GUI for source identification"),

    ##  temporary solution: create rec-array from Sources-instance; usage of
    ##                      instance in IdentSources will be implemented later
    pickData = sources.toRec(\
        ids=sources.ID[(sources.mag<reslim)&(sources.ID>=0)])

    pickData["x"] /= pixScale
    pickData["y"] /= pixScale

    app  = QtGui.QApplication(sys.argv)
    identifier = IdentSources(grabIma=dataIma, pickIma=identIma,
                              pickData=pickData)
    identifier.show()
    app.exec_()

    identified = identifier.getResults()


    ## adjust PSF FWHM (only if >=4 sources have a valid FWHM measurement)
    valid = (identified["fwhm_x"]>0)&(identified["fwhm_y"]>0)
    if valid.sum() < 4:
        logging.warn("Skipping FWHM measurement - too few sources ({0})"\
                         .format(valid.sum()))
##         fwhm = psf.values[psf.parameters.index("fwhm"),:].mean(axis=-1)
    else:
        prof.fwhm = 0.5*(identified['fwhm_x'][valid].mean() \
                            + identified['fwhm_y'][valid].mean())
        logging.info("Estimated Seeing FWHM: {0:.1f}pixel".format(prof.fwhm))
##         psf.values[psf.parameters.index("fwhm"),:] =  fwhm
        ## NOTE: will only write FWHM to values if FWHM is fitted
        ##       CHANGED: Parameter not updated, only class-property `fwhm`

    ## determine coordinate transformation using identified sources
    sources.loadIFUCoords(\
        data=identified[["x", "y"]].view(np.float64).reshape(-1,2).T,
        ids=identified["id"])
    sources.calcTrans()
    if posFit == 1: sources.ctFree[2:] = True
    elif posFit == 2: sources.ctFree[:] = True

    ## Source selection process 
    logging.info("Selecting sources to be used in PSF fitting.")

    ## get IFU coordinates for all (point) sources
    sources.loadIFUCoords()

    ## Position in Fov/distance to edge of FoV
    edgeDist = sources.edgeDist(fovShape=dataIma.shape)

    ## Photometric calibration:
    magPS = sources.mag[sources.ifuSelect]

    srcFlux = (10**(-0.4*magPS[edgeDist>0])).sum()
    magZ    = -2.5*np.log10(srcFlux/dataIma.sum())
    logging.info("Estimated photometric zeropoint ({0}): {1:.2f}".format(\
            passband, float(magZ)))


    ## distance to next brighter source
    distToBrighter, brighterID = sources.distToBrighter()
    if (distToBrighter < 1e-5).any:
        for i in (distToBrighter < 1e-5).nonzero()[0]:
            logging.warning("Source #{0} might be double.".format(\
                    sources.ID[i]))


    ## signal-to-noise
    signalToNoise = sources.signalToNoise(psf=prof, magZ=magZ, sigma=sigmaIma,
                                          r=psfRad)
    ##  modify S/N of sources close to brighter neighbor:
    close = distToBrighter<prof.fwhm    
    signalToNoise[close] *= (1.+0.9*np.log10(distToBrighter[close]/prof.fwhm))

    ##  penalize S/N of sources outside FoV:
    outside = edgeDist<0
    signalToNoise[outside] *= np.exp(-(edgeDist[outside]/prof.fwhm)**2)


    if not smallifu:
        shoulderDrop = sources.shoulderDrop(r=psfRad)

    ## determine status of all sources in catalogue:
    ##  4 - global sky spectrum
    ##  3 - resolved, own spectrum
    ##  2 - resolved, inferred spectrum
    ##  1 - resolved, but next to brighter source
    ##  0 - unresolved
    ## -1 - excluded

    ## Determine confusion limit, update parameter if requested
    mcut = magPS[edgeDist>0].min()
    while True:
        mcut += 0.1
        if (magPS[edgeDist>0]<mcut).sum() > confDens*dataIma.size \
                /(prof.fwhm*prof.fwhm):
            mcut -= 0.1
            break
        if not (magPS[edgeDist>0]>mcut).any():
            logging.info("No stars below m_{0}={1:.1f} detected.".format(\
                    passband, mcut))
            break
        if signalToNoise[edgeDist>0][magPS[edgeDist>0]>mcut].max() < snrcut:
            logging.info("No stars below m_{0}={1:.1f} pass S/N-cut.".format(\
                    passband, mcut))
            break
    logging.info("Estimated confusion limit: m_{0}={1:.1f}".format(\
            passband, mcut))
    if updateRL: reslim = mcut
    if useunres: unreslim = reslim+unresbin

    status  = -np.ones((sources.nsrc,), dtype=np.int16)
    primary = np.zeros((sources.nsrc,), dtype=np.int32)

    logging.info("Results of source selection:")
    logging.info("       ID   mag   S/N  d_bright  d_edge  status")
    for i in xrange(sources.nsrc):
        
        if sources.ID[i] < 0:
            status[i]  = 4
            primary[i] = sources.ID[i]
        elif sources.mag[i] <= reslim:
            if signalToNoise[i] > snrcut:
                if distToBrighter[i] < binrad*prof.fwhm:
                    status[i]  = 1
                    primary[i] = brighterID[i]
                else:
                    status[i]  = 2
                    primary[i] = sources.ID[i]
            else:
                if edgeDist[i] < 0:
                    continue
                if distToBrighter[i] < 2*binrad*prof.fwhm:
                    status[i]  = 1
                    primary[i] = brighterID[i]
                else:
                    status[i]  = 2
                    primary[i] = sources.ID[i]

            logging.info(\
                "   {0:6d}  {1:5.2f}  {2:5.1f}  {3:5.2f}  {4:5.2f}  {5:6d}"\
                             .format(int(sources.ID[i]),
                                     float(sources.mag[i]),
                                     float(signalToNoise[i]),
                                     float(distToBrighter[i]),
                                     float(edgeDist[i]),
                                     int(status[i])))
            
        elif useunres:
            if sources.mag[i]<=unreslim and edgeDist[i]>-prof.fwhm:
                status[i]  = 0
                primary[i] = -2

    sources.primary = primary
    sources.status  = status


    ## create dummy data: every spectrum is empty
    sources.loadSpectra(data=np.zeros((sources.ncmp, ndisp), dtype=np.float32),
                        crval=crval, cdelt=cdelt)


    if not smallifu:
        logging.info("Searching for useful PSF stars.")
        ## searching photometry results for usefuls PSF stars
        ## must fulfil following criteria:
        ## (1) at least 3 magnitudes brighter than provided resolution limit
        ##      (parameter `RESLIM')
        ## (2) at a distance of of at least 1 pixel from the edge of the FoV 
        ## (3) isolation criterium (relative contribution from neighboring
        ##      sources smaller than value of `APERNOIS'-parameter)

        logging.info("IDs of stars that fullfil search criteria:")
        logging.info("       ID    mag  shoulderDrop")
        for i in xrange(sources.nsrc):
            
            if sources.status[i] == 2 \
                    and sources.mag[i] <= (reslim-3) \
                    and edgeDist[i] > prof.fwhm \
                    and shoulderDrop[i] < apernois:

                sources.status[i] = 3
                logging.info("   {0:6d}  {1:5.2f}  {2:14.2f}".format(\
                        int(sources.ID[i]),
                        float(sources.mag[i]),
                        float(shoulderDrop[i])))

        if len(sources.psfIndices) == 0:
            logging.error("No useful PSF sources found!")
            raise ConfigurationError, "No useful PSF sources found!"
        elif len(sources.psfIndices) < 4:
            logging.warning("Found only {0} useful PSF source(s)".format(\
                    len(sources.psfIndices)))

        ## carry out aperture spectrophotometry for PSF sources
        logging.info("Aperture spectrophotometry on PSF stars.")

        sources.loadIFUCoords(ids=sources.ID[sources.status==3])

        aperData = sources.toRec(ids=sources.ID[sources.status==3])
        aperData["x"] = np.squeeze(sources.xdata[0])
        aperData["y"] = np.squeeze(sources.ydata[0])
 
        make_aper = AperPhoto(data=dataHDU, centers=aperData, mask=maskHDU)
        
        aperResults = make_aper.aper_photo(rAper=aperRad, psf=psf,
                                           calcMoments=True, thresh=3.,
                                           localBG=True)
        
        if aperResults.size > 3:
            ## determine wavelength-dependent coordinate transformation
            logging.info("Determining wavelength-dependent coord. transf.")
            stars     = (aperResults["id"]>=0)
            ifuCoords = aperResults[["x","y"]][stars].view(np.float32)\
                .reshape(stars.sum(), 2, -1)

            sources.loadIFUCoords(data=np.rollaxis(ifuCoords, 1, 0),
                                  ids=aperResults["id"][stars])
            sources.calcTrans()
            if posFit == 1:   sources.ctFree[2:] = True
            elif posFit == 2: sources.ctFree[:] = True


            ##  determine FWHM, E and THETA
            logging.info("Approximating PSF from measured moments.")
            x2 = aperResults[["x2"]][stars].view(np.float32)\
                .reshape(stars.sum(), -1)
            y2 = aperResults[["y2"]][stars].view(np.float32)\
                .reshape(stars.sum(), -1)
            xy = aperResults[["xy"]][stars].view(np.float32)\
                .reshape(stars.sum(), -1)

            psf.delValues()
            psf.resizeStructure(ndisp=x2.shape[1])
            psf.makeValues()

            ## get A, B:
            A = np.sqrt((x2+y2)/2.+np.sqrt(((x2-y2)/2.)**2+xy**2))
            B = np.sqrt((x2+y2)/2.-np.sqrt(((x2-y2)/2.)**2+xy**2))
            
            ## FWHM
            fwhm = np.sqrt(8.*np.log(2))*np.mean(A, axis=0)
##             psf.values[psf.parameters.index("fwhm")] = fwhm
            logging.info("   Mean FWHM: {0:.2f}pix.".format(fwhm.mean()))

            ## Ellipticity
            e    = 1. - np.mean(np.where(B!=0, A/B, 1.), axis=0)
            psf.values[psf.parameters.index("e")] = e
            logging.info("   Mean ellipticity: {0:.2f}".format(e.mean()))
        
            ## Position angle
            theta  = np.where(x2!=y2, 0.5*np.arctan(2*xy/(x2-y2)), 0.)
            theta -= (-1)**(theta<0)*np.pi/4.
            theta += (-1)**(xy>0)*np.pi/4.
            psf.values[psf.parameters.index("theta")] = theta.mean(axis=0)
            logging.info(\
                "   Mean position angle: {0:.2f}rad.".format(theta.mean()))

            ##  copy spectra to sources instance
            for i in xrange(aperResults.size):
                j = np.where(sources.ID==aperResults["id"][i])[0][0]
                sources.data[sources.specrow[j]] = aperResults["spectrum"][i]
            

    ## write results to file
    logging.info("Saving source selection in PRM-file:")
    logging.info("   {0}.prm.fits".format(prefix))

    primaryHDU = pyfits.PrimaryHDU()
    primaryHDU.header.update(\
        "hierarch PAMPELMUSE INITFIT UTC",
        datetime.datetime.utcnow().strftime("%y-%m-%d %H:%M:%S"))
    primaryHDU.header.update("hierarch PAMPELMUSE IFU", str(ifu))
    primaryHDU.header.update("hierarch PAMPELMUSE INITFIT VERSION", __version__)
    primaryHDU.header.update("hierarch PAMPELMUSE INITFIT PASSBAND", passband)
    primaryHDU.header.update("hierarch PAMPELMUSE INITFIT RESLIM", reslim)
    primaryHDU.header.update("hierarch PAMPELMUSE INITFIT UPDATERL", updateRL)
    primaryHDU.header.update("hierarch PAMPELMUSE INITFIT CONFDENS", confDens)
    primaryHDU.header.update("hierarch PAMPELMUSE INITFIT SNRCUT", snrcut)
    primaryHDU.header.update("hierarch PAMPELMUSE INITFIT BINRAD", binrad)
    primaryHDU.header.update("hierarch PAMPELMUSE INITFIT USEUNRES", useunres)
    if useunres:
        primaryHDU.header.update("hierarch PAMPELMUSE INITFIT UNRESLIM",
                                 unreslim)
    primaryHDU.header.update("hierarch PAMPELMUSE INITFIT GLOBSKY", globsky)

    sourceHDU  = sources.makeSourceHDU()
    spectraHDU = sources.makeSpecHDU(makeSigma=False)
    posparsHDU = sources.makeTransHDU(includePolyFit=False)
    psfparsHDU = psf.makePSFHDU(includePolyFit=False)

    HDUList = pyfits.HDUList([primaryHDU, sourceHDU, spectraHDU, posparsHDU,
                              psfparsHDU])

    if os.path.exists("{0}.prm.fits".format(prefix)):
        logging.warning("Overwriting existing PRM-file.")
        os.remove("{0}.prm.fits".format(prefix))
    HDUList.writeto("{0}.prm.fits".format(prefix))
