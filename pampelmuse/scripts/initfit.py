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
2013/03/26, v0.8.0.0, major modifications done to the code in order to
                      implement new way of calculating the PSF profiles.
2013/03/27, v0.8.0.1, some minor modifications; also implemented option to
                      restrict analysis to a certain wavelength range within
                      a cube.
2013/04/05, v0.8.0.2, updated code to make it compatible with updates in
                      class pampelmuse.core.sources.Sources.
2013/04/19, v0.8.0.3, now reading more wavelength information (keywords CTYPE?,
                      CUNIT?) from the header of the data cube. They are also
                      passed on to method 'Sources.loadSpectra'.
"""
__author__ = "Sebastian Kamann (skamann@aip.de)"
__version__ = "0.8.0.3"
import datetime, os, sys, logging
import numpy as np, pyfits
from scipy import interpolate, sparse
from PyQt4 import QtGui

## required PampelMuse routines
from pampelmuse.graphics.ident_sources import IdentSources
from pampelmuse.core.aperture import AperPhoto
from pampelmuse.core.new_profiles import Gaussian,Moffat
from pampelmuse.core.sources import Sources
from pampelmuse.core.psf import PsfAttributes


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
    layerrng = arguments.config["LAYERRNG"]         ##
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


    ## Initialize sources-class
    sources     = Sources(ids=ids, instrument=ifu)
    sources.x   = x
    sources.y   = y
    sources.mag = mag


    ## initialize PSF
    if psfType == "moffat":
        PSF = Moffat
        PSF.set_beta(psfPars[0])
        PSF.set_fwhm(psfPars[1])
    elif psfType == "gauss":
        PSF = Gaussian
        PSF.set_fwhm(psfPars[0])
    PSF.set_e(psfPars[-2])
    PSF.set_theta(psfPars[-1])

##     elif psfType == "lorentz": psfArgs = ["Ltheta", "e", "fwhm"]
##     elif psfType == "gaumoff": psfArgs = ["Gfwhm", "beta", "Mfwhm", "Mfrac",
##                                           "e", "fwhm"]

    psf_attributes = PsfAttributes() ## PSF used in analysis
    psf_attributes.makeNewPSF(profile=psfType)

    ## check which PSF parameters should be fitted and add parameters
    n = len(psf_attributes.parameters)
    for arg in psf_attributes.parameters[::-1]:
        if psfFit >= 2**n:
            psfFit -= 2**n
            psf_attributes.addParameter(\
                name=arg, value=psfPars[n-1], freepar=True)
        else:
            psf_attributes.addParameter(\
                name=arg, value=psfPars[n-1], freepar=False)
        n -= 1

    psf_attributes.makeValues()
    psf_attributes.info()


    ## open FITS-files:
    logging.info("Reading IFS data...")
        
    ## data
    dataFile = pyfits.open("{0}.dat.fits".format(prefix))

    ## check if calibration HDU for ARGUS exists
    if sources.instrument.name == "argus":
        try:
            setup_index = dataFile.index_of("FIBER_SETUP")
            sources.instrument.read_fiber_setup(hdu=dataFile[setup_index])
        except KeyError:
            logging.warning("No HDU with fibre setup found for ARGUS data.")
    
    ## open data HDU
    dataHDU  = dataFile[0]
    ndim     = dataHDU.header["NAXIS"]
    if sources.instrument.transform == None and ndim!=3:
        logging.error(\
            "Data must be 3dim. unless mapping from spaxels to x&y specified.")
        assert ndim==3, "Expected 3dim. datacube, got {0}dim".format(ndim)

    ## Reading information about wavelength coverage
    if ndim == 3:
        if sources.instrument.shape == None: ## set spatial shape if needed
            xdim = dataHDU.header["NAXIS1"]
            ydim = dataHDU.header["NAXIS2"]

            sources.instrument.shape = (ydim, xdim)

            x = np.mod(np.arange(xdim*ydim, dtype=np.int16), xdim)
            y = np.arange(xdim*ydim, dtype=np.int16)//xdim
            sources.instrument.transform = np.vstack((y,x)).T

        spectral_axis = 3
    else:
        spectral_axis = sources.instrument.spectral_axis

    cdelt = dataHDU.header['CDELT{0}'.format(spectral_axis)]
    crval = dataHDU.header['CRVAL{0}'.format(spectral_axis)]
    try:
        ctype = dataHDU.header["CTYPE{0}".format(spectral_axis)]
    except KeyError:
        ctype = "AWAV"
    try:
        cunit = dataHDU.header["CUNIT{0}".format(spectral_axis)]
    except KeyError:
        cunit = "Angstrom"
    ndisp = dataHDU.header['NAXIS{0}'.format(spectral_axis)]

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

    ## check if whole cube should be analyzed
    if layerrng[0] >= 0: start = layerrng[0]
    else: start = ndisp + layerrng[0]
    if layerrng[1] >= 0: stop = layerrng[1]
    else: stop = ndisp + layerrng[1]
    

    ## collapse data-cube to image in the provided photometric filter
    logging.info("Creating mock {0}-band image from datacube.".format(passband))
    filtercurve = np.genfromtxt(os.path.join(DATAPATH, passband), usecols=(0,1),
                                dtype=[("wl", np.float32), ("tp", np.float32)])

    f = interpolate.UnivariateSpline(\
        filtercurve["wl"], filtercurve["tp"], k=3, s=0)
    fMin = filtercurve["wl"].min()
    fMax = filtercurve["wl"].max()
    fWht = 0.

    ## make loop to avoid extensive use of memory for large data-cubes
    dataIma  = np.zeros(sources.instrument.nspaxel, dtype=np.float32)
    sigmaIma = np.zeros(dataIma.shape, dtype=np.float32)
    for i in xrange(start, stop):
        wave_i = crval+i*cdelt
        if ctype == "nm": wave_i *= 10
        if wave_i > fMin and wave_i < fMax:
            wht_i = f(wave_i)
            if ndim==3 :
                dataIma += dataHDU.section[i].flatten()*wht_i
                sigmaIma += sigmaHDU.section[i].flatten()*wht_i
            elif sources.instrument.spectral_axis==2:
                dataIma  += sources.instrument.get_science(\
                    dataHDU.section[i]*wht_i)
                sigmaIma += sources.instrument.get_science(\
                        sigmaHDU.section[i]*wht_i)
            else:
                dataIma  += sources.instrument.get_science(\
                    dataHDU.section[:,i]*wht_i)
                sigmaIma += sources.instrument.get_science(\
                    sigmaHDU.section[:,i]*wht_i)
            fWht += wht_i

    dataIma  /= fWht
    sigmaIma /= fWht


    ## create pseudo observation from photometry results
    ## PSF is same as in IFU, with FWHM scaled by `pixScale'-Parameter
    ## The zeropoint is such that a source with magnitude `reslim' will
    ## have a flux of 1000 counts and a peak-S/N of roughly 10
    logging.info("Creating mock image from photometry sources.")

    PSF.set_size(\
        (int((sources.y.max()/pixScale)+1), int((sources.x.max()/pixScale)+1)))
    PSF.set_A(1./pixScale)
    MZERO     = reslim + 7.5

    identIma  = np.zeros(PSF.size, dtype='float32')
    for i in xrange(sources.nsrc):
        if sources.ID[i] < 0: continue 

        psf = PSF(Xc=sources.x[i], Yc=sources.y[i], mag=sources.mag[i],
                  src_id=sources.ID[i], maxrad=10.)

        identIma[np.asarray(psf.y[psf.used], dtype=np.int16),
                 np.asarray(psf.x[psf.used], dtype=np.int16)] \
                 += psf.f*10**(0.4*MZERO)

    identIma += np.random.normal(loc=0., scale=10, size=PSF.size)

    ## start source identification GUI to identify bright(est) stars
    logging.info("Preparing GUI for source identification"),

    ##  temporary solution: create rec-array from Sources-instance; usage of
    ##                      instance in IdentSources will be implemented later
    pickData = sources.toRec(\
        ids=sources.ID[(sources.mag<reslim)&(sources.ID>=0)])

    pickData["x"] /= pixScale
    pickData["y"] /= pixScale

    app  = QtGui.QApplication(sys.argv)
    identifier = IdentSources(\
        grabIma=sources.instrument.make_grid(dataIma, fill_value=0.,),
        pickIma=identIma, pickData=pickData)

    identifier.show()
    app.exec_()

    identified = identifier.getResults()


##     ## adjust PSF FWHM (only if >=4 sources have a valid FWHM measurement)
##     valid = (identified["fwhm_x"]>0)&(identified["fwhm_y"]>0)
##     if valid.sum() < 4:
##         logging.warn("Skipping FWHM measurement - too few sources ({0})"\
##                          .format(valid.sum()))
## ##         fwhm = psf.values[psf.parameters.index("fwhm"),:].mean(axis=-1)
##     else:
##         prof.fwhm = 0.5*(identified['fwhm_x'][valid].mean() \
##                             + identified['fwhm_y'][valid].mean())
##         logging.info("Estimated Seeing FWHM: {0:.1f}pixel".format(prof.fwhm))
## ##         psf.values[psf.parameters.index("fwhm"),:] =  fwhm
##         ## NOTE: will only write FWHM to values if FWHM is fitted
##         ##       CHANGED: Parameter not updated, only class-property `fwhm`

    ## determine coordinate transformation using identified sources
    sources.loadIFUCoords(\
        data=identified[["x", "y"]].view(np.float64).reshape(-1,2).T,
        ids=identified["ID"])
    sources.calcTrans()
    if posFit == 1: sources.ctFree[2:] = True
    elif posFit == 2: sources.ctFree[:] = True

    ## add coordinate transformation to PSF
    i=0
    for prm in sources.parameters:
        getattr(PSF, "set_{0}".format(prm))(sources.ctValues[i,0])
        i+=1

    ## adapt PSF to IFU array
    if sources.instrument.transform != None:
        PSF.set_transform(sources.instrument.transform)
    else:
        PSF.set_size(sources.instrument.shape)

    ## Source selection process 
    logging.info("Selecting sources to be used in PSF fitting.")

    ## get IFU coordinates for all (point) sources
    sources.loadIFUCoords(index=0, ids=sources.ID[sources.ID>=0])

    ## Position in Fov/distance to edge of FoV
    edgeDist = sources.edgeDist()

    ## Photometric calibration:
    magPS = sources.mag[sources.loadedIFUCoords]

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

    ## collect FWHM of sources in single array
    fwhm = np.asarray([psf.fwhm for psf in PSF])

    ## signal-to-noise
    signalToNoise = sources.signalToNoise(class_psf=PSF, magZ=magZ,
                                          sigma=sigmaIma, r=psfRad)
    ##  modify S/N of sources close to brighter neighbor:
    close = distToBrighter<fwhm
    signalToNoise[close] *= (1.+0.9*np.log10(distToBrighter[close]/fwhm[close]))

    ##  penalize S/N of sources outside FoV:
    outside = edgeDist<0
    signalToNoise[outside] *= np.exp(-(edgeDist[outside]/fwhm[outside])**2)


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
                /(fwhm.mean()*fwhm.mean()):
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
                if distToBrighter[i] < binrad*fwhm[i]:
                    status[i]  = 1
                    primary[i] = brighterID[i]
                else:
                    status[i]  = 2
                    primary[i] = sources.ID[i]
            else:
                if edgeDist[i] < 0:
                    continue
                if distToBrighter[i] < 2*binrad*fwhm[i]:
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
            if sources.mag[i]<=unreslim and edgeDist[i]>-fwhm[i]:
                status[i]  = 0
                primary[i] = -2

    sources.primary = primary
    sources.status  = status


    ## create dummy data: every spectrum is empty
    sources.loadSpectra(\
        data=np.zeros((sources.ncmp, stop-start+1), dtype=np.float32),
        crval=crval+start*cdelt, cdelt=cdelt, ctype=ctype, cunit=cunit)


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
                    and edgeDist[i] > fwhm[i] \
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
 
        ## !!! NEEDS TO  BE UPDATED !!!
        ## include start/stop-values
        make_aper = AperPhoto(data=dataHDU, centers=aperData, mask=maskHDU)
        
        ## !!! NEEDS TO  BE UPDATED !!!
        aperResults = make_aper.aper_photo(rAper=aperRad, psf=psf,
                                           calcMoments=True, thresh=3.,
                                           localBG=True)
        
        if aperResults.size > 3:
            ## determine wavelength-dependent coordinate transformation
            logging.info("Determining wavelength-dependent coord. transf.")
            stars     = (aperResults["ID"]>=0)
            ifuCoords = aperResults[["x","y"]][stars].view(np.float32)\
                .reshape(stars.sum(), 2, -1)

            sources.loadIFUCoords(data=np.rollaxis(ifuCoords, 1, 0),
                                  ids=aperResults["ID"][stars])
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

            psf_attributes.delValues()
            psf_attributes.resizeStructure(ndisp=x2.shape[1])
            psf_attributes.makeValues()

            ## get A, B:
            A = np.sqrt((x2+y2)/2.+np.sqrt(((x2-y2)/2.)**2+xy**2))
            B = np.sqrt((x2+y2)/2.-np.sqrt(((x2-y2)/2.)**2+xy**2))
            
            ## FWHM
            fwhm = np.sqrt(8.*np.log(2))*np.mean(A, axis=0)
##             psf_attributes.values[psf_attributes.parameters.index("fwhm")] = fwhm
            logging.info("   Mean FWHM: {0:.2f}pix.".format(fwhm.mean()))

            ## Ellipticity
            e    = 1. - np.mean(np.where(B!=0, A/B, 1.), axis=0)
            psf_attributes.values[psf_attributes.parameters.index("e")] = e
            logging.info("   Mean ellipticity: {0:.2f}".format(e.mean()))
        
            ## Position angle
            theta  = np.where(x2!=y2, 0.5*np.arctan(2*xy/(x2-y2)), 0.)
            theta -= (-1)**(theta<0)*np.pi/4.
            theta += (-1)**(xy>0)*np.pi/4.
            psf_attributes.values[psf_attributes.parameters.index("theta")] \
                = theta.mean(axis=0)
            logging.info(\
                "   Mean position angle: {0:.2f}rad.".format(theta.mean()))

            ##  copy spectra to sources instance
            for i in xrange(aperResults.size):
                j = np.where(sources.ID==aperResults["ID"][i])[0][0]
                sources.data[sources.specrow[j]] = aperResults["spectrum"][i]
            

    ## write results to file
    logging.info("Saving source selection in PRM-file:")
    logging.info("   {0}.prm.fits".format(prefix))

    primaryHDU = pyfits.PrimaryHDU()
    primaryHDU.header.update(\
        "hierarch PAMPELMUSE INITFIT UTC",
        datetime.datetime.utcnow().strftime("%y-%m-%d %H:%M:%S"))
    primaryHDU.header.update("hierarch PAMPELMUSE IFU", str(ifu))
    primaryHDU.header.update("hierarch PAMPELMUSE START", start)
    primaryHDU.header.update("hierarch PAMPELMUSE STOP", stop)
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
    psfparsHDU = psf_attributes.makePSFHDU(includePolyFit=False)

    HDUList = pyfits.HDUList([primaryHDU, sourceHDU, spectraHDU, posparsHDU,
                              psfparsHDU])

    if os.path.exists("{0}.prm.fits".format(prefix)):
        logging.warning("Overwriting existing PRM-file.")
        os.remove("{0}.prm.fits".format(prefix))
    HDUList.writeto("{0}.prm.fits".format(prefix))
