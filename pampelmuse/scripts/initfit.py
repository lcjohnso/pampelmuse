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
2013/05/03, v0.8.0.4, exported the checking of the data for consistency to a
                      new class pampelmuse.core.cube.Cube
2013/05/13, v0.9.0.0, inplemented the new GUI to identify the sources. Also
                      modified the usage of the reference coordinates: they
                      are included in the 'Sources' catalog as provided now.
2013/05/14, v0.9.1.0, bugfix: reference coordinates in PSF instances were not
                              updated when recovering original ones.
2013/05/29, v0.10.0.0, major updates to the file to match the new way that the
                       configuration of PampelMuse is carried through the analysis.
"""
__author__ = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__version__ = "0.10.0.0"
import datetime, os, sys, logging
import numpy as np, pyfits
from scipy import interpolate, sparse
from PyQt4 import QtGui

## required PampelMuse routines
from pampelmuse.graphics.pampelmuse_gui import PampelMuseGui
from pampelmuse.core.aperture import AperPhoto
from pampelmuse.core.new_profiles import Gaussian, Moffat
from pampelmuse.core.sources import Sources
from pampelmuse.core.psf import PsfAttributes

DATAPATH = os.path.expandvars("$pampelmuse_path/data/")


## class defines the exception that is raised when the parameters
## in the PampelMuse-configuration are such that they do not allow
## for a proper analysis of the provided input
class ConfigurationError(EnvironmentError):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)


def initfit(pampelmuse):

    print ""
    print 21* " "+"***  InitFit, version {0}  ***".format(__version__)+21*" "
    print ""


    ## get results from crowded-field photometry, initialize sources
    logging.info("Reading photometry results from file '{0}'...".format(pampelmuse.initfit.sourcelist))

    cfpResults = np.genfromtxt(pampelmuse.initfit.sourcelist, usecols=pampelmuse.cfpcols, 
                               names="id,x,y,mag", dtype="<i8,<f4,<f4,<f4", comments="#")

    if pampelmuse.globsky:
        ids = np.r_[cfpResults['id'], -1]
        x   = np.r_[cfpResults['x'], 0]
        y   = np.r_[cfpResults['y'], 0]
        mag = np.r_[cfpResults["mag"], 0]
    else:
        ids = cfpResults['id']
        x   = cfpResults['x']
        y   = cfpResults['y']
        mag = cfpResults["mag"]


    ## Initialize sources-class
    logging.info("Initializing sources in the field...")
    sources     = Sources(ids=ids, instrument=pampelmuse.ifs_data.instrument)
    sources.x   = x
    sources.y   = y
    sources.mag = mag

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
    
    psf_attributes = PsfAttributes()
    psf_attributes.makeNewPSF(profile=pampelmuse.psf.profile)
    
    if pampelmuse.psf.profile == "moffat": PSF = Moffat
    elif pampelmuse.psf.profile == "gauss": PSF = Gaussian

    i=0
    while hasattr(pampelmuse.psf, "parname{0}".format(i)):
        parname_i = getattr(pampelmuse.psf, "parname{0}".format(i))
        parvalue_i = getattr(pampelmuse.psf, "parvalue{0}".format(i))
        
        psf_attributes.addParameter(name=parname_i, value=parvalue_i,
                                    freepar=getattr(pampelmuse.psf, "freepar{0}".format(i)))
        setattr(PSF, "set_{0}".format(parname_i),  parvalue_i)
        
        i += 1

    psf_attributes.makeValues()
    psf_attributes.info()

    
    ## collapse data-cube to image in the provided photometric filter
    logging.info("Creating mock {0}-band image from datacube...".format(pampelmuse.passband))
    filtercurve = np.genfromtxt(os.path.join(DATAPATH, pampelmuse.passband), usecols=(0,1),
                                dtype=[("wl", np.float32), ("tp", np.float32)])

    f = interpolate.UnivariateSpline(filtercurve["wl"], filtercurve["tp"], k=3, s=0)
    fMin = filtercurve["wl"].min()
    fMax = filtercurve["wl"].max()
    fWht = 0.

    ## make loop to avoid extensive use of memory for large data-cubes
    dataIma  = np.zeros(pampelmuse.ifs_data.instrument.transform.shape[0], dtype=np.float32)
    sigmaIma = np.zeros(dataIma.shape, dtype=np.float32)
    for i in xrange(pampelmuse.start, pampelmuse.stop):
        wave_i = pampelmuse.ifs_data.crval+i*pampelmuse.ifs_data.cdelt
        if pampelmuse.ifs_data.ctype == "nm": wave_i *= 10
        if wave_i > fMin and wave_i < fMax:
            wht_i = f(wave_i)
            mask_i = pampelmuse.ifs_data.getMaskForLayer(i)
            data_i, sigma_i = pampelmuse.ifs_data.getLayer(i)

            dataIma[mask_i==0]  += data_i*wht_i # pampelmuse.ifs_data.instrument.get_science(data_i*wht_i)
            sigmaIma[mask_i==0] += sigma_i*wht_i # pampelmuse.ifs_data.instrument.get_science(sigma_i*wht_i)
            fWht += wht_i

    dataIma  /= fWht
    sigmaIma /= fWht



    ## create pseudo observation from photometry results
    ## PSF is same as in IFU, with FWHM scaled by `pixScale'-Parameter
    ## The zeropoint is such that a source with magnitude `reslim' will
    ## have a flux of 1000 counts and a peak-S/N of roughly 10
    logging.info("Creating mock image from photometry sources...")

    PSF.set_size((int(sources.y.max()+1), int(sources.x.max()+1)))

    MZERO     = pampelmuse.reslim + 7.5

    identIma  = np.zeros(PSF.size, dtype='float32')
    for i in xrange(sources.nsrc):
        if sources.ID[i] <= 0: continue 

        psf = PSF(Xc=sources.x[i], Yc=sources.y[i], mag=sources.mag[i], src_id=sources.ID[i], maxrad=10.)

        identIma[np.asarray(psf.y[psf.used], dtype=np.int16), np.asarray(psf.x[psf.used], dtype=np.int16)] += psf.f*10**(0.4*MZERO)

    identIma += np.random.normal(loc=0., scale=10, size=PSF.size)

    ## start source identification GUI to identify bright(est) stars
    logging.info("Preparing GUI for source identification...")

    app  = QtGui.QApplication(sys.argv)
    identifier = PampelMuseGui(sources=sources, ifsData=pampelmuse.ifs_data, refImage=identIma, magCut=pampelmuse.reslim-1.)

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


    ## determine coordinate transformation using identified sources
    sources.loadIFUCoords(data=np.asarray(identified.values()).T[:2], ids=identified.keys())
    sources.calcTrans()
    if pampelmuse.posfit == 1: sources.ctFree[2:] = True
    elif pampelmuse.posfit == 2: sources.ctFree[:] = True

    ## add coordinate transformation to PSF
    i=0
    for prm in sources.parameters:
        getattr(PSF, "set_{0}".format(prm))(sources.ctValues[i,0])
        i+=1

    ## Source selection process 
    logging.info("Selecting sources to be used in PSF fitting...")

    ## get IFU coordinates for all (point) sources
    sources.loadIFUCoords(index=0, ids=sources.ID[sources.ID>=0])

    ## Position in Fov/distance to edge of FoV
    edgeDist = sources.edgeDist()

    ## Photometric calibration:
    magPS = sources.mag[sources.loadedIFUCoords]

    srcFlux = (10**(-0.4*magPS[edgeDist>0])).sum()
    magZ    = -2.5*np.log10(srcFlux/dataIma.sum())
    logging.info("Estimated photometric zeropoint ({0}) is {1:.2f}".format(pampelmuse.passband, float(magZ)))
    pampelmuse.initfit.photzero = magZ

    ## distance to next brighter source
    distToBrighter, brighterID = sources.distToBrighter()
    if (distToBrighter < 1e-5).any:
        for i in (distToBrighter < 1e-5).nonzero()[0]:
            logging.warning("Source #{0} might be double.".format(sources.ID[i]))

    ## collect FWHM of sources in single array
    fwhm = np.asarray([psf.fwhm for psf in PSF])

    ## signal-to-noise
    signalToNoise = sources.signalToNoise(class_psf=PSF, magZ=magZ, sigma=sigmaIma, r=pampelmuse.psfrad)

    ##  modify S/N of sources close to brighter neighbor:
    close = distToBrighter<fwhm
    signalToNoise[close] *= (1.+0.9*np.log10(distToBrighter[close]/fwhm[close]))

    ##  penalize S/N of sources outside FoV:
    outside = edgeDist<0
    signalToNoise[outside] *= np.exp(-(edgeDist[outside]/fwhm[outside])**2)


    if not pampelmuse.smallifu:
        shoulderDrop = sources.shoulderDrop(r=pampelmuse.psfrad)

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
#         if (magPS[edgeDist>0]<mcut).sum() > pampelmuse.confdens*dataIma.size/(fwhm.mean()*fwhm.mean()):
#             mcut -= 0.1
#             break
        if not (magPS[edgeDist>0]>mcut).any():
            logging.info("No stars below m_{0}={1:.1f} detected.".format(pampelmuse.passband, mcut))
            break
        if signalToNoise[edgeDist>0][magPS[edgeDist>0]>mcut].max() < pampelmuse.snrcut:
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
            if signalToNoise[i] > pampelmuse.snrcut:
                if distToBrighter[i] < pampelmuse.binrad*fwhm[i]:
                    status[i]  = 1
                    primary[i] = brighterID[i]
                else:
                    status[i]  = 2
                    primary[i] = sources.ID[i]
            else:
                if edgeDist[i] < 0:
                    continue
#                 if distToBrighter[i] < 2*pampelmuse.binrad*fwhm[i]:
#                     status[i]  = 1
#                     primary[i] = brighterID[i]
                else:
                    status[i]  = 0
                    primary[i] = -2 ## sources.ID[i]

            logging.info("   {0:6d}  {1:5.2f}  {2:5.1f}  {3:5.2f}  {4:5.2f}  {5:6d}".format(\
                int(sources.ID[i]), float(sources.mag[i]), float(signalToNoise[i]), float(distToBrighter[i]),
                float(edgeDist[i]), int(status[i])))
            
        elif pampelmuse.useunres:
            if sources.mag[i]<=unreslim and edgeDist[i]>-fwhm[i]:
                status[i]  = 0
                primary[i] = -2

    sources.primary = primary
    sources.status  = status


    ## create dummy data: every spectrum is empty
    sources.loadSpectra(data=np.zeros((sources.ncmp, pampelmuse.stop-pampelmuse.start+1), dtype=np.float32),
                        crval=pampelmuse.ifs_data.crval+pampelmuse.start*pampelmuse.ifs_data.cdelt, 
                        cdelt=pampelmuse.ifs_data.cdelt, ctype=pampelmuse.ifs_data.ctype, cunit=pampelmuse.ifs_data.cunit)


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
        logging.info("       ID    mag  shoulderDrop")
        for i in xrange(sources.nsrc):
            
            if sources.status[i] == 2 \
                    and sources.mag[i] <= (reslim-3) \
                    and edgeDist[i] > fwhm[i] \
                    and shoulderDrop[i] < pampelmuse.apernois:

                sources.status[i] = 3
                logging.info("   {0:6d}  {1:5.2f}  {2:14.2f}".format(int(sources.ID[i]), float(sources.mag[i]), float(shoulderDrop[i])))

        if len(sources.psfIndices) == 0:
            logging.error("No useful PSF sources found!")
            raise ConfigurationError, "No useful PSF sources found!"
        elif len(sources.psfIndices) < 4:
            logging.warning("Found only {0} useful PSF source(s)".format(len(sources.psfIndices)))

        ## carry out aperture spectrophotometry for PSF sources
        logging.info("Performing aperture spectrophotometry on PSF stars...")

        sources.loadIFUCoords(ids=sources.ID[sources.status==3])

        aperData = sources.toRec(ids=sources.ID[sources.status==3])
        aperData["x"] = np.squeeze(sources.xdata[0])
        aperData["y"] = np.squeeze(sources.ydata[0])
 
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

            psf_attributes.delValues()
            psf_attributes.resizeStructure(ndisp=x2.shape[1])
            psf_attributes.makeValues()

            ## get A, B:
            A = np.sqrt((x2+y2)/2.+np.sqrt(((x2-y2)/2.)**2+xy**2))
            B = np.sqrt((x2+y2)/2.-np.sqrt(((x2-y2)/2.)**2+xy**2))
            
            ## FWHM
            fwhm = np.sqrt(8.*np.log(2))*np.mean(A, axis=0)
##             psf_attributes.values[psf_attributes.parameters.index("fwhm")] = fwhm
            logging.info("Mean FWHM: {0:.2f}pix.".format(fwhm.mean()))

            ## Ellipticity
            e    = 1. - np.mean(np.where(B!=0, A/B, 1.), axis=0)
            psf_attributes.values[psf_attributes.parameters.index("e")] = e
            logging.info("Mean ellipticity: {0:.2f}".format(e.mean()))
        
            ## Position angle
            theta  = np.where(x2!=y2, 0.5*np.arctan(2*xy/(x2-y2)), 0.)
            theta -= (-1)**(theta<0)*np.pi/4.
            theta += (-1)**(xy>0)*np.pi/4.
            psf_attributes.values[psf_attributes.parameters.index("theta")] = theta.mean(axis=0)
            logging.info("Mean position angle: {0:.2f}rad.".format(theta.mean()))

            ##  copy spectra to sources instance
            for i in xrange(aperResults.size):
                j = np.where(sources.ID==aperResults["ID"][i])[0][0]
                sources.data[sources.specrow[j]] = aperResults["spectrum"][i]
            

    ## write results to file
    logging.info("Saving source selection in file {0}.".format(pampelmuse.filename))

    primaryHDU = pampelmuse.makeHeader()
    sourceHDU  = sources.makeSourceHDU()
    spectraHDU = sources.makeSpecHDU(makeSigma=False)
    posparsHDU = sources.makeTransHDU(includePolyFit=False)
    psfparsHDU = psf_attributes.makePSFHDU(includePolyFit=False)

    HDUList = pyfits.HDUList([primaryHDU, sourceHDU, spectraHDU, posparsHDU, psfparsHDU])

    if os.path.exists(pampelmuse.filename):
        logging.warning("Overwriting existing PRM-file.")
        os.remove(pampelmuse.filename)
    HDUList.writeto(pampelmuse.filename)
