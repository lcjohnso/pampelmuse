"""
INITFIT
=======
The routine is used as a wrapper for the 'indent_sources.py' and the
'create_prm.py' routine. It is called automatically if PampelMuse is
started using the `INITFIT`-option.
The purpose of those routines is to create the input data that is
required by the fitting routine of PampelMuse (option `CUBEFIT`).

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
"""
__author__ = "Sebastian Kamann (skamann@aip.de)"
__version__ = "0.2.0.0"
import os, sys, logging
import numpy as np, pyfits
from scipy import interpolate,sparse
from PyQt4 import QtGui

## required PampelMuse routines
from pampelmuse.graphics.ident_sources import IdentSources
from pampelmuse.core.aperture import AperPhoto
from pampelmuse.functions.trans_coords import trans_coords
from pampelmuse.core.select_sources import SelectSources
from pampelmuse.core.store_param import StoreParam 
from pampelmuse.core.my_profiles import Gaussian, Moffat


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
  
  
    ## List of required definitions: #################
    prefix   = arguments.prefix                     ##
    cfpfile  = arguments.sourcelist                 ##
    cfpcols  = arguments.config["CFPCOLS"]          ##
    passband = arguments.config["PASSBAND"]         ##
    smallifu = arguments.config["SMALLIFU"]         ##
##     aperrad  = arguments.config["APERRAD"]       ## deprecated, ->fitrad
    apernois = arguments.config["APERNOIS"]         ##
    reslim   = arguments.config["RESLIM"]           ##
    snrcut   = arguments.config["SNRCUT"]           ##
    binrad   = arguments.config["BINRAD"]           ##
    useunres = arguments.config["USEUNRES"]         ##
    unreslim = arguments.config["UNRESLIM"]         ##
    ewidth   = arguments.config["EWIDTH"]           ##
    globsky  = arguments.config["GLOBSKY"]          ##
    fitrad   = arguments.config["FITRAD"]           ##
    psfType  = arguments.config["PSFTYPE"]          ##
    psfPars  = arguments.config["PSFPARS"]          ##
    psfFit   = arguments.config["PSFFIT"]           ##
    posFit   = arguments.config["POSFIT"]           ##
    ##################################################
    DATAPATH = os.path.expandvars("$pampelmuse_path/data/")


    logging.info("Reading photometry results from file `{0}'".format(cfpfile))
    sys.stdout.flush()
    ## get results from crowded-field photometry
    cfpResults = np.copy(np.genfromtxt(cfpfile, 
                                       usecols=cfpcols,
                                       dtype=[('id','i8'), ('x','f8'),
                                              ('y','f8'), ('mag','f8')
                                              ],
                                       )
                         )
    cfpResults['x'] -= cfpResults['x'].min()
    cfpResults['y'] -= cfpResults['y'].min()

    if globsky:
        cfpResults.resize(cfpResults.size+1)
        cfpResults['id'][-1] = -1

    if useunres:
        resSrc = (cfpResults['mag']<unreslim)&(cfpResults['id']>=0)
    else:
        resSrc = (cfpResults['mag']<reslim)&(cfpResults['id']>=0)


    ## create pseudo observation from photometry results
    ## PSF is a Gaussian with  a FWHM of 4pix with PSFRAD=15.0pix
    ## The zeropoint is such that a source with magnitude `reslim' will
    ## have a flux of 1000 counts and a peak-S/N of roughly 10
    logging.info("Using photometry to create pseudo-observation of FoV")
    PSF = Gaussian(fwhm=4.)

    size = (int(cfpResults['y'].max()+1), int(cfpResults['x'].max()+1))

    MZERO = reslim + 7.5

    ident_ima = np.zeros(size, dtype='float32')
    for i in xrange(resSrc.sum()):
        ident_ima += PSF(size=size,
                         x_c=cfpResults['x'][resSrc][i],
                         y_c=cfpResults['y'][resSrc][i],
                         mag=(cfpResults['mag'][resSrc][i]-MZERO),
                         maxrad=15., return_sparse=True,
                         as_sparse_matrix=True).todense()

    ident_ima += np.random.normal(loc=0., scale=10, size=size)


    ## open FITS-files:
    logging.info("Reading IFS data: ")
    dat_data, dat_head = pyfits.getdata("{0}.dat.fits".format(prefix),
                                        header=True)
    wlStep   = dat_head['CDELT3']
    wlStart  = dat_head['CRVAL3']
    wlNsteps = dat_head['NAXIS3']

    logging.info("   {0}.dat.fits - OK".format(prefix))

    err_data           = pyfits.getdata("{0}.err.fits".format(prefix))
    err_data_ima       = np.mean(err_data, axis=0)
    logging.info("   {0}.err.fits - OK".format(prefix))

    try:
        msk_cube = pyfits.open("{0}.msk.fits".format(prefix))
        msk_data = np.asarray(msk_cube[0].data, dtype='bool')
        msk_cube.close()
        logging.info("   {0}.msk.fits - OK".format(prefix))
    except IOError:
        msk_data=None
        logging.info("   {0}.msk.fits - Not found".format(prefix))

    logging.info("Creating mock {0}-band image from datacube.".format(passband))
    filtercurve = np.genfromtxt(os.path.join(DATAPATH, passband), usecols=(0,1),
                                dtype=[("wl", ">f4"), ("tp", ">f4")])

    IP = interpolate.interp1d(filtercurve["wl"], filtercurve["tp"],
                              bounds_error=False, fill_value=0)

    wl_cube = np.arange(wlNsteps, dtype='float64')
    wl_cube *= wlStep
    wl_cube += wlStart

    dat_data_ima = np.average(dat_data, axis=0, weights=IP(wl_cube))


    logging.info("Preparing GUI for source identification"),
    ## start gui, identify bright(est) stars and extract their spectra
    app  = QtGui.QApplication(sys.argv)
    identifier = IdentSources(grabIma=dat_data_ima, pickIma=ident_ima,
                              pickData=cfpResults[resSrc])
    identifier.show()
    app.exec_()

    identified = identifier.getResults()


    ## use 'grab'-coordinates to get initial coordinate transformation
    pospars   = trans_coords(cfpResults, identified)
    (A,B,C,D) = pospars[0]

    transformation = np.ones((wlNsteps,),  dtype=pospars.dtype)
    for name in transformation.dtype.names:
            transformation[name] *= pospars[name][0]


    ## adjust PSF FWHM (only if >=4 sources have a valid FWHM measurement)
    valid = (identified["fwhm_x"]>0)&(identified["fwhm_y"]>0)
    if valid.sum() < 4:
        logging.warn("Skipping FWHM measurement - too few sources ({0})"\
                         .format(valid.sum()))
    else:
        fwhm = 0.5*(identified['fwhm_x'][valid].mean() \
                        + identified['fwhm_y'][valid].mean())
        logging.info("Estimated FWHM (Gaussian): {0:.1f}pixel".format(fwhm))


    ## preparing PSF for following source selection
    logging.info("Reading PSF information from configuration file.")
    if psfType == "gauss": ## TheGaussian PSF
        logging.info("   Analytical PSF model is a Gaussian.")
        dtype   = [('fwhm', '>f4'), ('e', '>f4'), ('theta', '>f4')]
        makePSF = lambda pars: Gaussian(fwhm=pars[0], e=pars[1], theta=pars[2])

    elif psfType == "moffat": ## The Moffat PSF
        logging.info("   Analytical PSF model is a Moffat.")
        dtype   = [('beta','>f4'), ('fwhm','>f4'), ('e','>f4'), ('theta','>f4')]
        makePSF = lambda pars: Moffat(beta=pars[0], fwhm=pars[1],
                                      e=pars[2], theta=pars[3])

    psfOutPars  = np.array(wlNsteps*[tuple(psfPars)], dtype=dtype)
    psfParNames = list(psfOutPars.dtype.names)

    ## check which PSF parameters should be fitted
    isfitted = np.zeros((len(psfParNames),), dtype='bool')
    expTwo   = 2**np.arange(1, isfitted.size+1)
    for i in np.arange(isfitted.size+1,0,-1):
        if psfFit >= (expTwo[isfitted].sum()+2**i):
            isfitted[i-1] = True



    makePSF(psfPars)
    if valid.sum() >= 4 and isfitted[psfParNames.index("fwhm")]:
        PSF.fwhm = fwhm ## update FWHM (if possible)
        psfOutPars['fwhm'][:] = fwhm


    ## Initialize source selection process 
    selector = SelectSources(sourcelist=cfpResults[resSrc],
                             bbImage=dat_data_ima,
                             psf=PSF, pospars=[A,B,C,D])

    logging.info("Selecting sources suited for PSF fitting.")
    ## determine status of all sources in catalogue:
    ##  4 - global sky spectrum
    ##  3 - resolved, own spectrum
    ##  2 - resolved, inferred spectrum
    ##  1 - resolved, but next to brighter source
    ##  0 - unresolved
    ## -1 - excluded
    resCheck = selector(criteria=['whatSNR', 'howFarFromBright',
                                  'howCloseToEdge'],
                        magcut=reslim, sigma=err_data_ima, psfrad=fitrad)

    sourceList = []

    logging.info("Results of source selection:")
    logging.info("       ID    mag     S/N  d_bright  status")
    for i in xrange(resCheck.size):
        
        listEntry = [resCheck['id'][i]]
        primary   = resCheck['id'][i]
        status    = 1

        if resCheck['whatSNR'][i] > snrcut:
            if resCheck['howFarFromBright'][i] < binrad*PSF.fwhm:
                primary = resCheck['brightNeighbor'][i]
            else:
                status += 1
##             if resCheck['howCloseToEdge'][i] < -(fitrad-1.):
##                 continue
        else:
            if resCheck['howFarFromBright'][i] < 2*binrad*PSF.fwhm:
                primary = resCheck['brightNeighbor'][i]
            else:
                status += 1
            if resCheck['howCloseToEdge'][i] < 0:
                continue

        listEntry.extend([status, primary])
        sourceList.append(tuple(listEntry))
        
        j = np.where(cfpResults['id']==resCheck['id'][i])[0][0]
        logging.info("   {0:6d}  {1:5.2f}  {2:6.1f}  {3:8.2f}  {4:6d}".format(\
                resCheck['id'][i], cfpResults['mag'][j], resCheck['whatSNR'][i],
                resCheck['howFarFromBright'][i], status)
                      )

    if useunres:
        resIDs = [sourceList[i][0] for i in xrange(len(sourceList))]

        unresCheck = selector(criteria=['howCloseToEdge'], magcut=unreslim)
        
        for i in xrange(unresCheck.size):
            if not unresCheck['id'][i] in resIDs:
                if unresCheck['howCloseToEdge'][i] > -ewidth:
                    sourceList.append((unresCheck['id'][i], 0, -1))

    if globsky:
        sourceList.append((-1, 4, -1)) ## the sky spectrum

    finalList = np.array(sourceList,
                         dtype=[('id', 'i8'),('status', 'i4'),('primary', 'i8')]
                         )

    
    if not smallifu:
        logging.info("Searching for useful PSF stars.")
        ## searching photometry results for usefuls PSF stars
        ## must fulfil following criteria:
        ## (1) at least 3 magnitudes brighter than provided resolution limit
        ##      (parameter `RESLIM')
        ## (2) at a distance of of at least 1 pixel from the edge of the FoV 
        ## (3) isolation criterium (relative contribution from neighboring
        ##      sources smaller than value of `APERNOIS'-parameter)

        psfCheck = selector(criteria=['howIsolated','howCloseToEdge'],
                             rad=fitrad, magcut=reslim-3, maglim=reslim)
        psfIDs   = psfCheck['id'][(psfCheck['howIsolated']<apernois)\
                                    &(psfCheck['howCloseToEdge']>1)]
        
        logging.info("IDs of stars that fullfil search criteria:")
        logging.info("   {0}".format(", ".join(map(str, psfIDs))))

        if len(psfIDs) == 0:
            logging.error("No useful PSF sources found!")
            raise ConfigurationError, "No useful PSF sources found!"
        elif len(psfIDs) < 4:
            logging.warning("Found only {0} useful PSF source(s)".format(\
                    len(psfIDs)))

        ## carry out aperture spectrophotometry for PSF sources
        logging.info("Performing aperture spectrophotometry on PSF sources.")

        selected = [np.where(cfpResults['id']==i)[0][0] for i in psfIDs]

        xAper = A*cfpResults['x'][selected]+B*cfpResults['y'][selected]+C
        yAper = A*cfpResults['y'][selected]-B*cfpResults['x'][selected]+D
        aperInput = np.array(zip(psfIDs, xAper, yAper),
                             dtype=[('id', 'i8'), ('x', 'f8'), ('y', 'f8')])

        make_aper = AperPhoto(data=dat_data, centers=aperInput, mask=msk_data)
        
        aperResults = make_aper.aper_photo(rAper=fitrad, psf=PSF,
                                           calcMoments=True, thresh=3.,
                                           localBG=True)
        
        ## modify `status' of PSF sources in list of sources to fit
        for i in xrange(finalList.size):
            if finalList['id'][i] in aperResults['id']:
                if finalList['id'][i] > -1:
                    finalList['status'][i] +=1

        if aperResults.size > 3:
            ## determine wavelength-dependent coordinate transformation
            logging.info("Determining wavelength-dependent coord. transf.")
            transformation = trans_coords(cfpResults[resSrc], aperResults)
            
            ##  determine FWHM, E and THETA
            logging.info("Approximating PSF parameters from measured moments.")

            ## get A, B:
            A = np.sqrt((aperResults['x2']+aperResults['y2'])/2.+np.sqrt(\
                    ((aperResults['x2']-aperResults['y2'])/2.)**2\
                        +aperResults['xy']**2))
            B = np.sqrt((aperResults['x2']+aperResults['y2'])/2.-np.sqrt(\
                    ((aperResults['x2']-aperResults['y2'])/2.)**2\
                        +aperResults['xy']**2))
            
            ## FWHM
            if isfitted[psfParNames.index("fwhm")]:
                fwhm = np.sqrt(8.*np.log(2))*np.mean(A, axis=0)
                psfOutPars['fwhm'] = fwhm
                logging.info("   Wavelength-dependent FWHM, mean={0:.2f}pix."\
                                 .format(fwhm.mean()))


            ## Ellipticity
            if isfitted[psfParNames.index("e")]:
                e    = 1. - np.mean(A/B, axis=0)
                psfOutPars['e'] = e
                logging.info("   Wavelength-dependent ellipt., mean={0:.2f}"\
                                 .format(e.mean()))
        

            ## Position angle
            if isfitted[psfParNames.index("theta")]:
                theta = 0.5*np.arctan(2*aperResults['xy']/\
                                          (aperResults['x2']-aperResults['y2']))
                theta -= (-1)**(theta<0)*np.pi/4.
                theta += (-1)**(aperResults['xy']>0)*np.pi/4.
                psfOutPars['theta'] = theta.mean(axis=0)
                logging.info("   Wavelength-dependent pos. angle, "+\
                                 "mean={0:.2f}rad.".format(theta.mean()))

    else:
        aperResults = np.array([], dtype=[('id', 'i8'), ('spectrum', 'f8')])


    ## write results to file
    logging.info("Writing results to prm-file.")

    storage = StoreParam(sourceList=cfpResults, wlStart=wlStart,
                         wlStep=wlStep, specPix=wlNsteps)
    storage.storeSourceCat(finalList)
    storage.storeSourceSpec(inputSpectra=aperResults, magZ=selector.mZ,
                            spatPix=dat_data_ima.size)
    storage.storePSF(psfType=psfType, psfPars=psfOutPars, psfFit=isfitted)
    storage.storePosPars(posPars=transformation, posFit=posFit)
    storage.writeto('{0}.prm.fits'.format(prefix), clobber=1)






    
