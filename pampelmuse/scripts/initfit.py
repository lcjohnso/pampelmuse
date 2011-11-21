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
"""
__author__ = "Sebastian Kamann (skamann@aip.de)"
__version__ = "0.1.2.0"
import pyfits, sys, logging
import numpy as np
from scipy import sparse
from PyQt4 import QtGui

## required PampelMuse routines
from pampelmuse.graphics.ident_sources import IdentSources
from pampelmuse.core.aperture import AperPhoto
from pampelmuse.functions.trans_coords import trans_coords
from pampelmuse.core.select_sources import SelectSources
from pampelmuse.core.store_param import StoreParam 
from pampelmuse.core.my_profiles import Gaussian


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
    globsky  = False ## True
    aperrad  = arguments.config["APERRAD"]          ##
    apernois = arguments.config["APERNOIS"]         ##
    reslim   = arguments.config["RESLIM"]           ##
    snrcut   = arguments.config["SNRCUT"]           ##
    binrad   = arguments.config["BINRAD"]           ##
    fitrad   = arguments.config["FITRAD"]           ##
    useunres = arguments.config["USEUNRES"]         ##
    unreslim = arguments.config["UNRESLIM"]         ##
    ewidth   = arguments.config["EWIDTH"]           ##
    ##################################################


    print " => Reading input data...",
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
    ## Gaussian PSF, FWHM=4.0pix, PSFRAD=15.0pix
    ## The zeropoint is such that a source with magnitude `reslim' will
    ## have a flux of 1000 counts and a peak-S/N of roughly 10 
    PSF = Gaussian(fwhm=4.0)

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
    dat_data, dat_head = pyfits.getdata("{0}.dat.fits".format(prefix),
                                        header=True)
    dat_data_ima       = np.mean(dat_data, axis=0) 
    wl_step  = dat_head['CDELT3']
    wl_start = dat_head['CRVAL3']

    err_data           = pyfits.getdata("{0}.err.fits".format(prefix))
    err_data_ima       = np.mean(err_data, axis=0)

    try:
        msk_cube = pyfits.open("{0}.msk.fits".format(prefix))
        msk_data = np.asarray(msk_cube[0].data, dtype='bool')
        msk_cube.close()
    except IOError:
        msk_data=None


    print "OK\n => Preparing GUI for source identification...",
    ## start gui, identify bright(est) stars and extract their spectra
    app  = QtGui.QApplication(sys.argv)
    identifier = IdentSources(grabIma=dat_data_ima, pickIma=ident_ima,
                              pickData=cfpResults[resSrc])
    print "OK"
    identifier.show()
    app.exec_()

    identified = identifier.getResults()

    ## estimate FWHM of sources
    fwhm = 0.5*(identified['fwhm_x'].mean() + identified['fwhm_y'].mean())
    PSF.fwhm = fwhm

    ## use 'grab'-coordinates to get initial coordinate transformation
    pospars   = trans_coords(cfpResults, identified)
    (A,B,C,D) = pospars[0]

    print " => Performing aperture spectrophotometry on selected sources...",
    sys.stdout.flush()
    ## find sources suited for aperture photometry
    ## must fulfil following criteria:
    ## (1) at least 3 magnitudes brighter than provided resolution limit
    ##      (parameter `RESLIM')
    ## (2) at a distance of of at least 1 pixel from the edge of the FoV 
    ## (3) isolation criterium (relative contribution from neighboring
    ##      sources smaller than value of `APERNOIS'-parameter)
    selector = SelectSources(sourcelist=cfpResults[resSrc],
                             shape=dat_data.shape[1:],
                             psf=PSF, pospars=[A,B,C,D])
    aperCheck = selector(criteria=['howIsolated','howCloseToEdge'],
                         rad=aperrad, magcut=reslim-3)
    aperIDs   = aperCheck['id'][(aperCheck['howIsolated']<apernois)\
                                    &(aperCheck['howCloseToEdge']>1)]
    if len(aperIDs) == 0:
        raise ConfigurationError,\
            "No sources suited for aperture spectrophotometry found"
    elif len(aperIDs) < 4:
        print ""
        logging.warning(\
            "Found only {0} source(s) suited for aperture spectrophotometry"\
                .format(len(aperIDs)))

    ## carry out aperture photometry
    selected = [np.where(cfpResults['id']==i)[0][0] for i in aperIDs]

    xAper = A*cfpResults['x'][selected]+B*cfpResults['y'][selected]+C
    yAper = A*cfpResults['y'][selected]-B*cfpResults['x'][selected]+D
    aperInput = np.array(zip(aperIDs, xAper, yAper),
                         dtype=[('id', 'i8'), ('x', 'f8'), ('y', 'f8')])

    make_aper = AperPhoto(data=dat_data, centers=aperInput, mask=msk_data)

    aperResults = make_aper.aper_photo(rAper=aperrad, psf=PSF,
                                       calcMoments=True, thresh=3.,
                                       localBG=True)

    print "OK"
    ## determine magnitude zeropoint
    meanFlux = np.zeros(aperResults[aperResults['id']>-1].shape,
                        dtype='float32')
    for i in xrange(meanFlux.size):
        meanFlux[i] = np.mean(aperResults['spectrum'][i])
    magZ = cfpResults['mag'][selected] + 2.5*np.log10(meanFlux)

    ## fine-tune coordinate transformation
    if aperResults.size > 3:
        print " => Determining coordinate transformation...",
        transformation = trans_coords(cfpResults[resSrc], aperResults)
        print "OK"
    else:
        transformation = np.ones((dat_head['NAXIS3'],),  dtype=pospars.dtype)
        for name in transformation.dtype.names:
            transformation[name] *= pospars[name][0]

    print " => Selecting sources for the fitting process...",
    ## determine status of all sources in catalogue:
    ##  4 - global sky spectrum
    ##  3 - resolved, own spectrum
    ##  2 - resolved, inferred spectrum
    ##  1 - resolved, but next to brighter source
    ##  0 - unresolved
    ## -1 - excluded
    resCheck = selector(criteria=['whatSNR', 'howFarFromBright',
                                  'howCloseToEdge'],
                        magcut=reslim, magZ=magZ.mean(), sigma=err_data_ima)

    finalList = []

    outList = open('list.dat', 'w')

    for i in xrange(resCheck.size):
        
        if resCheck['howCloseToEdge'][i] > 2:
            j = np.where(cfpResults['id']==resCheck['id'][i])[0][0]
            outList.write("{0}  {1}  {2}  {3}\n".format(\
                    resCheck['id'][i], cfpResults['mag'][j],
                    resCheck['whatSNR'][i], resCheck['howFarFromBright'][i]))

        listEntry = [resCheck['id'][i]]
        primary   = resCheck['id'][i]
        status    = 1

        if resCheck['whatSNR'][i] > snrcut:
            if resCheck['howFarFromBright'][i] < binrad*fwhm:
                primary = resCheck['brightNeighbor'][i]
            else:
                status += 1
                if resCheck['id'][i] in aperResults['id']:
                    status +=1
            if resCheck['howCloseToEdge'][i] < -fitrad:
                continue
        else:
            if resCheck['howFarFromBright'][i] < 2*binrad*fwhm:
                primary = resCheck['brightNeighbor'][i]
            else:
                status += 1
            if resCheck['howCloseToEdge'][i] < 0:
                continue

        listEntry.extend([status, primary])
        finalList.append(tuple(listEntry))
        
    outList.close()

    if useunres:
        resIDs = [finalList[i][0] for i in xrange(len(finalList))]

        unresCheck = selector(criteria=['howCloseToEdge'], magcut=unreslim)
        
        for i in xrange(unresCheck.size):
            if not unresCheck['id'][i] in resIDs:
                if unresCheck['howCloseToEdge'][i] > -ewidth:
                    finalList.append((unresCheck['id'][i], 0, -1))

    if globsky:
        finalList.append((-1, 4, -1)) ## the sky spectrum


    print "OK\n => Writing results to file...",
    ## write results to file
    storage = StoreParam(sourceList=cfpResults, wlStart=wl_start,
                         wlStep=wl_step)
    storage.storeSourceCat(np.array(finalList, dtype=[('id', 'i8'),
                                                      ('status', 'i4'),
                                                      ('primary', 'i8')]
                                    )
                           )
    storage.storeSourceSpec(inputSpectra=aperResults, magZ=None,
                            spatPix=dat_data_ima.size)
    storage.storePosPars(transformation)
    storage.writeto('{0}.prm.fits'.format(prefix), clobber=1)
    print "OK\n\n"





    
