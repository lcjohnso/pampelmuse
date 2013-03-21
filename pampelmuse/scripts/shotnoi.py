"""
SHOTNOI
=======
The routine is designed to calculate the shot noise in every spaxel of a data-
cube. To this purpose, the shot noise is defined as the fractional flux that
the brightest star contributes to the integrated light in that spaxel.

The common prefix of the names of the files associated to the datacube under
investigation must be provided. Optionally, a photometry file can be provided.
It none is provided, the source list included in the PRM-file is used. The
former option is useful if only the brghter stars are included in the PRM-file.

CHANGELOG
---------

2012/12/17, v0.1.0.0
2012/12/23, v0.1.0.1, added logging-entries that provide fraction of unresolved
                      light in datacube. 
2013/01/20, v0.1.1.1, hierarch keywords were named 'SUBTRES' instread of
                      'SHOTNOI'
"""
__author__ = "Sebastian Kamann (skamann@aip.de)"
__version__ = "0.1.1.1"
import datetime, logging, os, sys
import numpy as np, pyfits

## required PampelMuse-packages
from pampelmuse.core.sources import Sources
from pampelmuse.core.positions import CoordinateTransformation
from pampelmuse.core.psf import PsfAttributes

def shotnoi(arguments):

    print ""
    print 21* " "+"***  ShotNoi, version {0}  ***".format(__version__)+21*" "
    print ""

    ## List of required definitions: #######################
    prefix   = arguments.prefix                           ##
    cfpfile  = arguments.sourcelist                       ##
    psfRad   = arguments.config["PSFRAD"]                 ##
    passband = arguments.config["PASSBAND"]               ##
    copyHead = True
    ########################################################

    ## open PRM-file
    logging.info("Reading PampelMuse-file: {0}.prm.fits".format(prefix))
    prmFile  = pyfits.open("{0}.prm.fits".format(prefix))
    extNames = [prmFile[i].name for i in xrange(len(prmFile))]

    ## Check if PRM-file contains results of successful fit
    try: fitStatus = prmFile[0].header['PAMPELMUSE CUBEFIT STATUS']
    except KeyError:
        try: fitStatus = prmFile[0].header['PMFITIND']
        except KeyError: fitStatus = None

    if fitStatus != "SUCCESS":
        logging.warning("PRM-file does not contain results of successful fit.")

    ## get spatial size of datacube
    logging.info("Reading data cube: {0}.dat.fits".format(prefix))
    dataFits  = pyfits.open('{0}.dat.fits'.format(prefix))
    spatShape = (dataFits[0].header['NAXIS2'], dataFits[0].header['NAXIS1'])

    ## collapse coordinate transformation
    pospars = {}
    for prm in ["A", "B", "C", "D"]:
        i = np.where(prmFile["POSPARS"].data.field("name")==prm)[0][0]

        pospars[prm] = np.mean(prmFile["POSPARS"].data.field("value")[i])


    ## initialize coordinate transformation
    ct = CoordinateTransformation()
    ct.makeNewTrans(**pospars)


    ## Check if dedicated photometry file provided
    if cfpfile != None:
        cfpdata = np.genfromtxt(cfpfile, names=True, dtype=None, comments="#")

        ## it is assumed that the initial coordinates in the PRM-file are from
        ## the same data-set, i.e. they only differ from the ones in the
        ## photometry by a constant offset in x & y.
        
        ## find common entries in both lists
        match  = (cfpdata["id"][:,np.newaxis]\
                      ==prmFile["SOURCES"].data.field("id"))
        m1 = match.argmax(axis=0)[match.any(axis=0)]
        m2 = match.any(axis=0)

        ## get IFU pixel coordinates
        dx = np.mean(cfpdata["x"][m1]-prmFile["SOURCES"].data.field("x")[m2])
        dy = np.mean(cfpdata["y"][m1]-prmFile["SOURCES"].data.field("y")[m2])

        _x, _y = ct(cfpdata["x"]-dx, cfpdata["y"]-dy)

        ## select sources in/near field of view
        select = (_x[:,0]>-1.5)&(_x[:,0]<(spatShape[1]+1.))\
            &(_y[:,0]>-1.5)&(_y[:,0]<(spatShape[0]+1.))\
            &(np.in1d(np.arange(_x.size, dtype=np.int32), m1)==False)

        ## create combined table
        _nrows = prmFile["SOURCES"].data.shape[0]
        hdu = pyfits.new_table(prmFile["SOURCES"].columns,
                               nrows=_nrows+select.sum())
        hdu.data.field("id")[_nrows:]  = cfpdata["id"][select]
        hdu.data.field("x")[_nrows:]   = cfpdata["x"][select]-dx
        hdu.data.field("y")[_nrows:]   = cfpdata["y"][select]-dy
        hdu.data.field("mag")[_nrows:] = cfpdata[passband][select]
        hdu.data.field("status")[_nrows:]  = -1
        hdu.data.field("primary")[_nrows:] = -2    
        hdu.data.field("specrow")[_nrows:] \
            = prmFile["SOURCES"].data.field("specrow").max() + 1

        sources = Sources(data=hdu,  hdu=True)

    else:
        sources = Sources(data=prmFile["SOURCES"], hdu=True)

    sources.loadSpectra(data=prmFile["SPECTRA"], hdu=True)

    sources.makeNewTrans(**pospars)
    sources.makeValues()
    sources.info()


    ## PSF
    psf = PsfAttributes()
    if "PSFPARS" in extNames: ##  check if HDU exists
        logging.info("Reading 'PSFPARS' HDU.")

        ## collapse PSF parameters
        psfpars = {}
        for i in xrange(prmFile["PSFPARS"].data.shape[0]):
            
            psfpars[prmFile["PSFPARS"].data.field("name")[i]] \
                = np.mean(prmFile["PSFPARS"].data.field("value")[i])
            
        psf.makeNewPSF(profile=prmFile["PSFPARS"].header["PSFTYPE"], **psfpars)

    else: ## otherwise, assume Gaussian, round with FWHM=3.0pixel
        logging.error("No HDU 'PSFPARS' found. Assuming Gaussian PSF.")
        psf.makeNewPSF(profile="gauss", fwhm=3.0, e=0., theta=0.)
    psf.info()
    psf.makeValues()

    ## initialize PSF
    psf_0 = psf(index=0)

    ## load current IFU coords:
    sources.loadIFUCoords(index=0)

    ## initialize output array
    data_flux = np.zeros((2, sources.mag.size)+spatShape, dtype=np.float32)

    ## determine integrated fluxes of resolved and unresolved stars
    rsv_ifu = 0.
    rsv_cat = 0.
    unr_ifu = 0.
    unr_cat = 0.

    ## loop over sources:
    for r, indices in sources.indexmap.iteritems():

        for j in indices:

            if sources.status[j] == 4:
                unr_ifu += sources.data[r].sum()
            else:
                i = np.where(sources.ifuSelect==j)[0][0]
                x_i    = sources.xvalues[i,0]
                y_i    = sources.yvalues[i,0]

                _f = psf_0(\
                    size=spatShape, x_c=x_i, y_c=y_i, mag=0, maxrad=psfRad)

                data_flux[0, j] = _f*10**(-0.4*(sources.mag[j]-25.))

                if sources.status[j] <= 0:
                    data_flux[1, j] = _f

                mag_i  = sources.weights[r][list(indices).index(j)]
                norm_i = _f.sum()*10**(-0.4*mag_i)

                if sources.status[j] <= 0:
                    unr_cat += _f.sum()*10**(-0.4*(sources.mag[j]-25.))
                    if sources.status[j] == 0:
                        unr_ifu += norm_i*sources.data[r].sum()
                else:
                    rsv_cat += _f.sum()*10**(-0.4*(sources.mag[j]-25.))
                    rsv_ifu += norm_i*sources.data[r].sum()
                        
    logging.info("Contribution of unresovled to total light: ")
    logging.info("  Photometric catalogue: {0:.1f}%".format(\
            float(100.*unr_cat/(rsv_cat+unr_cat))))
    logging.info("  PampelMuse analysis: {0:.1f}%".format(\
            float(100.*unr_ifu/(rsv_ifu+unr_ifu))))
                 

    prmFile.close() ## finished reading source catalog.


    data_flux.sort(axis=1)


    data_flux_summed  = data_flux.cumsum(axis=1)

    data_flux_summed[0] /= data_flux_summed[0,-1]
    data_flux_summed[1] /= data_flux_summed[1,-1]

    outHDU = pyfits.PrimaryHDU(data_flux_summed[:,-1]-data_flux_summed[:,-2])
    if copyHead:
        outHDU.header = dataFits[0].header
    outHDU.header.update(\
        "hierarch PAMPELMUSE SHOTNOI UTC",
        datetime.datetime.utcnow().strftime("%y-%m-%d %H:%M:%S"))
    outHDU.header.update("hierarch PAMPELMUSE SHOTNOI VERSION", __version__)
    outHDU.header.update("hierarch PAMPELMUSE SHOTNOI PSFRAD", psfRad)
  
    if os.path.exists("{0}.shot.fits".format(prefix)):
        os.remove("{0}.shot.fits".format(prefix))
        logging.warning("Existing shot noise cube has been overwritten.")
    outHDU.writeto("{0}.shot.fits".format(prefix))

