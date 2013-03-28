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
2013/03/28, v0.2.0.0, modified code to implement usage of instrument-specific
                      characteristics via the classed defined in
                      pampelmuse.core.instruments and to use new way of
                      calculating PSF-profiles using the classes defined in
                      pampelmuse.core.new_profiles
"""
__author__ = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__version__ = "0.2.0.0"
import datetime, logging, os, sys
import numpy as np, pyfits

## required PampelMuse-packages
from pampelmuse.core.sources import Sources
from pampelmuse.core.positions import CoordinateTransformation
from pampelmuse.core.psf import PsfAttributes
from pampelmuse.core.instruments import Instrument, Pmas, Argus
from pampelmuse.core.new_profiles import Moffat, Gaussian

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


    ## Check provided datacube
    logging.info("Reading data cube: {0}.dat.fits".format(prefix))
    dataFits  = pyfits.open('{0}.dat.fits'.format(prefix))


    ## Check if information on used instrument available
    if "PAMPELMUSE IFU" in prmFile[0].header.keys():
        instname = prmFile[0].header["PAMPELMUSE IFU"]
    else:
        instname = None

    if instname == "argus":
        ## check if calibration HDU for ARGUS exists
        instrument = Argus()
        try:
            setup_index = dataFits.index_of("FIBER_SETUP")
            sources.instrument.read_fiber_setup(hdu=dataFits[setup_index])
            fiber_calib = True
        except KeyError:
            logging.warning("No HDU with fibre setup found for ARGUS data.")
    elif instname  == "pmas":
        instrument = Pmas()
    else:
        instrument = Instrument()

    ## Check format of datacube
    ndim     = dataFits[0].header["NAXIS"]

    if instrument.transform == None:
        if ndim!=3:
            logging.error(\
                "Data must be 3D unless mapping from spaxels to x&y specified.")
            assert ndim==3, "Expected 3dim. datacube, got {0}dim".format(ndim)

        ## initialize transformation for 'Instrument'-class
        xdim = dataFits[0].header['NAXIS1']
        ydim = dataFits[0].header['NAXIS2']

        x = np.mod(np.arange(xdim*ydim, dtype=np.int16), xdim)
        y = np.arange(xdim*ydim, dtype=np.int16)//xdim

        instrument.transform = np.vstack((y,x)).T
        instrument.shape = (ydim, xdim)

    ## collapse parameters of coordinate transformation
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
        select = (_x[:,0]>-1.5)&(_x[:,0]<(instrument.shape[1]+1.))\
            &(_y[:,0]>-1.5)&(_y[:,0]<(instrument.shape[0]+1.))\
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

        sources = Sources(data=hdu, hdu=True, instrument=instrument)

    else:
        sources = Sources(data=prmFile["SOURCES"], hdu=True,
                          instrument=instrument)

    sources.loadSpectra(data=prmFile["SPECTRA"], hdu=True)

    sources.makeNewTrans(**pospars)
    sources.makeValues()
    sources.info()



    ## PSF
    psf_attributes = PsfAttributes()
    if "PSFPARS" in extNames: ##  check if HDU exists
        logging.info("Reading 'PSFPARS' HDU.")

        ## collapse PSF parameters
        psfpars = {}
        for i in xrange(prmFile["PSFPARS"].data.shape[0]):
            
            psfpars[prmFile["PSFPARS"].data.field("name")[i]] \
                = np.mean(prmFile["PSFPARS"].data.field("value")[i])
            
        psf_attributes.makeNewPSF(\
            profile=prmFile["PSFPARS"].header["PSFTYPE"], **psfpars)

    else: ## otherwise, assume Gaussian, round with FWHM=3.0pixel
        logging.error("No HDU 'PSFPARS' found. Assuming Gaussian PSF.")
        psf_attributes.makeNewPSF(profile="gauss", fwhm=3.0, e=0., theta=0.)
    psf_attributes.info()
    psf_attributes.makeValues()


    ## Initialize PSF profiles of the sources:
    if psf_attributes.psftype == "gauss":
        PSF = Gaussian
    elif psf_attributes.psftype == "moffat":
        PSF = Moffat
    else:
        logging.critical("Invalid PSF type: '{0}'".format(\
                psf_attributes.psftype))
        raise IOError, "PSF type not understood."

    PSF.set_transform(sources.instrument.transform)
    for i in xrange(sources.nsrc):
        psf = PSF(Xc=sources.x[i], Yc=sources.y[i], mag=0, maxrad=psfRad,
                  osfactor=5, osradius=2)
        
    ## update PSF instances with current parameter fits
    i = 0 
    for prm in psf_attributes.parameters:
        getattr(PSF, "set_{0}".format(prm))(psf_attributes.values[i,0])
        i += 1
        
    i = 0
    for prm in sources.parameters:
        getattr(PSF, "set_{0}".format(prm))(sources.ctValues[i,0])
        i += 1



    ## initialize output array
    data_flux = np.zeros((2, sources.nsrc)+sources.instrument.shape,
                         dtype=np.float32)

    ## determine integrated fluxes of resolved and unresolved stars
    rsv_ifu = 0.
    rsv_cat = 0.
    unr_ifu = 0.
    unr_cat = 0.

    ## loop over sources:
    i=0
    unres_added = False
    for psf in PSF:
       
        j = sources.specrow[i]

        if j == sources.skyRow:
            unr_ifu += sources.data[j].sum()
        else:
            flux = np.zeros((sources.instrument.nspaxel,), dtype=np.float32)

            flux[psf.used] += 10**(-0.4*(sources.mag[i]-25.))*psf.f

            data_flux[0,i] = sources.instrument.make_grid(flux)

            if sources.status[i] <= 0:
                data_flux[1,i] = data_flux[0,i]
                unr_cat += psf.f.sum()*10**(-0.4*(sources.mag[i]-25.))
                if sources.status[i]==0:
                    unr_ifu += psf.f.sum()*10**(-0.4*sources.weights[i])\
                        *sources.data[j].sum()
            else:
                rsv_cat += psf.f.sum()*10**(-0.4*(sources.mag[i]-25.))
                rsv_ifu += psf.f.sum()*10**(-0.4*sources.weights[i])\
                    *sources.data[j].sum()

        i += 1

                        
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

