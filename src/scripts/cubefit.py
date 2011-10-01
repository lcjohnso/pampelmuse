"""
CUBEFIT
=======
The routine is a build around the main component of PampelMuse, namely
the fit_cube-routine. It is called automatically if PampelMuse is started
with the `FITCUBE' option. As input, it takes a dictionary containing the
keywords and their values after PampelMuse was started.

CHANGELOG
---------
2011/09/08, v0.0.0.0
"""
__author__ = "Sebastian Kamann (skamann@aip.de)"
__version__ = "0.0.0.0"
import numpy as np
import os, pyfits

## own packages
from fit_array import StarFit

def cubefit(arguments):

    print ""
    print 21* " "+"***  CubeFit, version {0}  ***".format(__version__)+21*" "
    print ""

    ## List of required definitions: #######################
    prefix   = arguments.prefix                           ##
##     flxfit   = arguments.config["FLXFIT"]                 ##
    posfit   = arguments.config["POSFIT"]                 ##
    psftype  = arguments.config["PSFTYPE"]                ##
    psfpars  = arguments.config["PSFPARS"]                ##
    psffit   = arguments.config["PSFFIT"]                 ##
    dobackup = False
    ########################################################


    ## open PRM-file, get no. of sources
    prmFile  = pyfits.open("{0}.prm.fits".format(prefix))
    extNames = [prmFile[i].name for i in xrange(len(prmFile))]

    Nsrc = prmFile['SOURCES'].header['NAXIS2'] ## no of sources
    Ndsp = prmFile['SPECTRA'].header['NAXIS1'] ## no of spectral pixels
    Nspc = prmFile['SPECTRA'].header['NAXIS2'] ## no of resolved components


    ## CREATE ARRAYS THAT CONTAIN PARAMETERS USED IN THE FIT
    ## Collect information about sources
    srcList = np.array(prmFile['SOURCES'].data,
                       dtype=prmFile['SOURCES'].data.dtype)

    ## collect the  spectra
    spcList = prmFile['SPECTRA'].data
    

    ## Collect information about coordinate transformation    
    isfitted = np.array(4*[False])
    if posfit in [1,2]: isfitted[2:] = True
    if posfit == 2:     isfitted[:2] = True

    if 'POSPARS' in extNames: ## check if HDU exists
        valueShape = (prmFile['POSPARS'].header['NAXIS1'],)

        posPars = np.array([('A', prmFile['POSPARS'].data[0], isfitted[0]),
                            ('B', prmFile['POSPARS'].data[1], isfitted[1]),
                            ('C', prmFile['POSPARS'].data[2], isfitted[2]),
                            ('D', prmFile['POSPARS'].data[3], isfitted[3])],
                           dtype=[('name', 'a1'),
                                  ('value', 'f8', valueShape),
                                  ('fit', 'b1')]
                           )

    else: ## otherwise, assume "identity" transformation
        if posfit in [1,2]: valueShape=(Ndsp,)
        else: valueShape = (1)

        posPars = np.array([('A',1, isfitted[0]),('B',0,isfitted[1]),
                            ('C',0, isfitted[2]),('D',0,isfitted[3])],
                           dtype=[('name', 'a1'),
                                  ('value', 'f8', valueShape),
                                  ('fit', 'b1')])


    ## Collect information about PSF
    ## PSF parameter definitions: ###################################
    psfProfiles = {'gauss':('fwhm', 'e', 'theta'),                 ##
                   'moffat':('beta', 'fwhm', 'e', 'theta')}        ##
    #################################################################
    thePSF = psfProfiles[psftype]

    ## check which PSF parameters should be fitted
    isfitted = np.zeros((len(thePSF),), dtype='bool')
    expTwo   = 2**np.arange(1, isfitted.size+1)
    for i in np.arange(isfitted.size+1,0,-1):
        if psffit >= (expTwo[isfitted].sum()+2**i):
            isfitted[i-1] = True

    if "PSFPARS" in extNames: ##  check if HDU exists
        if prmFile['PSFPARS'].header['NAXIS1'] != Ndsp:
            raise IOError, "Invalid shape of data in HDU 'PSFPARS'"

        Npars = prmFile['PSFPARS'].header['NAXIS2']

        params = []
        for i in xrange(Npars):
            params.append((prmFile['PSFPARS'].header["PSFPAR{0}".format(i+1)],
                           prmFile['PSFPARS'].data[i],
                           isfitted[i]))
        psfPars = np.array(params, dtype=[('name', 'a10'),
                                          ('value', 'f8', (Ndsp,)),
                                          ('fit', 'b1')])

        for requiredPar in thePSF: ## check if provided PSF information valid
            if not requiredPar in psfPars['name']:
                raise IOError, "Cannot find parameter `{0}'".format(\
                    requiredPar) +" required for PSF of type `{0}'".format(\
                    psftype)

    else: ## otherwise, use information from configuration file
        psfPars  = np.array([(thePSF[i], psfpars[i], isfitted[i]) \
                                 for i in xrange(len(thePSF))],
                            dtype=[('name', 'a8'),
                                   ('value', 'f8', (Ndsp,)),
                                   ('fit', 'b1')])

    prmFile.close() ## finished reading source catalog.



    ## open datacubes & check consistency
    dataFits = pyfits.open('{0}.dat.fits'.format(prefix))
    if dataFits[0].header['NAXIS3'] != Ndsp:
        raise IOError, "No. of spectral pixels in data-cube and prm-file do "\
            +" not match."
    
    sigmaFits = pyfits.open('{0}.err.fits'.format(prefix))
    if sigmaFits[0].header['NAXIS3'] != Ndsp:
        raise IOError, "No. of spectral pixels in sigma-cube and prm-file do "\
            +" not match."

    if os.path.exists("{0}.msk.fits".format(prefix)):
        maskFits = pyfits.open('{0}.msk.fits'.format(prefix))
        if maskFits[0].header['NAXIS3'] != Ndsp:
            raise IOError, "No. of spectral pixels in mask-cube and prm-file "\
                +"do not match."
    else:
        maskFits = [None]


    ## start fitting process:
        fitter = StarFit(dataFits=dataFits[0], sigmaFits=sigmaFits[0],
                         maskFits=maskFits[0])

        fitter.provideSources(sources=srcList, spectra=spcList,
                              coordTrans=posPars)

        fitter.providePSF(psfType=psftype, psfParameters=psfPars,
                          psfRad=10)
        
        fitter(fitRad=3)

    ## collect fitting results
    prmFile  = pyfits.open("{0}.prm.fits".format(prefix), mode='update')
    extNames = [prmFile[i].name for i in xrange(len(prmFile))]

    ## store spectra
    crval1 = prmFile['SPECTRA'].header['CRVAL1'] 
    cdelt1 = prmFile['SPECTRA'].header['CDELT1'] 
    crpix1 = prmFile['SPECTRA'].header['CRPIX1'] 

    newSpecHDU = pyfits.ImageHDU(fitter.spectra) 
    newSpecHDU.header.update('EXTNAME', 'SPECTRA')
    newSpecHDU.header.update('CRPIX1', crpix1)
    newSpecHDU.header.update('CRVAL1', crval1)
    newSpecHDU.header.update('CDELT1', cdelt1)

    if dobackup:
        if 'INSPECTRA' in extNames:
            prmFile.remove(prmFile['INSPECTRA'])
        prmFile['SPECTRA'].header.update('EXTNAME', 'INSPECTRA')
    else:
        prmFile.remove(prmFile['SPECTRA'])
    prmFile.insert(2, newSpecHDU)

    ## store coordinate transformation (only if it has been fitted)
    if fitter.coordTrans['fit'].any():
        newPosHDU = pyfits.ImageHDU(fitter.coordTrans['value'])
        newPosHDU.header.update('EXTNAME', 'POSPARS')
        newPosHDU.header.update('CRPIX1', crpix1)
        newPosHDU.header.update('CRVAL1', crval1)
        newPosHDU.header.update('CDELT1', cdelt1)
        if dobackup:
            if 'INPOSPARS' in extNames:
                prmFile.remove(prmFile['INPOSPARS'])
            if 'POSPARS' in extNames:
                prmFile['POSPARS'].header.update('EXTNAME', 'INPOSPARS')
        else:
            if 'POSPARS' in extNames:
                prmFile.remove(prmFile['POSPARS'])
        prmFile.insert(3, newPosHDU)


    ## store PSF (only if it as been fitted)
    if fitter.psfPars['fit'].any():
        newPSFHDU = pyfits.ImageHDU(fitter.psfPars['value']) 
        newPSFHDU.header.update('EXTNAME', 'PSFPARS')
        newPSFHDU.header.update('PSFTYPE', fitter.psfType)
        for i in xrange(fitter.psfPars.size):
            newPSFHDU.header.update('PSFPAR{0}'.format(i+1),
                                    fitter.psfPars['name'][i])
        newPSFHDU.header.update('CRPIX1', crpix1)
        newPSFHDU.header.update('CRVAL1', crval1)
        newPSFHDU.header.update('CDELT1', cdelt1)
        
        if dobackup:
            if 'INPSFPARS' in extNames:
                prmFile.remove(prmFile['INPSFPARS'])
            if 'PSFPARS' in extNames:
                prmFile['PSFPARS'].header.update('EXTNAME', 'INPSFPARS')
        else:
            if 'PSFPARS' in extNames:
                prmFile.remove(prmFile['PSFPARS'])
        prmFile.insert((3+('POSPARS' in extNames)), newPSFHDU)

    prmFile.close()
