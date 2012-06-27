"""
ADDWCS
======
The routine is designed to add WCS information to the header of the integral-
field data after the pixel positions of some sources (with known RA and Dec)
have been identified (and optionally fitted).
The routine currently requires `pyraf` and `astLib` to work.

ADDED: 2012/06/26

CHANGELOG
---------
2012/06/26, v0.0.0.0 
"""
__author__  = "Sebastian Kamann (skamann@aip.de)"
__version__ = "0.0.0.0"

import logging, os, sys
import numpy as np, pyfits
from astLib import astCoords
from pyraf.iraf import images

## required PampelMuse-routines
from pampelmuse.core.sources import Sources


OUTROWTEMPLATE = "{0}  {1}  {2:5.2f}  {3:5.2f}\n"
## template for a line of the temporary data-file that is created below.

def addwcs(arguments):

    ## arguments is a namespace that MUST contain the following entries:
    ## 1) prefix - the prefix common to all FITS-files to be used
    ## 2) sourcelist - the name of the ASCII-file containing the IDs and the
    ##                 world coordinates of the sources.
    ## 3) config - a dictionary containing the current PampelMuse configuration

    print ""
    print 21* " "+"***  AddWCS, version {0}  ***".format(__version__)+21*" "
    print ""
    
###### List of required definitions: #################
    prefix   = arguments.prefix                     ##
    catalog  = arguments.sourcelist                 ##
######################################################

    ## the routine needs to create two temporary files. An ASCII-file, named
    ## ROOT.dat will contain the source identifications, one per line with
    ## structure `HH:MM:SS.ss  +DD:MM:SS.s  XXX.xx  YYY.yy`. A FITS-file named
    ## ROOT.fits will contain the calculated WCS data in its header.
    ## By default, ROOT is `wcs_tmp`.  If either of the files already exists,
    ## the routine will append `0` to ROOT until no file is overwritten.
    tmpfileroot = "wcs_tmp"
    while True:
        if not os.path.exists("{0}.fits".format(tmpfileroot)):
            if not os.path.exists("{0}.dat".format(tmpfileroot)):
                break
        tmpfileroot += "0"

    ## reading source catalog
    logging.info("Reading source catalog: {0}".format(\
            os.path.split(catalog)[1]))
    catdata = np.genfromtxt(catalog, dtype=None, names=True)

    ## open PRM-file
    logging.info("Reading PampelMuse-file: {0}.prm.fits".format(prefix))
    prmFile  = pyfits.open("{0}.prm.fits".format(prefix))
    extNames = [prmFile[i].name for i in xrange(len(prmFile))]

    ## Spectra are not actually needed but must be loaded to initialize
    ## some attributes
    sources = Sources(data=prmFile["SOURCES"], hdu=True)
    sources.loadSpectra(prmFile["SPECTRA"], hdu=True)

    ## Positions
    logging.info("Obtaining source coordinates")
    if "POSITIONS" in extNames:
        logging.info("Reading 'POSITIONS' HDU.")
        sources.loadPositionsHDU(prmFile["POSITIONS"])

    elif 'POSPARS' in extNames:
        logging.info("Reading 'POSPARS' HDU.")
        sources.readTransHDU(prmFile["POSPARS"], backup=False)
        sources.resizeStructure(size=sources.ndisp)
        sources.makeValues()
        sources.info()
    else:
        logging.error("Cannot determine pixel coordinates.")

    ## load current IFU coords:
    sources.loadIFUCoords()

    tmp_coordfile = open("{0}.dat".format(tmpfileroot), "w")

    for i in xrange(sources.ncmp):
        j  = sources.indexmap[i][0]

        if sources.status[j] in [2,3] and len(sources.indexmap[i])==1:
            
            ID  = sources.ID[sources.indexmap[i][0]]

            x_j = np.mean(sources.xvalues[np.where(sources.ifuSelect==j)[0][0]])
            y_j = np.mean(sources.yvalues[np.where(sources.ifuSelect==j)[0][0]])

            k = np.where(catdata["id"]==ID)[0][0]

            tmp_coordfile.write(OUTROWTEMPLATE.format(\
                    astCoords.decimal2hms(catdata["RA"][k], delimiter=":"),
                    astCoords.decimal2dms(catdata["Dec"][k], delimiter=":"),
                    float(x_j),
                    float(y_j)))

    tmp_coordfile.close()
    prmFile.close()

    datFile  = pyfits.open("{0}.dat.fits".format(prefix), mode="update")
    spat_shape = (datFile[0].header["NAXIS2"], datFile[0].header["NAXIS1"])
    ndim       = datFile[0].header["NAXIS"]

    tmp_fitsfile = pyfits.PrimaryHDU(np.zeros((spat_shape), dtype=np.uint16))
    tmp_fitsfile.writeto("{0}.fits".format(tmpfileroot))
    

    images.imcoords.ccmap(\
        "{0}.dat".format(tmpfileroot),"{0}.db".format(tmpfileroot),
        image="{0}.fits".format(tmpfileroot),
        xcol=3, ycol=4, lngcol=1, latcol=2, update=True)

    
    wcs_info = pyfits.getheader("{0}.fits".format(tmpfileroot))

    datFile[0].header.update("RADECSYS", wcs_info["RADECSYS"],
                               after="NAXIS{0}".format(ndim))
    datFile[0].header.update("EQUINOX", wcs_info["EQUINOX"],
                               after="RADECSYS")
    datFile[0].header.update("WCSDIM", ndim, after="EQUINOX")

    for i in [1,2]:
        for key in ["CTYPE", "CRVAL", "CRPIX"]:
            fullkey = "{0}{1}".format(key, i)
            datFile[0].header.update(fullkey, wcs_info[fullkey])

    for i in xrange(ndim,0,-1):
        for j in xrange(ndim, 0, -1):
            fullkey = "CD{0}_{1}".format(i,j)

            if i in [1,2] and j in [1,2]:
                value = wcs_info[fullkey]
            elif i ==3 and j == 3:
                value = datFile[0].header["CDELT3"]
            else:
                value = 0.

            datFile[0].header.update(\
                fullkey, value, after="CTYPE{0}".format(ndim))

    del datFile[0].header["CDELT1"]
    del datFile[0].header["CDELT2"]

    datFile.flush()
    datFile.close()

    os.system("rm {0}.dat {0}.db {0}.fits".format(tmpfileroot))
