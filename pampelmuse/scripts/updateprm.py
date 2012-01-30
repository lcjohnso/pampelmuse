"""
UPDATEPRM
=========
The routine is designed to change the structure of a PampelMuse prm-file
so that it is compliant with versions of PampelMuse >= 0.2.0.0
The routine changes the structure of the POSPARS and PSFPARS HDUs from an
image to a binary FITS-table. Those HDUs will also contain columns to
contain the polynomial fits. The POLYFITS HDU (if present) is removed.

CHANGELOG
---------
2012/01/29, v0.0.0.0
2012/01/30, v0.0.1.0, fixed bug that loop-index `i' was re-assigned in loops
                      starting in rows #105 and #148.
                      replace 'PSFPARS' with 'POSPARS' in row #154.
"""
__author__ = "Sebastian Kamann (skamann@aip.de)"
__version__ = "0.0.1.0"

import logging
import numpy as np, pyfits

## Required PampelMuse-routines
## None


def updateprm(arguments):
    
    print ""
    print 21* " "+"***  UpdatePrm, version {0}  ***".format(__version__)+21*" "
    print ""

    ## List of required definitions: #######################
    prefix   = arguments.prefix                           ##

    ## open existing prm-file, check structure
    prmFile = pyfits.open("{0}.prm.fits".format(prefix), mode="update")

    try:
        indexSrc = prmFile.index_of("SOURCES")
    except KeyError:
        logging.critical("Input is not a valid PampelMuse PRM-file.")
        logging.critical("No HDU named 'SOURCES' detected.")

    try:
        indexSrc = prmFile.index_of("SPECTRA")
    except KeyError:
        logging.critical("No HDU named 'SPECTRA' detected.")

    Nsources = prmFile["SOURCES"].data.size
    Nspp     = prmFile["SPECTRA"].header["NAXIS1"]


    ## update header of SOURCES HDU
    prmFile["SOURCES"].header.update("PRMSTRUC", 2)

##     colXfit = pyfits.Column(name='x_isfitted', format='L',
##                             array=np.zeros(Nsources))
##     colXdat = pyfits.Column(name='x_polyfit', format='{0}E'.format(int(Nspp)),
##                             array=np.zeros((Nsources, Nspp)))
##     colYfit = pyfits.Column(name='y_isfitted', format='L',
##                             array=np.zeros(Nsources))
##     colYdat = pyfits.Column(name='y_polyfit', format='{0}E'.format(int(Nspp)),
##                             array=np.zeros((Nsources, Nspp)))
## 
## 
##     newTable = pyfits.new_table([colXfit, colXdat, colYfit, colYdat])
##     fullTable= pyfits.new_table(prmFile["SOURCES"].columns + newTable.columns)
##     fullTable.header.update("PRMSTRUC", 2)
##     fullTable.header.update("EXTNAME", "SOURCES")
##     prmFile["SOURCES"] = fullTable
## 
##     logging.info("Updated HDU 'SOURCES'.")


    ## looking for POLYFITS HDU
    try:
        logging.info("HDU named 'POLYFITS' was found.")
        fitHDU = prmFile["POLYFITS"]
    except KeyError:
        logging.info("No HDU named 'POLYFITS' was found.")
        fitHDU = None
        
    ## working on PSFPARS HDU
    try:
        psfHDU = prmFile["PSFPARS"]
        Nspp   = prmFile["PSFPARS"].header["NAXIS1"]
        dtype  = [("name", "|S10"),
                  ("value", ">f4", (Nspp,)),
                  ("freepar", "i1"),
                  ("isfitted", "i1"),
                  ("polyfit", ">f4", (Nspp,))]
    
        if psfHDU.header["NAXIS2"]  == 4:
            psftype = "moffat"
            outData = np.zeros(4, dtype=dtype)
            outData["name"] = ["beta", "fwhm", "e", "theta"]
        elif psfHDU.header["NAXIS2"] == 3:
            psftype = "gauss"
            outData = np.zeros(3, dtype=dtype)
            outData["name"] = ["fwhm", "e", "theta"]
        else:
            logging.critical(\
                "No. of PSF parameters does not match a known profile")

        for i in xrange(outData.size):
            outData["value"][i] = prmFile["PSFPARS"].data[i]
            if fitHDU != None:
                try:
                    j = np.where((fitHDU.data.field("hdu")=="PSFPARS")\
                                     &(fitHDU.data.field("row")==i))[0][0]
                    polpars = fitHDU.data.field('fit')[j]
                except IndexError:
                    polpars = None
    
                if polpars != None:
                    outData["freepar"][i] = True
                    outData["isfitted"][i] = True
                    outData["polyfit"][i] = np.polyval(polpars, np.arange(Nspp))

        outHDU = pyfits.BinTableHDU(outData)
        outHDU.header.update("EXTNAME", "PSFPARS")
        outHDU.header.update("PSFTYPE", psftype)

        prmFile["PSFPARS"] = outHDU

        logging.info("Updated HDU 'PSFPARS'.")

    except KeyError:
        logging.warn("No HDU named 'PSFPARS' was found.")


    ## working on POSPARS HDU
    try:
        psfHDU = prmFile["POSPARS"]
        Nspp   = prmFile["POSPARS"].header["NAXIS1"]
        dtype  = [("name", "|S10"),
                  ("value", ">f4", (Nspp,)),
                  ("freepar", "i1"),
                  ("isfitted", "i1"),
                  ("polyfit", ">f4", (Nspp,))]
    
        if psfHDU.header["NAXIS2"]  == 4:
            outData = np.zeros(4, dtype=dtype)
            outData["name"] = ["A", "B", "C", "D"]
        else:
            logging.critical("No. of parameters of coord. tranformation not 4.")

        for i in xrange(outData.size):
            outData["value"][i] = prmFile["POSPARS"].data[i]
            if fitHDU != None:
                try:
                    j = np.where((fitHDU.data.field("hdu")=="POSPARS")\
                                     &(fitHDU.data.field("row")==i))[0][0]
                    polpars = fitHDU.data.field('fit')[j]
                except IndexError:
                    polpars = None
    
                if polpars != None:
                    outData["freepar"][i]  = True
                    outData["isfitted"][i] = True
                    outData["polyfit"][i] = np.polyval(polpars, np.arange(Nspp))

        outHDU = pyfits.BinTableHDU(outData)
        outHDU.header.update("EXTNAME", "POSPARS")

        prmFile["POSPARS"] = outHDU
        
        logging.info("Updated HDU 'POSPARS'.")

    except KeyError:
        logging.warn("No HDU named 'POSPARS' was found.")

    if fitHDU != None:
        del prmFile["POLYFITS"]
        logging.info("Deleted HDU 'POLYFITS'.")
        
    prmFile.flush()
