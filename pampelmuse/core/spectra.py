"""
SPECTRA
=======
The routine is designed to handle a set of spectra used by PampelMuse. The
methods deifned in the `Spectra'-class are designed to open a set of spectra
form a FITS image and provide the InspectFitData-routine of PampelMuse (the
backbone of the POLYFIT-script) with all  necessary information about the
spectra.

CHANGELOG
---------
2012/02/28, v0.1.0.0
2012/02/28, v0.1.0.1, some minor changing about format of DEBUG-messages.
2012/03/01, v0.1.0.2, renamed property `spectra' to `data'. Simplified
                      access to unresolved and sky spectra.
                      Added methods to access and update data, added a method
                      to write the current data into a FITS-ImageHDU.
2012/03/02, v0.1.1.2, added names for HDUs containing spectra and sigma spc.
2012/03/07, v0.2.0.0, implemented the methods from the Spectra-class in the
                      Sources-class. Usage of this routine is now deprecated.
"""
__author__ = "Sebastian Kamann (skamann@aip.de)"
__version__ = "0.2.0.0"

import logging
import numpy as np, pyfits
from PyQt4 import QtCore


def matchOnID(parent, child, assume_unique=False):
    """
    For each element in `children', return indix of element in `parent'.

    Parameters
    ----------
    parent : array_like, 1dim.
        The list that that is scanned
    child : array_like, 1dim.
        The list containing the entries that `parent' is scanned for.
    assume_unique : boolean
        Whether the entries of the two lists are unique.

    Returns
    -------
    indices : nd_array
        The index of every `child'-entry in the `parent' list (if found).
    """
##     indices \
##         = np.nonzero(np.in1d(parent, child, assume_unique=assume_unique))[0]
##     return indices

    return np.nonzero((child[:,np.newaxis]-parent)==0)[1]



class Spectra:


    def __init__(self, sources, hdu=False):
        logging.error("Usage of Spectra-class is deprecated.")
        logging.error("Sources-class should  be used instead.")

        if hdu:
            logging.info("Initializing Spectra instance from HDU.")
            sources = np.array(sources.data,
                               dtype=sources.data.dtype)
        else:
            logging.info("Initializing Spectra instance.")


        for name in ["id", "status"]:
            if not name in sources.dtype.names:
                logging.critical(\
                    "No field `{0}' found in provided list.".format(name))

        self.ID     = sources[["id"]].view(sources["id"].dtype).reshape(-1,)
        self.status = sources[["status"]].view(sources["status"].dtype)\
            .reshape(-1,)
        self.psf    = np.where(self.status==3)[0]  ## indices of PSF stars

        
        if not "specrow" in sources.dtype.names:
            if (self.status==0).any() or (self.status==1).any():
                logging.critical("No field `specrow' found in provided list.")
            else:
                logging.warning("No field `specrow' found in provided list.")
                logging.warning("Assuming spectra are ordered wrt. sources.")
            self.specrow = np.arange(self.ID.size, dtype=np.int32)
        else:
            self.specrow = sources[["specrow"]].view(sources["specrow"].dtype)\
                .reshape(-1,)


        if not "mag" in sources.dtype.names:
            if (self.status==0).any() or (self.status==1).any():
                logging.critical("No field `mag' found in provided list.")
        else:
            self.mag = sources[["mag"]].view(sources["mag"].dtype).reshape(-1,)

        self.ncmp = self.specrow.max()+1

        ## create dictionary with components:  specrow as key, indices as values
        logging.debug("Structure of source catalog:")
        logging.debug("Component:   IDs (& mags):")

        ## `map' is a dictionary mapping the `specrow' (keys) to the
        ## index/indices (values) in the `sources'-table
        match    = np.searchsorted(np.arange(self.ncmp), self.specrow)
        self.map = dict(zip(\
                np.arange(self.ncmp),
                [np.nonzero(match==i)[0] for i in xrange(self.ncmp)]))

        ## `weights' is a dictionary mapping the `specrow' (keys) to the
        ## relative magnitudes (values) of its component(s)
        self.weights = {}
        for row, ind in self.map.iteritems():
            self.weights[row] = \
                self.mag[ind] + 2.5*np.log10((10**(-0.4*self.mag[ind])).sum())

            ## only logging
            logString = "{0:10d}".format(int(row))
            j = 0

            for i in ind:
                logString += "   {0} ({1:.2f})".format(\
                    int(self.ID[i]), float(self.weights[row][j]))
                j +=1
                if not np.mod(j, 2):
                    logging.debug(logString)
                    logString = 10*" "
            if np.mod(j, 2)>0: logging.debug(logString)
        
        self.unResRow = None
        self.skyRow   = None

        self.hasUnRes = (self.status==0).any()
        if self.hasUnRes:
            self.unResRow =self.specrow[np.where(self.status==0)[0][0]]

        self.hasSky = (self.status==4).any()
        if self.hasSky:
            self.skyRow = self.specrow[np.where(self.status==4)[0][0]]



    def loadSpectra(self, data, sigma=None, hdu=False, crval=1., cdelt=1.,
                    crpix=1, ctype="AWAV"):
                    
        self.crval = crval
        self.cdelt = cdelt
        self.crpix = crpix
        self.ctype = ctype

        if hdu:
            self.data = data.data
            if "CRVAL1" in data.header: self.crval = data.header["CRVAL1"]
            if "CDELT1" in data.header: self.cdelt = data.header["CDELT1"]
            if "CRPIX1" in data.header: self.crpix = data.header["CRPIX1"]
            if "CTYPE1" in data.header: self.ctype = data.header["CTYPE1"]
        else:
            self.data = data

        if self.data.shape[0] != self.ncmp:
            logging.error("Number of provided spectra and components differ.")

        if hdu and sigma != None:
            self.sigma = sigma.data
        else:
            self.sigma = sigma

        if self.sigma!=None:
            if self.sigma.shape != self.data.shape:
                logging.error("Spectra and sigma arrays have different shapes.")

        self.ndisp = self.data.shape[1]

        

    def provideNames(self):

        ownSpec = np.in1d(self.status, np.arange(2,4))

        self.names   = QtCore.QStringList(self.ID[ownSpec].astype("|S6"))
        self.nameMap = dict((zip(self.ID[ownSpec].astype("|S6"),
                                 self.specrow[ownSpec])))
        if self.hasSky:
            self.names.append("Sky")
            self.nameMap["Sky"] = self.skyRow
        if self.hasUnRes:
            self.names.append("Unresolved")
            self.nameMap["Unresolved"] = self.unResRow
        
        return self.names



    def viewSpectrum(self, name=None, ID=None):
        
        if name==None and ID==None:
            logging.error("Either name or ID must be provided.")
            return

        if name != None:
            return self.data[self.nameMap[name]]
        else:
            return self.data[self.specrow[np.where(self.ID==ID)[0][0]]]



    def viewData(self, index, ID=None):
        
        index = np.atleast_1d(index)
        
        if ID == None:
            return self.data[:,index]
        else:
            ID = np.atleast_1d(ID)
            select = self.specrow[matchOnID(self.ID, ID)]
            if select.size < ID.size:
                logging.error("Cannot return data: Invalid ID(s) provided.")
                return None
            return self.data[select[:,np.newaxis], index]



    def addData(self, data, index, ID=None):

        index = np.atleast_1d(index)
        
        if ID == None:
            self.data[:,index] += data[:,np.newaxis]
        else:
            ID = np.atleast_1d(ID)
            select = self.specrow[matchOnID(self.ID, ID)]
            if select.size < ID.size:
                logging.error("Cannot add to data: Invalid ID(s) provided.")
                return
            self.data[select[:,np.newaxis], index] += data




    def putSigma(self, sigma, index, ID=None):

        if self.sigma==None:
            self.sigma = np.zeros(self.data.shape, dtype=self.data.dtype)
            
        index = np.atleast_1d(index)
        
        if ID == None:
             self.sigma[:,index] = sigma
        else:
            ID = np.atleast_1d(ID)
            select = self.specrow[matchOnID(self.ID, ID)]
            if select.size < ID.size:
                logging.error("Cannot put sigma: Invalid ID(s) provided.")
                return
            self.sigma[select[:,np.newaxis], index] = sigma



    def makeHDU(self, makeSigma=True):

        outHDU = pyfits.ImageHDU(self.data, name="SPECTRA")
        outHDU.header.update("CRPIX1", self.crpix)
        outHDU.header.update("CRVAL1", self.crval)
        outHDU.header.update("CDELT1", self.cdelt)
        outHDU.header.update("CTYPE1", self.ctype)

        if makeSigma:
            if self.sigma != None:
                sigmaHDU = pyfits.ImageHDU(self.sigma, name="SIGMASPC")
                sigmaHDU.header.update("CRPIX1", self.crpix)
                sigmaHDU.header.update("CRVAL1", self.crval)
                sigmaHDU.header.update("CDELT1", self.cdelt)
                sigmaHDU.header.update("CTYPE1", self.ctype) 
                return outHDU, sigmaHDU
            else:
                logging.error("Cannot create Sigma HDU: No sigma data")
                return outHDU, None
        else:
            return outHDU
