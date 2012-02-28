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
"""
__author__ = "Sebastian Kamann (skamann@aip.de)"
__version__ = "0.1.0.0"

import logging
import numpy as np
from PyQt4 import QtCore


def matchOnID(parent, child, assume_unique=False):

    indices = np.nonzero(np.in1d(parent, child, assume_unique=assume_unique))[0]
    
    return indices




class Spectra:


    def __init__(self, sources, hdu=False):

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

        match    = np.searchsorted(np.arange(self.ncmp), self.specrow)
        self.map = dict(zip(\
                np.arange(self.ncmp),
                [np.nonzero(match==i)[0] for i in xrange(self.ncmp)]))

        self.weights = {}
        for row, ind in self.map.iteritems():
            self.weights[row] = \
                self.mag[ind] + 2.5*np.log10((10**(-0.4*self.mag[ind])).sum())

            ## only logging
            logging.debug("{0:10d}".format(int(row)))
            j = 0
            logString = " "
            for i in ind:
                logString += "  {0} ({1:5.2f})".format(\
                    int(self.ID[i]), float(self.weights[row][j]))
                j +=1
                if not np.mod(j, 5):
                    logging.debug(logString)
                    logString = " "
            if np.mod(j, 5)>0: logging.debug(logString)
        
        self.hasUnres = (self.status==0).any()
        


    def loadSpectra(self, data, sigma=None, hdu=False, crval=1., cdelt=1.,
                    crpix=1, ctype="AWAV"):
                    
        self.crval = crval
        self.cdelt = cdelt
        self.crpix = crpix
        self.ctype = ctype

        if hdu:
            self.spectra = data.data
            if "CRVAL1" in data.header: self.crval = data.header["CRVAL1"]
            if "CDELT1" in data.header: self.cdelt = data.header["CDELT1"]
            if "CRPIX1" in data.header: self.crpix = data.header["CRPIX1"]
            if "CTYPE1" in data.header: self.ctype = data.header["CTYPE1"]
        else:
            self.spectra = data

        if self.spectra.shape[0] != self.ncmp:
            logging.error("Number of provided spectra and components differ.")

        if hdu and sigma != None:
            self.sigma = sigma.data
        else:
            self.sigma = sigma

        if self.sigma!=None:
            if self.sigma.shape != self.spectra.shape:
                logging.error("Spectra and sigma arrays have different shapes.")

        self.ndisp = self.spectra.shape[1]

        

    def provideNames(self):

        ownSpec = np.in1d(self.status, np.arange(2,4))

        self.names   = QtCore.QStringList(self.ID[ownSpec].astype("|S6"))
        self.nameMap = dict((zip(self.ID[ownSpec].astype("|S6"),
                                 self.specrow[ownSpec])))
        if (self.status==4).any():
            self.names.append("Sky")
            self.nameMap["Sky"] = self.specrow[np.where(self.status==4)[0][0]]
        if self.hasUnres:
            self.names.append("Unresolved")
            self.nameMap["Unresolved"]\
                = self.specrow[np.where(self.status==0)[0][0]]
        
        return self.names


    def viewSpectrum(self, name=None, ID=None):
        
        if name==None and ID==None:
            logging.error("Either name or ID must be provided.")
            return

        if name != None:
            return self.spectra[self.nameMap[name]]
        else:
            return self.spectra[self.specrow[np.where(self.ID==ID)[0][0]]]
