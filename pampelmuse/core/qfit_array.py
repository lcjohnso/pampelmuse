"""
The class `QStarFit` is a wrapper around the class `StarFit` that implements
the usage of a GUI using PyQt4 and matplotlib. It is only available if those
libraries exist on the system. The bulk of its functionality is the same as
that of `StarFit`, so check the documentation of that class for further
reference.

CHANGELOG
---------
2012/03/23, v0.5.0.0, created own file for class `QStarFit` because of issues
                      when loading StarFit without PyQt4 and/or matplotlib-
                      support.

"""
__author__  = "Sebastian Kamann (skamann@aip.de)"
__version__ = "0.5.0.0"

import logging
import numpy as np
from PyQt4 import QtCore, QtGui

##  required PampelMuse-routines
from pampelmuse.core.fit_array import StarFit
from pampelmuse.graphics.fitting_gui import FittingGui




class QStarFit(StarFit, FittingGui):


    def __init__(self, **kwargs):

        super(QStarFit, self).__init__(**kwargs)

        self.BUSY     = False ## possible states of the fitting loop 
        self.FINISHED = False ## not defined in FittingGui-class



    def __call__(self, jumpStart=False, **kwargs):

        super(QStarFit, self).__call__(**kwargs)

        ## start GUI that controls fitting process
        super(QStarFit, self).show()

        if jumpStart:
            super(QStarFit, self).startIt()



    def timeout(self):
        """
        Method is called every time the `FittingGui`-instance encounters a
        `timeout`. Depending on the current status of the fitting routine, 
        the method calls the appropriate methods to analyze it.
        """
        if not self.STARTED:
            return

        if self.BUSY:
            return

        if self.ABORTED:
            logging.warning("Fit stopped on user request.")
            self.timer.stop()
            self.close()
    
        if self.PAUSED:
            self.sBar.showMessage("PAUSED")
            return

        self.jj = (self.specPix>>1) - 1 - int(self.i/2) \
            + self.i*np.mod(self.i, 2) + np.mod(self.specPix, 2)
        ## jj is a counter that starts at the central spectral pixel
        ## and proceeds simultaneoulsy to the red and blue end.

        if self.i == self.specPix: ## if data-cube is entirely analysed
            self.timer.stop()
            self.save = True
            self.close()
        else: ## update information in GUI
            self.prefixString = "[{0:0{1}d}/{2}]".format(\
                self.i, int(np.log10(self.specPix))+1, self.specPix)
            self.sBar.showMessage("{0} Analyzing slice #{1}".format(\
                    self.prefixString, self.jj))

            self.iterFitImage()

            self.i += 1



    def iterFitImage(self):
        """
        The method is called for every slice of the datacube by the `timeout`-
        method. It is basically a wrapper that calls all the necessary
        routines required to analyze an individual slice from the datacube.
        It is also used to load the data section from the HDUs that is
        currently analysed, update the information in the GUI and calculate
        the uncertainties associated with the fluces once the analysis is
        completed.
        """

        ## change state of routine to indicate a fit is carried out
        self.BUSY = True

        self.sBar.showMessage("{0} Fitting fluxes".format(self.prefixString))

        super(QStarFit, self).iterFitImage()

        ## plot image that has been fitted
        self.dataCanvas.addImage(self.data.reshape(self.spatShape))

        ## create 2dim. residuals for plotting
        if self.hasMask:
            resi1d = np.zeros((self.spatPix,), dtype='float32')
            resi1d[self.pixMap>-1] = self.current
            self.resi2d = (resi1d*self.sigma).reshape(self.spatShape)
        else:
            self.resi2d = (self.current*self.sigma).reshape(self.spatShape)
            
        self.resiCanvas.addImage(self.resi2d)

        self.pBar.setValue(100.*float(self.i+1)/self.specPix)

        self.BUSY = False ## change status to indicste fit is completed


###############################################################################
##             self.sBar.showMessage(\
##                 "{0} Fitting PSF to Source #{1} ({2}/{3})".format(\
##                     self.prefixString, self.sources.ID[i], j+1,
##                     len(self.sources.psfIndices)))
##             QtGui.QApplication.processEvents()
