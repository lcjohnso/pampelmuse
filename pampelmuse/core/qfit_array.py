"""
qfit_array.py
=============

Provides
--------
pampelmuse.core.qfit_array.QStarFit

Purpose
-------
The class 'QStarFit' is an extension of the class pampelmuse.core.fit_arrray.StarFit
that implements the usage of a GUI using PyQt4 and matplotlib. It is only available
if those modules exist on the system. The bulk of its functionality is the same as
in StarFit, so check the documentation of that class for further reference.

Latest SVN release
------------------
232, 2015/01/30
"""
__author__ = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__revision__ = 232

import logging

import numpy as np


# required PampelMuse-routines
from pampelmuse.core.fit_array import StarFit
from pampelmuse.graphics.fitting_gui import FittingGui


class QStarFit(StarFit, FittingGui):
    """
    The class pampelMuse.core.qfit_array.QStarFit extends the functionality
    of the class pampelmuse.core.fit_array.StarFit by a loading a GUI that
    displays the current results of the analysis (by showing the layer that
    is currently being analysed and the fit residuals for that layer). The
    GUI also allows the user to pause the fit after a layer has been analysed.
    """
    def __init__(self, **kwargs):
        """
        Initialize an instance of the class pampelmuse.core.qfit_array.QStarFit
        
        Parameters
        ----------
        All keyword arguments parsed to this method will be carried on to the
        initialization of the super-class pampelmuse.core.fit_array.StarFit.
        Consult the documentation of that class for the supported arguments.
        """

        super(QStarFit, self).__init__(**kwargs)

        self.BUSY = False  # possible states of the fitting loop
        self.FINISHED = False  # not defined in FittingGui-class


    def __call__(self, jumpstart=False, **kwargs):
        """
        Execute the analysis of the IFS data.

        Parameters
        ----------
        jumpStart : bool, optional
            Flag indicating whether the fit should  start immediately or only
            after the user presses the respective button on the GUI.
        kwargs
            All other keyword arguments parsed to this method will be carried
            on to the execution method of the super-class
            pampelmuse.core.fit_array.StarFit.
            Consult the documentation of that class for the supported
            arguments.
        """
        super(QStarFit, self).__call__(**kwargs)

        # start GUI that controls fitting process
        super(QStarFit, self).show()

        if jumpstart:
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

        self.jj = (self.specPix >> 1) - 1 - int(self.i / 2) + self.i * np.mod(self.i, 2) + np.mod(self.specPix, 2)
        # jj is a counter that starts at the central spectral pixel and proceeds simultaneously to the red and blue end.

        if self.i == self.specPix:  # if data-cube is entirely analysed
            self.timer.stop()
            self.save = True
            self.close()
        else:  # update information in GUI
            self.prefix_string = "[{0:0{1}d}/{2}]".format(self.i, int(np.log10(self.specPix)) + 1, self.specPix)
            self.sBar.showMessage("{0} Analyzing layer #{1}".format(self.prefix_string, self.jj))

            self.iter_fit_layer()

            self.i += 1

    def iter_fit_layer(self):
        """
        The method is called for every layer of the IFS data by the method
        QStarFit.timeout(). It is basically a wrapper that calls all the
        necessary routines required to analyze an individual layer of the
        data. It is also used to load the data section from the HDUs that is
        currently analysed, update the information in the GUI and calculate
        the uncertainties associated with the fluces once the analysis is
        completed.
        """
        # change state of routine to indicate a fit is carried out
        self.BUSY = True

        self.sBar.showMessage("{0} Fitting fluxes".format(self.prefix_string))

        super(QStarFit, self).iter_fit_layer()

        # plot image that has been fitted
        self.dataCanvas.addImage(self.data.reshape(self.spatShape))

        # create 2dim. residuals for plotting
        if self.hasMask:
            resi1d = np.zeros((self.spatPix,), dtype='float32')
            resi1d[self.pixMap > -1] = self.current
            self.resi2d = (resi1d * self.sigma).reshape(self.spatShape)
        else:
            self.resi2d = (self.current * self.sigma).reshape(self.spatShape)

        self.resiCanvas.addImage(self.resi2d)

        self.pBar.setValue(100. * float(self.i + 1) / self.specPix)

        self.BUSY = False  # change status to indicste fit is completed


###############################################################################
#             self.sBar.showMessage(\
#                 "{0} Fitting PSF to Source #{1} ({2}/{3})".format(\
#                     self.prefixString, self.sources.ID[i], j+1,
#                     len(self.sources.psf_indices)))
#             QtGui.QApplication.processEvents()
