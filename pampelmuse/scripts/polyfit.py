"""
POLYFIT
=======
The routine is designed to inspect the fit results of PampelMuse. It can also
be used to fit a step-polynomial to a subset of the fit parameters (e.g PSF
parameters or source positions). The changes performed can be stored in the
PRM-file.
The graphical backbone of this routine is the InspectFitData-routine that is
located in the `graphics'-folder.

CHANGELOG
---------
... 2012/01/30, <v.0.2.0.0
2012/01/30, v0.2.0.0, major updates to the InspectFitData-class so that
                      the routine can handle the updated structure of the
                      PRM-files. This class is now located in a separate file
                      in the `graphics'-folder.
2012/01/30, v0.3.0.0, again major updates in InspectFitData-class which was
                      modified so that it now uses the dedicated classes `PSF',
                      `Positions' and `Spectra'. Also included usage of the
                      DOBACKUP-parameter.
2013/05/15, v0.4.0.0, replaced existing GUI with the new PampelMuse GUI. The
                      new GUI also displays the IFU data and allows one to load
                      a reference image.
"""
__author__  = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__version__ = "0.4.0.0"


import logging, sys
from PyQt4 import QtCore, QtGui

##  required PampelMuse-packages
from pampelmuse.graphics.pampelmuse_gui import PampelMuseGui





def polyfit(arguments):

    print ""
    print 21* " "+"***  PolyFit, version {0}  ***".format(__version__)+21*" "
    print ""

    ## List of required definitions: #######################
    prefix   = arguments.prefix                           ##
    refImage = arguments.reference                        ##
##     dobackup = arguments.config["DOBACKUP"]            ##
    ########################################################

    app = QtGui.QApplication(sys.argv)
    ui = PampelMuseGui(fitsFile="{0}.prm.fits".format(prefix), 
                       fitsIfu="{0}.dat.fits".format(prefix), 
                       refImage=refImage)
    ui.show()
    
    sys.exit(app.exec_())

