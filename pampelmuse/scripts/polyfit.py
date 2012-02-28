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
"""
__author__  = "Sebastian Kamann (skamann@aip.de)"
__version__ = "0.3.0.0"


import logging, sys
from PyQt4 import QtGui

##  required PampelMuse-packages
from pampelmuse.graphics.inspect_fit_data import InspectFitData




def polyfit(arguments):

    print ""
    print 21* " "+"***  PolyFit, version {0}  ***".format(__version__)+21*" "
    print ""

    ## List of required definitions: #######################
    prefix   = arguments.prefix                           ##
    dobackup = arguments.config["DOBACKUP"]               ##
    ########################################################

    app  = QtGui.QApplication(sys.argv)
    fitter = InspectFitData(fitsFile="{0}.prm.fits".format(prefix),
                            dobackup=dobackup)
    fitter.show()
    app.exec_()



## def main(filename):
## 
##     print ""
##     print 21* " "+"***  PolyFit, version {0}  ***".format(__version__)+21*" "
##     print ""
## 
##     app  = QtGui.QApplication(sys.argv)
##     fitter = PolyFit(fitsFile=filename)
##     fitter.show()
##     app.exec_()
## 
## if __name__ == "__main__":
##     main()
