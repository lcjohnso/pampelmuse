"""
INSPECTFITDATA
==============
The InspectFitData-class is the graphical backbone of the routine POLYFIT-
routine of PampelMuse. It provides possibilities to open a PRM-file, inspect
the results, fit polynomials to the data and store the results in the same
PRM-file.


CHANGELOG
---------
... 2012/01/30, <v.0.2.0.0
2012/01/30, v0.2.0.0, major updates so that the routine can handle the updated
                      structure of the PRM-files.
2012/02/27, v0.3.0.0, modified the class to make use of new dedicated classes
                      for PSF, positions and spectra.
2012/02/28, v0.3.0.1, small change when saving HDUs to make sure `INPOSITIONS'
                      is inserted after `POSITIONS'.
2012/03/03, v0.3.0.2, added some HIERARCH-keys to the header of the updated
                      PampelMuse PRM-file.
2012/04/13, v0.3.0.3, added Keyborad-shortcuts to "Plot"- and "Fit"-buttons.
2012/05/14, v0.3.1.3, bugfix: previous versions crashed if the results where
                              saved and no source coordinates had been fitted.
2012/07/03, v0.3.2.3, bugfix: previous versions did not properly grab the names
                              of the PSF and coord. trans. parameters (did not
                              use `provideParNames` method).
2012/11/10, v0.4.0.0, included usage of new 'Instruments'-class. Added option
                      to specify the type of polynomial that is used in the
                      fit (step polynomial, Chebyshev)
2013/02/04, v0.5.0.0, now using wavelength as x-axis instead of pixels. Added
                      labels to x- and y-axes. Polynomial fits are now displayed
                      as solid red lines.
2013/03/28, v0.5.0.1, some minor changes.
"""
__author__  = "Sebastian Kamann (skamann@aip.de)"
__version__ = "0.5.0.1"
import datetime, logging, sys
import numpy as np, pyfits
from PyQt4 import QtCore, QtGui

##  required PampelMuse-packages
from pampelmuse.graphics.dynamic_plot_canvas import DynamicPlotCanvas
from pampelmuse.core.psf import PsfAttributes
from pampelmuse.core.sources import Sources


class InspectFitData(QtGui.QMainWindow):

    def __init__(self, fitsFile, parent=None, dobackup=False):
        
        super(InspectFitData, self).__init__(parent)
        
        self.dobackup=dobackup

        self.setWindowTitle('PampelMuse - Parameter inspection and fitting')

        ## Build central widget that will contain the plot
        self.dataPlot = DynamicPlotCanvas()
 
        ## Initialize plot that will contain data.
        self.dataPlot.addData(name='data', xData=[], yData=[],
                              ls='-', c='b', marker='None')

        ## Initialize plot that will contain current fit.
        self.dataPlot.addData(name='fit', xData=[], yData=[],
                              ls='--', c='r', lw=1.8, marker='None')

        plotWidget = QtGui.QWidget()
        plotWidget.setMinimumSize(600,200)
        plotWidget.setContextMenuPolicy(QtCore.Qt.ActionsContextMenu)

        plotFrame = QtGui.QHBoxLayout(plotWidget)
        plotFrame.addWidget(self.dataPlot)
        plotFrame.setAlignment(QtCore.Qt.AlignCenter)

        self.setCentralWidget(plotWidget)


        ## dock widget for history
        self.logWidget = QtGui.QDockWidget("Log of events", self)
        self.logWidget.setObjectName("History Widget")
        self.logWidget.setAllowedAreas(QtCore.Qt.BottomDockWidgetArea)

        self.txtBrowser = QtGui.QTextBrowser()
        self.logWidget.setWidget(self.txtBrowser)

        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.logWidget)


        ## create status bar
        self.status = self.statusBar()
        self.status.setSizeGripEnabled(False)
        self.status.showMessage("Ready", 5000)


        ## creating Actions
        ## FILE
        fileSaveAction = self.createAction(\
            "&Save", slot=self.saveResults, shortcut="Ctrl+S",
            tip="Save results in file")

        fileQuitAction = self.createAction(\
            "&Quit", slot=self.close, shortcut="Ctrl+Q",
            tip="Close window")

        ## EDIT
        fitAllAction = self.createAction(\
            "Fit &all positions?", slot=self.isFitAll, shortcut="Alt+A",
            checkable=True, tip="Fit source coordinates simultaneously?",
            signal="toggled(bool)")
        commonSlopeAction = self.createAction(\
            "Force &common slope", slot=self.isCommonSlope, shortcut="Alt+C",
            checkable=True, tip="Force position fits to all have same slope?",
            signal="toggled(bool)")

        ## HELP
        usageAction = self.createAction("&Usage", shortcut="Ctrl+H",
                                        slot=self.usage,
                                        tip="Instructions on usage")


        ## Add menu bar to widget
        fileMenu = self.menuBar().addMenu("&File")
        self.addActions(fileMenu, (fileSaveAction, fileQuitAction))

        editMenu = self.menuBar().addMenu("&Edit")
        self.addActions(editMenu, (fitAllAction, commonSlopeAction))

        helpMenu = self.menuBar().addMenu("&Help")
        self.addActions(helpMenu, (usageAction,))


        ## Add toolbar to widget
##         fileToolbar = self.addToolBar("File")
##         fileToolbar.setObjectName("FileToolbar")
##         self.addActions(fileToolbar, (fileSaveAction, fileQuitAction))

        dataToolbar = self.addToolBar("Data")
        dataToolbar.setObjectName("DataToolBar")

        ## Label & drop-down menu for selecting the extension
        self.hduLabel  = QtGui.QLabel("<b>Data:</b> ", self)
        self.hduLabel.setFont(QtGui.QFont("Helvetica", 12))

        self.hduSelect = QtGui.QComboBox()

        ##SpinBox for selecting the data row to plot
        self.rowLabel  = QtGui.QLabel(" - ", self)
        self.rowLabel.setFont(QtGui.QFont("Helvetica", 12))

        self.rowSelect = QtGui.QComboBox()

        self.plotButton =  QtGui.QPushButton("&Plot")

        dataToolbar.addWidget(self.hduLabel)
        dataToolbar.addWidget(self.hduSelect)
        dataToolbar.addWidget(self.rowLabel)
        dataToolbar.addWidget(self.rowSelect)
        dataToolbar.addWidget(self.plotButton)


        fitToolbar = self.addToolBar("Fitting")
        fitToolbar.setObjectName("FitToolBar")

        ## Drop-down menu for fitting functions
        self.fctLabel  = QtGui.QLabel("<b>Polynomial:</b> ")
        self.fctLabel.setFont(QtGui.QFont("Helvetica",12))
        self.fctSelect = QtGui.QComboBox()
        self.fctSelect.addItem("Step")
        self.fctSelect.addItem("Chebyshev")
        self.fctSelect.setFont(QtGui.QFont("Helvetica", 12))

        ## Spin box for selecting degree of polynomial
        self.degLabel  = QtGui.QLabel(" - Order: ", self)
        self.degLabel.setFont(QtGui.QFont("Helvetica", 12))
        self.degSelect = QtGui.QSpinBox()
        self.degSelect.setRange(0, 10)

        ## Spin box for specifying number of bins
        self.binLabel  = QtGui.QLabel(" - Bins: ", self)
        self.binLabel.setFont(QtGui.QFont("Helvetica", 12))
        self.binSelect = QtGui.QSpinBox()
        self.binSelect.setRange(1, 10)

        self.fitButton = QtGui.QPushButton("&Fit")

        fitToolbar.addWidget(self.fctLabel)
        fitToolbar.addWidget(self.fctSelect)
        fitToolbar.addWidget(self.degLabel)
        fitToolbar.addWidget(self.degSelect)
        fitToolbar.addWidget(self.binLabel)
        fitToolbar.addWidget(self.binSelect)
        fitToolbar.addWidget(self.fitButton)


        ## Initial status of GUI
        self.fitAll = True
        fitAllAction.setChecked(True) ## default is to fit pos. simultaneously
        self.commonSlope=False
        commonSlopeAction.setChecked(True) ## default is that fits to
                                           ## positions have same slopes

        ## Interaction with data:
        self.hduMap = {}

        ## open FITS-file
        self.fitsFile = fitsFile
        self.openFile()

        ## connect data selection process to masking method
        self.connect(self.dataPlot, QtCore.SIGNAL('rangeSelected'),
                     self.modifyMask)

        ## adjust range of data rows if new HDU is selected
        self.connect(self.hduSelect, QtCore.SIGNAL('activated(QString)'),
                     self._updateRowSelect)

        ## update plot if button is pressed
        self.connect(self.plotButton, QtCore.SIGNAL('clicked()'),
                     self._updatePlotData)
        
        ## update fit if button is pressed
        self.connect(self.fitButton, QtCore.SIGNAL('clicked()'),
                     self._fitData)



    def createAction(self, text, slot=None, shortcut=None, icon=None, tip=None,
                     checkable=False, signal="triggered()"):

        action = QtGui.QAction(text, self)
        if icon is not None:
            action.setIcon(QtGui.QIcon(":/{0}.png".format(icon)))
        if shortcut is not None:
            action.setShortcut(shortcut)
        if tip is not None:
            action.setToolTip(tip)
            action.setStatusTip(tip)
        if slot is not None:
            self.connect(action, QtCore.SIGNAL(signal), slot)
        if checkable:
            action.setCheckable(True)
        return action



    def addActions(self, target, actions):
        for action in actions:
            if action is None:
                target.addSeparator()
            else:
                target.addAction(action)



    def usage(self):
        
        helpString = "<u><b>Instructions:</b></u><ol>"
        helpString += "<li>To inspect or fit a data row, select the apropriate HDU and row and press <font color=grey><b>Plot</b></font>.</li>"
        helpString += "<li>A region can be excluded from a fit by moving the mouse over the corresponding part of the figure while pressing the LEFT button.</li>"
        helpString += "<li>An excluded region can be (re)included by moving the mouse over the corresponding part of the figure while pressing the RIGHT button.</li>"
        helpString += "<li>The order of the polynomial that is fitted to the data can be changed in the field named <font color=grey><b>Order</b></font>.</li>"
        helpString += "<li>The number of polynomials of given order used to fit the data can be changed in the field named <font color=grey><b>Bins</b></font>.</li>"
        helpString += "<li>Press <font color=grey><b>Fit</b></font> to fit the data and display the results.</li>"
        helpString += "<li>To store the current results in the opened FITS-file, select <font color=grey><b>Save</b></font> from the File-Toolbar.</li>"
        helpString += "<li>To quit, select <font color=grey><b>Quit</b></font> from the File-Toolbar.</li></ol><br />"

        self.updateHistory("")
        self.updateHistory(helpString)



    def isFitAll(self, on): ## whether fits to all positions should be carried
                          ## out simultaneously
        self.fitAll = on



    def isCommonSlope(self, on): ## whether fits to all positions should have
                               ## same slope
        self.commonSlope = on



    def openFile(self): ## open FITS-file, read information from various HDUs

        self.HDUList  = pyfits.open(self.fitsFile, mode="update")

        self.updateHistory(\
            "<b>Opened file:</b> <font color=grey><i>{0}</i></font>".format(\
                self.fitsFile))

        historyString = "<u>Detected HDUs:</u><ul>"

        self.extNames = []
        historyString += "<li>{0}</li>".format(self.HDUList[0].name)
        if len(self.HDUList) > 1:
            for i in xrange(1, len(self.HDUList)):
##                 if 'ImageHDU' in self.HDUList[i]._summary():
                self.extNames.append(self.HDUList[i].name)
                historyString += "<li>{0}</li>".format(self.HDUList[i].name)
        historyString += "</ul><br />"
        self.updateHistory(historyString)


        ## Check which instrument is being used
        if "PAMPELMUSE IFU" in self.HDUList[0].header.keys():
            instrument = self.HDUList[0].header["PAMPELMUSE IFU"]
        else:
            instrument = None

        ## Handle source spectra ###############################################
        if not "SOURCES" in self.extNames:
            logging.critical("Input is not a valid PampelMuse PRM-file.")
            logging.critical("No HDU named 'SOURCES' detected.")
        if not "SPECTRA" in self.extNames:
            logging.critical("Input is not a valid PampelMuse PRM-file.")
            logging.critical("No HDU named 'SPECTRA' detected.")

        self.sources = Sources(data=self.HDUList["SOURCES"], hdu=True,
                               instrument=instrument)
        self.sources.loadSpectra(data=self.HDUList["SPECTRA"], hdu=True)

        ## create wavelength array:
        self.wave  = np.arange(self.sources.ndisp,  dtype=np.float64)
        self.wave -= (self.sources.crpix-1)
        self.wave *= self.sources.cdelt
        self.wave += self.sources.crval

        self.sourceList = {}
        self.sourceList["Spectra"] = self.sources.provideNames()

        self.hduSelect.addItem("Spectra")
        self.hduMap["Spectra"] = "SPECTRA"

        ## Handle Source positions #############################################
        ## The routine will search for HDUs named POSPARS, POSITIONS and
        ## INPOSITIONS. Note that since in case POSITIONS and INPOSITIONS
        ## contain similar data, that from POSITIONS is used.
        if "POSPARS" in self.extNames:
            self.sources.readTransHDU(self.HDUList["POSPARS"])

            self.hduSelect.addItem("x Positions")
            self.hduMap["x Positions"] = "POSITIONS"
            self.hduSelect.addItem("y Positions")
            self.hduMap["y Positions"] = "POSITIONS"
            self.hduSelect.addItem("Coord. Trans.")
            self.hduMap["Coord. Trans."] = "POSPARS"

            self.sourceList["Positions"] = self.sources.providePosIDs()
            self.sourceList["Coord. Trans."] = self.sources.provideParNames()

            if "POSITIONS" in self.extNames:
                self.sources.loadPositionsHDU(self.HDUList["POSITIONS"])

            self.sources.loadIFUCoords()

        elif "POSITIONS" in self.extNames:

            self.hduSelect.addItem("x Positions")
            self.hduMap["x Positions"] = "POSITIONS"
            self.hduSelect.addItem("y Positions")
            self.hduMap["y Positions"] = "POSITIONS"

            if "INPOSITIONS" in self.extNames:
                self.sources.loadPositionsHDU(self.HDUList["INPOSITIONS"])
            self.sources.loadPositionsHDU(self.HDUList["POSITIONS"])

            self.sourceList["Positions"] = self.sources.providePosIDs()

            self.sources.loadIFUCoords()

        ## Handle PSF ##########################################################
        if "PSFPARS" in self.extNames:
            self.psf = PsfAttributes()
            self.psf.readFitsHDU(self.HDUList["PSFPARS"])

            self.hduSelect.addItem("PSF")
            self.hduMap["PSF"] = "PSFPARS"

            self.sourceList["PSF"] = self.psf.provideParNames()

        ########################################################################

        self.mask = np.ones((self.sources.ndisp,), dtype=np.bool)

        self._updateRowSelect(self.hduSelect.currentText())



    def _updateRowSelect(self, name):

        self.curHDU  = self.hduMap[str(name)]
        self.curProp = str(name)

        self.rowSelect.clear()  

        if name in ["x Positions", "y Positions"]:
            self.rowSelect.addItems(self.sourceList["Positions"])
        else:
            self.rowSelect.addItems(self.sourceList[str(name)])
              


    def _updatePlotData(self):

        self.curRowName = str(self.rowSelect.currentText())
        
        ## retrieve current data for selected HDU and property
        if self.curProp == "Spectra":
            self.yFit  = None
            self.yData = self.sources.viewSpectrum(name=self.curRowName)
            yLabel = r"$f_\lambda\,[\mathrm{counts}/\mathrm{layer}]$"

        elif self.curProp == "x Positions":
            self.yData, self.yFit \
                = self.sources.getIFUCoords(coord="x", ID=int(self.curRowName))
            yLabel = r"$x\,[\mathrm{spaxel}]$"

        elif self.curProp == "y Positions":
            self.yData, self.yFit \
                = self.sources.getIFUCoords(coord="y", ID=int(self.curRowName))
            yLabel = r"$y\,[\mathrm{spaxel}]$"

        elif self.curProp == "PSF":
            i = self.psf.parameters.index(str(self.curRowName))
            self.yData = self.psf.data[0,i]
            if self.psf.poly[i]: self.yFit = self.psf.data[1,i]
            else: self.yFit = None
            yLabel=None

        elif self.curProp == "Coord. Trans.":
            i = self.sources.parameters.index(str(self.curRowName))
            self.yData = self.sources.ctData[0,i]
            if self.sources.ctPoly[i]: self.yFit = self.sources.ctData[1,i]
            else: self.yFit = None
            yLabel=None

        ## update plot
        self.dataPlot.updateData(name='data',xData=self.wave,yData=self.yData,
                                 xLabel=r"$\lambda\,[\mathrm{\AA}]$",
                                 yLabel=yLabel)

        if self.yFit != None:
            self.dataPlot.updateData(\
                name='fit', xData=self.wave, yData=np.copy(self.yFit))
        else:
            self.dataPlot.updateData(name='fit', xData=[], yData=[])

        self.plotMask()

        ## write out info about selected HDU and data row
        self.updateHistory("")
        rowName = " - <font color=blue><i>{0}</i><\font>".format(\
            self.curRowName)
        printString  = "<u>Current selection:</u> "
        printString += "<b>Row <font color=blue>#{0}</font> of ".format(\
            self.curRowName)
        printString += "HDU <font color=blue>'{0}'</font><\b>".format(\
            self.curHDU)
        printString += rowName
        self.updateHistory(printString)



    def _fitData(self):

        ## if no data has been selected/plotted:
        if not "yData" in dir(self):
            logging.error("Please select data to fit first.")
            return

        function = str(self.fctSelect.currentText())
        order = int(self.degSelect.value())
        bins  = int(self.binSelect.value())

        if self.curProp == "Spectra":
            logging.warning("Fitting spectra is not supported.")
            logging.warning("(and not that useful anyway, ol' goofball!)")
            return

        elif self.curProp == "PSF":
            self.psf.polyFitParameter(name=self.curRowName, order=order,
                                      bins=bins, mask=self.mask,
                                      function=function.lower())
            self._updatePlotData()

        elif self.curProp in ["x Positions", "y Positions"]:

            if self.fitAll: ids=None
            else: ids=int(self.curRowName)

            if self.curProp == "x Positions": coord="x"
            else: coord="y"

            self.sources.fitIFUCoords(\
                ids=ids, coord=coord, order=order, bins=bins, mask=self.mask,
                function=function.lower(), commonSlope=self.commonSlope)
            self._updatePlotData()
                
        elif self.curProp == "Coord. Trans.":
            logging.warning("Fitting the coord. transformation is deprecated.")
            logging.warning("Directly fit IFU  x- & y-coordiates instead.")

##         self._printFit()

    
##     def _printFit(self):
##         polyFit = self.plotHistory[self.curKey][-1]
## 
##         n = polyFit.size-1
## 
##         ## write fit results into history widget
##         printString  = "<font color=green><i>Polynomial fit</i> "
##         printString += "(n={0}):</font>".format(n)
##         printString += "<font color=grey> {0:e}".format(polyFit[-1])
##         if n > 0:
##             for i in xrange(1,n+1):
##                 if polyFit[-(i+1)] < 0:
##                     printString += " -"
##                 else:
##                     printString += " +"
##                 printString += " {0:e} x<sup>{1}</sup>".format(\
##                     abs(polyFit[-(i+1)]), i)
##         printString += "</font>"
##         self.updateHistory(printString)



    def modifyMask(self, limits, mask):
##         print limits, mask
        iMin = np.argmin(abs(self.wave-limits[0])) ## max(0, int(limits[0]))
        iMax = np.argmin(abs(self.wave-limits[1])) ## min(int(limits[1])+1, self.mask.shape[0])
        
        self.mask[iMin:iMax] = (mask==False)

        self.plotMask()



    def plotMask(self):
        searchMask = np.diff(np.r_[1, self.mask, 1])
        maskStart  = np.where(searchMask==-1)[0]
        maskEnd    = np.where(searchMask==1)[0]

        ## plot masked areas
        for i in xrange(maskStart.size):
            if "mask{0}".format(i) in self.dataPlot.thePlots.keys():
                self.dataPlot.updateData(\
                    name="mask{0}".format(i),
                    xData=self.wave[maskStart[i]:maskEnd[i]],
                    yData=self.yData[maskStart[i]:maskEnd[i]])
        
            else:
                self.dataPlot.addData(\
                    name="mask{0}".format(i),
                    xData=self.wave[maskStart[i]:maskEnd[i]],
                    yData=self.yData[maskStart[i]:maskEnd[i]],
                    ls='-', c='w', marker='None', alpha=0.5)

        if maskStart.size == 0:
            i = -1

        i += 1
        ## remove masks that are not in use anymore
        while "mask{0}".format(i) in self.dataPlot.thePlots.keys():
            self.dataPlot.updateData(name="mask{0}".format(i),
                                     xData=[], yData=[])
            i += 1


    def updateHistory(self, string):       
        try:
            text = unicode(string)
            self.txtBrowser.append(string)
        except:
            self.txtBrowser.append("<font color=red>{0} is invalid!</font>"\
                                       .format(string))


    def saveResults(self):

        update=False

        if "POSPARS" in self.extNames or "POSITIONS" in self.extNames:

##             if self.sources.unsavedIFUCoords:
##             ## check if unsaved changes exist

                ## Results of polynomial fit are written to `POSITIONS'-HDU,
                ## Input positions (if any) are saved in `INPOSITIONS'-HDU
                ## (only if backup-flag is set)

            if self.dobackup \
                    and "INPOSITIONS" not in self.extNames \
                    and "POSITIONS" in self.extNames \
                    and not self.sources.verifyTrans:

                logging.info("Creating backup of existing HDU 'POSITIONS'.")
                self.HDUList["POSITIONS"].name = "INPOSITIONS"

                self.dobackup = False

            newPosHDU   = self.sources.makeCoordinatesHDU(polyFit=True)
            
            if newPosHDU != None:
                try: self.HDUList["POSITIONS"] = newPosHDU
                except: self.HDUList.insert(4, newPosHDU)

            update=True


        if "PSFPARS" in self.extNames and self.psf.free.any():
            if self.psf.unsavedPSF:  ## check if unsaved changes exist
                self.HDUList["PSFPARS"] = self.psf.makePSFHDU()

                update = True

        if update: ##  check if any HDU was updated at all
            self.HDUList.flush()
            self.updateHistory("<font color=green>Saving OK!</font>")
        else:
            logging.warning("No changes need to be saved.")



    def close(self):

        self.HDUList[0].header.update(\
            "hierarch PAMPELMUSE POLYFIT UTC",
            datetime.datetime.utcnow().strftime("%y-%m-%d %H:%M:%S"))
        self.HDUList[0].header.update(\
            "hierarch PAMPELMUSE POLYFIT VERSION", __version__)
        self.HDUList.close()
        QtGui.QMainWindow.close(self)
