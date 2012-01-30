"""
INSPECTFITDATA
==============
The InspectFitData-class is the graphical backbone of the routine POLYFIT-
routine of PampelMuse. It provides possibilities to open a PRM-file, inspect
the results, fit polynomials to  the data and store the results in the same
PRM-file.


CHANGELOG
---------
... 2012/01/30, <v.0.2.0.0
2012/12/30, v0.2.0.0, major updates so that the routine can handle the updated
                      structure of the PRM-files.
"""
__author__  = "Sebastian Kamann (skamann@aip.de)"
__version__ = "0.2.0.0"
import logging, sys
import numpy as np, pyfits
from PyQt4 import QtCore, QtGui

##  required PampelMuse-packages
from pampelmuse.graphics.dynamic_plot_canvas import DynamicPlotCanvas
from pampelmuse.functions.step_poly_fit import StepPolyFit



class CalcPixCoord:
    """
    Class to conveniently calculate pixel coordinates using the input
    coordinates and the parameters of the coordinate transformation.
    """
    def __init__(self, parameters):

        self.A = parameters["A"]
        self.B = parameters["B"]
        self.C = parameters["C"]
        self.D = parameters["D"]


    def __call__(self, u, v):

        x = self.A*u + self.B*v + self.C
        y = self.A*v - self.B*u + self.D

        return x, y



class InspectFitData(QtGui.QMainWindow):

    def __init__(self, fitsFile, parent=None):
        
        super(InspectFitData, self).__init__(parent)
        
        self.setWindowTitle('PampelMuse - Parameter inspection and fitting')

        ## Build central widget that will contain the plot
        self.dataPlot = DynamicPlotCanvas()
 
        ## Initialize plot that will contain data.
        self.dataPlot.addData(name='data', xData=[], yData=[],
                              ls='-', c='b', marker='None')

        ## Initialize plot that will contain current fit.
        self.dataPlot.addData(name='fit', xData=[], yData=[],
                              ls='--', c='g', lw=1.5, marker='None')

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


        ## HELP
        usageAction = self.createAction("&Usage", shortcut="Ctrl+H",
                                        slot=self.usage,
                                        tip="Instructions on usage")


        ## Add menu bar to widget
        fileMenu = self.menuBar().addMenu("&File")
        self.addActions(fileMenu, (fileSaveAction, fileQuitAction))

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

        self.plotButton =  QtGui.QPushButton("Plot")

        dataToolbar.addWidget(self.hduLabel)
        dataToolbar.addWidget(self.hduSelect)
        dataToolbar.addWidget(self.rowLabel)
        dataToolbar.addWidget(self.rowSelect)
        dataToolbar.addWidget(self.plotButton)


        fitToolbar = self.addToolBar("Fitting")
        fitToolbar.setObjectName("FitToolBar")

        ## Spin box for selecting degree of polynomial
        self.degLabel  = QtGui.QLabel("<b>Polynomial:</b> Order: ", self)
        self.degLabel.setFont(QtGui.QFont("Helvetica", 12))
        self.degSelect = QtGui.QSpinBox()
        self.degSelect.setRange(0, 10)

        ## Spin box for specifying number of bins
        self.binLabel  = QtGui.QLabel(" - Bins: ", self)
        self.binLabel.setFont(QtGui.QFont("Helvetica", 12))
        self.binSelect = QtGui.QSpinBox()
        self.binSelect.setRange(1, 10)

        self.fitButton = QtGui.QPushButton("Fit")

        fitToolbar.addWidget(self.degLabel)
        fitToolbar.addWidget(self.degSelect)
        fitToolbar.addWidget(self.binLabel)
        fitToolbar.addWidget(self.binSelect)
        fitToolbar.addWidget(self.fitButton)


        self.plotHistory = {} ## will contain the result of the polynomial fit
                              ## for all parameters that wer already fitted:

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

##         self.savButton = QtGui.QPushButton("Save")
##         self.quiButton = QtGui.QPushButton("Quit")
## 
##         ## Save results
##         self.connect(self.savButton, QtCore.SIGNAL('clicked()'),
##                      self.saveResults)
## 
##         ## close GUI
##         self.connect(self.quiButton, QtCore.SIGNAL('clicked()'),
##                      self.close)



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
        helpString += "<li>Press <font color=grey><b>Fit</b></font> to fit the data and display the results.</li>"
        helpString += "<li>Press <font color=grey><b>Save</b></font> to store the results of the fitting process in a FITS-table in extension `POLYFITS' of the provided FITS-file (will be created if not present).</li></ol><br />"

        self.updateHistory("")
        self.updateHistory(helpString)



    def openFile(self):

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

        if not "SOURCES" in self.extNames:
            logging.critical("Input is not a valid PampelMuse PRM-file.")
            logging.critical("No HDU named 'SOURCES' detected.")
        if not "SPECTRA" in self.extNames:
            logging.critical("Input is not a valid PampelMuse PRM-file.")
            logging.critical("No HDU named 'SPECTRA' detected.")

        self.Nsrc = self.HDUList["SOURCES"].data.size
        self.Ndsp = self.HDUList["SPECTRA"].header["NAXIS1"]

        self.name2Row     = {}
        self.sourceList = {"Spectra":QtCore.QStringList(),
                           "Positions":QtCore.QStringList()}

        unres = False
        for i in xrange(self.Nsrc):
            
            if self.HDUList["SOURCES"].data.field("status")[i] in (2,3):
                curID = str(self.HDUList["SOURCES"].data.field("id")[i])
            elif self.HDUList["SOURCES"].data.field("status")[i] == 4:
                curID = "Sky"
            elif self.HDUList["SOURCES"].data.field("status")[i] == 0 \
                    and not unres:
                curID = "Unresolved"
                unres = True
            else:
                continue

            self.sourceList["Spectra"].append(curID)
            if not curID in ["Sky", "Unresolved"]:
                self.sourceList["Positions"].append(curID)
            self.name2Row[curID] = i


        self.hduSelect.addItem("Spectra")
        self.hduMap["Spectra"] = "SPECTRA"

        self.mask = np.ones((self.Ndsp,), dtype=np.bool)


        if "POSPARS" in self.extNames:
            self.hduSelect.addItem("x Positions")
            self.hduMap["x Positions"] = "SOURCES"
            self.hduSelect.addItem("y Positions")
            self.hduMap["y Positions"] = "SOURCES"
            self.hduSelect.addItem("Coord. Trans.")
            self.hduMap["Coord. Trans."] = "POSPARS"

            parameters = {}
            for i in xrange(self.HDUList["POSPARS"].data.size):
                curName = self.HDUList["POSPARS"].data.field("name")[i]
                self.name2Row[curName] = i
                parameters[curName] \
                    = self.HDUList["POSPARS"].data.field("value")[i]

            self.coordCalculator = CalcPixCoord(parameters)

            if "POSITIONS" in self.extNames:
                self.posFits      = self.HDUList["POSITIONS"].data
                self.posFitsExist = True
            else:
                self.posFitsExist = False
                self.posFits      = np.zeros((2, self.Ndsp, self.Nsrc),
                                             dtype=np.float32) 


        if "PSFPARS" in self.extNames:
            self.hduSelect.addItem("PSF")
            self.hduMap["PSF"] = "PSFPARS"

            for i in xrange(self.HDUList["PSFPARS"].data.size):
                curName = self.HDUList["PSFPARS"].data.field("name")[i]
                self.name2Row[curName] = i

        self._updateRowSelect(self.hduSelect.currentText())



    def _updateRowSelect(self, name):

        self.curHDU  = self.hduMap[str(name)]
        self.curProp = str(name)

        self.rowSelect.clear()        
        if name in ["Coord. Trans.", "PSF"]:
            self.rowSelect.addItems(\
                self.HDUList[self.curHDU].data.field("name"))
        elif name == "Spectra":
            self.rowSelect.addItems(self.sourceList["Spectra"])
        elif name in ["x Positions", "y Positions"]:
            self.rowSelect.addItems(self.sourceList["Positions"])
        


    def _updatePlotData(self):

        self.curRowName = str(self.rowSelect.currentText())
        self.curRow     = self.name2Row[self.curRowName]
        self.yFit       = None
        
        if self.curProp in ["Coord. Trans.", "PSF"]:
            self.yData \
                = self.HDUList[self.curHDU].data.field("value")[self.curRow]

            if self.HDUList[self.curHDU].data.field("isfitted")[self.curRow]:
                self.yFit = self.HDUList[self.curHDU].data.field(\
                    "polyfit")[self.curRow]
            
        elif self.curProp == "Spectra":
            self.yData = self.HDUList["SPECTRA"].section[\
                self.HDUList["SOURCES"].data.field("specrow")[self.curRow]]

        elif self.curProp in ["x Positions", "y Positions"]:
            x_c, y_c = self.coordCalculator(\
                self.HDUList["SOURCES"].data.field("x")[self.curRow],
                self.HDUList["SOURCES"].data.field("y")[self.curRow])
            if self.curProp == "x Positions":
                self.yData = x_c
                if self.posFitsExist:
                    self.yFit = self.posFits[0,:,self.curRow]
            else:
                self.yData = y_c
                if self.posFitsExist:
                    self.yFit = self.posFits[1,:,self.curRow]
            
        self.xData = np.arange(self.yData.size)

        self.dataPlot.updateData(\
            name='data', xData=self.xData, yData=self.yData)

        if self.yFit != None:
            self.dataPlot.updateData(\
                name='fit', xData=self.xData, yData=self.yFit)
        else:
            self.dataPlot.updateData(name='fit', xData=[], yData=[])

        self.plotMask()

        ## write out info about selected HDU and data row
        self.updateHistory("")
        rowName = " - <font color=blue><i>{0}</i><\font>".format(\
            self.curRowName)
        printString  = "<u>Current selection:</u> "
        printString += "<b>Row <font color=blue>#{0}</font> of ".format(\
            self.curRow)
        printString += "HDU <font color=blue>'{0}'</font><\b>".format(\
            self.curHDU)
        printString += rowName
        self.updateHistory(printString)


    def _fitData(self):

        ## if no data has been selected/plotted:
        if not "yData" in dir(self):
            logging.error("Please select data to fit first.")
            return

        order = int(self.degSelect.value())
        bins  = int(self.binSelect.value())

        self.yFit = StepPolyFit(self.yData, order=order, bins=bins,
                                mask=self.mask)

        self.dataPlot.updateData(name='fit', xData=self.xData, yData=self.yFit)


        if self.curProp in ["Coord. Trans.", "PSF"]:
            self.HDUList[self.curHDU].data.field("polyfit")[self.curRow] \
                = self.yFit
            self.HDUList[self.curHDU].data.field("isfitted")[self.curRow] = 1

        elif self.curProp in ["x Positions", "y Positions"]:
            for i in xrange(self.posFits.shape[2]):
                if self.HDUList["SOURCES"].data.field("status")[i] < 4:
                    x_i, y_i = self.coordCalculator(\
                        self.HDUList["SOURCES"].data.field("x")[i],
                        self.HDUList["SOURCES"].data.field("y")[i])
                    self.posFits[0,:,i] = StepPolyFit(\
                        x_i, order=order, bins=bins, mask=self.mask)
                    self.posFits[1,:,i] = StepPolyFit(\
                        y_i, order=order, bins=bins, mask=self.mask)
            self.posFitsExist = True
                
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

        iMin = max(0, int(limits[0]))
        iMax = min(int(limits[1])+1, self.mask.shape[0])

        self.mask[iMin:iMax] = (mask==False)

        self.plotMask()



    def plotMask(self):
        searchMask = np.diff(np.r_[True, self.mask, True])
        maskStart  = np.where(searchMask==-1)[0]
        maskEnd    = np.where(searchMask==1)[0]

        ## plot masked areas
        for i in xrange(maskStart.size):
            if "mask{0}".format(i) in self.dataPlot.thePlots.keys():
                self.dataPlot.updateData(\
                    name="mask{0}".format(i),
                    xData=self.xData[maskStart[i]:maskEnd[i]],
                    yData=self.yData[maskStart[i]:maskEnd[i]])
        
            else:
                self.dataPlot.addData(\
                    name="mask{0}".format(i),
                    xData=self.xData[maskStart[i]:maskEnd[i]],
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

##         nrows=len(self.plotHistory)
## 
##         fitHDU = pyfits.new_table(pyfits.ColDefs([\
##                 pyfits.Column(name='hdu', format='10A'),
##                 pyfits.Column(name='row', format='I'),
##                 pyfits.Column(name='fit', format='PD()',array=np.array(\
##                             [np.asarray([i]) for i in range(nrows)],
##                             dtype=np.object)
##                               )
##                 ]),
##                                   nrows=nrows)
##         fitHDU.header.update('EXTNAME', 'POLYFITS')
## 
##         self.updateHistory("")
## 
##         row = 0
##         for name in self.extNames:
##             nRows = self.HDUList[name].header['NAXIS2']
## 
##             for i in xrange(nRows):
##                 key = "{0}{1}".format(name[:-1], i+1)
##         
##                 if key in self.plotHistory.keys():
##                 
##                     fitHDU.data.field('hdu')[row] = name
##                     fitHDU.data.field('row')[row] = i
##                     fitHDU.data.field('fit')[row] = self.plotHistory[key][-1]
## 
##                     row += 1
## 
##         if self.saveIndex > -1:
##             self.HDUList[self.saveIndex].data = fitHDU.data
##             self.HDUList[self.saveIndex].verify()
##         else:    
##             self.HDUList.append(fitHDU)
##             self.saveIndex = len(self.HDUList)-1

        if self.posFitsExist:
            positionsHDU = pyfits.ImageHDU(self.posFits)
            positionsHDU.header.update("EXTNAME", "POSITIONS")
            try:
                self.HDUList["POSITIONS"] = positionsHDU
            except KeyError:
                self.HDUList.insert(3, positionsHDU)
            
        self.HDUList.flush()
        self.updateHistory("<font color=green>Saving OK!</font>")


    def close(self):

        self.HDUList.close()
        QtGui.QMainWindow.close(self)
