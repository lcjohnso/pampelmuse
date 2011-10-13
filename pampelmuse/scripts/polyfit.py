import numpy as np, pyfits, sys
from PyQt4 import QtCore, QtGui

##  required PampelMuse-packages
from pampelmuse.graphics.dynamic_plot_canvas import DynamicPlotCanvas

__author__  = "Sebastian Kamann (skamann@aip.de)"
__version__ = "0.0.0.0"


class PolyFit(QtGui.QMainWindow):

    def __init__(self, fitsFile):

        QtGui.QMainWindow.__init__(self)
        
        self.resize(800,800)
        self.setWindowTitle('Data inspection and polynomial fitting')

        self._addMenu()

        self.frame = QtGui.QWidget(self) ##  will be the central widget 
        vbox = QtGui.QVBoxLayout(self.frame)
   
        txtWidget = QtGui.QWidget() ## Widget containing the history text
        self.txtBrowser = QtGui.QTextBrowser()
        txtFrame  = QtGui.QHBoxLayout(txtWidget) 
        txtFrame.addWidget(self.txtBrowser)

        self.plotHistory = {} ## will contain an array for every row of data
                              ## that was already fitted:
                              ## [<mask>, <polynomial fit parameters>]
                              ## Key is <HDU><row>

        self.fitsFile = fitsFile ## open FITS-file
        self.openFile()

        plotWidget = QtGui.QWidget(self.frame) ## widget containing the plot

        self.dataPlot = DynamicPlotCanvas(parent=plotWidget)
        self.dataPlot.addData(name='data', xData=[], yData=[],
                              ls='-', c='b', marker='None')
        self.dataPlot.addData(name='fit', xData=[], yData=[],
                              ls='--', c='g', lw=1.5, marker='None')
        self.connect(self.dataPlot, QtCore.SIGNAL('rangeSelected'),
                     self.modifyMask)

        plotFrame = QtGui.QHBoxLayout(plotWidget)
        plotFrame.addWidget(self.dataPlot)

        selWidget = QtGui.QWidget() ## Widget containing the buttons

        ## Drop-down menu for selecting the HDU
        self.hduLabel  = QtGui.QLabel('HDU:', self)
        self.hduLabel.setFont(QtGui.QFont('Helvetica', 12))
        self.hduSelect = QtGui.QComboBox()
        for name in self.extNames:
            self.hduSelect.addItem(name)

        ##SpinBox for selecting the data row to plot
        self.rowLabel  = QtGui.QLabel('Row:', self)
        self.rowLabel.setFont(QtGui.QFont('Helvetica', 12))
        self.rowSelect = QtGui.QSpinBox()
        self._updateRowSelect(self.hduSelect.currentText())
        self._updatePlotData() ## Plot first data row

        ## adjust range of data rows if new HDu is selected
        self.connect(self.hduSelect,
                     QtCore.SIGNAL('activated(QString)'),
                     self._updateRowSelect)

        self.plotButton = QtGui.QPushButton("Plot")

        self.connect(self.plotButton, QtCore.SIGNAL('clicked()'),
                     self._updatePlotData)
        

        selFrame = QtGui.QHBoxLayout(selWidget)
        selFrame.addStretch(0)
        selFrame.addWidget(self.hduLabel)
        selFrame.addWidget(self.hduSelect)
        selFrame.addWidget(self.rowLabel)
        selFrame.addWidget(self.rowSelect)
        selFrame.addWidget(self.plotButton)


        fitWidget = QtGui.QWidget() ## Widget to control fitting process
        
        ## Spin box for selecting degree of polynomial
        self.degLabel  = QtGui.QLabel('Order:', self)
        self.degLabel.setFont(QtGui.QFont('Helvetica', 12))
        self.degSelect = QtGui.QSpinBox()
        self.degSelect.setRange(0, 10)
    
        self.fitButton = QtGui.QPushButton("Fit")
        self.connect(self.fitButton, QtCore.SIGNAL('clicked()'),
                     self._fitData)

        fitFrame = QtGui.QHBoxLayout(fitWidget)
        fitFrame.addStretch(0)
        fitFrame.addWidget(self.degLabel)
        fitFrame.addWidget(self.degSelect)
        fitFrame.addWidget(self.fitButton)


        sysWidget = QtGui.QWidget() ## Widget to save results and or quit


        self.savButton = QtGui.QPushButton("Save")
        self.quiButton = QtGui.QPushButton("Quit")

        ## Save results
        self.connect(self.savButton, QtCore.SIGNAL('clicked()'),
                     self.saveResults)

        ## close GUI
        self.connect(self.quiButton, QtCore.SIGNAL('clicked()'),
                     self.close)

        sysFrame = QtGui.QHBoxLayout(sysWidget)
        sysFrame.addStretch(0)
        sysFrame.addWidget(self.savButton)
        sysFrame.addWidget(self.quiButton)


        ## Combine widgets to get overall structure
        vbox.addWidget(plotWidget)
        vbox.addWidget(selWidget)
        vbox.addWidget(fitWidget)
        vbox.addWidget(txtWidget)
        vbox.addWidget(sysWidget)
        self.setCentralWidget(self.frame)



    def _addMenu(self):
        ## File menu
        self.file_menu = self.menuBar().addMenu("&File")

        quit_action = self.__action("&Quit", shortcut="Ctrl+Q", slot=self.close,
                                    tip="Close window")
        self.file_menu.addAction(quit_action)

        ## Options menu
##         self.help_menu = self.menuBar().addMenu("&Options")


        ## Help menu
        self.help_menu = self.menuBar().addMenu("&Help")

        usage_action = self.__action("&Usage", shortcut="Ctrl+H",
                                     slot=self.usage, tip="Usage instructions")
        self.help_menu.addAction(usage_action)


    def __action(self, text, shortcut=None, tip=None, slot=None,
                 signal="triggered()"):

        new_action = QtGui.QAction(text, self)
        if shortcut is not None:
            new_action.setShortcut(shortcut)
        if tip is not None:
            new_action.setStatusTip(tip)
        if slot is not None:
            self.connect(new_action, QtCore.SIGNAL(signal), slot)
        return new_action


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

        self.updateHistory("<b>Opened file:</b> <font color=grey><i>{0}</i></font>".format(self.fitsFile))

        historyString = "<u>Available HDUs:</u><ul>"

        self.extNames = []
        if self.HDUList[0].header['NAXIS'] == 2:
            self.extNames.append(self.HDUList[0].name)
            historyString += "<li>{0}</li>".format(self.HDUList[0].name)
        if len(self.HDUList) > 1:
            for i in xrange(1, len(self.HDUList)):
                if 'ImageHDU' in self.HDUList[i]._summary():
                    self.extNames.append(self.HDUList[i].name)
                    historyString += "<li>{0}</li>".format(self.HDUList[i].name)
        historyString += "</ul><br />"
        self.updateHistory(historyString)
        
        try: ## check if file has been previously analyzed
            self.saveIndex = self.HDUList.index_of('POLYFITS')
        except KeyError:
            self.saveIndex = -1

        if self.saveIndex > -1:
            for i in xrange(self.HDUList[self.saveIndex].data.size):
                hdu = self.HDUList[self.saveIndex].data[i][0]
                row = self.HDUList[self.saveIndex].data[i][1]
                fit = self.HDUList[self.saveIndex].data[i][2]
                self.plotHistory["{0}{1}".format(hdu[:-1], row+1)] = [\
                    np.ones(self.HDUList[hdu].header['NAXIS1'], dtype='bool'),
                    fit
                    ]
            self.updateHistory("File has been analyzed before, read existing results.")


    def _updateRowSelect(self, hduName):

        self.curHDU = str(hduName)
        self.rowSelect.setRange(1, self.HDUList[self.curHDU].header['NAXIS2'])

        
    def _updatePlotData(self):

        self.curRow = self.rowSelect.value()
        self.curKey = "{0}{1}".format(self.curHDU[:-1], self.curRow)

        self.yData = self.HDUList[self.curHDU].section[self.curRow-1]
        self.xData = np.arange(self.yData.size)

        if not self.curKey in self.plotHistory:
            self.dataPlot.updateData(name='fit', xData=[], yData=[])
            self.mask = np.ones((self.HDUList[self.curHDU].header['NAXIS1'],),
                                dtype='bool')
        else:
            yPlotData = np.polyval(\
                self.plotHistory[self.curKey][1], self.xData)
            self.dataPlot.updateData(\
                name='fit', xData=self.xData, yData=yPlotData)
            self.mask = self.plotHistory[self.curKey][0]
        self.plotMask()

        self.dataPlot.updateData(name='data',
                                 xData=self.xData, yData=self.yData)

        ## write out info about selected HDU and data row
        self.updateHistory("")
        try:
            rowName = " - <font color=blue><i>{0}</i><\font>".format(\
                self.HDUList[self.curHDU].header[self.curKey])
        except KeyError:
            rowName = ""
        printString  = "<u>Current selection:</u> "
        printString += "<b>Row <font color=blue>#{0}</font> of ".format(\
            self.curRow)
        printString += "HDU <font color=blue>'{0}'</font><\b>".format(\
            self.curHDU)
        printString += rowName
        self.updateHistory(printString)

        if self.curKey in self.plotHistory:
            self._printFit()


    def _fitData(self):

        n = int(self.degSelect.value())

        polyFit = np.polyfit(x=self.xData[self.mask], y=self.yData[self.mask],
                             deg=n)

        self.dataPlot.updateData(name='fit',
                                 xData=self.xData,
                                 yData=np.polyval(polyFit, self.xData))

        self.plotHistory[self.curKey] = [self.mask, polyFit]
        self._printFit()

    
    def _printFit(self):
        polyFit = self.plotHistory[self.curKey][-1]

        n = polyFit.size-1

        ## write fit results into history widget
        printString  = "<font color=green><i>Polynomial fit</i> "
        printString += "(n={0}):</font>".format(n)
        printString += "<font color=grey> {0:e}".format(polyFit[-1])
        if n > 0:
            for i in xrange(1,n+1):
                if polyFit[-(i+1)] < 0:
                    printString += " -"
                else:
                    printString += " +"
                printString += " {0:e} x<sup>{1}</sup>".format(\
                    abs(polyFit[-(i+1)]), i)
        printString += "</font>"
        self.updateHistory(printString)


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

        nrows=len(self.plotHistory)

        fitHDU = pyfits.new_table(pyfits.ColDefs([\
                pyfits.Column(name='hdu', format='10A'),
                pyfits.Column(name='row', format='I'),
                pyfits.Column(name='fit', format='PD()',array=np.array(\
                            [np.asarray([i]) for i in range(nrows)],
                            dtype=np.object)
                              )
                        ]),
                                  nrows=nrows)
        fitHDU.header.update('EXTNAME', 'POLYFITS')

        self.updateHistory("")

        row = 0
        for name in self.extNames:
            nRows = self.HDUList[name].header['NAXIS2']

            for i in xrange(nRows):
                key = "{0}{1}".format(name[:-1], i+1)
        
                if key in self.plotHistory.keys():
                
                    fitHDU.data.field('hdu')[row] = name
                    fitHDU.data.field('row')[row] = i
                    fitHDU.data.field('fit')[row] = self.plotHistory[key][-1]

                    row += 1

        if self.saveIndex > -1:
            self.HDUList[self.saveIndex].data = fitHDU.data
            self.HDUList[self.saveIndex].verify()
        else:    
            self.HDUList.append(fitHDU)
            self.saveIndex = len(self.HDUList)-1

        self.updateHistory("<font color=green>Saving OK!</font>")


    def close(self):

        self.HDUList.flush()
        QtGui.QMainWindow.close(self)



def polyfit(arguments):

    print ""
    print 21* " "+"***  PolyFit, version {0}  ***".format(__version__)+21*" "
    print ""

    ## List of required definitions: #######################
    prefix   = arguments.prefix                           ##
    ########################################################

    app  = QtGui.QApplication(sys.argv)
    fitter = PolyFit(fitsFile="{0}.prm.fits".format(prefix))
    fitter.show()
    app.exec_()



def main(filename):

    print ""
    print 21* " "+"***  PolyFit, version {0}  ***".format(__version__)+21*" "
    print ""

    app  = QtGui.QApplication(sys.argv)
    fitter = PolyFit(fitsFile=filename)
    fitter.show()
    app.exec_()

if __name__ == "__main__":
    main()
