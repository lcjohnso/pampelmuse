import matplotlib.pyplot, sys
import numpy as np
from PyQt4 import QtCore, QtGui

## Required PampelMuse-packages
from pampelmuse.graphics.dynamic_ima_canvas import DynamicImaCanvas



class PickCanvas(DynamicImaCanvas):

##     def selectBright(self, maxmag):
##         self.allData['x'][self.allData['mag']>maxmag] = np.ma.masked
##         self.allDataPlot.set_xdata(self.allData['x'])
## Does not work like this - Method to update data of scatterplot?     

    def startPicking(self):
        self.unpick=False
        self.cidPick = self.fig.canvas.mpl_connect('pick_event', self._picker)

    
    def startUnpicking(self):
        self.unpick=True
        self.cidPick = self.fig.canvas.mpl_connect('pick_event', self._picker)


    def _picker(self, event):

        if len(event.ind) != 1:
            return

        i = event.ind[0]

        if not i in self.pickData:
            if not self.unpick:
                self.edgecolors[:] = self.basecolor
                self.edgecolors[i] = 1-self.basecolor
                self.emit(QtCore.SIGNAL("pickedSource"), event.ind[0])

        else:
            if self.unpick:
                self.emit(QtCore.SIGNAL("unpickedSource"), event.ind[0])
            
        self.allDataPlot.set_edgecolors(self.edgecolors)
        self.draw()


    def plotPicked(self):
        
        self.edgecolors[:] = self.basecolor
        self.facecolors = self.allData.size*['none']
        for i in self.pickData:
            self.facecolors[i] = self.basecolor
        
        self.allDataPlot.set_edgecolors(self.edgecolors)
        self.allDataPlot.set_facecolors(self.facecolors)
        self.draw()

        
    def stopPicking(self):

        self.edgecolors[:] = self.basecolor
        self.allDataPlot.set_edgecolors(self.edgecolors)
        self.draw()

        self.fig.canvas.mpl_disconnect(self.cidPick)
 

    def stopUnpicking(self):
        
        self.fig.canvas.mpl_disconnect(self.cidPick)




class GrabCanvas(DynamicImaCanvas):

    def startGrabbing(self):

        self.cidGrab = self.fig.canvas.mpl_connect('button_press_event',
                                                   self._grabber)


    def _grabber(self, event):

        if event.inaxes != self.axes:
            return

        ## use position of mouse click as initial guess, calculate first
        ## order moments and use them as source positions

        if self.calcMoments:
            rad = 2

            x_init = int(round(event.xdata))
            y_init = int(round(event.ydata))

            Y, X = np.indices((2*rad+1,2*rad+1), dtype='int32')
            Y += (y_init-rad)
            X += (x_init-rad)

            _valid = (X>=0)&(X<self.image.shape[1])&(Y>=0)\
                &(Y<self.image.shape[0])

            tmpIma  = self.image[Y[_valid],X[_valid]]
            tmpIma -= tmpIma.min()

            total = float(tmpIma.sum())
            x  = (X[_valid]*tmpIma).sum()/total
            y  = (Y[_valid]*tmpIma).sum()/total

            x2 = (X[_valid]*X[_valid]*tmpIma).sum()/total - x**2
            y2 = (Y[_valid]*Y[_valid]*tmpIma).sum()/total - y**2        
            xy = (X[_valid]*Y[_valid]*tmpIma).sum()/total - x*y

        else:
            x = event.xdata
            y = event.ydata

            x2 = 0
            y2 = 0
            xy = 0

        fwhmx = 2.355*np.sqrt(x2)
        fwhmy = 2.355*np.sqrt(y2)

        self.curGrabPlot.set_data([x], [y])
        self.draw()

        self.emit(QtCore.SIGNAL("grabbedCoords"), (x, y, fwhmx, fwhmy))


    def plotGrabbed(self):

        self.curGrabPlot.set_data([], [])
        self.grabDataPlot.set_data(self.grabXData, self.grabYData)
        self.draw()

        
    def stopGrabbing(self):

        self.fig.canvas.mpl_disconnect(self.cidGrab)



## Create the GUI

CUTLEVELS = ["100%", "99.9%", "99.5%", "99%", "98%", "95%", "user"]

class IdentSources(QtGui.QMainWindow):

    def __init__(self, pickIma, grabIma, pickData, parent=None):

        super(IdentSources, self).__init__(parent)
        

        self.setWindowTitle('PampelMuse - Source identification')

        self.pickData = pickData

        ## Plot image data into central widget
        imaWidget = QtGui.QWidget()
        imaWidget.setContextMenuPolicy(QtCore.Qt.ActionsContextMenu)
        
        self.leftIma  = PickCanvas(imaWidget)
        self.leftIma.addImage(pickIma)
        self.leftIma.addData(pickData, marker='o',
                             edgecolor='r', picker=5, alpha=0.5)
        self.rightIma = GrabCanvas(imaWidget)
        self.rightIma.addImage(grabIma)
        self.rightIma.grabCoords(marker='x', mew=1.6, mec='r', ls='None', ms=15)

        imaFrame = QtGui.QHBoxLayout(imaWidget)
        imaFrame.addWidget(self.leftIma)
        imaFrame.addWidget(self.rightIma)


        self.setCentralWidget(imaWidget)


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
        fileQuitAction = self.createAction(\
            "&Quit", slot=self.close, shortcut="Ctrl+Q",
            tip="Close window")

        ## EDIT
        self.zoomMode = False
        enableMomentsAction = self.createAction(\
            "&Moment calculation", slot=self.enableMoments, shortcut="Alt+M",
            checkable=True, tip="Calculate moments for grabbed sources?",
            signal="toggled(bool)")

        changeCutsAction = self.createAction(\
            "&Cuts...", slot=self.changeCuts, shortcut="Alt+C",
            tip="Change cuts in active image")

        imageSelectGroup = QtGui.QActionGroup(self)
        selLeftImaAction = self.createAction(\
            "&Left", slot=self.selLeftIma, shortcut="Ctrl+L", checkable=True,
            tip="Make left image active", signal="toggled(bool)")
        selRightImaAction = self.createAction(\
            "&Right", slot=self.selRightIma, shortcut="Ctrl+R", checkable=True,
            tip="Make right image active", signal="toggled(bool)")
        imageSelectGroup.addAction(selLeftImaAction)
        imageSelectGroup.addAction(selRightImaAction)

        modeSelectGroup = QtGui.QActionGroup(self)
        selZoomAction   = self.createAction(\
            "&Zoom", slot=self.selZoom, shortcut="Alt+L", checkable=True,
            tip="Enable Zoom-Mode", signal="toggled(bool)")
        selPickAction   = self.createAction(\
            "&Pick", slot=self.selPick, shortcut="Alt+P", checkable=True,
            tip="Enable selection of sources", signal="toggled(bool)")
        selUnpickAction = self.createAction(\
            "&Unpick", slot=self.selUnpick, shortcut="Alt+U", checkable=True,
            tip="Enable deselection of sources", signal="toggled(bool)")
        modeSelectGroup.addAction(selZoomAction)
        modeSelectGroup.addAction(selPickAction)
        modeSelectGroup.addAction(selUnpickAction)


        ## HELP
        usageAction = self.createAction("&Usage", shortcut="Ctrl+H",
                                        slot=self.usage,
                                        tip="Instructions on usage")

        ## Add menu bar to widget
        fileMenu = self.menuBar().addMenu("&File")
        self.addActions(fileMenu, (fileQuitAction,))

        editMenu = self.menuBar().addMenu("&Edit")
        self.addActions(editMenu, (changeCutsAction, enableMomentsAction))
        selectMenu = editMenu.addMenu("&Select")
        self.addActions(selectMenu, (selLeftImaAction, selRightImaAction))
        modeMenu = editMenu.addMenu("&Mode")
        self.addActions(modeMenu, (selZoomAction, selPickAction,
                                   selUnpickAction))

        helpMenu = self.menuBar().addMenu("&Help")
        self.addActions(helpMenu, (usageAction,))


        ## Add toolbar to widget
##         fileToolbar = self.addToolBar("File")
##         fileToolbar.setObjectName("FileToolbar")
##         self.addActions(fileToolbar, (fileQuitAction))

        editToolbar = self.addToolBar("Edit")
        editToolbar.setObjectName("EditToolBar")
        self.addActions(editToolbar, (selZoomAction, selPickAction,
                                      selUnpickAction))
    
        cutsToolbar = self.addToolBar("Cuts")
        cutsToolbar.setObjectName("CutsToolBar")

        self.cutsComboBox = QtGui.QComboBox()
        self.cutsComboBox.addItems(CUTLEVELS)
        self.cutsComboBox.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.cutsComboBox.setFixedWidth(80)
        self.connect(self.cutsComboBox,
                     QtCore.SIGNAL("currentIndexChanged(const QString&)"),
                     self.changeCuts)
        self.leftImaCutIndex  = 0 ## used to save current cut levels when
        self.rightImaCutIndex = 0 ## switching between the images

##         self.cutsValidator  = QtGui.QDoubleValidator()
##         self.cutsValidator.setRange(self.leftIma.cuts[0],
##                                     self.leftIma.cuts[1])
##         self.cutsValidator.setDecimals(3)

        self.lcutLineEdit = QtGui.QLineEdit('{0}'.format(self.leftIma.cuts[0]))
        self.lcutLineEdit.setFixedWidth(80)
##         self.lcutLineEdit.setValidator(self.cutsValidator)

        self.hcutLineEdit = QtGui.QLineEdit('{0}'.format(self.leftIma.cuts[1]))
        self.hcutLineEdit.setFixedWidth(80)
##         self.hcutLineEdit.setValidator(self.cutsValidator)

        self.connect(self.lcutLineEdit, QtCore.SIGNAL("returnPressed()"),
                     self.updateLcutLineEdit)

        self.connect(self.hcutLineEdit, QtCore.SIGNAL("returnPressed()"),
                     self.updateHcutLineEdit)


        cutsToolbar.addWidget(QtGui.QLabel('Cut levels: '))
        cutsToolbar.addWidget(self.cutsComboBox)
        cutsToolbar.addWidget(QtGui.QLabel('='))
        cutsToolbar.addWidget(self.lcutLineEdit)
        cutsToolbar.addWidget(QtGui.QLabel(':'))
        cutsToolbar.addWidget(self.hcutLineEdit)


        ## Initial status of GUI
        selLeftImaAction.setChecked(True)    ## initially select left image
        enableMomentsAction.setChecked(True) ## moment calc. is on by default
        selPickAction.setChecked(True)       ## initially in `pick'-mode
        self.changeCuts("100%")

        ## interaction with images:
        self.curPick = None
        self.matched = {}
        self.connect(self.leftIma, QtCore.SIGNAL('pickedSource'),
                     self.collectPick)
        self.connect(self.rightIma, QtCore.SIGNAL('grabbedCoords'),
                     self.collectGrab)
        self.connect(self.leftIma, QtCore.SIGNAL('unpickedSource'),
                     self.collectUnpick)
        self.connect(self.leftIma, QtCore.SIGNAL('changedLoCut'),
                     self.updateLcutLineEdit)
        self.connect(self.leftIma, QtCore.SIGNAL('changedHiCut'),
                     self.updateHcutLineEdit)
        self.connect(self.rightIma, QtCore.SIGNAL('changedLoCut'),
                     self.updateLcutLineEdit)
        self.connect(self.rightIma, QtCore.SIGNAL('changedHiCut'),
                     self.updateHcutLineEdit)


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
        helpstring = """\nInstructions:
----------------------
 1) Use buttons to <Add> a new source or <Remove> a previously
    added source.
 2) (De)Select a source on the image on the left-hand side
 3) If a new source is added, click on its approximate position
    on the image on the right-hand side.
 4) Press <Quit> to finish selection and close GUI.\n"""

        self.updateHistory(helpstring)


        
    def selLeftIma(self, on):
        if on:
            self.activeIma = self.leftIma
            if self.zoomMode:
                self.activeIma.zoomImage()
                self.rightIma.stopZoom()
            self.cutsComboBox.setCurrentIndex(self.leftImaCutIndex)
            self.lcutLineEdit.setText(unicode(self.leftIma.cuts[0]))
            self.hcutLineEdit.setText(unicode(self.leftIma.cuts[1]))
        else:
            self.leftImaCutIndex = self.cutsComboBox.currentIndex()


    def selRightIma(self, on):
        if on:
            self.activeIma = self.rightIma
            if self.zoomMode:
                self.activeIma.zoomImage()
                self.leftIma.stopZoom()
            self.cutsComboBox.setCurrentIndex(self.rightImaCutIndex)
            self.lcutLineEdit.setText(unicode(self.rightIma.cuts[0]))
            self.hcutLineEdit.setText(unicode(self.rightIma.cuts[1]))
        else:
            self.rightImaCutIndex = self.cutsComboBox.currentIndex()


    def enableMoments(self, on):
        
        self.rightIma.calcMoments = on


    def changeCuts(self, value):
        if value == "user":
            self.lcutLineEdit.setEnabled(True)
            self.hcutLineEdit.setEnabled(True)
            return
        else:
            self.lcutLineEdit.setEnabled(False)
            self.hcutLineEdit.setEnabled(False)
            self.activeIma.changeCuts(vrange=float(value[:-1]))
        

    def updateLcutLineEdit(self, locut=None):
        if not self.lcutLineEdit.isEnabled():
            self.lcutLineEdit.setText(unicode(locut))
        else:
            self.activeIma.changeCuts(\
                locut=float(self.lcutLineEdit.displayText()))
 

    def updateHcutLineEdit(self, hicut=None):
        if not self.hcutLineEdit.isEnabled():
            self.hcutLineEdit.setText(unicode(hicut))
        else:
            self.activeIma.changeCuts(\
                hicut=float(self.hcutLineEdit.displayText()))


    def selZoom(self, on):
        self.zoomMode = on
        if on:
            self.activeIma.zoomImage()
        else:
            self.activeIma.stopZoom()


    def selPick(self, on):
        if on:
            self.leftIma.startPicking()
            self.rightIma.startGrabbing()
        else:
            self.curPick = None
            self.leftIma.stopPicking()
            self.rightIma.stopGrabbing()

    
    def selUnpick(self, on):
        if on:
            self.leftIma.startUnpicking()
        else:
            self.leftIma.stopUnpicking()



    def updateHistory(self, string):
        if string == "left":
            string = self.leftIma.STRING

        try:
            text = unicode(string)
            self.txtBrowser.append(string)
        except:
            self.txtBrowser.append("<font color=red>{0} is invalid!</font>"\
                                       .format(string))


    def collectPick(self, n):

        self.curPick = n
        self.status.showMessage("Source #{0}: m={1:.1f} @({2:.1f},{3:.1f})"\
                                    .format(self.pickData['id'][n],
                                            self.pickData['mag'][n],
                                            self.pickData['x'][n],
                                            self.pickData['y'][n]), 5000)


    def collectGrab(self,  coords):
        
        statusMessage = "Coordinates: ({0:.1f},{1:.1f})".format(\
                coords[0], coords[1])
        if self.rightIma.calcMoments:
            statusMessage += " - FWHM: ({0:.1f},{1:.1f})".format(\
                coords[2], coords[3])
        self.status.showMessage(statusMessage, 5000)
        
        if self.curPick != None:
            n = self.curPick
            
            matchString = "Identified source #{0} (m={1:.1f}): ".format(\
                self.pickData['id'][n], self.pickData['mag'][n])
            matchString += "({0:.1f},{1:.1f}) -> ({2:.1f},{3:.1f})".format(\
                self.pickData['x'][n],self.pickData['y'][n],coords[0],coords[1])
            self.updateHistory(matchString)

            self.matched[self.pickData['id'][n]] = coords

            self.leftIma.pickData.append(n)
            self.leftIma.plotPicked()

            self.rightIma.grabXData.append(coords[0])
            self.rightIma.grabYData.append(coords[1])
            self.rightIma.plotGrabbed()

            self.curPick = None


    def collectUnpick(self, n):

        self.status.showMessage("Source #{0}: m={1:.1f} @({2:.1f},{3:.1f})"\
                                    .format(self.pickData['id'][n],
                                            self.pickData['mag'][n],
                                            self.pickData['x'][n],
                                            self.pickData['y'][n]), 5000)

        matchString = "Removed identification for source #{0} -".format(\
            self.pickData['id'][n])
        matchString += " m={0:.1f} @({0:.1f},{2:.1f})".format(\
            self.pickData['mag'][n],self.pickData['x'][n],self.pickData['y'][n])
        self.updateHistory(matchString)

        index2Del = self.leftIma.pickData.index(n)

        del self.leftIma.pickData[index2Del]
        self.leftIma.plotPicked()

        del self.rightIma.grabXData[index2Del]
        del self.rightIma.grabYData[index2Del]
        self.rightIma.plotGrabbed()

        del self.matched[self.pickData['id'][n]]



    def getResults(self):
        results = np.zeros(len(self.matched),
                           dtype=[('id', 'i8'),('x','f8'),('y', 'f8'),
                                  ('fwhm_x', 'f8'), ('fwhm_y','f8')]
                           )
        i=0
        for key, value in self.matched.iteritems():
            results[i][0] = key
            for j in xrange(len(value)):
                results[i][j+1] = value[j]
            i += 1
        return results
            

def main():
    app  = QtGui.QApplication(sys.argv)
    identifier = IdentSources()
    identifier.show()
    app.exec_()
 ##    print identifier.getResults()

if __name__ == "__main__":
    main()
