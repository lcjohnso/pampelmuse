import matplotlib.pyplot, sys
import numpy as np
from PyQt4 import QtCore, QtGui

## Required PampelMuse-packages
from pampelmuse.graphics.dynamic_ima_canvas import DynamicImaCanvas



class PickCanvas(DynamicImaCanvas):

##     def selectBright(self, maxmag):
##         self.allData['x'][self.allData['mag']>maxmag] = np.ma.masked
##         self.allDataPlot.set_xdata(self.allData['x'])
## Does not work like this - Method to update date of scatterplot?     

    def pickPoint(self):
        self.unpick=False
        self.cidPick = self.fig.canvas.mpl_connect('pick_event', self._picker)

    
    def unpickPoint(self):
        self.unpick=True
        self.cidPick = self.fig.canvas.mpl_connect('pick_event', self._picker)


    def _picker(self, event):

        if len(event.ind) != 1:
            return

        self.INDEXPICK = event.ind[0]
        i = self.INDEXPICK

        if not self.INDEXPICK in self.pickData:
            if not self.unpick:
                self.pickData.append(self.INDEXPICK)
                self.facecolors[self.INDEXPICK] = self.edgecolor
            else:
                return

        else:
            if self.unpick:
                SELINDEX = self.pickData.index(self.INDEXPICK)
                self.pickData.remove(self.INDEXPICK)
                self.facecolors[self.INDEXPICK] = 'none'
            else:
                return
            

        self.allDataPlot.set_facecolors(self.facecolors)
        self.draw()
        self.fig.canvas.mpl_disconnect(self.cidPick)
 
        self.emit(QtCore.SIGNAL("newString"),
                  " ID={0} @({1:.2f},{2:.2f})"\
                      .format(self.allIDs[i], self.allData['x'][i],
                              self.allData['y'][i]))

        if not self.unpick:
            self.emit(QtCore.SIGNAL("pickSuccess"))
        else:
            self.emit(QtCore.SIGNAL("unpickSuccess"), SELINDEX)




class GrabCanvas(DynamicImaCanvas):

    def grabPoint(self):
        self.cidGrab = self.fig.canvas.mpl_connect('button_press_event',
                                                   self._grabber)


    def ungrabPoint(self, index):
        del self.grabXData[index]
        del self.grabYData[index]
        del self.grabXFWHM[index]
        del self.grabYFWHM[index]

        self.grabDataPlot.set_data(self.grabXData, self.grabYData)
        self.draw()
        self.emit(QtCore.SIGNAL("ungrabSuccess"))


    def _grabber(self, event):

        if event.inaxes != self.axes:
            return

        ## use position of mouse click as initial guess, calculate first
        ## order moments and use them as source positions
        x_init = int(round(event.xdata))
        y_init = int(round(event.ydata))

        rad = 2
        Y, X = np.indices((2*rad+1,2*rad+1), dtype='int32')
        Y += (y_init-rad)
        X += (x_init-rad)

        _valid = (X>=0)&(X<self.image.shape[1])&(Y>=0)&(Y<self.image.shape[0])

        tmpIma  = self.image[Y[_valid],X[_valid]]
        tmpIma -= tmpIma.min()

        total = float(tmpIma.sum())
        x  = (X[_valid]*tmpIma).sum()/total
        y  = (Y[_valid]*tmpIma).sum()/total

        x2 = (X[_valid]*X[_valid]*tmpIma).sum()/total - x**2
        y2 = (Y[_valid]*Y[_valid]*tmpIma).sum()/total - y**2        
        xy = (X[_valid]*Y[_valid]*tmpIma).sum()/total - x*y


        self.grabXData.append(x)
        self.grabYData.append(y)
        self.grabXFWHM.append(2.355*np.sqrt(x2))
        self.grabYFWHM.append(2.355*np.sqrt(y2))

        self.grabDataPlot.set_data(self.grabXData, self.grabYData)
        self.draw()

        self.fig.canvas.mpl_disconnect(self.cidGrab)
        self.emit(QtCore.SIGNAL("newString"),
                  "  ==> @({0:.2f},{1:.2f})".format(event.xdata, event.ydata))
        self.emit(QtCore.SIGNAL("grabSuccess"))



## Create the GUI

class IdentSources(QtGui.QMainWindow):

    def __init__(self, pickIma, grabIma, pickData):

        QtGui.QMainWindow.__init__(self)
        
        self.resize(800,800)
        self.setWindowTitle('Identify sources')

        self._addMenu()

        self.frame = QtGui.QWidget(self) ##  will be the central widget 
        vbox = QtGui.QVBoxLayout(self.frame)


        txtWidget = QtGui.QWidget() ## Widget containing the history text
        self.txtBrowser = QtGui.QTextBrowser()
        txtFrame  = QtGui.QHBoxLayout(txtWidget) 
        txtFrame.addWidget(self.txtBrowser)


        imaWidget = QtGui.QWidget(self.frame) ## widget containing the images
        
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


        cutsWidget = QtGui.QWidget(self.frame)  ## widget containing the cuts

        validCuts = QtGui.QDoubleValidator()
        validCuts.setRange(0.0, 100.0, decimals=1)


        lcutWidget = QtGui.QWidget(cutsWidget)  ## left cuts
        lcutVal  = QtGui.QLabel('Cut level:')
        lcutEdit = QtGui.QLineEdit('100.0', lcutWidget, maxLength=5)
        lcutEdit.setValidator(validCuts)
        lcutEdit.setFocusPolicy(QtCore.Qt.ClickFocus)
        lcutEdit.setFixedSize(50,25)
        self.connect(lcutEdit, QtCore.SIGNAL('textEdited(QString&)'),
                     lcutEdit, QtCore.SIGNAL('setText(QString&)'))
        self.connect(lcutEdit, QtCore.SIGNAL('returnPressed()'), 
                     lambda ima=self.leftIma: ima.changeCuts(lcutEdit.text())
                     )

        self.lzoomButton = QtGui.QPushButton("Zoom")
        self.connect(self.lzoomButton, QtCore.SIGNAL('clicked()'),
                     self.leftIma.zoomImage)
        self.connect(self.lzoomButton, QtCore.SIGNAL('clicked()'),
                     lambda off=1: self.switchButtons(off))
        self.connect(self.leftIma, QtCore.SIGNAL('zoomEnd'),
                     lambda off=0: self.switchButtons(off))

        lcutFrame = QtGui.QHBoxLayout(lcutWidget)
        lcutFrame.addStretch(1)
        lcutFrame.addWidget(self.lzoomButton)
        lcutFrame.addWidget(lcutVal)
        lcutFrame.addWidget(lcutEdit)



        rcutWidget = QtGui.QWidget(cutsWidget)  ##  right cuts
        rcutVal  = QtGui.QLabel('Cut level:')
        rcutEdit = QtGui.QLineEdit('100.0', rcutWidget, maxLength=5)
        rcutEdit.setValidator(validCuts)
        rcutEdit.setFocusPolicy(QtCore.Qt.ClickFocus)
        rcutEdit.setFixedSize(50,25)
        self.connect(rcutEdit, QtCore.SIGNAL('textEdited(QString&)'),
                     rcutEdit, QtCore.SIGNAL('setText(QString&)'))
        self.connect(rcutEdit, QtCore.SIGNAL('returnPressed()'), 
                     lambda ima=self.rightIma: ima.changeCuts(rcutEdit.text())
                     )

        self.rzoomButton = QtGui.QPushButton("Zoom")
        self.connect(self.rzoomButton, QtCore.SIGNAL('clicked()'),
                     self.rightIma.zoomImage)
        self.connect(self.rzoomButton, QtCore.SIGNAL('clicked()'),
                     lambda off=1: self.switchButtons(off))
        self.connect(self.rightIma, QtCore.SIGNAL('zoomEnd'),
                     lambda off=0: self.switchButtons(off))

        rcutFrame = QtGui.QHBoxLayout(rcutWidget)
        rcutFrame.addStretch(1)
        rcutFrame.addWidget(self.rzoomButton)
        rcutFrame.addWidget(rcutVal)
        rcutFrame.addWidget(rcutEdit)


        cutsFrame = QtGui.QHBoxLayout(cutsWidget)
        cutsFrame.addWidget(lcutWidget)
        cutsFrame.addWidget(rcutWidget)


        butWidget = QtGui.QWidget() ## Widget containing the buttons

        self.addButton = QtGui.QPushButton("Add")
        self.remButton = QtGui.QPushButton("Remove")
        self.quiButton = QtGui.QPushButton("Quit")

        ## add new point or remove existing point
        self.connect(self.addButton, QtCore.SIGNAL('clicked()'),
                     self.leftIma.pickPoint)
        self.connect(self.addButton, QtCore.SIGNAL('clicked()'),
                     lambda off=1: self.switchButtons(off))
        self.connect(self.addButton, QtCore.SIGNAL('clicked()'),
                     lambda string="Adding new source:": \
                         self.updateHistory(string))

        self.connect(self.remButton, QtCore.SIGNAL('clicked()'),
                     self.leftIma.unpickPoint)
        self.connect(self.remButton, QtCore.SIGNAL('clicked()'),
                     lambda off=1: self.switchButtons(off))
        self.connect(self.remButton, QtCore.SIGNAL('clicked()'),
                     lambda string="Removing selected source:": \
                         self.updateHistory(string))

        ## work on left image
        self.connect(self.leftIma, QtCore.SIGNAL('pickSuccess'),
                     self.rightIma.grabPoint)
        self.connect(self.leftIma, QtCore.SIGNAL('unpickSuccess'),
                     self.rightIma.ungrabPoint)
        self.connect(self.leftIma, QtCore.SIGNAL('newString'),
                     self.updateHistory)


        ## work on right image
        self.connect(self.rightIma, QtCore.SIGNAL('grabSuccess'),
                     lambda off=0: self.switchButtons(off))
        self.connect(self.rightIma, QtCore.SIGNAL('ungrabSuccess'),
                     lambda off=0: self.switchButtons(off))
        self.connect(self.rightIma, QtCore.SIGNAL('newString'),
                     self.updateHistory)

        ## close GUI
        self.connect(self.quiButton, QtCore.SIGNAL('clicked()'),
                     self.close)

        butFrame = QtGui.QHBoxLayout(butWidget)
        butFrame.addStretch(0)
        butFrame.addWidget(self.addButton)
        butFrame.addWidget(self.remButton)
        butFrame.addWidget(self.quiButton)

        

        vbox.addWidget(imaWidget)
        vbox.addWidget(cutsWidget)
        vbox.addWidget(butWidget)
        vbox.addWidget(txtWidget)
        self.setCentralWidget(self.frame)
        


    def switchButtons(self, off):
        
        for button in [self.lzoomButton, self.rzoomButton, self.addButton,
                       self.remButton,  self.quiButton]:
            button.setDisabled(off)


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
        helpstring = """\nInstructions:
----------------------
 1) Use buttons to <Add> a new source or <Remove> a previously
    added source.
 2) (De)Select a source on the image on the left-hand side
 3) If a new source is added, click on its approximate position
    on the image on the right-hand side.
 4) Press <Quit> to finish selection and close GUI.\n"""

        self.updateHistory(helpstring)


    def updateHistory(self, string):
        if string == "left":
            string = self.leftIma.STRING

        try:
            text = unicode(string)
            self.txtBrowser.append(string)
        except:
            self.txtBrowser.append("<font color=red>{0} is invalid!</font>"\
                                       .format(string))


    def getResults(self):
        return np.array(zip(self.leftIma.allIDs[self.leftIma.pickData],
                            self.rightIma.grabXData,
                            self.rightIma.grabYData,
                            self.rightIma.grabXFWHM,
                            self.rightIma.grabYFWHM),
                        dtype=[('id', 'i8'),('x','f8'),('y', 'f8'),
                               ('fwhm_x', 'f8'), ('fwhm_y','f8')]
                        )
                        
        

            

def main():
    app  = QtGui.QApplication(sys.argv)
    identifier = IdentSources()
    identifier.show()
    app.exec_()
 ##    print identifier.getResults()

if __name__ == "__main__":
    main()
