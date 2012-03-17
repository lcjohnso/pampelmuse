"""
FITTINGGUI
==========
The graphical back-bone of the StarFit-class defined in the fit_array.py
file.

CHANGELOG
---------
<v0.1.0.0, ...2012/01/31, (c). S. Kamann
0.1.0.0, 2012/01/31, changed interval of timer from 5s to 0.5s to avoid
                     ridicolous overheads when fitting only fluxes.
0.1.0.1, 2012/02/13, shortened interval to 0.1s (0.5s still included overhead)
"""
__author__  = "Sebastian Kamann (skamann@aip.de)"
__version__ = "0.1.0.1"

import numpy as np, sys, time
from PyQt4 import QtCore, QtGui

## Required PampelMuse-packages
from pampelmuse.graphics.dynamic_ima_canvas import DynamicImaCanvas


INTERVAL=10 ## timer interval, in milli-seconds

class FittingGui(QtGui.QMainWindow):

    styleSheetContinue = """
    QPushButton {color : #009933; font : bold 14px} 
    """
    
    styleSheetPause = """
    QPushButton {font : bold 14px} 
    """

    styleSheetAbort = """
    QPushButton {color : #ff3300, font : bold 16px}
    """

    def __init__(self, parent=None):

        super(FittingGui, self).__init__(parent)

        self.STARTED = False
        self.PAUSED  = False
        self.ABORTED = False

        self.setWindowTitle("PampelMuse - Fitting process")

        self.imageLabel  = QtGui.QWidget()
        self.dataCanvas = DynamicImaCanvas(self.imageLabel)
        self.resiCanvas = DynamicImaCanvas(self.imageLabel)
        self.imageLabel.setMinimumSize(600,400)
##         self.imageLabel.setAlignment(QtCore.Qt.AlignCenter)
##         self.imageLabel.setContextmenuPolicy(QtCore.Qt.ActionsContextMenu)

        imaFrame = QtGui.QHBoxLayout(self.imageLabel)
        imaFrame.addWidget(self.dataCanvas)
        imaFrame.addWidget(self.resiCanvas)

        self.setCentralWidget(self.imageLabel)


        self.interactToolbar = self.addToolBar("Interaction")
        self.interactToolbar.setObjectName("InteractToolBar")

        self.startButton = QtGui.QPushButton("Start")
        self.startButton.setStyleSheet(self.styleSheetContinue)
        self.abortButton = QtGui.QPushButton("Abort")
        self.abortButton.setStyleSheet(self.styleSheetAbort)

        self.connect(self.startButton, QtCore.SIGNAL('released()'),
                     self.startIt)
        self.connect(self.abortButton, QtCore.SIGNAL('released()'),
                     self.abortIt)

        self.interactToolbar.addWidget(self.startButton)
        self.interactToolbar.addWidget(self.abortButton)


        self.pBar = QtGui.QProgressBar(self)
        self.pBar.setMaximumWidth(100)

        self.sBar = self.statusBar()
        self.sBar.setSizeGripEnabled(False)
        self.sBar.addPermanentWidget(self.pBar)

 
        self.timer = QtCore.QTimer()
        QtCore.QObject.connect(self.timer, QtCore.SIGNAL("timeout()"),
                               self.timeout)



        
    def startIt(self):
        
        if not self.STARTED:
            self.timer.start(INTERVAL)
            self.STARTED = True
            self.startButton.setText("Pause")
            self.startButton.setStyleSheet(self.styleSheetPause)
        else:
            self.PAUSED = not self.PAUSED
            self.updateButton()


    def updateButton(self):

            self.startButton.setText(\
                    ["Pause", "Continue"][self.PAUSED])
            self.startButton.setStyleSheet(\
                [self.styleSheetPause, self.styleSheetContinue][self.PAUSED])



    def abortIt(self):

        self.PAUSED = True

        self.ABORTED = self.okToCancel()

        if self.ABORTED:
            self.abortButton.setDisabled(True)
        else:
            self.updateButton()


    def okToCancel(self):
        
        reply = QtGui.QMessageBox.question(\
            self,
            "WARNING: About to cancel fit!",
            "Save current results?",
            QtGui.QMessageBox.No|QtGui.QMessageBox.Yes|QtGui.QMessageBox.Cancel
            )
        if reply == QtGui.QMessageBox.Cancel:
            return False

        if reply == QtGui.QMessageBox.Yes:
            self.save = True
        else:
            self.save = False
        return True


    def timeout(self): ##  a placeholder, replaced by method in sub-class

        return
