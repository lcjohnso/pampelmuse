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
0.4.0.0, 2012/03/19, updated documentation; timeout-interval is now provided
                     as a parameter to the __init__-instance of `FittingGui`;
                     adapted version number to match overall PampelMuse-
                     versioning
"""
__author__  = "Sebastian Kamann (skamann@aip.de)"
__version__ = "0.4.0.0"

import numpy as np, sys, time
from PyQt4 import QtCore, QtGui

## Required PampelMuse-packages
from pampelmuse.graphics.dynamic_ima_canvas import DynamicImaCanvas


## INTERVAL=10 ## timer interval, in milli-seconds
               ## provided as a paremeter to __init__ now

class FittingGui(QtGui.QMainWindow):
    """
    The class provides the graphical user interface (GUI) for the CUBEFIT
    routine of PampelMuse. So far, the GUI does not provide that much
    functionality:
    *Two `DynamicImaCanvas`-instances are initialized as the central widgets
     and can be used e.g to display the slice that is currently being analysed
     and the fit reiduals.
    *Start/Stop buttons that can be used to pause the fit or abort it.
    *A progress bar (usually used to show the overall  status of the analysis)
    *A status bar showing what is currently going on.

    Note that the possibiliy to pause/stop the fit is implemented by calling a
    timeout-function after a certain time interval. The interval can be
    specified upon creation of a new `FittingGui`-instance.
    """

    ## Some class-attributes to format stert/stop buttons:
    styleSheetContinue = """
    QPushButton {color : #009933; font : bold 14px} 
    """
    
    styleSheetPause = """
    QPushButton {font : bold 14px} 
    """

    styleSheetAbort = """
    QPushButton {color : #ff3300, font : bold 16px}
    """

    def __init__(self, parent=None, interval=10):
        """
        Initialize a new instance of the `FittingGui`-class.

        Parameters
        ----------
        parent : optional
            The parameter is passed directly `QtGui.QMainWindow`-superclass
            that is initialized at the beginning of initialization.
        interval : int, optional
            The interval (in milli-seconds) that ellapses between two
            consecutive calls to the timeout-function.

        Return
        ------
            The nwely created instance.
        """
        self.interval = interval

        super(FittingGui, self).__init__(parent) ## initialize superclass

        self.STARTED = False ## Flag whether process has been started
        self.PAUSED  = False ## Flag whether process has been paused
        self.ABORTED = False ## Flag whether process has been aborted

        self.setWindowTitle("PampelMuse - Fitting process") ## the title name

        ## define central widget: two `DynamicImaCanvas`-instances
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


        ##  prepare buttons to interact with process
        self.interactToolbar = self.addToolBar("Interaction")
        self.interactToolbar.setObjectName("InteractToolBar")

        self.startButton = QtGui.QPushButton("Start")
        self.startButton.setStyleSheet(self.styleSheetContinue)
        self.abortButton = QtGui.QPushButton("Abort")
        self.abortButton.setStyleSheet(self.styleSheetAbort)

        ## connect to callback-functions
        self.connect(self.startButton, QtCore.SIGNAL('released()'),
                     self.startIt) 
        self.connect(self.abortButton, QtCore.SIGNAL('released()'),
                     self.abortIt)

        self.interactToolbar.addWidget(self.startButton)
        self.interactToolbar.addWidget(self.abortButton)

        ##  prepare toolbar and status-bar
        self.pBar = QtGui.QProgressBar(self)
        self.pBar.setMaximumWidth(100)

        self.sBar = self.statusBar()
        self.sBar.setSizeGripEnabled(False)
        self.sBar.addPermanentWidget(self.pBar)


        ## intialize timer-function
        self.timer = QtCore.QTimer()
        QtCore.QObject.connect(self.timer, QtCore.SIGNAL("timeout()"),
                               self.timeout)



        
    def startIt(self):
        """
        The callback-funtion that is called when the start-button is pressed.
        """
        if not self.STARTED:
            self.timer.start(self.interval)
            self.STARTED = True
            self.startButton.setText("Pause")
            self.startButton.setStyleSheet(self.styleSheetPause)
        else:
            self.PAUSED = not self.PAUSED
            self.updateButton()


    def updateButton(self):
        """
        The function is used to update the text on the start/pause/continue
        button when the state of the process is changes.
        """
        self.startButton.setText(\
            ["Pause", "Continue"][self.PAUSED])
        self.startButton.setStyleSheet(\
            [self.styleSheetPause, self.styleSheetContinue][self.PAUSED])



    def abortIt(self):
        """
        callback-funtion that is called when the abort-button is pressed.
        """
        self.PAUSED = True

        self.ABORTED = self.okToCancel()

        if self.ABORTED:
            self.abortButton.setDisabled(True)
        else:
            self.updateButton()


    def okToCancel(self):
        """
        Method that prepares the dialog that is loaded when the abort-button
        is pressed.
        """
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


    def timeout(self):
        """
        The timeout-funtion. This is essentially a placeholder that should be
        replaced by a dedicated method in any sub-class of `FittingGui`.
        """
        return
