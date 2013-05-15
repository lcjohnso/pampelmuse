# -*- coding: utf-8 -*-

"""
Module implementing exitUnsavedDialog.
"""

from PyQt4.QtGui import QDialog
from PyQt4.QtCore import pyqtSignature

from Ui_exit_dialog import Ui_exitUnsavedDialog

class exitUnsavedDialog(QDialog, Ui_exitUnsavedDialog):
    """
    Class documentation goes here.
    """
    def __init__(self, parent = None):
        """
        Constructor
        """
        QDialog.__init__(self, parent)
        self.setupUi(self)
        
        self.status = None
    
    @pyqtSignature("")
    def on_buttonBox_accepted(self):
        """
        Slot documentation goes here.
        """
        # TODO: not implemented yet
        self.status = True
        QDialog.close(self)

    
    @pyqtSignature("")
    def on_buttonBox_rejected(self):
        """
        Slot documentation goes here.
        """
        # TODO: not implemented yet
        self.status = False
        QDialog.close(self)

