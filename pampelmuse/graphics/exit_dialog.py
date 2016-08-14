# -*- coding: utf-8 -*-
"""
exit_dialog.py
==============
Copyright 2013-2016 Sebastian Kamann

This file is part of PampelMuse.

PampelMuse is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.

PampelMuse is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with PampelMuse. If not, see <http://www.gnu.org/licenses/>.


+++
Provides
--------
class pampelmuse.graphics.exit_dialog.exitUnsavedDialog

Purpose
-------
This module implements the class exitUnsavedDialog that is
used to display a dialog window if the PampelMuse GUI is about
to get closed with unsaved changes.
The dialog itself was created using QtDesigner and is defined
in pampelmuse.graphics.Ui_exit_dialog.

Latest SVN revision
-------------------
340, 2016/07/29
"""
from PyQt4.QtGui import QDialog
from PyQt4.QtCore import pyqtSignature
from .Ui_exit_dialog import Ui_exitUnsavedDialog


__author__ = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__revision__ = 340


class exitUnsavedDialog(QDialog, Ui_exitUnsavedDialog):
    """
    Class to implement a dialog that pops up if the PampelMuse GUI is
    about to be closed although the latest changes have not been saved.
    """
    def __init__(self, parent = None):
        """
        Initialize a new instance of the class exitUnsavedDialog.
        
        Parameters
        ----------
        parent : 
            The 'parent' argument, if not None, causes self to be 
            owned by Qt instead of PyQt. (copied from PyQt4 documentation)
        """
        QDialog.__init__(self, parent)
        self.setupUi(self)
        
        self.status = None
    
    @pyqtSignature("")
    def on_buttonBox_accepted(self):
        """
        Handle event when the button to accept the closing
        of the GUI is pressed.
        """
        self.status = True
        QDialog.close(self)

    @pyqtSignature("")
    def on_buttonBox_rejected(self):
        """
        Handle event when the button to reject the closing
        of the GUI is pressed.
        """
        self.status = False
        QDialog.close(self)


