# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'exit_dialog.ui'
#
# Created: Mon Dec  7 15:39:02 2015
#      by: PyQt4 UI code generator 4.9.1
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

class Ui_exitUnsavedDialog(object):
    def setupUi(self, exitUnsavedDialog):
        exitUnsavedDialog.setObjectName(_fromUtf8("exitUnsavedDialog"))
        exitUnsavedDialog.resize(400, 300)
        exitUnsavedDialog.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedKingdom))
        self.buttonBox = QtGui.QDialogButtonBox(exitUnsavedDialog)
        self.buttonBox.setGeometry(QtCore.QRect(30, 240, 341, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName(_fromUtf8("buttonBox"))
        self.label = QtGui.QLabel(exitUnsavedDialog)
        self.label.setGeometry(QtCore.QRect(60, 110, 271, 20))
        font = QtGui.QFont()
        font.setPointSize(13)
        font.setBold(False)
        font.setWeight(50)
        self.label.setFont(font)
        self.label.setObjectName(_fromUtf8("label"))

        self.retranslateUi(exitUnsavedDialog)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("accepted()")), exitUnsavedDialog.accept)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("rejected()")), exitUnsavedDialog.reject)
        QtCore.QMetaObject.connectSlotsByName(exitUnsavedDialog)

    def retranslateUi(self, exitUnsavedDialog):
        exitUnsavedDialog.setWindowTitle(QtGui.QApplication.translate("exitUnsavedDialog", "Leave PampelMuse", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("exitUnsavedDialog", "Unsaved changes exist! Proceed?", None, QtGui.QApplication.UnicodeUTF8))

