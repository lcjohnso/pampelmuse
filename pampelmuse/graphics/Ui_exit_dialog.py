# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'exit_dialog.ui'
#
# Created by: PyQt5 UI code generator 5.8.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_exitUnsavedDialog(object):
    def setupUi(self, exitUnsavedDialog):
        exitUnsavedDialog.setObjectName("exitUnsavedDialog")
        exitUnsavedDialog.resize(400, 300)
        exitUnsavedDialog.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedKingdom))
        self.buttonBox = QtWidgets.QDialogButtonBox(exitUnsavedDialog)
        self.buttonBox.setGeometry(QtCore.QRect(30, 240, 341, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.label = QtWidgets.QLabel(exitUnsavedDialog)
        self.label.setGeometry(QtCore.QRect(60, 110, 271, 20))
        font = QtGui.QFont()
        font.setPointSize(13)
        font.setBold(False)
        font.setWeight(50)
        self.label.setFont(font)
        self.label.setObjectName("label")

        self.retranslateUi(exitUnsavedDialog)
        self.buttonBox.accepted.connect(exitUnsavedDialog.accept)
        self.buttonBox.rejected.connect(exitUnsavedDialog.reject)
        QtCore.QMetaObject.connectSlotsByName(exitUnsavedDialog)

    def retranslateUi(self, exitUnsavedDialog):
        _translate = QtCore.QCoreApplication.translate
        exitUnsavedDialog.setWindowTitle(_translate("exitUnsavedDialog", "Leave PampelMuse"))
        self.label.setText(_translate("exitUnsavedDialog", "Unsaved changes exist! Proceed?"))

