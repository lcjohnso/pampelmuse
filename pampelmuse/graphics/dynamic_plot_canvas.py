"""
DYNAMICPLOTCANVAS
=================
Routine is used to create a lineplot and interact with it. The routine is
called by the POLYFIT routine of PampelMuse.

CHANGELOG:
----------
2012/01/25, v0.0.1.1, added some documentation. Fixed bug that sometimes the
                      vertical line indicating the current cursor position did
                      not disappear upon updating  the data to be plotted. 
"""
__author__ = "Sebastian Kamann (skamann@aip.de)"
__version__ = "0.0.1.1"

import matplotlib.pyplot
import numpy as np
from PyQt4 import QtCore
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas


class DynamicPlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=8, height=4, dpi=100):

        self.fig  = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_axes((0.12, 0.12, 0.85, 0.85))

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        self.thePlots = {}

        self.vLine, = self.axes.plot([], [], ls='--', color='r')
        
        self.rect = self.axes.bar([0.],[0.], width=0., bottom=0., alpha=0.5,
                                  fc='r', lw=0)

        self.enterAxes = self.fig.canvas.mpl_connect('axes_enter_event',
                                                     self._uponEntering)
        self.leaveAxes = self.fig.canvas.mpl_connect('axes_leave_event',
                                                     self._uponLeaving)


    def addData(self, name, xData, yData, **kwargs):

        self.thePlots[name] = self.axes.plot(xData, yData, **kwargs)[0]
        
        self.axes.minorticks_on()

        self.draw()


    def updateData(self, name, xData=None, yData=None):

        if xData==None and yData==None:
            return

        if xData!=None:
            self.thePlots[name].set_xdata(xData)
        if yData!=None:
            self.thePlots[name].set_ydata(yData)

        self.vLine.set_data([],[])

        self.axes.axes.relim()
        self.axes.autoscale_view(tight=False)

        self.draw()


    def _uponEntering(self, event):

        if event.inaxes != self.axes:
            return
        
        self.Init = self.fig.canvas.mpl_connect('button_press_event',
                                                self._grabCoordsInit)
        self.Motion= self.fig.canvas.mpl_connect('motion_notify_event',
                                                 self._motionNotify)

        

    def _uponLeaving(self, event):

        self.vLine.set_data([],[])
        self.draw()
        
        if event.inaxes != self.axes:
            return

        self.fig.canvas.mpl_disconnect(self.Init)
        self.fig.canvas.mpl_disconnect(self.Motion)




    def _grabCoordsInit(self,  event):

        self.xInit = event.xdata
        self.rect[0].set_x(self.xInit)
        self.rect[0].set_height(self.axes.get_ylim()[1]-self.axes.get_ylim()[0])
        self.rect[0].set_y(self.axes.get_ylim()[0])


        self.MotionKey = self.fig.canvas.mpl_connect(\
            'motion_notify_event', self._motionNotifyKeyPressed)
        self.Final = self.fig.canvas.mpl_connect(\
            'button_release_event', self._grabCoordsFinal)


    def _grabCoordsFinal(self, event):
        
        self.rect[0].set_height(0)
        self.rect[0].set_width(0)

        self.fig.canvas.mpl_disconnect(self.MotionKey)
        self.fig.canvas.mpl_disconnect(self.Final)

        if event.inaxes == self.axes:
            self.xFinal = event.xdata

        if event.xdata > self.xInit:
            self.emit(QtCore.SIGNAL("rangeSelected"),
                      *[(self.xInit, self.xFinal), event.button==1])
        elif event.xdata < self.xInit:
            self.emit(QtCore.SIGNAL("rangeSelected"),
                      *[(self.xFinal, self.xInit), event.button==1])


    def _motionNotify(self, event):

        if event.inaxes == self.axes:
            self.vLine.set_data([event.xdata,event.xdata],
                                self.axes.get_ylim())
            self.draw()


    def _motionNotifyKeyPressed(self, event):

        if event.inaxes != self.axes:
            return

        self.xFinal = event.xdata
        self.rect[0].set_width(self.xFinal-self.xInit)
