import matplotlib.pyplot
import numpy as np
from PyQt4 import QtCore
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas



class DynamicImaCanvas(FigureCanvas):

    def __init__(self, parent=None, width=4, height=4, dpi=100):
        
        self.fig  = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_axes((0.05,0.05,0.9,0.9))
##         self.axes.hold(False)

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)


    def addImage(self, image, cmap=matplotlib.pyplot.cm.gray_r):

        self.image = image
        self.imagePlot = self.axes.imshow(image, cmap=cmap)
        
        self.axes.set_autoscale_on(False)
        self.axes.set_xticks([])
        self.axes.set_yticks([])


    def changeCuts(self, newcuts):

        values = self.image.copy().reshape(-1)
        values.sort()

        cutlev = float(newcuts)/100.
        npix  = values.shape[0]

        cutpix = int((1-cutlev)*npix)

        cmin = values[cutpix/2]
        cmax = values[-cutpix/2]
        if cutpix == 0:
            cmax = values[-1]

        self.imagePlot.set_clim(cmin, cmax)
        self.draw()


    def zoomImage(self):
        
        self.zoomCoords = np.zeros((2,2), dtype='int32')

        self.zoomFrame, = self.axes.bar(left=self.zoomCoords[0,0], height=0.,
                                        width=0., bottom=self.zoomCoords[0,1],
                                        color='k', edgecolor='k', linewidth=1.2,
                                        alpha=0.2)

        self.cidZoomInit = self.fig.canvas.mpl_connect('button_press_event',
                                                       self._zoomInit)


    def _zoomInit(self, event):

        if event.inaxes != self.axes:
            return
        else:
            if event.button == 1:
                self.fig.canvas.mpl_disconnect(self.cidZoomInit)

                self.zoomCoords[0] = [event.xdata, event.ydata]
                self.cidZoomDrag = self.fig.canvas.mpl_connect(\
                    'motion_notify_event', self._zoomDrag)
                self.cidZoomEnd = self.fig.canvas.mpl_connect(\
                    'button_release_event', self._zoomEnd)
            elif event.button == 3:
                self.fig.canvas.mpl_disconnect(self.cidZoomInit)

                self.zoomCoords[1,0] = self.image.shape[1]
                self.zoomCoords[1,1] = self.image.shape[0]

                self.axes.set_xlim(self.zoomCoords[:,0])
                self.axes.set_ylim(self.zoomCoords[:,1])

                self.draw()
                self.emit(QtCore.SIGNAL("zoomEnd"))
            else:
                return



    def _zoomDrag(self, event):
        
        if event.inaxes != self.axes:
            return
        else:
            self.zoomCoords[1] = [event.xdata, event.ydata]

            self.zoomFrame.set_x(min(self.zoomCoords[:,0]))
            self.zoomFrame.set_y(min(self.zoomCoords[:,1]))
            self.zoomFrame.set_height(abs(self.zoomCoords[1,1]-
                                          self.zoomCoords[0,1]))
            self.zoomFrame.set_width(abs(self.zoomCoords[1,0]-
                                          self.zoomCoords[0,0]))
            self.draw()


    
    def _zoomEnd(self, event):
        
        if event.button != 1:
            return
        else:
            self.fig.canvas.mpl_disconnect(self.cidZoomDrag)
            self.fig.canvas.mpl_disconnect(self.cidZoomEnd)

            self.axes.set_xlim(np.sort(self.zoomCoords[:,0]))
            self.axes.set_ylim(np.sort(self.zoomCoords[:,1]))

            self.zoomFrame.set_height(0)
            self.zoomFrame.set_width(0)

            self.draw()

            self.emit(QtCore.SIGNAL("zoomEnd"))



    def addData(self, data, **kwargs):
        ## Overplot a list of sources on an image. Also a dummy function for
        ## purpose of picking sources if `picker`-keyword is set. It can be
        ## expanded by defining subclasses of DynamicalImaCanvas with
        ## apropriate methods
        self.allData = np.ma.array(data)

        sizes = 100*(10**(-0.4*(self.allData['mag']-min(self.allData['mag']))))
        sizes[sizes<1]=1.

        self.facecolors = self.allData.size*['none'] 

        self.allDataPlot = self.axes.scatter(self.allData['x'], 
                                             self.allData['y'], s=sizes,
                                             facecolors=self.facecolors,
                                             **kwargs)
        self.draw()

        self.edgecolor = self.allDataPlot.get_edgecolors()[0]

        if self.allDataPlot.get_picker() != None:
            self.pickData = []

        try:
            self.allIDs = self.allData['id']
        except  ValueError:
            self.allIDs = np.arange(self.allData.size, dtype='int32')



    def grabCoords(self, **kwargs):
        ## Dummy function for purpose of grabbing coordinates from an image
        ## Can be expanded by defining subclasses of DynamicalImaCanvas with
        ## apropriate methods
        self.grabXData = []
        self.grabYData = []
        self.grabXFWHM = []
        self.grabYFWHM = []


        self.grabDataPlot, = self.axes.plot(self.grabXData, self.grabYData,
                                            **kwargs)
        self.draw()

