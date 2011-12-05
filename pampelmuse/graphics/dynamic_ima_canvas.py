import matplotlib.pyplot
import numpy as np
from PyQt4 import QtCore
from matplotlib.figure import Figure
from matplotlib.colors import ColorConverter
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas



class DynamicImaCanvas(FigureCanvas):

    def __init__(self, parent=None, width=4, height=4, dpi=100):
        
        self.fig  = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_axes((0.05,0.05,0.9,0.9))
##         self.axes.hold(False)

        self.image = None

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)


    def addImage(self, image, cmap=matplotlib.pyplot.cm.gray_r,
                 interpolation='nearest'):

        if self.image == None:
            self.imagePlot = self.axes.imshow(image, cmap=cmap,
                                              interpolation=interpolation)
        else:
            self.imagePlot.set_data(image)

        self.image = image

        self.cuts  =  (self.image.min(), self.image.max())
        
        self.axes.set_autoscale_on(False)
        self.axes.set_xticks([])
        self.axes.set_yticks([])

        self.draw()


    def changeCuts(self, vrange=None, locut=None, hicut=None):

        oldCuts = self.cuts

        if vrange != None:
            cutlev = float(vrange)/100.
            npix   = self.image.size

            cutpix = int((1-cutlev)*npix)
            
            if cutpix == 0:
                self.cuts = (self.image.min(), self.image.max())
            else:
                values = self.image.copy().reshape(-1)
                values.sort()
                self.cuts = (values[cutpix/2], values[-cutpix/2])
                
        else:
            if locut != None:
                self.cuts = (locut, self.cuts[1])
            if hicut != None:
                self.cuts = (self.cuts[0], hicut)

        self.imagePlot.set_clim(self.cuts[0], self.cuts[1])
        self.draw()

        if self.cuts[0] != oldCuts[0]:
            self.emit(QtCore.SIGNAL("changedLoCut"), self.cuts[0])
        if self.cuts[1] != oldCuts[1]:
            self.emit(QtCore.SIGNAL("changedHiCut"), self.cuts[1])


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

            self.cidZoomInit = self.fig.canvas.mpl_connect('button_press_event',
                                                           self._zoomInit)

    
    def stopZoom(self):
        
        self.fig.canvas.mpl_disconnect(self.cidZoomInit)



    def addData(self, data, **kwargs):
        ## Overplot a list of sources on an image. Also a dummy function for
        ## purpose of picking sources if `picker`-keyword is set. It can be
        ## expanded by defining subclasses of DynamicalImaCanvas with
        ## apropriate methods
        self.allData = np.ma.array(data)

        self.sizes = 100*(10**(-0.4*(self.allData['mag']\
                                         -min(self.allData['mag']))))
        self.sizes[self.sizes<1]=1.

        self.facecolors = self.allData.size*['none'] 

        self.allDataPlot = self.axes.scatter(self.allData['x'], 
                                             self.allData['y'], s=self.sizes,
                                             facecolors=self.facecolors,
                                             **kwargs)
        self.draw()

        edgecolors = self.allDataPlot.get_edgecolors()
        self.basecolor  = edgecolors[0]
        if edgecolors.shape[0] == 1:
            self.edgecolors = np.resize(\
                edgecolors, (self.allData.size,edgecolors.shape[1]))
        else:
            self.edgecolors = edgecolors


        if self.allDataPlot.get_picker() != None:
            self.pickData = []

        try:
            self.allIDs = self.allData['id']
        except  ValueError:
            self.allIDs = np.arange(self.allData.size, dtype='int32')



    def grabCoords(self, calcMoments=True, **kwargs):
        ## Dummy function for purpose of grabbing coordinates from an image
        ## Can be expanded by defining subclasses of DynamicalImaCanvas with
        ## apropriate methods

        self.calcMoments = calcMoments
        
        self.grabXData = []
        self.grabYData = []

        self.grabDataPlot, = self.axes.plot(self.grabXData, self.grabYData,
                                            **kwargs)

        self.curGrabPlot, = self.axes.plot([],[], **kwargs)

        ## invert colors in plot of current selection
        mec = self.grabDataPlot.get_mec()
        self.curGrabPlot.set_mec(self.invertColor(mec))

        mfc = self.grabDataPlot.get_mfc()
        self.curGrabPlot.set_mfc(self.invertColor(mfc))

        c = self.grabDataPlot.get_c()
        self.curGrabPlot.set_c(self.invertColor(c))

        self.draw()


    def invertColor(self, color):
        
        if color == "none":
            return "none"
        else:
            return 1 - np.asarray(ColorConverter().to_rgba(color))

        
