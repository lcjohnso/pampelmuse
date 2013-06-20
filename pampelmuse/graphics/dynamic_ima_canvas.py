"""
DYNAMIC_IMA_CANVAS
==================
DynamicImaCanvas defines a class that is designed to display FITS data
and to interact with it. It provides the basic functionailty like a zoom
option or a pan option and also allows one to interact with the data, like
grabbing coordinates or picking data. The class is used by the PampelMuse
Gui to display IFU data and reference data.


CHANGELOG:
----------
2013/02/03, v0.2.0.0, started to add documentation and increased the size of
                      the circles used to indicate data points.
2013/05/06, v0.3.0.0, added support for viewing layers of datacubes.
2013/05/15, v0.4.0.0, documented the code and revised zoom mode, also added
                      a mode to pan across the data and a history that can be
                      used to go back and forth between different views. 
2013/05/23, v0.4.0.1, added option to provide an instance of the Cube class
                      of PampelMuse when loading IFS data.
2013/05/31, v0.4.0.2, modified method 'addData' so that an existing data set
                      is automatically removed before a new one is plotted.
                      Also improved updating of axes limits when new data is
                      plotted.
2013/06/13, v0.4.0.3, modified plotting of frame around IFU FoV so that the
                      Path instance available in
                      pampelmuse.core.instrument.Instrument is used.
2013/06/19, v0.4.1.3, it happened occasionally that the code showed unexpected
                      behavior in zoom or pan mode when leaving and re-entering
                      the axes because multiple events were triggered. This should
                      be fixed now.
2013/06/20, v0.5.0.0, now using animations to only redraw the elements that are
                      actually affected by a user interaction. This makes the inter-
                      actions much smoother.
"""
__author__ = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__version__ = "0.5.0.0"

import logging
import matplotlib.transforms, matplotlib.artist, matplotlib.pyplot, numpy as np
from PyQt4 import QtCore
from matplotlib import path, patches
from matplotlib.figure import Figure
from matplotlib.colors import ColorConverter
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.patches import Rectangle

##  required PampelMuse routines
from pampelmuse.core.cube import Cube

class DynamicImaCanvas(FigureCanvas):
    """
    Class that has the purpose of displaying FITS images or layers of IFU data.
    It also provides the basic options to modify the data view, such as changing
    the cuts, zooming in/out of the data or panning over it.
    """

    def __init__(self, parent=None, width=4, height=4, dpi=100):
        """
        Initialize a new instance of the DynamicImaCanvas' class.
        
        Parameters
        ----------
        width : float,  optional
            The width (in cm) of the figure canvas used to display
            the data.
        height : float, optional
            The height (in cm) of the figure canvas used to display
            the data.
        dpi : int, optional
            The resolution of the figure canvas.
        """
        
        ## initialize the figure canvas
        self.fig  = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_axes((0.05,0.05,0.9,0.9))

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        
        ## Default setup
        self.cuts = None
        self.vrange = 100

        self.image = None
        self.cube = None
        self.layer = -1
        self.wave = 0
        
        ## The routine keeps a history of previously selected axis limits
        ## in memory. This history can be used to recover previous views
        ## of the data.
        self.curViewIndex = 0
        self.viewHistory = [(self.axes.get_xlim(), self.axes.get_ylim()), ]

        ## Turn of autoscaling so the limits are not changes when
        ## a data set is overplotted on an image.
        self.axes.set_autoscale_on(False)

        ## Initialize the frame used to indicate a zoom into the data.
        self.zoomFrame = Rectangle((0., 0.),  width=0., height=0., fc="None", lw=1.0, ls="solid", ec="0.5",
                                   transform=self.axes.transData, gid="zoomFrame")
        self.axes.add_patch(self.zoomFrame)

        ## start in normal mode (zoom and pan switched off)
        self.zoom = False
        self.pan = False
        self.inEvent = False ## will be True if pan or zoom is going on
        
        ## Handle events when axes instance is left or entered
        self.enterAxes = self.fig.canvas.mpl_connect('axes_enter_event', self._uponEntering)
        self.leaveAxes = self.fig.canvas.mpl_connect('axes_leave_event', self._uponLeaving)


    def draw(self):
        """
        Update the plot to display the latest changes. Following the
        recommendations for animations in matplotlib, we only redraw
        the animated parts of a plot and blit the static parts during
        an animation. To this aim, we save the static background to a
        variable immediately after each call to draw().
        
        """
        ## draw() shows unanimated parts of plot that we also save as a static background for later recovery
        super(DynamicImaCanvas, self).draw()
        self.background = self.figure.canvas.copy_from_bbox(self.axes.bbox)
        
        ## show animated part - needs to be called because draw() only plots static parts
#        if self.vLine.get_animated(): self.axes.draw_artist(self.vLine)
        
        ## and blit just the redrawn (e.g. animated) area
#        self.figure.canvas.blit(self.axes.bbox)


    def addImage(self, image, **kwargs):
        """
        Add a new data to the figure. This method can only
        handle 2dim. data. To load IFU data, the method
        'loadCube' should be used.
        
        Parameters
        ----------
        image : nd_array, 2-dimensional
            The image to be displayed.
        kwargs
            Any keyword arguments that are provided are passed
            on to the call of matplotlib.pyplot.imshow that is
            used to display the data.
        """

        ## Check if an imshow instance does exist, if so, update its data
        if self.image == None:
            self.imagePlot = self.axes.imshow(image, **kwargs)
        else:
            self.imagePlot.set_data(image)

        self.image = image

        self.changeCuts() ## adapt cuts to the new data

        ## adjust axes limits
        self.updateAxesLimits()
        
        ## reset history of views
        self.curViewIndex = 0
        self.viewHistory  = [(self.axes.get_xlim(), self.axes.get_ylim()), ]

        ## do not display any ticks when plotting images
        self.axes.set_xticks([])
        self.axes.set_yticks([])

        self.draw() ## update canvas view



    def loadCube(self, input, instrument=None, checks={}, **kwargs):
        """
        Load an IFU data set, either by providing a filename or a valid
        instance of the 'Cube' class of PampelMuse. Note that if a filename is
        provided, it may reference 2dim. or 3dim. data. In the case that 2dim.
        data is provided, however, 'instrument' must hold information about the
        instrument used to observe the data (either as a string giving the
        instrument name or an instance of a pampelmuse.core.instruments.Instrument
        class).
        No data is actually displayed when calling this method. Use 'loadLayer'
        for this purpose.
        
        Parameters
        ----------
        name : string or an instance of pampelmuse.core.cube.Cube
            If a string is provided, it must be the name of the FITS file containing the
            IFU data.
        instrument : string or pampelmuse.core.instruments.Instrument instance, optional
            Can be either the name of the instrument that observed the data or an
            'Instrument' instance. If neither is provided, 'name' must contain 3dim.
            data. Note that this keyword only has an effect if 'name' contains a filename. 
        checks : dict, optional
            For each key in 'checks', the routine will verify that the FITS header of
            'name' contains a keyword set to the respective value.
        """

        ## load cube using the dedicated PampelMuse class for this purpose that
        ## also checks whether the data is valid (unless an instance was provided right away).
        if isinstance(input, str):
            self.cube = Cube(name, instrument=instrument, checks=checks)
        else:
            self.cube = input
      
        ## define a Path that indicates the edges of the field of view of the instrument
        ## and add it to the figure canvas.
        ## such a Path is available in the pampelmuse.cube.instrument.Instrument instance,
        ## yet the x- and y-axes need to be swapped.
        t = matplotlib.transforms.Affine2D(np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]))
        plotpath = self.cube.instrument.edgepath.transformed(t)
      
        edgepatch = patches.PathPatch(plotpath, facecolor="None", edgecolor="r", lw=2, alpha=0.5)
        self.axes.add_patch(edgepatch)
      
      
    def loadLayer(self, i,  **kwargs):
        """
        Display an individual layer from a loaded datacube. To display the
        data, the method will call 'addImage'.
        
        Parameters
        ----------
        i : int
            The layer of the IFU data that should be loaded.
        kwargs
            Any keyword arguments provided to this function will be passed
            on to the call of 'addImage' that is used to display the layer.
        """
        ## check if a cube has been loaded, complain if not
        if self.cube == None:
            logging.error("Cannot load layer as no cube is open.")
            return
            
        ## get the requested layer, this depends on structure of the data
        elif self.cube.ndim==3:
            self.addImage(self.cube.data_hdu.section[i], **kwargs)
        elif self.cube.instrument.spectral_axis==2:
            self.addImage(self.cube.instrument.make_grid(self.cube.data_hdu.section[i],  fill_value=0.), **kwargs)
        else:
            self.addImage(self.cube.instrument.make_grid(self.cube.data_hdu.section[:, i],  fill_value=0.), **kwargs)

        ## store currrent wavelength and number of layer
        self.wave = self.cube.wave[i]
        self.layer = i


    def changeCuts(self, vrange=None, locut=None, hicut=None):
        """
        Modify the cuts used to display the data thas is currently loaded
        in the canvas. Absolute values as well as relative cuts can be provided.
        
        Parameters
        ----------
        vrange : float, optional
            A relative cut level in percent. The cuts will be changed to encompass
            'vrange'  percent of the pixel values.
        locut : float, optional
            The absolute value of the lower cut level.
        hicut : float, optional
            The absolute value of the upper cut level.
        """

        ## get current cut levels
        oldCuts = self.imagePlot.get_clim()
        if self.cuts == None: self.cuts = oldCuts

        ## check if absolute cuts have been provided
        if locut!=None or hicut != None:
            if locut != None:
                self.cuts = (locut, self.cuts[1])
            if hicut != None:
                self.cuts = (self.cuts[0], hicut)
                
        ## check if relative cuts have been provided, if so, translate
        ##  them into absolute cut levels.
        elif vrange != None:
            self.vrange = vrange
            cutlev = float(self.vrange)/100.
            npix   = self.image.size
            cutpix = int((1-cutlev)*npix)
            
            if self.layer != -1: ## if cube loaded, only use science fibers to determine cuts
                if self.cube.ndim == 3 or self.cube.instrument.spectral_axis==2:
                    values = self.cube.instrument.get_science(self.cube.data_hdu.section[self.layer])
                else:
                    values = self.cube.instrument.get_science(self.cube.data_hdu.section[:, i])
            else:
                values = self.image.copy().reshape(-1)
            values.sort()
            self.cuts = (values[cutpix/2], 0.5*(values[-cutpix/2-1]+values[-cutpix/2-2]))

        ## change cuts and update canvas
        self.imagePlot.set_clim(self.cuts[0], self.cuts[1])
        self.draw()

        ## emit signal that cuts were changed
        if self.cuts[0] != oldCuts[0]:
            self.emit(QtCore.SIGNAL("changedLoCut"), self.cuts[0])
        if self.cuts[1] != oldCuts[1]:
            self.emit(QtCore.SIGNAL("changedHiCut"), self.cuts[1])


    def updateAxesLimits(self):
        """
        Automatically adjust the axis limits when a new image is loaded or
        when new data is overplotted on the image.
        The reason for doing this in a dedicated method is that the method
        matplotlib.axes.Axes.relim() does not work with instances of
        matplotlib.Collection and each scatter plot is an instance of the
        sub-class PathCollection.
        """

        axesMin = [0, 0]
        axesMax = [1, 1]

        if self.image is not None:
            axesMin = [-0.5, -0.5]
            axesMax = [self.image.shape[1]-0.5, self.image.shape[0]-0.5]

        if hasattr(self, "dataPlot") and isinstance(self.dataPlot, matplotlib.artist.Artist):
            dataMin, dataMax = self.dataPlot.get_datalim(self.axes.transData).get_points()
            
            dataMinRound = [int(dataMin[0]-1*(dataMin[0]<0)), int(dataMin[1]-1*(dataMin[1]<0))] 
            dataMaxRound = [int(dataMax[0]+1), int(dataMax[1]+1)]
        
            axesMin = np.min([axesMin, dataMinRound], axis=0)
            axesMax = np.max([axesMax, dataMaxRound], axis=0)

        self.axes.set_xlim(axesMin[0], axesMax[0])
        self.axes.set_ylim(axesMin[1], axesMax[1])


    def getNearestSpaxel(self, x, y):
        """
        Return the spectrum of the nearest spaxel relative to a provided
        (x, y) position.
        
        Parameters
        ----------
        x : float
            The x-coordinate
        y : float
            The y-coordinate
        """
        
        ## return if no cube is loaded
        if self.cube == None: return
        
        ## get requested section of HDU
        elif self.cube.ndim == 3:
            return self.cube.data_hdu.section[:, int(round(y)), int(round(x))]
        else:
            dr = self.cube.instrument.dist2spax(x, y)

            ## need to take into account calibration/sky fibers that might be present
            ## in the data.
            i_min  = dr.argmin()
            f_min  = np.where((np.cumsum(self.cube.instrument.isscience)-1)==i_min)[0][0]
        
            if self.cube.instrument.spectral_axis==2:
                return self.cube.data_hdu.section[:, f_min]
            else:
                return self.cube.data_hdu.section[f_min]


    def addData(self, x, y, ids=None, **kwargs):
        """
        Overplot a dataset on an image as a scatterplot. If the `picker`-keyword
        is provided within kwargs, it is possible to interactively select data from
        the canvas once the data set is shown.
        
        Parameters
        ----------
        x : array_like
            The x-coordinates of the data.
        y : array_like
            The y-coordinates of the data. Must of course have same size
            as 'x'.
        ids : array_like, optional
            Can be used to provide a list containing an ID of every source
            in the dataset. The ID is not used in this method, it is defined
            as a placeholder for methods that deal with the picking of the
            data points that build up on this method.
        kwargs
            Any keyword arguments provided to this function are passed on to the
            call of matplotlib.pyplot.scatter that is used to diisplay the data.
        """
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        
        assert self.x.shape == self.y.shape, "x- and y-coords must have same shape."

        ## if previously data have been added, that is removed
        if hasattr(self, "dataPlot") and isinstance(self.dataPlot, matplotlib.artist.Artist):
            self.dataPlot.remove()

        self.dataPlot = self.axes.scatter(self.x, self.y, **kwargs)
    
        ## change axes limits to show all the data
        self.updateAxesLimits()
        
        ## reset history of views
        self.curViewIndex = 0
        self.viewHistory  = [(self.axes.get_xlim(), self.axes.get_ylim()), ]
        
        self.draw()

        ## check if IDs were provided, otherwise numerate sources
        if ids != None:
            self.ids = ids
        else:
            self.ids = np.arange(self.x.size, dtype='int32')


    @property
    def edgecolors(self):
        """
        Return the edgecolors of the markers of the data set overplotted
        on an image. The return value will be an array of the same length
        as the dataset even if all  markers have the same edgecolor.
        """

        ## return if no plot is available
        if not "dataPlot" in self.__dict__: return None
    
        ## get current edgecolors from dataset, they are returned as
        ## RGBA values, so the length of the second dimension of the
        ## return value is 4, the length of the first depends on how
        ## the colors were initialized
        edgecolors = self.dataPlot.get_edgecolors()
        
        if edgecolors.shape[0] == 0: ## if colors were initialized to "None"
            return np.zeros((self.x.size, edgecolors.shape[1]), dtype=np.float32)
        elif edgecolors.shape[0] == 1: ## if a single color was provided
            return np.resize(edgecolors, (self.x.size, edgecolors.shape[1]))
        else: ## if different colours were provided
            return edgecolors


    @property
    def facecolors(self):
        """
        Return the facecolors of the markers of the data set overplotted
        on an image. The return value will be an array of the same length
        as the dataset even if all markers have the same facecolor.
        """
        
        ## return if no plot is available
        if not "dataPlot" in self.__dict__: return None
        
        ## get current facecolors from dataset, they are returned as
        ## RGBA values, so the length of the second dimension of the
        ## return value is 4, the length of the first depends on how
        ## the colors were initialized
        facecolors = self.dataPlot.get_facecolors()

        if facecolors.shape[0] == 0: ## if colors were initialized to "None"
            return np.zeros((self.x.size, facecolors.shape[1]), dtype=np.float32)
        elif facecolors.shape[0] == 1: ## if a single color was provided
            return np.resize(facecolors, (self.x.size, facecolors.shape[1]))
        else: ## if different colours were provided
            return facecolors


    def _uponEntering(self, event):
        """
        Handle event when the axes instance is entered by mouse. Connect
        mouse click event to begin of a zoom or a pan event if the current
        mode is either of the two.
        
        Parameters
        ----------
        event : a matplotlib.backend_bases.LocationEvent instance
            The event that triggered the method call.
        """
        if event.inaxes != self.axes:
            return ## do noting if wrong axes were entered
        
        try: ## disconnect initialization handler that might still be active
            self.fig.canvas.mpl_disconnect(self.Init)
        except:
            pass
        
        ## initialize handler for mouse click unless a zoom or pan event is still active
        if not self.inEvent:
            self.Init = self.fig.canvas.mpl_connect('button_press_event', self.on_mouseButton_pressed)

        

    def _uponLeaving(self, event):
        """
        Handle event when the axes instance is left by mouse. Disconnect
        the handler for a mouse click event.
        
        Parameters
        ----------
        event : a matplotlib.backend_bases.LocationEvent instance
            The event that triggered the method call.
        """
        if event.inaxes != self.axes:
            return

        ## disconnect handlers for mouse movement and mouse clicks
        self.fig.canvas.mpl_disconnect(self.Init)




    def on_mouseButton_pressed(self, event):
        """
        Handle event when a mouse button is pressed over the axis
        instance.
        What precisely the method does depends on the active mode.
        In normal mode, currently nothing happens. In zoom or pan
        mode, a new selection is initialized.
        
        Parameters
        ----------
        event : a matplotlib.backend_bases.MouseEvent instance
            The event that triggered the method call.
        """

        if not self.zoom and not self.pan:
            return

        self.button = event.button
        self.xInit = event.xdata
        self.yInit = event.ydata

        ## In zoom mode, right mouse button causes recovery of
        ## original view.
        if self.zoom:
            if event.button == 3: ## recover original view.
                self.gotoView(0)
                return
            self.zoomFrame.set_xy((self.xInit,self.yInit))
            self.zoomFrame.set_animated(True)
        elif self.pan: ## for panning, we need current axis limits
            self.panWindowX = self.axes.get_xlim()
            self.panWindowY = self.axes.get_ylim()

        ## disconnect initialization event to avoid multiple events.
        self.fig.canvas.mpl_disconnect(self.Init)
        self.inEvent = True

        ## initialize actions for mouse movement with button pressed and
        ## release of button
        self.MotionKey = self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouseMotion_notify)
        self.Final = self.fig.canvas.mpl_connect('button_release_event', self.on_mouseButton_released)


    def on_mouseButton_released(self, event):
        """
        Handle event when a mouse button is released. This event implies
        that a selection (a zoom or a pan) is completed. So we reset the status of
        the axis to be prepared for the start of the next selection. In zoom mode,
        the axis limits are updated to the selected window.
        
        Parameters
        ----------
        event : a matplotlib.backend_bases.MouseEvent instance 
            The event that triggered the method call.
        """
        ## check if the same button was released that triggered
        ## the initialization call to '_grabCoordsInit'
        if event.button != self.button:  return
        
        ## disconnect handlers for button release and movement
        ## with mouse button pressed.
        self.inEvent = False
        self.fig.canvas.mpl_disconnect(self.Final)
        self.fig.canvas.mpl_disconnect(self.MotionKey)

        ## get coordinates where event took place and edges of
        ## selected rectangle.
        if event.inaxes == self.axes:
            self.xFinal = event.xdata
            self.yFinal = event.ydata

        if self.xInit != None and self.yInit != None:
            xmin, xmax = sorted([self.xInit, self.xFinal])
            ymin, ymax = sorted([self.yInit, self.yFinal])
    
            if self.zoom: ## if zoom, change limits to selected window
                if xmin==xmax or ymin==ymax:
                    logging.error("Area selection must be 2dimensional.")
                else:
                    self.axes.set_xlim(xmin, xmax)
                    self.axes.set_ylim(ymin, ymax)
                    self.draw()
                self.zoomFrame.set_width(0) ## minimize zoom window
                self.zoomFrame.set_height(0)
                self.zoomFrame.set_animated(False)
    
            self.curViewIndex += 1
            self.viewHistory.insert(self.curViewIndex, (self.axes.get_xlim(), self.axes.get_ylim()))
            if len(self.viewHistory) > self.curViewIndex+1:
                self.viewHistory = self.viewHistory[:self.curViewIndex+1]

        ## initialize handler for next selection event
        self.Init = self.fig.canvas.mpl_connect('button_press_event', self.on_mouseButton_pressed)



    def on_mouseMotion_notify(self, event):
        """
        Handle event when mouse is moved over the axis with
        a button pressed. The behavior depends on the selected
        mode. In normal mode, nothing happens. In zoom mode, a
        rectangle is spanned indicating the selected window.
        And in pan mode, the x- and y-limits of the plots are
        updated while the mouse moves.
        
        Parameters
        ----------
        event : a matplotlib.backend_bases.MouseEvent instance
            The event that triggered the method call.
        """

        if event.inaxes != self.axes:
            return

        if self.xInit == None or self.yInit == None:
            return

        ## in case the button is released outside the axes, we use the
        ## coordinates of the last motion event inside the axis
        self.xFinal = event.xdata
        self.yFinal = event.ydata

        if self.zoom: ##  span up zoom window
            self.zoomFrame.set_width(self.xFinal-self.xInit)
            self.zoomFrame.set_height(self.yFinal-self.yInit)

            self.figure.canvas.restore_region(self.background)
            self.axes.draw_artist(self.zoomFrame)
            self.figure.canvas.blit(self.axes.bbox)

        elif self.pan: ## change x- and y-limits
            dx = (self.xInit - self.xFinal)/2.
            dy = (self.yInit - self.yFinal)/2.
            self.axes.set_xlim(self.panWindowX[0]+dx, self.panWindowX[1]+dx)
            self.axes.set_ylim(self.panWindowY[0]+dy, self.panWindowY[1]+dy)
            self.draw()


    def gotoView(self,  index):
        """
        Recover a previously used view of the data from the history of the
        displayed axis limits.
        
        Parameters
        ----------
        index : int
            The index of the view to be recovered. The current index can be
            accessed via the property 'curViewIndex'.
        """
        
        ##  check if requested index is valid
        if index >= len(self.viewHistory) or index < -len(self.viewHistory):
            logging.error("Requested view index is out of range.")
            return
    
        ## update current index
        self.curViewIndex = index

        ## change axis limits
        self.axes.set_xlim(self.viewHistory[index][0])
        self.axes.set_ylim(self.viewHistory[index][1])
        
        self.zoomFrame.set_width(0) ## minimize zoom  frame
        self.zoomFrame.set_height(0)
        
        self.draw()
