"""
DYNAMICPLOTCANVAS
=================
Routine is used to create a lineplot and interact with it. The PampelMuse Gui
uses this routine to display spectral properties (source spectra, x- & y-
positions, PSF parameters) and their polynomial fits.

CHANGELOG:
----------
2012/01/25, v0.0.1.1, added some documentation. Fixed bug that sometimes the
                      vertical line indicating the current cursor position did
                      not disappear upon updating  the data to be plotted.
2012/01/25, v0.0.1.2, modified the dimensions of the plot to make it fit better
                      into the surrounding frame.
2013/02/04, v0.1.0.0, added possibility to add/update x- and y-labels. Minor
                      changes to how canvas/axes are plotted.
2013/05/06, v0.2.0.0, made code more robust with respect to user interaction.
                      It should not happen any more that the 'masker' does
                      not disappear after the button is released.
                      Methods to zoom into a plot or out of it and to pan have
                      been added.
2013/05/14, v0.3.0.0, significantly enhanced documentation of the code.
2013/05/15, v0.3.1.0, bugfix: a '*' was missing when emitting the 'rangeSelected'
                              signal.
2013/05/23, v0.3.2.0, bugfix: when updating the data, the lines that reset the
                              heights of the rectangles were commented out. For this
                              reason, the y limits of the plot were not adjusted
                              properly.
2013/06/20, v0.4.0.0, now using animations to only redraw the elements that are
                      actually affected by a user interaction. This makes the inter-
                      actions much smoother.
2013/06/21, v0.4.1.0, for some reason I cannot figure out calling the blit()-method
                      in call() causes the GUI to not update the axes ticks and labels
                      until the GUI gets out of focus. So I commented that line out.
"""
__author__  = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__version__ = "0.4.1.0"

import inspect, logging
import matplotlib.pyplot
import numpy as np
from PyQt4 import QtCore
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory
matplotlib.pyplot.rcParams["axes.linewidth"]=1.5

class DynamicPlotCanvas(FigureCanvasQTAgg):
    """
    This class can be used to display a matplotlib lineplot and
    interact with it. The possible types of interaction include
    event handling and modifications of the data. It currrently
    supports three modes of event handling:
    normal: by mouse clicks and movements, horizontal ranges can
            be selected.
    zoom: One can zoom in and out of the plot using mouse clicks
          and movements.
    pan: Move mouse over axes with button pressed to change x- and
         y-limits while keeping the size of the window.
    """

    def __init__(self, parent=None, width=9, height=5, dpi=100):
        """
        Create a new instance of the class 'DynamicPlotCanvas'
        
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
        
        ## prepare figure canvas
        self.fig  = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_axes((0.15, 0.20, 0.82, 0.76))
        self.canvas = self.fig.canvas

        super(DynamicPlotCanvas, self).__init__(self.fig)
        self.setParent(parent)

        ## dictionary to keep track of the data sets that are plotted.
        self.thePlots = {}

        ## a vertical line that can later be used to hightlight the cursor position on the canvas.
        self.vLine, = self.axes.plot([0.5, 0.5], [0, 1], ls='--', color='r')
        self.vLine.set_animated(True) ## make animated for smoother interaction

        ## prepare structure to hold a set of rectangles to be
        ## overplotted on th data. In PampelMuse, this is used to
        ## display masks.

        ## to achieve axes units (bottom to top = 0 to 1) in
        ## vertical direction and data units (wavelength) in
        ## horizontal direction:
        self.bar_transform = blended_transform_factory(self.axes.transData, self.axes.transAxes)
        self.bar_gid = [] ## will contain the ID of every rectangle
        
        ## the mask selector. It will be spanned by mouse mouvements over the canvas while the left button is pressed.
        selector = Rectangle((0.,0.), width=0., height=1., alpha=0.5,fc='r', lw=0, transform=self.bar_transform, gid="selector")
        self.axes.add_patch(selector)
        self.bar_gid.append(selector)
        
        ## The frame used to indicate a zoom into the data.
        zoomFrame = Rectangle((0., 0.), width=0., height=0., fc="None", lw=1.0, ls="solid", ec="0.5", transform=self.axes.transData, gid="zoomFrame")
        self.axes.add_patch(zoomFrame)
        self.bar_gid.append(zoomFrame.get_gid())

        ## Handle events when axes instance is left or entered
        self.enterAxes = self.fig.canvas.mpl_connect('axes_enter_event', self._uponEntering)
        self.leaveAxes = self.fig.canvas.mpl_connect('axes_leave_event', self._uponLeaving)

        ## The initial data dimensions of the axes instance
        self.fullXRange = (0, 1)
        self.fullYRange = (0, 1)

        ## start in normal mode (zoom and pan switched off)
        self.zoom = False
        self.pan = False
        self.inEvent = False ## will be True if selection, pan or zoom is going on

        self.draw()


    def draw(self):
        """
        Update the plot to display the latest changes. Following the
        recommendations for animations in matplotlib, we only redraw
        the animated parts of a plot and blit the static parts during
        an animation. To this aim, we save the static background to a
        variable immediately after each call to draw().
        
        """
        ## draw() shows unanimated parts of plot that we also save as a static background for later recovery
        super(DynamicPlotCanvas, self).draw()
        self.background = self.figure.canvas.copy_from_bbox(self.axes.bbox)
        
        ## for debugging: check where the call was triggered
        curframe = inspect.currentframe()
        calframe = inspect.getouterframes(curframe, 2)
        logging.debug("call to draw() triggered by: {0} in {1}[@{2}]".format(calframe[1][3], calframe[1][1], calframe[1][2]))
        
        ## show animated part - needs to be called because draw() only plots static parts
        if self.vLine.get_animated():
            self.axes.draw_artist(self.vLine)
        
            ## and blit just the redrawn (e.g. animated) area
            ## self.figure.canvas.blit(self.axes.bbox) ## for some reason I do not understand, uncommenting this line causes the plot to not update the labels
                                                       ## and ticks of the axes until the GUI is out of focus...


    def addData(self, name, xData, yData, **kwargs):
        """
        Add a new data set to the axes.
        
        Parameters
        ----------
        name : string
            Each plot must bbe given a name that is used to identify it.
        xData : array_like
            The data along the x-axis.
        yData : array_like
            The data along the y-axis. Must of course have the same length
            as 'xData'.
        kwargs
            The keyword arguments are passed on to the matplotlib.pyplot.plot
            command that is used to initialize the data plot.
        """

        ## plot the data using matplotlib
        self.thePlots[name] = self.axes.plot(xData, yData, **kwargs)[0]
        
        ## set minor ticks on x- and y-axis on
        self.axes.minorticks_on()

        ## increase strength of major and minor ticks to avoid hair lines.
        for line in self.axes.xaxis.get_ticklines() \
                + self.axes.yaxis.get_ticklines() \
                + self.axes.xaxis.get_minorticklines() \
                + self.axes.yaxis.get_minorticklines():
            line.set_markeredgewidth(1.5)

        ## update canvas
        self.draw()


    def addRectangle(self, xy, width, height=1, **kwargs):
        """
        Add a new rectangle to the canvas if its ID is unknown,
        otherwise update an existing rectangle.
        NOTE that the scaling is in data units along the x-axis
        but in relative axis units along the y-axis.
        
        Parameters
        ----------
        xy : (float, float)
            The coordinates of the lower left corner of the rectangle.
        width : float
            The width of the rectangle in data units
        height : float, optional
            The height of the rectangle in realtive units, i.e height=1
            means from bottom to top.
        kwargs
            Any keyword argument is passed on to the matplotlib.patches.Rectangle
            command used to prepare the rectangle.
        """

        ## Check if an identifier has been provided, otherwise define one. 
        if not "gid" in kwargs:
            kwargs["gid"] = "rect{0}".format(len(self.axes.patches))

        ##  check if rectangle is a new one or an update to an existing one
        ## using the identifier.
        if kwargs["gid"] in self.bar_gid: ## update existing rectangle
            i = self.bar_gid.index(kwargs["gid"])
            self.axes.patches[i].set_xy(xy)
            self.axes.patches[i].set_width(width)
        else: ## add new rectangle
            rect = Rectangle(xy, width, height, transform=self.bar_transform,
                             **kwargs)

            self.axes.add_patch(rect)
            self.bar_gid.append(rect.get_gid())
            

    def delRectangle(self, gid):
        """
        Delete an existing rectangle from the collection. The identification
        is done using the 'gid'.
        
        gid : string
            The identifier of the rectangle to be deleted.
        """
        
        try: ## check if provided identifier exists, if so, remove rectangle
            i = self.bar_gid.index(gid)
            del self.axes.patches[i]
            del self.bar_gid[i]
        except IndexError: ## otherwise complain
            logging.error("No rectangle instance has gid '{0}'.".format(gid))


    def updateData(self, name, xData=None, yData=None, xLabel=None,
                   yLabel=None, updateYRange=True):
        """
        Update an existing data plot.
        
        Parameters
        ----------
        name : string
            The identifier of the data set to be updated.
        xData : array_like, optional
            The new data along the x-axis. Note that it can have a different size
            than the current data.
        yData : array_like, optional
            The new data along the y-axis. Must of course have the same length
            as the new 'xData'.
        xLabel : string, optional
            The new label for the x-axis.
        yLabel : string, optional
            The new label for the y-axis.
        updateYRange : bool, optional
            If this flag is set, the axes limits will be adjusted so that all the
            new data is shown.
        """

        ## return if no new data is provided
        if xData==None and yData==None:
            return

        ## check if plot to be updated exists
        if not name in self.thePlots.keys():
            logging.error("Cannot update plot '{0}': unknown identifier.".format(name))

        ## update data
        if xData!=None:
            self.thePlots[name].set_xdata(xData)
        if yData!=None:
            self.thePlots[name].set_ydata(yData)

        ## reset vertical line used to highlight cursor position
        self.vLine.set_data([],[])

        for i in xrange(len(self.axes.patches)):
            self.axes.patches[i].set_height(0)

        ## update axes limits if requested
        if updateYRange:
            self.axes.axes.relim()
            self.axes.autoscale(tight=False)

            self.fullXRange = self.axes.get_xlim()
            self.fullYRange = self.axes.get_ylim()  

        ## update labels if requested
        if xLabel != None:
            self.axes.set_xlabel(xLabel, fontsize=20)

        if yLabel != None:
            self.axes.set_ylabel(yLabel, fontsize=20)

        ## modify height of rectangles.
        for i in xrange(len(self.axes.patches)):
            self.axes.patches[i].set_height(1)

        self.draw() ## update canvas.


    def _uponEntering(self, event):
        """
        Handle event when the axes instance is entered by mouse.
        
        Parameters
        ----------
        event : a matplotlib.backend_bases.LocationEvent instance
            The event that triggered the method call.
        """

        if event.inaxes != self.axes:
            return ## do noting if wrong axes were entered
        
        try: ## disconnect handlers that might still be active
            self.fig.canvas.mpl_disconnect(self.Init)
            self.fig.canvas.mpl_disconnect(self.Motion)
        except:
            pass

        ## initialize handlers for mouse movement & click unless a zoom or pan event is still active
        if not self.inEvent:
            self.Init = self.fig.canvas.mpl_connect('button_press_event', self._grabCoordsInit)
            self.Motion= self.fig.canvas.mpl_connect('motion_notify_event', self._motionNotify)

        

    def _uponLeaving(self, event):
        """
        Handle event when the axes instance is left by mouse.
        
        Parameters
        ----------
        event : a matplotlib.backend_bases.LocationEvent instance
            The event that triggered the method call.
        """

        ## reset cursor indicator
        self.vLine.set_data([],[])
        self.axes.draw_artist(self.vLine)
        
        if event.inaxes != self.axes:
            return

        ## disconnect handlers for mouse movement and mouse clicks
        self.fig.canvas.mpl_disconnect(self.Init)
        self.fig.canvas.mpl_disconnect(self.Motion)




    def _grabCoordsInit(self,  event):
        """
        Handle event when a mouse button is pressed over the axis
        instance.
        What precisely the method does depends on the active mode.
        In any case, the routine will keep track opf the x- & y-
        coordinates where the event took place.
        
        Parameters
        ----------
        event : a matplotlib.backend_bases.MouseEvent instance
            The event that triggered the method call.
        """
        ## change status to make sure no second event is triggered
        self.inEvent = True

        ##  Only one such event is allowed.
        self.figure.canvas.mpl_disconnect(self.Init)

        ## disable normal motion event
        self.figure.canvas.mpl_disconnect(self.Motion)

        self.button = event.button
        self.xInit = event.xdata
        self.yInit = event.ydata

        ## In zoom mode, behaviour depends on clicked mouse button:
        ## *right mouse button causes recovery of original view.
        ## *left mouse button: animate zoom rectangle and set its origin to the current mouse location
        if self.zoom:
            if event.button == 3:
                self.axes.set_xlim(self.fullXRange)
                self.axes.set_ylim(self.fullYRange)
                self.axes.patches[1].set_width(0)
                self.axes.patches[1].set_height(0)
                self.draw()
                self.inEvent = False ## event ends immediately 
                return
            self.axes.patches[1].set_animated(True)
            self.axes.patches[1].set_xy((self.xInit,self.yInit))
        elif self.pan: ## for panning, we need current axis limits
            self.panWindowX = self.axes.get_xlim()
            self.panWindowY = self.axes.get_ylim()
        else: ## in normal mode, set start coordinates for 'selector' rectangle and animate zoom rectangle
            self.axes.patches[0].set_animated(True)
            self.axes.patches[0].set_xy((self.xInit,0))
            

        ## initialize actions for mouse movement with button pressed and release of button
        self.MotionKey = self.fig.canvas.mpl_connect('motion_notify_event', self._motionNotifyKeyPressed)
        self.Final = self.fig.canvas.mpl_connect('button_release_event', self._grabCoordsFinal)


    def _grabCoordsFinal(self, event):
        """
        Handle event when a mouse button is released. This event implies
        that a selection (a zoom window, a masked range or a pan) is completed.
        So we reset the status of the axis to be prepared for the start of the
        next selection. In zoom mode, the axis limits are updated to the selected
        window.
        
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
        self.fig.canvas.mpl_disconnect(self.Final)
        self.fig.canvas.mpl_disconnect(self.MotionKey)

        ## get coordinates where event took place and edges of
        ## selected rectangle.
        if event.inaxes == self.axes:
            self.xFinal = event.xdata
            self.yFinal = event.ydata

        xmin, xmax = sorted([self.xInit, self.xFinal])
        ymin, ymax = sorted([self.yInit, self.yFinal])

        if self.zoom: ## if zoom, change limits to selected window
            if xmin==xmax or ymin==ymax:
                logging.error("Area selection must be 2dimensional.")
            else:
                self.axes.set_xlim(xmin, xmax)
                self.axes.set_ylim(ymin, ymax)
                self.draw()

            ## recover initial state of zoom window
            self.axes.patches[1].set_width(0)
            self.axes.patches[1].set_height(0)

            ## turn off animation of zoom rectangle
            self.axes.patches[1].set_animated(False)
    
        elif not self.pan: ## if normal mode was one, trigger signal about a newly selected range
            self.emit(QtCore.SIGNAL("rangeSelected"), *[(xmin, xmax), event.button==1])

            ## recover initial state of selection rectangle
            self.axes.patches[0].set_width(0)

            ## turn off animation of zoom rectangle
            self.axes.patches[0].set_animated(False)

        ## initialize handler for next selection event
        self.Init = self.fig.canvas.mpl_connect('button_press_event', self._grabCoordsInit)

        ## intialize handler for normal motions again
        self.Motion= self.fig.canvas.mpl_connect('motion_notify_event', self._motionNotify)

        ## no event taking place anymore
        self.inEvent = False



    def _motionNotify(self, event):
        """
        Handle event when mouse is moved over the axis without
        a button pressed. The position of the vertical line is
        updated to keep track of the mouse position.
        
        Parameters
        ----------
        event : a matplotlib.backend_bases.MouseEvent instance
            The event that triggered the method call.
        """

        if self.zoom or self.pan: return
        if event.inaxes != self.axes: return
        
        self.vLine.set_data([event.xdata,event.xdata], self.axes.get_ylim())
        
        ##restore the background region
        self.figure.canvas.restore_region(self.background)

        ## redraw just the current rectangle
        self.axes.draw_artist(self.vLine)

        ## blit just the redrawn area
        self.figure.canvas.blit(self.axes.bbox)



    def _motionNotifyKeyPressed(self, event):
        """
        Handle event when mouse is moved over the axis with
        a button pressed. The behavior depends on the selected
        mode. In normal mode, a rectangle indicates the selected
        range along the x-axis. In zoom mode, a rectangle is
        spanned indicating the selected window. And in pan mode,
        the x- and y-limits of the plots are updated while the
        mouse moves.
        
        Parameters
        ----------
        event : a matplotlib.backend_bases.MouseEvent instance
            The event that triggered the method call.
        """

        if event.inaxes != self.axes: return
        if self.thePlots == {}: return

        ## in case the button is released outside the axes, we use the
        ## coordinates of the last motion event inside the axis
        self.xFinal = event.xdata
        self.yFinal = event.ydata
            
        if self.pan: ## change x- and y-limits
            dx = (self.xInit - self.xFinal)/2.
            dy = (self.yInit - self.yFinal)/2.
            self.axes.set_xlim(self.panWindowX[0]+dx, self.panWindowX[1]+dx)
            self.axes.set_ylim(self.panWindowY[0]+dy, self.panWindowY[1]+dy)
            
            ## in panning mode, we need to redraw complete canvas (slow!)
            self.draw()
        
        else: ## in other modes, we only redraw animated elements (fast!)
            if self.zoom: ##  span up zoom window
                self.axes.patches[1].set_width(self.xFinal-self.xInit)
                self.axes.patches[1].set_height(self.yFinal-self.yInit)
            else: ## modify width of 'selector' rectangle
                self.axes.patches[0].set_width(self.xFinal-self.xInit)
            
            ## restore the background region
            self.figure.canvas.restore_region(self.background)

            ## redraw just the current rectangle
            if self.zoom: self.axes.draw_artist(self.axes.patches[1])
            else: self.axes.draw_artist(self.axes.patches[0])

            ## blit just the redrawn area
            self.figure.canvas.blit(self.axes.bbox)
