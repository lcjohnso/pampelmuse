"""
dynamic_ima_canvas.py
=====================
Copyright 2013-2024 Sebastian Kamann

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
class pampelmuse.graphics.dynamic_ima_canvas.DynamicImaCanvas

Purpose
-------
This module provides the class DynamicImaCanvas defines that is designed
to display FITS image data (2D or 3D are supported) and to interact with it.
It provides basic functionality like grabbing coordinates or picking data.
The class is used by the PampelMuse GUI to display IFS data and reference
data. In order to handle 2dim. IFS data, the Cube class is loaded from
pampelmuse.core.cube

Latest Git revision
-------------------
2024/04/08
"""
import logging
import numpy as np
from matplotlib import collections, patches
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from PyQt5 import QtCore, QtWidgets
from ..instruments import Instrument
from ..utils.fits import open_ifs_data


__author__ = "Sebastian Kamann (s.kamann@ljmu.ac.uk)"
__revision__ = 20240408


class DynamicImaCanvas(FigureCanvasQTAgg):
    """
    Class that has the purpose of displaying FITS images or layers of IFU data.
    It also provides the basic options to modify the data view, such as changing
    the cuts.
    """
    background = None
    image_plot = None
    data_plot = None
    x = None  # x-values of data_plot
    y = None  # y-values of data_plot
    ids = None

    # define custom signals
    layer_changed = QtCore.pyqtSignal(int, name='layerChanged')
    lower_cut_changed = QtCore.pyqtSignal(float, name='loCutChanged')
    upper_cut_changed = QtCore.pyqtSignal(float, name='hiCutChanged')
    pointer_position_changed = QtCore.pyqtSignal(int, int, float, name='coordsChanged')

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

        # initialize the figure canvas
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_axes((0.05, 0.05, 0.9, 0.9))

        # FigureCanvasQTAgg.__init__(self, self.fig)
        super(DynamicImaCanvas, self).__init__(self.fig)
        self.setParent(parent)

        FigureCanvasQTAgg.setSizePolicy(self,
                                        QtWidgets.QSizePolicy.Expanding,
                                        QtWidgets.QSizePolicy.Expanding)
        FigureCanvasQTAgg.updateGeometry(self)

        # Default setup
        self.cuts = None
        self.vrange = 100

        self.image = None
        self.cube = None
        self.layer = -1
        self.wave = 0

        self.wcs = None

        self.data_plot = None

        # Turn of autoscaling so the limits are not changes when a data set is overplotted on an image.
        self.axes.set_autoscale_on(False)

        # Connect to callback function that will trigger QtCore.SIGNAL containing current x, y and flux information
        self.mpl_connect('motion_notify_event', self.on_mouseMotion_notify)

    def draw(self):
        """
        Update the plot to display the latest changes. Following the
        recommendations for animations in matplotlib, we only redraw
        the animated parts of a plot and blit the static parts during
        an animation. To this aim, we save the static background to a
        variable immediately after each call to draw().
        
        """
        # draw() shows unanimated parts of plot that we also save as a static background for later recovery
        super(DynamicImaCanvas, self).draw()
        self.background = self.figure.canvas.copy_from_bbox(self.axes.bbox)

    def addImage(self, image, update_axlims=True, wcs=None, **kwargs):
        """
        Add a new data to the figure. This method can only handle 2dim. data.
        To load IFS data, the method 'loadCube' should be used.
        
        Parameters
        ----------
        image : nd_array, 2-dimensional
            The image to be displayed.
        update_axlims : boolean, optional
            Flag indicating  whether or not the limits of the x- and y-
            axis should be updated to display the whole image.
        wcs : instance of astropy.wcs.WCS, optional
            Can be used to convert pixel to world coordinates and vice versa.
        kwargs
            Any additional keyword arguments that are provided to this
            method are passed on to the call of matplotlib.pyplot.imshow
            that is used to display the data.
        """
        # sanity check
        if not isinstance(image, (list, np.ndarray)):
            logging.error('Cannot display data: {0}'.format(image))
            return

        # Check if an imshow instance does exist, if so, update its data
        if self.image is None:
            self.image_plot = self.axes.imshow(image, **kwargs)
        else:
            self.image_plot.set_data(image)

        self.image = image
        logging.debug('Size of image to be loaded: {0}x{1} pixels'.format(self.image.shape[1], self.image.shape[0]))

        self.changeCuts()  # adapt cuts to the new data

        # adjust axes limits
        if update_axlims:
            self.updateAxesLimits()

        # do not display any ticks when plotting images
        self.axes.set_xticks([])
        self.axes.set_yticks([])

        if wcs is not None:
            self.wcs = wcs

        self.draw()  # update canvas view
        logging.debug('Loading of new image completed.')

    def loadCube(self, input, wcs=None, **kwargs):
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
        input : string or an instance of a pampelmuse.instruments class
            If a string is provided, it must be the name of the FITS file containing the
            IFU data.
        wcs : instance of astropy.wcs.WCS, optional
            Can be used to convert pixel to world coordinates and vice versa.
        """
        # load cube using the dedicated PampelMuse class for this purpose that
        # also checks whether the data is valid (unless an instance was provided right away).
        if isinstance(input, str):
            self.cube = open_ifs_data(input)
        else:
            assert(isinstance(input, Instrument)), '"input" must be filname or instance of pampelmuse.instruments'
            self.cube = input

        # define a Path that indicates the edges of the field of view of the instrument and add it to the figure canvas.
        # Such a Path is available in the pampelmuse.cube.instrument classes.
        if self.cube.edgepath is not None:
            edgepatch = patches.PathPatch(self.cube.edgepath, facecolor="None", edgecolor="r", lw=2, alpha=0.5)
            self.axes.add_patch(edgepatch)

        if wcs is not None:
            self.wcs = wcs

    def loadLayer(self, i, residuals=False, **kwargs):
        """
        Display an individual layer from a loaded data cube. To display the
        data, the method will call 'addImage'.
        
        Parameters
        ----------
        i : int
            The layer of the IFU data that should be loaded.
        residuals : boolean, optional
            Set this flag if the fit residuals should be displayed instead
            of the data.
        kwargs
            Any keyword arguments provided to this function will be passed
            on to the call of 'addImage' that is used to display the layer.
        """
        # check if a cube has been loaded, complain if not
        if self.cube is None:
            logging.error("Cannot load layer as no cube is open.")
            return

        # get the requested layer and pass it on to the addImage-method
        if residuals and self.cube.residualshdu is None:
            logging.error("Cannot display fit residuals because no residuals HDU is loaded.")
            residuals = False
        self.addImage(self.cube.get_image(i, return_residuals=residuals)[0], **kwargs)

        # store current wavelength and number of layer
        self.wave = self.cube.wave[i]
        self.layer = i

        # emit QtCore.SIGNAL notifying about index of new layer
        self.layer_changed.emit(i)

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
        # get current cut levels
        oldcuts = self.image_plot.get_clim()
        if self.cuts is None:
            self.cuts = oldcuts

        # check if absolute cuts have been provided
        if locut is not None or hicut is not None:
            if locut is not None:
                self.cuts = (locut, self.cuts[1])
            if hicut is not None:
                self.cuts = (self.cuts[0], hicut)

        # check if relative cuts have been provided, if so, translate them into absolute cut levels.
        elif vrange is not None:
            self.vrange = vrange
            cutlev = float(self.vrange) / 100.
            npix = self.image.size
            cutpix = int((1 - cutlev) * npix)

            _values = self.image.copy().reshape(-1)
            values = _values[_values == _values]  # remove NaN pixels

            if values.size == 0:
                logging.error("Requested layer contains only NaN values.")
                self.cuts = (0, 1)
            else:
                # sort pixel values
                values.sort()
                self.cuts = (values[cutpix//2], 0.5 * (values[-cutpix//2 - 1] + values[-cutpix//2 - 2]))

        # change cuts and update canvas
        logging.debug("Updated display cuts are min={0:.3g}, max={1:.3g}".format(
            float(self.cuts[0]), float(self.cuts[1])))
        self.image_plot.set_clim(self.cuts[0], self.cuts[1])
        self.draw()

        # emit signal that cuts were changed
        if self.cuts[0] != oldcuts[0]:
            self.lower_cut_changed.emit(self.cuts[0])
        if self.cuts[1] != oldcuts[1]:
            self.upper_cut_changed.emit(self.cuts[1])

    def updateAxesLimits(self):
        """
        Automatically adjust the axis limits when a new image is loaded or
        when new data is overplotted on the image.
        The reason for doing this in a dedicated method is that the method
        matplotlib.axes.Axes.relim() does not work with instances of
        matplotlib.Collection and each scatter plot is an instance of the
        sub-class PathCollection.
        """
        axes_min = [0, 0]
        axes_max = [1, 1]

        if self.image is not None:
            axes_min = [-0.5, -0.5]
            axes_max = [self.image.shape[1] - 0.5, self.image.shape[0] - 0.5]

        if isinstance(self.data_plot, collections.PathCollection):
            data_min, data_max = self.data_plot.get_datalim(self.axes.transData).get_points()
            data_min_round = [int(data_min[0] - 1 * (data_min[0] < 0)), int(data_min[1] - 1 * (data_min[1] < 0))]
            data_max_round = [int(data_max[0] + 1), int(data_max[1] + 1)]

            axes_min = np.min([axes_min, data_min_round], axis=0)
            axes_max = np.max([axes_max, data_max_round], axis=0)

        logging.debug("Updated axes limits are x=[{0:.1f},{1:.1f}], y=[{2:.1f},{3:.1f}]".format(
            axes_min[0], axes_max[0], axes_min[1], axes_max[1]))
        self.axes.set_xlim(axes_min[0], axes_max[0])
        self.axes.set_ylim(axes_min[1], axes_max[1])

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
        # return if no cube is loaded
        if self.cube is None:
            return

        dr = self.cube.distance_to_spaxels(x, y)
        i_min = dr.argmin()
        logging.debug("Loading spectrum for spaxel #{0}, x={1}, y={2}...".format(
            i_min, self.cube.x[i_min], self.cube.y[i_min]))
        return self.cube.get_spectrum(i=i_min)[0]

    def addData(self, x, y, ids=None, update_axes_limits=True, **kwargs):
        """
        Overplot a dataset on an image as a scatter plot. If the `picker`-keyword
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
        update_axes_limits : boolean, optional
            Flag indicating if the axes limits should be updated after the
            plot is included.
        kwargs
            Any keyword arguments provided to this function are passed on to the
            call of matplotlib.pyplot.scatter that is used to diisplay the data.
        """
        self.x = np.asarray(x)
        self.y = np.asarray(y)

        assert(self.x.shape == self.y.shape), "x- and y-coords must have same shape."
        logging.debug('Overplotting {0} data points on image ...'.format(self.x.shape[0]))

        # if previously data have been added, that is removed
        if isinstance(self.data_plot, collections.Collection):
            self.data_plot.remove()

        self.data_plot = self.axes.scatter(self.x, self.y, **kwargs)

        # change axes limits to show all the data
        if update_axes_limits:
            self.updateAxesLimits()

        self.draw()

        # check if IDs were provided, otherwise numerate sources
        if ids is not None:
            self.ids = ids
        else:
            self.ids = np.arange(self.x.size, dtype='int32')

    @property
    def edgecolors(self):
        """
        Return the edgeColors of the markers of the data set overplotted
        on an image. The return value will be an array of the same length
        as the dataset even if all  markers have the same edgecolor.
        """
        # return if no plot is available
        if self.data_plot is None:
            return None

        # get current edgeColors from dataset, they are returned as RGBA values, so the length of the second dimension
        # of the return value is 4, the length of the first depends on how the colors were initialized
        edgecolors = self.data_plot.get_edgecolors()

        if edgecolors.shape[0] == 0:  # if colors were initialized to "None"
            return np.zeros((self.x.size, edgecolors.shape[1]), dtype=np.float32)
        elif edgecolors.shape[0] == 1:  # if a single color was provided
            return np.resize(edgecolors, (self.x.size, edgecolors.shape[1]))
        else:  # if different colours were provided
            return edgecolors

    @property
    def facecolors(self):
        """
        Return the facecolors of the markers of the data set overplotted
        on an image. The return value will be an array of the same length
        as the dataset even if all markers have the same facecolor.
        """
        # return if no plot is available
        if self.data_plot is None:
            return None

        # get current facecolors from dataset, they are returned as RGBA values, so the length of the second dimension
        # of the return value is 4, the length of the first depends on how the colors were initialized
        facecolors = self.data_plot.get_facecolors()

        if facecolors.shape[0] == 0:  # if colors were initialized to "None"
            return np.zeros((self.x.size, facecolors.shape[1]), dtype=np.float32)
        elif facecolors.shape[0] == 1:  # if a single color was provided
            return np.resize(facecolors, (self.x.size, facecolors.shape[1]))
        else:  # if different colours were provided
            return facecolors

    def on_mouseMotion_notify(self, event):
        """
        Handle event when mouse is moved over the axis with a button pressed.
        The method emits a signal providing the x- and y-coordinates of the
        pixel and its flux.
        
        Parameters
        ----------
        event : a matplotlib.backend_bases.MouseEvent instance
            The event that triggered the method call.
        """

        if event.inaxes != self.axes:
            return

        x = int(round(event.xdata))
        y = int(round(event.ydata))
        try:
            f = float(self.image[y, x])
        except IndexError:
            f = np.nan

        self.pointer_position_changed.emit(x, y, f)
