"""
dynamic_plot_canvas.py
======================
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
class pampelmuse.graphics.dynamic_plot_canvas.DynamicPlotCanvas

Purpose
-------
The module provides the class DynamicPlotCanvas that is designed to
display and interact with 1dim. data, such as spectra, IFS coordinates,
or parameters of the PSF or coordinate transformation. The class
supports the simultaneous display of more than one dataset. This is
useful, e.g., to display data together with a polynomial fit.
The possibilities to interact with the data include the masking of
certain (wavelength) ranges, zooming, and panning.

The class DynamicPlotCanvas is used by the PampelMuse GUI to show the
top panel, named 'Spectrum Viewer'.

Latest SVN revision
-------------------4
360, 2016/08/12
"""
import inspect
import logging
import matplotlib.pyplot
import numpy as np
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory
from matplotlib.widgets import Cursor, SpanSelector
from PyQt4 import QtCore
matplotlib.pyplot.rcParams["axes.linewidth"] = 1.5


__author__ = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__revision__ = 360


class DynamicPlotCanvas(FigureCanvasQTAgg):
    """
    This class can be used to display a matplotlib lineplot and interact with
    it by selecting or deselecting horizontal ranges. In PampelMuse, this
    feature is used to create a spectrumMask in wavelength space.
    """
    background = None

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
        # prepare figure canvas
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.canvas = self.fig.canvas

        self.axes = self.fig.add_axes((0.15, 0.20, 0.82, 0.76))

        super(DynamicPlotCanvas, self).__init__(self.fig)
        self.setParent(parent)

        # dictionary to keep track of the data sets that are plotted.
        self.thePlots = {}

        # a vertical line is used to highlight the cursor position on the canvas.
        self.cursor = Cursor(self.axes, color='r', lw=1.5, ls='--', horizOn=False, vertOn=True, useblit=True)

        # define two selectors for adding a horizontal range to the rectangles or subtracting it.
        self.selector = SpanSelector(self.axes, self.on_select, direction='horizontal', useblit=True,
                                     rectprops=dict(facecolor='red', alpha=0.5), button=1)
        self.deselector = SpanSelector(self.axes, self.on_deselect, direction='horizontal', useblit=True,
                                       rectprops=dict(facecolor='0.5', alpha=0.5), button=3)

        # prepare structure to hold the set of rectangles that are overplotted on th data.
        # to achieve axes units (bottom to top = 0 to 1) in  vertical direction and data units (wavelength) in
        # horizontal direction, we define a corresponding transform. Also a list of IDs is initialized to keep
        # track of the number of available rectangles and their names.
        self.bar_transform = blended_transform_factory(self.axes.transData, self.axes.transAxes)
        self.bar_gid = ['selector', 'deselector']  # will contain the ID of every rectangle

        self.draw()

    def draw(self):
        """
        Update the plot to display the latest changes. Following the
        recommendations for animations in matplotlib, we only redraw
        the animated parts of a plot and blit the static parts during
        an animation. To this aim, we save the static background to a
        variable immediately after each call to draw().
        
        """
        # draw() shows unanimated parts of plot that we also save as a static background for later recovery
        super(DynamicPlotCanvas, self).draw()
        self.background = self.figure.canvas.copy_from_bbox(self.axes.bbox)

        # for debugging: check where the call was triggered
        curframe = inspect.currentframe()
        calframe = inspect.getouterframes(curframe, 2)
        logging.debug("call to draw() triggered by: {0} in {1}[@{2}]".format(
            calframe[1][3], calframe[1][1], calframe[1][2]))

    def addData(self, name, x_data, y_data, **kwargs):
        """
        Add a new data set to the axes.
        
        Parameters
        ----------
        name : string
            Each plot must bbe given a name that is used to identify it.
        x_data : array_like
            The data along the x-axis.
        y_data : array_like
            The data along the y-axis. Must of course have the same length
            as 'xData'.
        kwargs
            The keyword arguments are passed on to the matplotlib.pyplot.plot
            command that is used to initialize the data plot.
        """
        # plot the data using matplotlib
        self.thePlots[name] = self.axes.plot(x_data, y_data, **kwargs)[0]

        # set minor ticks on x- and y-axis on
        self.axes.minorticks_on()

        # increase strength of major and minor ticks to avoid hair lines.
        self.axes.tick_params(axis="both", which="both", width=1.5)

        # update canvas
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
            The height of the rectangle in relative units, i.e height=1
            means from bottom to top.
        kwargs
            Any keyword argument is passed on to the matplotlib.patches.Rectangle
            command used to prepare the rectangle.
        """

        # Check if an identifier has been provided, otherwise define one.
        if "gid" not in kwargs:
            kwargs["gid"] = "rect{0}".format(len(self.axes.patches))

        # check if rectangle is a new one or an update to an existing one using the identifier.

        if kwargs["gid"] in self.bar_gid:  # update existing rectangle
            i = self.bar_gid.index(kwargs["gid"])
            self.axes.patches[i].set_xy(xy)
            self.axes.patches[i].set_width(width)
        else:  # add new rectangle
            rect = Rectangle(xy, width, height, transform=self.bar_transform, **kwargs)

            self.axes.add_patch(rect)
            self.bar_gid.append(rect.get_gid())

    def delRectangle(self, gid):
        """
        Delete an existing rectangle from the collection. The identification
        is done using the 'gid'.
        
        gid : string
            The identifier of the rectangle to be deleted.
        """

        try:  # check if provided identifier exists, if so, remove rectangle
            i = self.bar_gid.index(gid)
            del self.axes.patches[i]
            del self.bar_gid[i]
        except IndexError:  # otherwise complain
            logging.error("No rectangle instance has gid '{0}'.".format(gid))

    def updateData(self, name, x_data=None, y_data=None, x_label=None, y_label=None, update_y_range=True):
        """
        Update an existing data plot.
        
        Parameters
        ----------
        name : string
            The identifier of the data set to be updated.
        x_data : array_like, optional
            The new data along the x-axis. Note that it can have a different size
            than the current data.
        y_data : array_like, optional
            The new data along the y-axis. Must of course have the same length
            as the new 'xData'.
        x_label : string, optional
            The new label for the x-axis.
        y_label : string, optional
            The new label for the y-axis.
        update_y_range : bool, optional
            If this flag is set, the axes limits will be adjusted so that all the
            new data is shown.
        """
        # return if no new data is provided
        if x_data is None and y_data is None:
            return

        # check if plot to be updated exists
        if name not in self.thePlots.keys():
            logging.error("Cannot update plot '{0}': unknown identifier.".format(name))

        # update data
        if x_data is not None:
            self.thePlots[name].set_xdata(x_data)
        if y_data is not None:
            self.thePlots[name].set_ydata(y_data)

        for i in xrange(len(self.axes.patches)):
            self.axes.patches[i].set_height(0)

        # update axes limits if requested
        if update_y_range:
            xlim = (np.inf, -np.inf)
            ylim = (np.inf, -np.inf)
            for plot in self.thePlots.values():
                if len(plot.get_xdata()) == 0:
                    continue
                xdata = plot.get_xdata()
                xlim = (min(xlim[0], np.nanmin(xdata)), max(xlim[1], np.nanmax(xdata)))
                ydata = plot.get_ydata()
                ylim = (min(ylim[0], np.nanmin(ydata)), max(ylim[1], np.nanmax(ydata)))
            if xlim[0] == xlim[1]:
                xlim = (xlim[0] - 0.1, xlim[0] + 0.1)
            self.axes.set_xlim(xlim)
            if ylim[0] == ylim[1]:
                ylim = (ylim[0] - 0.1, ylim[0] + 0.1)
            self.axes.set_ylim(ylim)

            # somehow this does not work:
            # self.axes.axes.relim()  # widget.Cursor instance seems to corrupt auto-scaling of axes
            # self.axes.autoscale()

        # update labels if requested
        if x_label is not None:
            self.axes.set_xlabel(x_label, fontsize=20)

        if y_label is not None:
            self.axes.set_ylabel(y_label, fontsize=20)

        # modify height of rectangles.
        for i in xrange(len(self.axes.patches)):
            self.axes.patches[i].set_height(1)

        self.draw()  # update canvas.

    def on_select(self, x_click, x_release):
        """
        This method is the callback function for the selection of an
        x-range. It finds the minimum and maximum x-values of the selection
        and emits a signal indicating that an x-range has been selected.

        Parameters
        ----------
        x_click : float
            The x-value were the selection was started.
        x_release : float
            The x-axis were the selection was completed.

        Returns
        -------
        Nothing is returned but a QtCore.SIGNAL is emitted that contains the
        min. and max. x-values of the selection and a flag indicating that
        the selection was positive.

        """
        x_min = min(x_click, x_release)
        x_max = max(x_click, x_release)
        logging.info('Selected range: {0:.1f} - {1:.1f}'.format(x_min, x_max))
        self.emit(QtCore.SIGNAL("rangeSelected"), *[(x_min, x_max), True])

    def on_deselect(self, x_click, x_release):
        """
        This method is the callback function for the deselection of an
        x-range. It finds the minimum and maximum x-values of the deselection
        and emits a signal indicating that an x-range has been deselected.

        Parameters
        ----------
        x_click : float
            The x-value were the deselection was started.
        x_release : float
            The x-axis were the deselection was completed.

        Returns
        -------
        Nothing is returned but a QtCore.SIGNAL is emitted that contains the
        min. and max. x-values of the deselection and a flag indicating that
        the selection was negative.

        """
        x_min = min(x_click, x_release)
        x_max = max(x_click, x_release)
        logging.info('Deselected range: {0:.1f} - {1:.1f}'.format(x_min, x_max))
        self.emit(QtCore.SIGNAL("rangeSelected"), *[(x_min, x_max), False])
