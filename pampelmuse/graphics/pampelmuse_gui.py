# -*- coding: utf-8 -*-
"""
pampelmuse_gui.py
=================
Copyright 2013-2017 Sebastian Kamann

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
class pampelmuse.graphics.pampelmuse_gui.PampelMuseGui

Purpose
-------
Module implementing the GUI of PampelMuse. It is designed to facilitate the
polynomial fitting of the coordinates and the PSF variables and to allow for
a visual inspection of the fitting results. The GUI was designed in PyQt4 with
the dedicated tool QtDesginer.
The python code that actually contains the design of the GUI is contained in
the class pampelmuse.graphics.Ui_pampelmuse_gui.Ui_PampelMuseGui. This module
contains the dialogue code for Ui_pampelmuse_gui, in the sense that it
defines the methods that are called by Ui_PampelMuseGui.

Latest Git revision
-------------------
2017/11/23
"""
import logging
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches, colors
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from astropy import wcs
from astropy.io import fits
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import QMainWindow, QHBoxLayout, QVBoxLayout, QFileDialog
from .dynamic_plot_canvas import DynamicPlotCanvas
from .dynamic_ima_canvas import DynamicImaCanvas
from .Ui_pampelmuse_gui import Ui_PampelMuseGui
from .exit_dialog import exitUnsavedDialog
from ..core import Sources
from ..core.coordinates import Transformation
from ..psf import Variables
from ..utils.fits import save_prm, read_config_from_header, make_header_from_config


__author__ = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__revision__ = 20171123


class ToolbarWithoutMessage(NavigationToolbar):
    """
    A small subclass of the navigation toolbar that does not display the
    coordinates under the mouse pointer. This is used for the reference
    image and IFS data because to save space and because this information
    is already displayed in the status bar.
    """
    def set_message(self, msg):
        pass


class PampelMuseGui(QMainWindow, Ui_PampelMuseGui):
    """
    The class defines the various actions that can be triggered by the
    user to interact with the GUI. The GUI serves two purposes:
    (i)  a visual inspection of the fit results.
    (ii) performing polynomial fits to the source positions and/or
         PSF variables.
    """
    spectrumMask = None  # mask along spectral axis

    identifiedSources = {}  # a dictionary {ID:(x,y)} of sources identified in the reference and the IFU view
    edgeColors = None  # placeholder for array containing th colors used to plot the sources in the IFS and ref. view

    ifsInteractMode = None
    ifsPlot = None
    ifsToolbar = None
    ifsOffsets = None  # placeholder for array containing the coordinates that are plotted in the IFS view
    ifsPlotCurrentlyMarked = None
    ifsPlotPreviouslyMarked = None
    ifsPositionsKnown = False

    refInteractMode = None
    refPlot = None
    refToolbar = None
    refOffsets = None  # placeholder for array containing coordinates that are plotted in the reference view
    currentRefID = None

    unsavedChanges = False  # flag indicating whether there are unsaved changes
    outfileName = None  # name of the FITS file used to store the results
    config = None  # may be a dictionary containing the configuration written to the header of the stored FITS file.

    plotted_mag_range = [0., 99.]  # the magnitude range of the stars that are currently being displayed
    plotted_statuses = []  # the statuses of the sources that are currently being displayed

    # define custom signals
    ifs_interact_mode_changed = pyqtSignal(str, name='ifsInteractModeChanged')
    ref_interact_mode_changed = pyqtSignal(str, name='refInteractModeChanged')

    def __init__(self, sources=None, fitsFile=None, ifsData=None, refImage=None, parent=None, **kwargs):
        """
        Construct a new instance of the class 'PampelMuseGui'.
        
        Parameters
        ----------
        sources : instance of either numpy.recarray or fits.BinTableHDU, optional
            The components (stellar sources, background components) that
            are present in the analysed field, either provided as a FITS
            HDU (binary table) or a record array.
        fitsFile : string
            A FITS-file containing the catalogue of sources and/or information about
            the sources such as their spectra or positions or about the PSF. An example
            for such a file is a valid prm-file as it is created by PampelMuse.
        ifsData : string or instance of a pampelmuse.instruments class, optional
            A FITS-file containing the datacube that is analysed or an instance of one
            of the classes defined in the pampelmuse.instruments package.
        refImage : string or nd_array, optional
            A reference image, provided either as a FITS file or a 2dim image array.
            Note that the idea is that the reference coordinates used by PampelMuse
            are the pixel coordinates in this image.
        parent : 
            The 'parent' argument, if not None, causes self to be 
            owned by Qt instead of PyQt. (copied from PyQt4 documentation)
        **kwargs
            Any keyword args are passed to the methods called by __init__.
        """
        QMainWindow.__init__(self, parent)
        self.setupUi(self)

        # Initial setup for the spectrum viewer:
        self.dataPlot = DynamicPlotCanvas()
        plotFrame = QVBoxLayout(self.spectrumWidget)
        plotFrame.addWidget(self.dataPlot)
        plotFrame.setAlignment(Qt.AlignCenter)

        # add navigation toolbar
        navi_toolbar = NavigationToolbar(self.dataPlot, self)
        self.dataPlot.navigation_toolbar = navi_toolbar
        self.spectrumHeaderHBox.addWidget(navi_toolbar)

        # Initialize plot that will contain data.
        self.dataPlot.addData(name='data', x_data=[], y_data=[], ls='-', c='b', marker='None')

        # Initialize plot that will contain current fit.
        self.dataPlot.addData(name='fit', x_data=[], y_data=[], ls='--', c='r', lw=1.8, marker='None')

        # add a dictionary that will contain the available data that can be displayed/fitted.
        self.available_data = {}

        # Start working on source catalog (if any provided)
        self.sources = None
        self.psf = None

        # check if valid instance of Sources provided
        if sources:
            self.loadSources(sources)
        # open prm-file if provided
        if fitsFile:
            self.openFile(fitsFile)

        # connect data selection process to masking method
        self.dataPlot.range_selected.connect(self.modifyMask)

        # connect change in interaction mode to callback function
        self.ifs_interact_mode_changed.connect(self.changeIfsInteractMode)
        self.ref_interact_mode_changed.connect(self.changeRefInteractMode)

        # load cube into GUI if one is provided
        if ifsData:
            self.openIfs(ifsData)

        # load reference image into GUI if one is provided
        if refImage is not None:
            self.openReference(refImage, **kwargs)

        self.statusBar.showMessage("Ready", 5000)

    @property
    def current_hdu(self):
        """
        Returns
        -------
        The current text in the HDU selection ComboBox as a string.
        """
        return str(self.spectrumShowHduComboBox.currentText())

    @property
    def current_row_in_hdu(self):
        """
        Returns
        -------
        The current row in the spectrum selection ComboBox as a string.
        """
        return str(self.spectrumShowIdentComboBox.currentText())

    @property
    def instrument(self):
        """
        Returns
        -------
        The instrument used to observe the IFS data. This attribute
        is copied from the instance of pampelmuse.core.cube.Cube if
        a data cube has been loaded.
        """
        if self.ifsLoaded:
            return self.ifsPlot.cube
        else:
            return None

    @property
    def start(self):
        """
        Returns
        -------
        The index of the first layer of the IFU data that is analyzed 
        """
        if hasattr(self.namespace, "start"):
            return self.namespace.start
        else:
            return None

    @property
    def ifsLoaded(self):
        """
        :return: boolean flag indicating if a data cube has been loaded
        """
        return self.ifsPlot is not None

    @property
    def refLoaded(self):
        """
        :return: boolean flag indicating if a reference image has been loaded
        """
        return self.refPlot is not None

    def loadSources(self, data):
        """
        Initialize the catalogue of sources to work with.
        
        Parameters
        ----------
        data : instance of pampelmuse.core.Sources or record array or fits.BinTableHDU instance
            The source catalogue: at least the columns 'ID', 'x' and 'y'
            should be available.
        """
        if isinstance(data, Sources):
            self.sources = data
        else:
            self.sources = Sources(catalog=data)

        # add new property indicating which stars are currently plotted
        self.sources.plotted = pd.Index([], dtype=np.bool)

        self.actionSaveAs.setEnabled(True)

        if np.isfinite(self.sources.mag[~self.sources.is_background]).any():
            self.minMagSpinBox.setValue(self.sources.mag[~self.sources.is_background].min() - 0.1)
            self.maxMagSpinBox.setValue(self.sources.mag[~self.sources.is_background].max() + 0.1)
        self.dataLoaded()

    def loadSpectra(self, data=None):
        """
        Read the source spectra from a 2dim. array or a fits ImageHU.
        
        Parameters
        ----------
        data : fits.ImageHDU instance, optional
            The set of spectra to be loaded. Note that this method may only be
            called after the source catalogue has been initialized and that a
            check is carried out whether the number of spectra matches the number
            of components in the catalogue.
            If no data are provided, it is assumed that the 'Sources' instance
            that is currently active includes spectra.
        """
        if self.sources is None:
            logging.error("Cannot load spectra unless information about sources available.")
            return

        if data is not None:
            self.sources.open_spectra_hdu(data)
        self.sources.load_spectra()  # read all spectra into memory

        # create entry about loaded data for data selection via QComboBox if not existent
        if self.spectrumShowHduComboBox.findText("Spectra") == -1:
            self.spectrumShowHduComboBox.addItem("Spectra")
        self.available_data["Spectra"] = self.sources.provideComponentNames()

        self.dataLoaded()  # update status of GUI

    def loadCoordTrans(self, data=None):
        """
        Read the coordinate transformation from reference to IFU coordinates
        form a FITS table.
        If no data are provided, it is checked whether the instance of 'Sources'
        that is currently active contains a valid coordinate transformation.

        Parameters
        ----------
        data : fits.BinTableHDU instance
            The HDU containing the coordinate transformation, one parameter per
            row.
        """
        if self.sources is None:
            logging.error("Cannot load coord. transf. unless information about sources available.")
            return

        if data is not None:
            self.sources.transformation = Transformation(hdu=data)
            # make sure transformation has same dispersion axis as Sources instance
            if self.sources.ndisp != self.sources.transformation.n_dispersion == 1:
                self.sources.transformation.resize(self.sources.ndisp)

            # make sure the same dispersion axis is used
            self.sources.transformation.data.index = self.sources.wave

        # create entries for positions and coord. transf. in QComboBox if not existent
        if self.sources.transformation.valid:
            for entry in ["x Positions", "y Positions", "Coord. Transf."]:
                if self.spectrumShowHduComboBox.findText(entry) == -1:
                    self.spectrumShowHduComboBox.addItem(entry)
            self.available_data["Positions"] = self.sources.providePositionIds()
            self.available_data["Coord. Transf."] = self.sources.transformation.names

        self.ifsPositionsKnown = True
        self.dataLoaded()  # update status of GUI

    def loadPositions(self, data=None):
        """
        Read the IFU coordinates from a 3dim. ImageHDU.
        
        Parameters
        ----------
        data : fits.ImageHDU instance, optional
            The coordinates of the sources on the IFU data as a function of
            wavelength. NOTE that this method may only be called after a
            catalogue of sources and their spectra have been loaded.
            If no data are provided, it is checked whether IFU coordinates for
            some/all sources are available in the instance of 'Sources' that
            is currently active.
        """
        if self.sources is None:
            logging.error("Cannot load positions unless information about sources available.")
            return

        if data is not None:
            self.sources.open_coordinates_hdu(data)

        # create entries for IFU positions in QComboBox if not existent
        for entry in ["x Positions", "y Positions"]:
            if self.spectrumShowHduComboBox.findText(entry) == -1:
                self.spectrumShowHduComboBox.addItem(entry)
        self.available_data["Positions"] = self.sources.providePositionIds()

        self.ifsPositionsKnown = True
        self.dataLoaded()  # update status of GUI

    def loadPsfParameters(self, data):
        """
        Read the parameters of the PSF from a fits.BinTableHDU.
        
        Parameters
        ----------
        data : a fits.BinTableHDU instance or a pampelmuse.core.psf.Variables instance
            The parameters of the PSF, given one per line.
        """
        if isinstance(data, Variables):
            self.psf = data
        else:
            self.psf = Variables(hdu=data)  # initialize PSF

        # make sure PSF parameters have same dispersion axis as Sources instance
        if self.sources.n_dispersion != self.psf.n_dispersion == 1:
            self.psf.resize(self.sources.wave)

        # make sure the same dispersion axis is used
        self.psf.data.index = self.sources.wave

        # update QComboBox for newly loaded data
        if self.spectrumShowHduComboBox.findText("PSF") == -1:
            self.spectrumShowHduComboBox.addItem("PSF")
        self.available_data["PSF"] = self.psf.names

        self.dataLoaded()  # update status of GUI

    def dataLoaded(self):
        """
        This method serves as a callback function whenever new data is loaded.
        It updates the status of the GUI according to the amount of available
        data.
        """
        if len(self.available_data) > 0:
            # enable data selection if required
            if not self.spectrumShowGo.isEnabled():
                self.spectrumShowGo.setEnabled(True)

            # get current text of HDU and row QComboBoxes and emit the signal that the highlighted row of
            # the current HDU is selected
            self.on_spectrumShowHduComboBox_activated(self.spectrumShowHduComboBox.currentText())
            self.on_spectrumShowGo_pressed()

        # initialize spectrumMask if needed
        if self.spectrumMask is None or self.spectrumMask.size != self.sources.n_dispersion:
            self.spectrumMask = np.ones((self.sources.n_dispersion,), dtype=np.bool)

        # check if interaction states need to be changed
        if self.ifsLoaded is None:
            mode = None
        elif self.ifsPositionsKnown:
            mode = "pick"
        else:
            mode = "grab"
        if mode != self.ifsInteractMode:
            self.ifs_interact_mode_changed.emit(mode)
            logging.debug('Interaction mode for IFS view set to "{0}".'.format(mode))

        if not self.refLoaded or self.sources is None:
            mode = None
        else:
            mode = 'pick'
        if mode != self.refInteractMode:
            self.ref_interact_mode_changed.emit(mode)
            logging.debug('Interaction mode for reference view set to "{0}".'.format(mode))

    def openFile(self, filename):
        """
        Open prm-file and read information from various HDUs. This method
        is called automatically when a new instance is created and the parameter
        'fitsFile' is provided.
        
        Parameters
        ----------
        filename : string
            The name of the FITS file to be loaded. The method will scan through
            its various HDUs and check whether they contain useful data. Otherwise
            the HDUs are ignored.
        """
        # to later write the results to this file, we need its absolute path
        self.outfileName = os.path.abspath(filename)

        hdu_list = fits.open(filename, mode="update")

        # search headers for HIERARCH PAMPELMUSE keywords.
        self.config = read_config_from_header(header=hdu_list[0].header, existing_configuration=self.config)

        # for logging only:
        hdu_string = ""

        # loop over HDUs in file.
        for hdu in hdu_list:

            hdu_string += "{0} ".format(hdu.name)

            if hdu.name == "SOURCES":
                self.loadSources(hdu)
            elif hdu.name == "SPECTRA":
                self.loadSpectra(hdu)
            elif hdu.name == "POSPARS":
                self.loadCoordTrans(hdu)
            elif hdu.name == "POSITIONS":
                self.loadPositions(hdu)
            elif hdu.name == "INPOSITIONS" and "Coord. Transf." not in self.available_data.keys():
                self.loadPositions(hdu)
            elif hdu.name == "PSFPARS":
                self.loadPsfParameters(hdu)

        self.statusBar().showMessage(
            "Opened file <{0}>, detected HDUs: {1}".format(self.outfileName, hdu_string),  5000)

        # enable saving to new filename
        self.actionSaveAs.setEnabled(True)

    def openIfs(self, filename):
        """
        Open an IFS data cube and display it, using the widget foreseen
        for this purpose.
        
        Parameters
        ----------
        filename : string or a pampelmuse.core.cube.Cube instance
            If a string is provided, it must be the name of the FITS file
            containing the data. Note that in agreement with the the other
            routines of PampelMuse, 2dim. data is okay if the instrument
            used in the observation is known (otherwise data must be 3dim.).
        """
        logging.debug("Loading IFS data into GUI...")

        # load cube using instance of DynamicImaCanvas
        self.ifsPlot = DynamicImaCanvas()
        self.ifsPlot.loadCube(filename)

        # enable display of fit residuals if they are available
        if self.ifsPlot.cube.residualshdu is not None:
            self.actionResiduals.setEnabled(True)

        # include this instance in GUI
        ifsframe = QHBoxLayout(self.ifuViewWidget)
        ifsframe.addWidget(self.ifsPlot)
        ifsframe.setAlignment(Qt.AlignCenter)

        # add navigation toolbar
        self.ifsToolbar = ToolbarWithoutMessage(self.ifsPlot, self)
        self.ifuHeadHBox.insertWidget(1, self.ifsToolbar)

        # load first layer by default
        self.ifsPlot.loadLayer(0, residuals=self.actionResiduals.isChecked(), cmap=plt.cm.Blues,
                               interpolation="nearest")

        # adapt layer range to match spectral dimension of data cube and show central layer
        self.ifuLayerSelectSlider.setMaximum(self.ifsPlot.cube.wave.size-1)
        self.ifuLayerSelectSlider.setValue(self.ifsPlot.cube.wave.size//2)
        self.on_ifuLayerSelectSlider_sliderMoved(self.ifsPlot.cube.wave.size//2)

        # qt designer does not offer a method to define a validator, so we do it here
        self.ifuMinCutLineEdit.setValidator(QDoubleValidator())
        self.ifuMaxCutLineEdit.setValidator(QDoubleValidator())

        # default cut level is 100%
        self.ifuCutsComboBox.setCurrentIndex(0)
        logging.debug("Successfully loaded IFS data into GUI.")

        # log current position of mouse cursor in status bar
        self.ifsPlot.pointer_position_changed.connect(lambda x, y, f: self.statusBar.showMessage(
            "x={0}, y={1}, f={2:.3g}".format(x, y, f), 5000))

        self.dataLoaded()

    def openReference(self, filename, **kwargs):
        """
        Open an image as a reference to compare with the IFU data. Note that the idea
        here is that the reference coordinates that are used by PampelMuse are the pixel
        coordinates on that image.
        
        Parameters
        ----------
        filename : string or nd_array
            Either the name of a FITS file containing the image or the image data as
            a 2dim. array must be provided.
        """
        logging.debug("Loading reference image into GUI...")

        # initialize an instance of DynamicImaCanvas that will handle the data.
        self.refPlot = DynamicImaCanvas()

        # either the name of a FITS file or an ndarray can be provided
        # TODO: add kwargs to control how image is displayed.
        if isinstance(filename, str):
            fitsdata, fitsheader = fits.getdata(str(filename), header=True)
            self.refPlot.addImage(fitsdata, cmap=plt.cm.Blues, interpolation="nearest", wcs=wcs.WCS(fitsheader))
        else:
            self.refPlot.addImage(filename, cmap=plt.cm.Blues, interpolation="nearest")

        # add instance just created to GUI
        logging.debug('Adding reference image view to GUI ...')
        refFrame = QHBoxLayout(self.refViewWidget)
        refFrame.addWidget(self.refPlot)
        refFrame.setAlignment(Qt.AlignCenter)

        # add navigation toolbar
        logging.debug('Adding navigation toolbar to reference view ...')
        self.refToolbar = ToolbarWithoutMessage(self.refPlot, self)
        self.refHeaderHBox.insertWidget(1, self.refToolbar)

        # Qt Designer does not offer a method to define a validator, so we do it here
        self.refMinCutLineEdit.setValidator(QDoubleValidator())
        self.refMaxCutLineEdit.setValidator(QDoubleValidator())

        # default cut level is 100%
        self.refCutsComboBox.setCurrentIndex(0)

        logging.debug("Successfully loaded reference image into GUI.")
        self.dataLoaded()

    def addFovToReference(self, dx, dy, angle=0, scaling=1.):
        """
        Add the footprint of the IFS to the reference image plot.
        
        Parameters
        ----------
        dx : float
            The shift applied in x-direction
        dy : float
            The shift applied in y-direction
        angle : float, optional
            The rotation angle applied (in radian)
        scaling : float, optional
            The scaling factor applied to the field of view, e.g. to
            account for a different pixel size.
        """
        if self.refLoaded:
            # get center of native FoV of IFS
            bbox = self.instrument.edgepath.get_extents()
            xc = scaling*(bbox.xmax+bbox.xmin)/2.
            yc = scaling*(bbox.ymax+bbox.ymin)/2.

            # initialize transformation from native position to position on reference image
            t = matplotlib.transforms.Affine2D(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))

            # scale the field of view if needed
            if scaling != 1:
                t.scale(scaling)

            # translate it
            t.translate(dx-xc, dy-yc)

            # rotate it around central coordinate if needed
            if angle != 0:
                t.rotate_around(dx, dy, angle)

            # perform transformation of IFS-Edgepath, as defined in Instrument-class
            plotpath = self.instrument.edgepath.transformed(t)

            edgepatch = patches.PathPatch(plotpath, facecolor="None", edgecolor="#00FF00", lw=2, alpha=0.5)
            self.refPlot.axes.add_patch(edgepatch)
            self.refPlot.draw()
        else:
            logging.error("Can not overplot IFS FoV as no reference image has been loaded.")

    def changeIfsInteractMode(self, new_mode):
        """
        This is the callback function used when the interaction mode for the
        IFS data changes. It sets the new interaction mode and prepares the
        environment needed for the new mode.

        Parameters
        ----------
        new_mode : string
            The new interaction mode. Can be None, 'pick', or 'grab'
        """
        # check if prerequisites are fulfilled
        if new_mode is None:
            pass
        else:
            assert self.ifsLoaded, 'Cube needs to be loaded to interact with it.'
            if new_mode == 'grab':
                # We need two plots, one to hold the currently selected position and one that contains previously
                # grabbed coordinates (the second one is needed to initialize the coordinate transformation between
                # the reference and the IFU positions)
                self.ifsPlotCurrentlyMarked, = self.ifsPlot.axes.plot([], [], ls="None", marker="x", mew=1.6, ms=12,
                                                                      mec="r", alpha=0.5)
                self.ifsPlotPreviouslyMarked, = self.ifsPlot.axes.plot([], [], ls="None", marker="x", mew=1.6, ms=12,
                                                                       mec="r")
                self.ifsPlot.figure.canvas.mpl_connect("button_press_event", self.on_ifsGrab_event)
            else:
                assert new_mode == 'pick', 'Unknown interaction mode for IFS data: "{0}"'.format(new_mode)
                assert self.ifsPositionsKnown, 'Source positions on IFS need to be known for source picking.'

                # The sources will be overlaid on the image with circles. The edge colours depend on the source
                # statuses:
                # a) PSF sources (status==3, if any) data points will be overplotted as light green circles
                # b) Other resolved sources (status==2) will be displayed as black circles
                # c) sources close to a brighter source (status==1) will be displayed as grey circles.
                if self.edgeColors is None:
                    self.edgeColors = pd.Series(np.zeros(self.sources.n_sources, dtype="<U7"), index=self.sources.ids)
                    self.edgeColors[self.sources.status == 3] = "#00FF00"
                    self.edgeColors[self.sources.status == 2] = "0.0"
                    self.edgeColors[self.sources.status == 1] = "0.5"

                # load the IFS coordinates for all resolved sources and store them as a separate variable for later
                # usage
                if self.ifsOffsets is None:
                    self.sources.load_ifs_coordinates()

                    # use polynomial fits if they exist for all sources, direct coordinates otherwise
                    if self.sources.ifs_coordinates_fitted[~self.sources.is_background].all():
                        self.ifsOffsets = {
                            'x': self.sources.ifs_coordinates.xs(('x', 'fit'), level=('axis', 'type'), axis=1),
                            'y': self.sources.ifs_coordinates.xs(('y', 'fit'), level=('axis', 'type'), axis=1)}
                    else:
                        self.ifsOffsets = {
                            'x': self.sources.ifs_coordinates.xs(('x', 'value'), level=('axis', 'type'), axis=1),
                            'y': self.sources.ifs_coordinates.xs(('y', 'value'), level=('axis', 'type'), axis=1)}
                        # interpolate NaN values
                        self.ifsOffsets['x'].fillna(method='pad', inplace=True)
                        self.ifsOffsets['y'].fillna(method='pad', inplace=True)

                if not self.plotStarsCheckBoxPSF.isEnabled():
                    self.plotStarsCheckBoxPSF.setEnabled(True)
                if not self.plotStarsCheckBoxResolved.isEnabled():
                    self.plotStarsCheckBoxResolved.setEnabled(True)
                if not self.plotStarsCheckBoxNearby.isEnabled():
                    self.plotStarsCheckBoxNearby.setEnabled(True)
                if not self.plotStarsCheckBoxUnresolved.isEnabled() and self.sources.is_unresolved.any():
                    self.plotStarsCheckBoxUnresolved.setEnabled(True)

                self.plot_sources_on_ifs(update_sources=True)
                self.ifsPlot.figure.canvas.mpl_connect("pick_event", self.on_ifsPick_event)

        self.ifsInteractMode = new_mode

    def changeRefInteractMode(self, new_mode, **kwargs):
        """
        This is the callback function used when the interaction mode for the
        reference image changes. It sets the new interaction mode and prepares
        the environment needed for the new mode.

        Parameters
        ----------
        new_mode : string
            The new interaction mode. Can be None or 'pick'
        """
        if new_mode is None:
            pass
        else:
            assert self.refLoaded, 'Reference image needs to be loaded to interact with it.'
            assert self.sources is not None, 'Need to open source catalogue before picking sources.'
            assert new_mode == 'pick', 'Unknown interaction mode for reference data: "{0}"'.format(new_mode)

            # for INITFIT, make sure all sources are overplotted on the reference image
            if self.sources.is_resolved.sum() == 0 and 0 not in self.plotted_statuses:
                self.on_plotStarsCheckBoxUnresolved_stateChanged(state=2)

            # check if pixel coordinates of sources on reference image can be derived
            if self.refOffsets is None and self.refPlot.wcs is not None:
                if self.sources.ra is not None and self.sources.dec is not None:
                    logging.debug('(Re)calculating pixel coordinates for sources on reference image.')
                    x, y = self.refPlot.wcs.wcs_world2pix(self.sources.ra, self.sources.dec, 0)
                    self.refOffsets = {'x': pd.Series(x, index=self.sources.ids),
                                       'y': pd.Series(y, index=self.sources.ids)}

            self.plot_sources_on_ref()
            self.refPlot.figure.canvas.mpl_connect("pick_event", self.on_refPick_event)

        self.refInteractMode = new_mode

    def on_ifsGrab_event(self, event):
        """
        Handle event when coordinates are grabbed from the IFU viewer.
       
        Parameters
        ----------
        event : a matplotlib.backend_bases.MouseEvent instance
            The event that triggered the method call.
        """

        # do nothing if event happened outside axes
        if event.inaxes != self.ifsPlot.axes or self.ifsInteractMode != 'grab' or self.ifsToolbar._active is not None:
            return

        # check if click was inside field of view of IFU
        if not self.instrument.infov([[event.xdata, event.ydata]])[0]:
            self.statusBar.showMessage("Out of bounds.", 5000)
            return

        # only do something when left mouse button was clicked
        if event.button != 1:
            return

        # call handler to show spectrum in nearest spaxel
        self.on_ifuPlot_grabbed(event.xdata, event.ydata)

    def on_ifsPick_event(self, event):
        """
        Handle action if one or more sources are picked from the IFU canvas.
        
        Parameters
        ----------
        event : a matplotlib.backend_bases.PickEvent instance
            The mouse event that triggered the method call.
        """
        if self.ifsInteractMode != 'pick' or self.ifsToolbar._active is not None:
            return
        if len(event.ind) != 1:
            self.statusBar.showMessage("Multiple sources selected", 5000)
            return

        # get index of source
        i = event.ind[0]

        # call handler to update GUI
        self.on_ifsPlot_picked(i)

    def on_refPick_event(self, event):
        """
        Handle event when a source is picked from the reference viewer.
        
        Parameters
        ----------
        event : a matplotlib.backend_bases.PickEvent instance
            The mouse event that triggered the method call.
        """
        if self.refInteractMode != 'pick' or self.refToolbar._active is not None:
            return

        # do nothing if more than one source selected
        if len(event.ind) != 1:
            self.statusBar.showMessage("Multiple sources selected.")
            return

        # get index of selected source.
        i = event.ind[0]

        # call method that updates the GUI
        if event.mouseevent.button == 1:
            self.on_refPlot_picked(i, True)
        elif event.mouseevent.button == 3:
            self.on_refPlot_picked(i, False)

# The following methods deal with actions that affect the overall state of the GUI.
    @pyqtSlot()
    def on_actionQuit_triggered(self):
        """
        If GUI should be closed, check if unsaved changes exist. If so, open dialog and
        ask whether it should  be closed anyway. Otherwise, call the method to  save the
        changes.
        """
        if self.unsavedChanges:
            dialog = exitUnsavedDialog()
            dialog.exec_()

            if dialog.status:
                QMainWindow.close(self)
            else:
                return
        else:
            QMainWindow.close(self)

    @pyqtSlot()
    def on_actionSave_triggered(self):
        """
        Save changes to the currently opened prm-file.
        """

        # if no file is currently open, open dialog for selecting a filename
        if self.outfileName is None:
            self.on_actionSaveAs_triggered()

        # if configuration is available, store it in header
        if self.config is not None:
            header = make_header_from_config(self.config)
        else:
            header = None

        save_prm(sources=self.sources, psf_attributes=self.psf, filename=self.outfileName, header=header)

        # update status of GUI.
        logging.info("Wrote results to file {0}.".format(self.outfileName))
        self.statusBar.showMessage("Saving OK!", 5000)
        self.unsavedChanges = False
        self.actionSave.setEnabled(False)

    @pyqtSlot()
    def on_actionSaveAs_triggered(self):
        """
        Save changes to a new file.
        """
        if self.outfileName is None:
            directory = "./"
        else:
            directory = self.outfileName

        # Open dialog to select filename
        self.outfileName = QFileDialog.getOpenFileName(self, caption="Please select a filename", directory=directory,
                                                       filter="*.fits")
        if self.outfileName == "":
            self.statusBar().showMessage("Can't do, no filename selected!")
            return

        # TODO: dialog on whether file should be overwritten

        # call method to save data to a file with known filename
        self.on_actionSave_triggered()

# The following methods deal with actions that affect the state of the spectrum viewer.
    def modifyMask(self, lower, upper, mask):
        """
        A region defined by 'limits' is added to or removed from the spectrumMask.
        The region is added if 'spectrumMask' is True, otherwise it is removed.
        
        Parameters:
        -----------
        lower : float
            The min. value of the range to be masked.
        upper : float
            The max. value of the range to be masked.
        spectrumMask : bool
            Whether the selected region should be included in the spectrumMask
            (if True) or excluded from it (if False).
        """
        if self.spectrumMask is None:
            logging.warning('Missing spectrum mask. Mask modification has no effect.')
            return

        i_min = np.nanargmin(abs(self.sources.wave - lower))
        i_max = np.nanargmin(abs(self.sources.wave - upper))

        self.spectrumMask[i_min:i_max] = not mask
        self.plotMask()

    def plotMask(self):
        """
        Overplot the spectrumMask onto the curve that is currently active in
        the Spectrum Viewer.
        """
        searchMask = np.diff(np.r_[1, self.spectrumMask, 1])
        maskStart = self.sources.wave[np.where(searchMask == -1)[0]]
        maskEnd = self.sources.wave[np.where(searchMask == 1)[0]]

        # add/update one rectangle for each range that is masked out
        # the 'addRectangle' method of 'data_plot' will know from
        # the provided 'gid' whether an existing rectangle can be
        # updated or a new one must be created.
        i = 0
        while i < maskStart.size:
            start = maskStart[i]
            end = maskEnd[i]
            self.dataPlot.addRectangle((start, 0), width=end-start, edgecolor="w", facecolor="k", alpha=0.5,
                                       gid="spectrumMask{0}".format(i))
            i += 1

        # remove rectangles that are not needed any more.
        while "spectrumMask{0}".format(i) in self.dataPlot.bar_gid:
            self.dataPlot.delRectangle("spectrumMask{0}".format(i))
            i += 1

        self.dataPlot.draw()

    @pyqtSlot(str)
    def on_spectrumShowHduComboBox_activated(self, name):
        """
        update the selection in the spectrumShowIdentComboBox
        when a new HDU is selected.
        """
        self.spectrumShowIdentComboBox.clear()

        if name in ["x Positions", "y Positions"]:
            self.spectrumShowIdentComboBox.addItems(self.available_data["Positions"])
        else:
            self.spectrumShowIdentComboBox.addItems(self.available_data[str(name)])

    @pyqtSlot()
    def on_spectrumShowGo_pressed(self, updateYRange=True):
        """
        Handle event that a new curve should be plotted.
        
        Parameters
        ----------
        updateYRange : bool, optional
            Flag indicating whether the y-limits of the plot canvas should
            be updated to fit the new data.
        """
        # retrieve current data for selected HDU and property
        if self.current_hdu == "Spectra":
            data = self.sources.viewSpectrum(name=self.current_row_in_hdu)
            fit = None
            label = r"$f_\lambda\,[\mathrm{counts}/\mathrm{layer}]$"

        elif self.current_hdu in ["x Positions", "y Positions"]:
            source_id = int(self.current_row_in_hdu)
            ifs_coordinates = self.sources.viewIfsCoordinates(source_id=source_id)
            axis = 'x' if self.current_hdu == 'x Positions' else 'y'
            data = ifs_coordinates[(axis, 'value')]
            fit = ifs_coordinates[(axis, 'fit')] if self.sources.ifs_coordinates_fitted[source_id] else None
            label = r"${0}_\mathrm{{IFS}}\,[\mathrm{{spaxel}}]$".format(axis)

        elif self.current_hdu == "PSF":
            parameter = self.current_row_in_hdu
            data = self.psf.data[(parameter, 'value')]
            fit = self.psf.data[(parameter, 'fit')]
            label = self.psf.LABELS[parameter]

        elif self.current_hdu == "Coord. Transf.":
            parameter = self.current_row_in_hdu
            data = self.sources.transformation.data[(parameter, 'value')]
            fit = self.sources.transformation.data[(parameter, 'fit')]
            label = r"${{\rm {0}}}$".format(self.current_row_in_hdu)

        else:
            logging.error('Unknown HDU encountered: "{0}"'.format(self.current_hdu))
            return

        # update plot
        if self.sources.cunit == "m":
            x_label = r"$\lambda\,[\mathrm{m}]$"
        elif self.sources.cunit == "um":
            x_label = r"$\lambda\,[\mathrm{\mu m}]$"
        elif self.sources.cunit == "nm":
            x_label = r"$\lambda\,[\mathrm{nm}]$"
        else:
            x_label = r"$\lambda\,[\mathrm{\AA}]$"

        valid = np.isfinite(data)
        self.dataPlot.updateData(name='data', x_data=data[valid].index.values, y_data=data[valid].values,
                                 x_label=x_label, y_label=label, update_y_range=updateYRange)
        if fit is not None and np.isfinite(fit).any():
            valid = np.isfinite(fit)
            self.dataPlot.updateData(name='fit', x_data=fit[valid].index.values, y_data=fit[valid].values,
                                     update_y_range=updateYRange)
        else:
            self.dataPlot.updateData(name='fit', x_data=[], y_data=[], update_y_range=updateYRange)

        # show spectrumMask
        self.plotMask()

        # write out some info
        self.statusBar.showMessage("Fetching data successful.", 5000)

        # if IFU and/or reference data loaded, highlight currently selected source in the respective plot.
        try:
            source_id = int(self.current_row_in_hdu)
        except ValueError:  # if current_row_in_hdu is not an ID.
            return
        else:
            if self.ifsPlot.ids is not None and source_id in self.ifsPlot.ids:
                self.on_ifsPlot_picked(np.where(self.ifsPlot.ids == source_id)[0][0])
            if self.refLoaded and source_id in self.refPlot.ids:
                self.on_refPlot_picked(np.where(self.refPlot.ids == source_id)[0][0], True)

    @pyqtSlot()
    def on_spectrumFitGo_pressed(self):
        """
        Handle event when the currently displayed spectral data should be fitted.
        """
        # get fit function and degree of polynomial
        function = str(self.spectrumFitTypeComboBox.currentText())
        order = int(self.spectrumFitOrderSpinBox.value())

        if self.current_hdu == "Spectra":
            self.statusBar.showMessage("Fitting spectra is not supported.",  5000)
            return

        elif self.current_hdu == "PSF":
            self.psf.polynomial_fit(name=self.current_row_in_hdu, order=order, mask=~self.spectrumMask,
                                    polynom=function.lower())

        elif self.current_hdu in ["x Positions", "y Positions"]:

            if self.actionFit_all_positions.isChecked():
                ids = None
            else:
                ids = int(self.current_row_in_hdu)

            self.sources.fitIfsCoordinates(ids=ids, order=order, mask=~self.spectrumMask, polynom=function.lower(),
                                           common_slope=self.actionForce_common_slope.isChecked(),
                                           statusBar=self.statusBar, instrument=self.instrument)

        elif self.current_hdu == "Coord. Transf.":
            self.statusBar.showMessage(
                "Fitting coord. transf. parameters is deprecated. Directly fit IFS coordinates instead.",  5000)
            return
        else:
            logging.error('Encountered unexpected HDU: "{0}"'.format(self.current_hdu))
            return

        self.unsavedChanges = True
        self.actionSave.setEnabled(True)
        self.on_spectrumShowGo_pressed(updateYRange=False)

# The following methods handle actions that affect the IFU viewer
    @pyqtSlot(int, int)
    def on_ifuLayerSelectSlider_rangeChanged(self, min, max):
        """
        If the range of the slider through the layers of the datacube
        changes, update range of the spin box accordingly.
        
        Parameters
        ----------
        min : int
            The minimum value of the slider range
        max : int
            The maximum value of the slider range
        """
        self.ifuLayerSelectSpinBox.setRange(min, max)

    @pyqtSlot(int)
    def on_ifuLayerSelectSlider_sliderMoved(self, position):
        """
        Update IFU viewer if a new layer is selected via
        the slider and synchronize the spin box.
        NOTE: this method is only triggered if the value is
              actually changed.
        
        Parameters
        ----------
        position : int
            The new position of the slider.
        """

        # synchronize spin box - the spin box will also check whether a cube is loaded
        # and will synchronise the pointer in the spectrum viewer
        self.ifuLayerSelectSpinBox.setValue(position)

    @pyqtSlot(int)
    def on_ifuLayerSelectSpinBox_valueChanged(self, p0):
        """
        Update IFU cube plot if a new layer is selected via
        the spin box and synchronize the slider.
        NOTE: this method is only triggered if the value is
              actually changed.

        Parameters
        ----------
        p0 : The new value of the spinbox
        """
        # synchronize slider
        self.ifuLayerSelectSlider.setValue(p0)

        # show new layer if IFU data
        if self.ifsLoaded:
            self.ifsPlot.loadLayer(p0, residuals=self.actionResiduals.isChecked(), update_axlims=False,
                                   reset_history=False)

            # change cuts unless absolute values were set by the user:
            if self.ifuCutsComboBox.currentText() != "User":
                self.on_ifuCutsComboBox_currentIndexChanged(self.ifuCutsComboBox.currentText())

            # #  also update position of pointer if it is currently animated
            # if self.data_plot.vLine.get_animated():
            #     self.data_plot.vLine.set_data([self.ifsPlot.wave, self.ifsPlot.wave], self.data_plot.axes.get_ylim())
            #
            #     self.data_plot.figure.canvas.restore_region(self.data_plot.background)
            #     self.data_plot.axes.draw_artist(self.data_plot.vLine)
            #     self.data_plot.figure.canvas.blit(self.data_plot.axes.bbox)

    @pyqtSlot(str)
    def on_ifuCutsComboBox_currentIndexChanged(self, p0):
        """
        Handle event that a new cut level for the IFU cube is selected
        from the combo box.
        
        p0 : string
            The text that is currently displayed in the combo box.
        """
        # if new level is 'User', enable the boxes that show the min. and max. cuts for user input
        if p0 == "User":
            self.ifuMinCutLineEdit.setEnabled(True)
            self.ifuMaxCutLineEdit.setEnabled(True)
            return
        # otherwise disable them and update cuts using new cut level.
        else:
            self.ifuMinCutLineEdit.setEnabled(False)
            self.ifuMaxCutLineEdit.setEnabled(False)
            self.ifsPlot.changeCuts(vrange=float(p0[:-1]))

            self.ifuMinCutLineEdit.setText("{0:.5g}".format(float(self.ifsPlot.cuts[0])))
            self.ifuMaxCutLineEdit.setText("{0:.5g}".format(float(self.ifsPlot.cuts[1])))

    @pyqtSlot()
    def on_ifuMinCutLineEdit_returnPressed(self):
        """
        Handle event that minimum cut of the IFU plot is changed via the line
        edit. NOTE that this method is only triggered when <ENTER> is hit.
        """
        self.ifsPlot.changeCuts(locut=float(self.ifuMinCutLineEdit.text()))

    @pyqtSlot()
    def on_ifuMaxCutLineEdit_returnPressed(self):
        """
        Handle event that maximum cut of the IFU plot is changed via the line
        edit. NOTE that this method is only triggered when <ENTER> is hit.
        """
        self.ifsPlot.changeCuts(hicut=float(self.ifuMaxCutLineEdit.text()))

    def on_ifuPlot_grabbed(self, x, y):
        """
        If a position is grabbed from the IFU viewer, show the spectrum
        of the nearest spaxel.
        
        Parameters
        ----------
        x : float
            The x coordinate of the grab event.
        y : float
            The y-coordinate of the grab event.
        """
        # show spectrum in spectrum viewer
        self.dataPlot.updateData(name='data', x_data=self.ifsPlot.cube.wave, y_data=self.ifsPlot.getNearestSpaxel(x, y),
                                 x_label=r"$\lambda$", y_label=r"$f_\lambda\,[\mathrm{counts}]$")

        # indicate position where grab event took place in IFU viewer
        self.ifsPlotCurrentlyMarked.set_data(x, y)

        # if the GUI is used to identify sources in the IFU data, add the event to the selection
        if self.currentRefID is not None:
            self.identifiedSources[self.currentRefID] = [x, y, 0., 0.]
            logging.info(
                'Identified source #{0} (x_ref={1:.1f}, y_ref={2:.1f}) at (x_ifs={3:.1f}, y_ifs={4:.1f})'.format(
                    self.currentRefID, self.sources.x[self.currentRefID], self.sources.y[self.currentRefID], x, y))
            self.updateMarkHistPlot()

        self.ifsPlot.draw()

    def on_ifsPlot_picked(self, index):
        """
        If a source is picked from the IFS image via a mouse click,
        show its spectrum or position.
        
        Parameters
        ----------
        index : int
            The index of the selected source in the IFU viewer plot.
        """
        # recover ID
        ident = self.ifsPlot.ids[index]

        # mark source in datacube image by setting the facecolor of its marker to the edgecolor of its marker,
        # all other symbols with transparent face
        facecolors = np.zeros((len(self.ifsPlot.ids), 4), dtype=np.float32)
        facecolors[index] = self.ifsPlot.edgecolors[index]
        self.ifsPlot.data_plot.set_facecolors(facecolors)

        # check if ID is available in HDU that is currently loaded in the
        # spectrum viewer. If so, show the corresponding curve.
        indexInComboBox = self.spectrumShowIdentComboBox.findText(str(ident))
        if indexInComboBox >= 0 and self.spectrumShowIdentComboBox.currentText() != str(ident):
            self.spectrumShowIdentComboBox.setCurrentIndex(indexInComboBox)
            self.on_spectrumShowGo_pressed()
        self.ifsPlot.figure.canvas.draw()

        # update status bar
        self.statusBar.showMessage("Selected source: {0}".format(ident), 5000)

# The following methods deal with actions that affect the state of the reference view.
    @pyqtSlot(str)
    def on_refCutsComboBox_currentIndexChanged(self, p0):
        """
        Handle event that a new cut level for the reference image is
        selected from the combo box.
        
        p0 : string
            The text that is currently displayed in the combo box.
        """
        if "refPlot" not in self.__dict__:
            return

        # if new level is 'User', enable the boxes that show the min.
        # and max. cuts for user input
        if p0 == "User":
            self.refMinCutLineEdit.setEnabled(True)
            self.refMaxCutLineEdit.setEnabled(True)
            return
        # otherwise disable them and update cuts using new cut level.
        else:
            self.refMinCutLineEdit.setEnabled(False)
            self.refMaxCutLineEdit.setEnabled(False)
            self.refPlot.changeCuts(vrange=float(p0[:-1]))

            self.refMinCutLineEdit.setText("{0:.5g}".format(float(self.refPlot.cuts[0])))
            self.refMaxCutLineEdit.setText("{0:.5g}".format(float(self.refPlot.cuts[1])))

    @pyqtSlot()
    def on_refMinCutLineEdit_returnPressed(self):
        """
        Handle event that minimum cut of the reference plot is changed
        via the line edit. NOTE that this method is only triggered when
        <ENTER> is hit.
        """
        if "refPlot" not in self.__dict__:
            return

        self.refPlot.changeCuts(locut=float(self.refMinCutLineEdit.text()))

    @pyqtSlot()
    def on_refMaxCutLineEdit_returnPressed(self):
        """
        Handle event that maximum cut of the reference plot is changed
        via the line edit. NOTE that this method is only triggered when
        <ENTER> is hit.
        """
        if "refPlot" not in self.__dict__:
            return

        self.refPlot.changeCuts(hicut=float(self.refMaxCutLineEdit.text()))

    def on_refPlot_picked(self, index, select):
        """
        Update status of GUI after a source has been selected from the
        reference viewer. This method is called by 'on_refPick_event'.
        
        Parameters
        ----------
        index : int
            The index of the selected source in the reference view plot.
        select : bool
            A flag indicating whether the source should be selected (if
            True) or de-selected (if False).
        """
        # get source ID
        current_id = self.refPlot.ids[index]

        # source is selected
        if select:
            # show information on selected source
            statusMessage = "Picked source #{0}: ".format(int(current_id))
            statusMessage += "x={0:.2f}, y={1:.2f}".format(float(self.sources.x[current_id]),
                                                           float(self.sources.y[current_id]))
            if self.sources.mag is not None:
                statusMessage += ", mag={0:.2f}".format(float(self.sources.mag[current_id]))
            self.statusBar.showMessage(statusMessage, 5000)

            # set face color of selected marker to its edge color, all other sources are transparent
            facecolors = np.zeros((len(self.refPlot.ids), 4), dtype=np.float32)
            facecolors[index] = self.refPlot.edgecolors[index]

            if self.ifsInteractMode == "grab":
                self.currentRefID = current_id

                # mark previously selected sources
                for source_id in self.identifiedSources.keys():
                    if source_id in self.refPlot.ids:
                        i = np.flatnonzero(self.refPlot.ids == source_id)[0]
                        facecolors[i] = self.refPlot.edgecolors[i]

                # make facecolor of selected marker 50% transparent
                facecolors[index, -1] /= 2.

            elif self.ifsInteractMode == "pick":
                # check whether selected source is included in the currently loaded dataset in the spectrum viewer.
                # If so, load the data. same source in IFU viewer; Note that the method called below
                # will also take care that the source is highlighted in the IFU viewer if possible.
                indexInComboBox = self.spectrumShowIdentComboBox.findText(str(current_id))
                if indexInComboBox >= 0 and self.spectrumShowIdentComboBox.currentText() != str(current_id):
                    self.spectrumShowIdentComboBox.setCurrentIndex(indexInComboBox)
                    self.spectrumShowGo.click()

            self.refPlot.data_plot.set_facecolors(facecolors)
            self.refPlot.draw()

        else:
            if current_id in self.identifiedSources.keys():
                del self.identifiedSources[current_id]
                self.currentRefID = None

                if self.ifsInteractMode == "grab":
                    self.updateMarkHistPlot()
                logging.info('Unselected source #{0}.'.format(current_id))

    def updateMarkHistPlot(self):
        """
        Update the plot containing the identification history, i.e.
        the sources from the reference viewer for which a position was
        identified in the IFU viewer.
        """

        # update data in ifuViewer
        if self.identifiedSources is None:
            self.ifsPlotPreviouslyMarked.set_data([], [])
        else:
            self.ifsPlotPreviouslyMarked.set_data([v[0] for v in self.identifiedSources.values()],
                                                  [v[1] for v in self.identifiedSources.values()])
        self.ifsPlot.draw()

        # mark sources on reference viewer:
        # all symbols transparent, except selected, they are colored using the edgecolor
        facecolors = np.zeros((len(self.refPlot.ids), 4), dtype=np.float32)
        for ident in self.identifiedSources.keys():
            if ident in self.refPlot.ids:
                i = np.where(self.refPlot.ids == ident)[0][0]
                facecolors[i] = self.refPlot.edgecolors[i]
        self.refPlot.data_plot.set_facecolors(facecolors)
        self.refPlot.draw()

    @pyqtSlot(bool)
    def on_actionData_triggered(self, checked):
        """
        This method is called if the button to switch from the display of
        the residuals to the IFS data is hit.

        :param checked: New (boolean) state of the actionData button.
        :return: None
        """
        self.actionResiduals.setChecked(not checked)

        self.ifsPlot.loadLayer(self.ifuLayerSelectSlider.value(), residuals=False)

        # change cuts unless absolute values were set by the user:
        if self.ifuCutsComboBox.currentText() is not "User":
            self.on_ifuCutsComboBox_currentIndexChanged(self.ifuCutsComboBox.currentText())

    @pyqtSlot(bool)
    def on_actionResiduals_triggered(self, checked):
        """
        This method is called if the button to switch from the display of
        the IFS data to the residuals is hit.

        :param checked: New (boolean) state of the actionResiduals button.
        :return: None
        """
        self.actionData.setChecked(not checked)

        self.ifsPlot.loadLayer(self.ifuLayerSelectSlider.value(), residuals=True)

        # change cuts unless absolute values were set by the user:
        if self.ifuCutsComboBox.currentText() is not "User":
            self.on_ifuCutsComboBox_currentIndexChanged(self.ifuCutsComboBox.currentText())

    def plot_sources_on_ref(self):
        """
        This method overlays the sources on the reference image.

        :return: None
        """
        logging.debug('Overplotting sources on reference image ...')
        if self.refPlot.data_plot is not None:
            self.refPlot.data_plot.remove()
            self.refPlot.data_plot = None

        if len(self.sources.plotted) == 0:
            self.refPlot.draw()

        else:
            # if sources also overplotted on IFS data, copy properties from there
            kwargs = {'picker': 5}
            if self.ifsPlot.data_plot and self.ifsInteractMode == 'pick':
                kwargs['edgecolors'] = self.ifsPlot.data_plot.get_edgecolors()
                kwargs['facecolors'] = self.ifsPlot.data_plot.get_facecolors()
                kwargs['s'] = self.ifsPlot.data_plot.get_sizes()
            else:
                kwargs['edgecolor'] = 'r'
                kwargs['facecolor'] = None
                if self.sources.mag is not None:
                    # determine size of datapoints representing the sources (as for the IFU plot, the magnitudes from
                    # the source catalogue are used for this purpose if available)
                    logging.debug('Calculating sizes of source symbols ...')
                    sizes = 2.*(10**(-0.2*(self.sources.mag[self.sources.plotted] - self.sources.mag.max())))
                    sizes[sizes <= 1] = 0
                else:
                    sizes = 10*np.ones(self.sources.n_sources, dtype=np.float32)
                kwargs['s'] = sizes

            logging.debug('Updating plot ...')
            # if dedicated pixel coordinates for reference image available, use them, otherwise reference coordinates
            # from Sources class
            if self.refOffsets is not None:
                self.refPlot.addData(self.refOffsets['x'][self.sources.plotted],
                                     self.refOffsets['y'][self.sources.plotted],
                                     ids=self.sources.plotted, update_axes_limits=False, **kwargs)
            else:
                self.refPlot.addData(self.sources.x[self.sources.plotted], self.sources.y[self.sources.plotted],
                                     ids=self.sources.plotted, update_axes_limits=False, **kwargs)

            # recover sources previously identified on IFS data (if INITFIT is being used)
            self.updateMarkHistPlot()

    def plot_sources_on_ifs(self, update_sources=False):
        """
        This method overlays the sources on the IFS image.

        :param update_sources: Flag indicating if a new plot needs to be
        created (if True) because the number of sources changed compared
        to the previous plot or if the plot can be updated in place.
        :return: None
        """
        index = self.sources.wave[self.ifuLayerSelectSpinBox.value()]

        if update_sources:
            if self.ifsPlot.data_plot is not None:
                # TODO: combine the following two commands.
                self.ifsPlot.data_plot.remove()  # remove scatter plot from Axes instance
                self.ifsPlot.data_plot = None  # update state of DynamicImaCanvas instance

            if not len(self.sources.plotted):
                self.ifsPlot.draw()

            else:
                x = self.ifsOffsets['x'].loc[index, self.sources.plotted]
                y = self.ifsOffsets['y'].loc[index, self.sources.plotted]

                # if magnitudes are available in the sources catalogue, we use them to scale the circles representing
                # the sources.
                if not np.isnan(self.sources.mag).all():
                    sizes = 10.*(10**(-0.2*(self.sources.mag[self.sources.plotted] - max(self.sources.mag))))
                else:
                    sizes = 20*np.ones(len(self.sources.plotted), dtype=np.float32)

                edgecolors = colors.ColorConverter().to_rgba_array(self.edgeColors[self.sources.plotted].values)
                facecolors = np.zeros((len(self.sources.plotted), 4), dtype=np.float32)

                self.ifsPlot.addData(x, y, s=sizes, ids=self.sources.plotted, picker=3, edgecolors=edgecolors,
                                     facecolors=facecolors, update_axes_limits=False)
                # NOTE: self.ifsPlot.draw() is unnecessary because it is executed by the addData method.

        else:  # if the number of sources stays the same, update the scatter plot in place.
            x_offsets = self.ifsOffsets['x'].loc[index, self.sources.plotted]
            y_offsets = self.ifsOffsets['y'].loc[index, self.sources.plotted]
            assert(len(self.soures.plotted) == self.ifsPlot.dataPlot.get_offsets()[0])
            self.ifsPlot.dataPlot.set_offsets(np.stack((x_offsets, y_offsets), axis=1))

    @pyqtSlot(int)
    def on_plotStarsCheckBoxPSF_stateChanged(self, state):
        """
        This method is called whenever the state of the check box changes that
        indicates if the PSF stars should be overlaid on the reference and/or
        IFS data plot.

        Parameters
        ----------
        state : boolean
            The new state of the check box. Note that it is not a boolean flag
            but an integer (0 if disabled, 2 if enabled)
        """
        self.deselect_sources_by_status(status=3) if not state else self.select_sources_by_status(status=3)

    @pyqtSlot(int)
    def on_plotStarsCheckBoxResolved_stateChanged(self, state):
        """
        This method is called whenever the state of the check box changes that
        indicates if the resolved stars should be overlaid on the reference
        and/or the IFS data plot.

        Parameters
        ----------
        state : boolean
            The new state of the check box. Note that it is not a boolean flag
            but an integer (0 if disabled, 2 if enabled)
        """
        self.deselect_sources_by_status(status=2) if not state else self.select_sources_by_status(status=2)

    @pyqtSlot(int)
    def on_plotStarsCheckBoxNearby_stateChanged(self, state):
        """
        This method is called whenever the state of the check box changes that
        indicates if the nearby stars should be overlaid on the reference
        and/or the IFS data plot.

        Parameters
        ----------
        state : boolean
            The new state of the check box. Note that it is not a boolean flag
            but an integer (0 if disabled, 2 if enabled)
        """
        self.deselect_sources_by_status(status=1) if not state else self.select_sources_by_status(status=1)

    @pyqtSlot(int)
    def on_plotStarsCheckBoxUnresolved_stateChanged(self, state):
        """
        This method is called whenever the state of the check box changes that
        indicates if the unresolved stars should be overlaid on the reference
        and/or the IFS data plot.

        Parameters
        ----------
        state : boolean
            The new state of the check box. Note that it is not a boolean flag
            but an integer (0 if disabled, 2 if enabled)
        """
        self.deselect_sources_by_status(status=0) if not state else self.select_sources_by_status(status=0)

    def select_sources_by_status(self, status):
        """
        The method adds stars with the given status to the selection of stars
        that are overplotted on the IFS/reference GUI.
        
        Parameters
        ----------
        status : int
            The status of the sources to be added.
        """
        self.sources.plotted = self.sources.plotted.union(
            self.sources.ids[(self.sources.status == status) & (
                self.sources.mag >= self.plotted_mag_range[0]) & (
                self.sources.mag <= self.plotted_mag_range[1])])
        self.plotted_statuses.append(status)

        if self.ifsInteractMode == 'pick':
            self.plot_sources_on_ifs(update_sources=True)
        if self.refInteractMode == 'pick':
            self.plot_sources_on_ref()

    def deselect_sources_by_status(self, status):
        """
        The method removes stars with the given status from the selection of
        stars that are overplotted on the IFS/reference GUI.
        
        Parameters
        ----------
        status: int
            The status of the sources to be removed.
        """
        self.sources.plotted = self.sources.plotted.intersection(self.sources.ids[self.sources.status != status])
        self.plotted_statuses.remove(status)

        if self.ifsInteractMode == 'pick':
            self.plot_sources_on_ifs(update_sources=True)
        if self.refInteractMode == 'pick':
            self.plot_sources_on_ref()

    @pyqtSlot(float)
    def on_minMagSpinBox_valueChanged(self, double):
        """
        This method is called whenever the value in the spin box with
        the lower limit of the selected magnitude range changes.

        Parameters
        ----------
        double : float
            The new value in the spin box.
        """
        if double > self.plotted_mag_range[0]:
            self.sources.plotted = self.sources.plotted.intersection(self.sources.ids[self.sources.mag >= double])
        else:
            for i in self.plotted_statuses:
                self.sources.plotted = self.sources.plotted.union(
                    self.sources.ids[(self.sources.status == i) & (self.sources.mag >= double) & (
                        self.sources.mag <= self.plotted_mag_range[0])])
        self.plotted_mag_range[0] = double

        if self.ifsInteractMode == 'pick':
            self.plot_sources_on_ifs(update_sources=True)
        if self.refInteractMode == 'pick':
            self.plot_sources_on_ref()

    @pyqtSlot(float)
    def on_maxMagSpinBox_valueChanged(self, double):
        """
        This method is called whenever the value in the spin box with
        the upper limit of the selected magnitude range changes.
        
        Parameters
        ----------
        double : float
            The new value in the spin box.
        """
        if double < self.plotted_mag_range[1]:
            self.sources.plotted = self.sources.plotted.intersection(self.sources.ids[self.sources.mag <= double])
        else:
            for i in self.plotted_statuses:
                self.sources.plotted = self.sources.plotted.union(
                    self.sources.ids[(self.sources.status == i) & (self.sources.mag >= self.plotted_mag_range[1]) & (
                        self.sources.mag <= double)])
        self.plotted_mag_range[1] = double

        if self.ifsInteractMode == 'pick':
            self.plot_sources_on_ifs(update_sources=True)
        if self.refInteractMode == 'pick':
            self.plot_sources_on_ref()

    @pyqtSlot()
    def on_actionLoadImage_triggered(self):
        """
        Open a dialog to load a reference image from disk.
        """
        if self.outfileName is None:
            directory = "./"
        else:
            directory = self.outfileName

        # Open dialog to select filename
        filename = QFileDialog.getOpenFileName(self, caption="Please select a filename", directory=directory,
                                               filter="*.fits")
        if filename == "":
            self.statusBar().showMessage("Can't do, no filename selected!")
            return

        # call method to save data to a file with known filename
        self.openReference(filename)
