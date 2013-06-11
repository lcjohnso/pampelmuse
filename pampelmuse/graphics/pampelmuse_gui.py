# -*- coding: utf-8 -*-

"""
Module implementing the GUI of PampelMuse. It is designed to facilitate the
polynomial fitting of the coordinates and PSF parameters and to allow for
a visual inspection of the fitting results.
This file contains the dialogue code for Ui_pampelmuse_gui.py that is created
automatically by QtDesigner.

CHANGELOG
=========
v0.6.0.0, 2013/05/06, migrated an older existing code (inspect_fit_data.py)
                      that was so far used to handle the GUI.
v0.7.0.0, 2013/05/13, merged the existing GUIs for source identification and
                      inspecting and fitting the results.
v0.8.0.0, 2013/05/15, added documentation to the code and streamlined it. Also
                      added options to pan across the reference view and to
                      go scroll through the history of previously displlayed
                      cutouts of the data.
v0.9.0.0, 2013/05/24, increased flexibility of the code; when a cube is loaded
                      the routine also accepts an instance of the Cube class of
                      PampelMuse; when source spectra or coordinates are loaded
                      and no data is provided, the active instance of the Sources
                      class is checked for the data.
v0.9.0.1, 2013/05/29, implemented usage of the namespace provided by 
                      PampelMuseConfig instead of defining an own namespace.
v0.9.1.1, 2013/05/31, so far, when both IFU coordinates and polynomial fits were
                      provided, both were overplotted on the datacube. This has
                      been changed.
v0.9.1.2, 2013/06/11, when overplotting the source list on the reference view,
                      the sky is now excluded.
"""
__author__ = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__version__ = "0.9.1.2"

import datetime, logging, re, os, sys
import matplotlib.pyplot as plt, numpy as np, pyfits
from PyQt4.QtGui import QMainWindow, QHBoxLayout, QDoubleValidator, QFileDialog
from PyQt4.QtCore import pyqtSignature, Qt, SIGNAL

from Ui_pampelmuse_gui import Ui_PampelMuseGui
from exit_dialog import exitUnsavedDialog

## Required PampelMuse packages
from pampelmuse.config import Namespace, PampelMuseConfig
from pampelmuse.core.psf import PsfAttributes
from pampelmuse.core.sources import Sources
from pampelmuse.graphics.dynamic_plot_canvas import DynamicPlotCanvas
from pampelmuse.graphics.dynamic_ima_canvas import DynamicImaCanvas




class PampelMuseGui(QMainWindow, Ui_PampelMuseGui):
    """
    The class defines the various actions that can be triggered by the
    user to interact with the GUI. The GUI serves two purposes:
    (i)  a visual inspection of the fit results.
    (ii) performing polynomial fits to the source positions and/or
         PSF parameters.
    """
    def __init__(self, sources=None, fitsFile=None, ifsData=None, refImage=None, 
                 parent=None, namespace=None, **kwargs):
        """
        Construct a new instance of the class 'PampelMuseGui'.
        
        Parameters
        ----------
        sources : instance of either numpy.recarray or pyfits.BinTableHDU, optional
            The components (stellar sources, background components) that
            are present in the analysed field, either provided as a FITS
            HDU (binary table) or a record array.
        fitsFile : string
            A FITS-file containing the catalogue of sources and/or information about the
            sources such as their spectra or positions or about the PSF. An example
            for such a file is a valid prm-file as it is created by PampelMuse.
        ifsData : string or instance of pampelmuse.core.cube.Cube, optional
            A FITS-file containing the datacube that is analysed or an instance of the
            Cube class of PampelMuse holding the IFS data.
        refImage : string or nd_array, optional
            A reference image, provided either as a FITS file or a 2dim image array.
            Note that the idea is that the refrence coordinates used by PampelMuse are
            the pixel coordinates in this image.
        parent : 
            The 'parent' argument, if not None, causes self to be 
            owned by Qt instead of PyQt.
            (copied from PyQt4 documentation)
        namespace : a python namespace, optional
            The class uses a configuration namespace that contains information, e.g.
            about the instrument or the version of the code. By default, an empty
            namespace is initialized. To add the configuration to an existing one,
            this namespace should be provided here.
        **kwargs
            Any keyword args are passed to the methods called by __init__.
        """
        QMainWindow.__init__(self, parent)
        self.setupUi(self)

        ## Initialize variables for later use:

        ## PampelMuse uses a suite of hierarch keywords to store its configuration
        ## that we collect in a namespace
        if namespace == None:
            self.namespace = PampelMuseConfig()
        else:
            self.namespace = namespace

        self.mask = None  ## mask along spectral axis
        self.cubeLoaded = False  ## flag indicating whether a cube is loaded
        self.refLoaded = False  ## flag indicating whether a reference image is loaded
        self.ifuInteractMode = None ## status of the interaction mode with the IFU data
        self.curId = None ## ID of currently selected star
        self.unsavedChanges = False ## flag indicating whether there are unsaved changes
        self.outfileName = None ## name of the FITS file used to store the results
        self.identifiedSources = {} ## a dictionary {ID:(x,y)} of sources identified in
                                    ## the reference and the IFU viewer

        ## Initial setup for the spectrum viewer:
        self.dataPlot = DynamicPlotCanvas()

        plotFrame = QHBoxLayout(self.spectrumWidget)
        plotFrame.addWidget(self.dataPlot)
        plotFrame.setAlignment(Qt.AlignCenter)

        ## Initialize plot that will contain data.
        self.dataPlot.addData(name='data', xData=[], yData=[], ls='-', c='b', marker='None')

        ## Initialize plot that will contain current fit.
        self.dataPlot.addData(name='fit', xData=[], yData=[], ls='--', c='r', lw=1.8, marker='None')

        ## check if valid instance of Sources provided
        if sources != None:
            assert isinstance(sources, Sources), "If provided, 'sources' must be an instance of pampelmuse.core.sources.Sources"
            self.sources = sources

            self.sourceList = {} ## a dictionary that will contain the available HDUs
                                ## and the included spectral properties that can be
                                ## displayed/fitted.
            self.pickSourcesOnRef() ## initialize source selection via reference image
                                ## if possible
            self.actionSaveAs.setEnabled(True)
        
        ## open prm-file if provided
        if fitsFile != None:
            self.openFile(fitsFile)
        else:
            self.hasPrm = False

        ## Initial status of GUI
        self.fitAll = True  ## default is to fit pos. simultaneously
        self.commonSlope = True ##  default is to use common slope when fitting positions
        
        ## connect data selection process to masking method
        self.connect(self.dataPlot, SIGNAL('rangeSelected'), self.modifyMask)
        
        ## load datacube into GUI if one is provided
        if ifsData != None:
            self.openIfu(ifsData)
        
        ## load reference image into GUI if one is provided 
        if refImage != None:
            self.openReference(refImage, **kwargs)
        
        self.statusBar.showMessage("Ready", 5000)
    
    
    @property
    def instrument(self):
        """
        The instrument used to observe the IFU data
        """
        if hasattr(self.namespace, "ifu"):
            return self.namespace.ifu
        else:
            return None
            
    @property
    def start(self):
        """
        The index of the first layer of the IFU data that is analyzed 
        """
        if hasattr(self.namespace, "start"):
            return self.namespace.start
        else:
            return None


    def loadSources(self, data, instrument=None):
        """
        Initialize the catalogue of sources to work with.
        
        Parameters
        ----------
        data : record array or pyfits.BinTableHDU instance
            The source catalogue: at least the columns 'ID', 'x' and 'y'
            should be available.
        instrument : string or pampelmuse.core.instruments.Instrument instance
            The instrument used to observe the IFU data.
        """
        self.sources = Sources(data=data,  instrument=instrument)
        self.sourceList = {} ## a dictionary that will contain the available HDUs
                             ## and the included spectral properties that can be
                             ## displayed/fitted.
        self.pickSourcesOnRef() ## initialize source selection via reference image
                                ## if possible
    
    
    def loadSpectra(self, data=None):
        """
        Read the source spectra from a 2dim. array or a pyfits ImageHU.
        
        Parameters
        ----------
        data : nd_array or pyfits.ImageHDU instance, optional
            The set of spectra to be loaded. Note that this method may only be
            called after the source catalogue has been initialized and that a
            check is carried out whether the number of spectra matches the number
            of components in the catalogue.
            If no data are provided, it is assumed that the 'Sources' instance
            that is currently active includes spectra.
        """
        if not "sources" in self.__dict__:
            logging.error("Cannot load spectra unless information about sources available.")
            return

        if data != None:
            self.sources.loadSpectra(data=data)
        
        ## create entry about loaded data for data selection via QComboBox if not existent
        if self.spectrumShowHduComboBox.findText("Spectra") == -1:
            self.spectrumShowHduComboBox.addItem("Spectra")
        self.sourceList["Spectra"] = self.sources.provideNames()
    
        self.dataLoaded() ## update status of GUI
    
    
    def loadCoordTrans(self, data=None):
        """
        Read the coordinate transformation from reference to IFU coordinates
        form a FITS table.
        If no data are provided, it is checked whether the instance of 'Sources'
        that is currently active contains a valid coordinate transformation.

        Parameters
        ----------
        data : pyfits.BinTableHDU instance
            The HDU containing the coordinate transformation, one parameter per
            row.
        """

        if not "sources" in self.__dict__:
            logging.error("Cannot load coord. transf. unless information about sources available.")
            return
        
        if data != None:
            self.sources.readTransHDU(self.HDUList["POSPARS"])
    
        ## create entries for positions and coord. transf. in QComboBox if not existent
        if self.sources.verifyTrans:
            for entry in ["x Positions", "y Positions", "Coord. Transf."]:
                if self.spectrumShowHduComboBox.findText(entry) == -1:
                    self.spectrumShowHduComboBox.addItem(entry)
            self.sourceList["Positions"] = self.sources.providePosIDs()
            self.sourceList["Coord. Transf."] = self.sources.provideParNames()
        
            ## NOTE: currently all IFU coordinates are loaded into memory, this
            ## can be very demanding for large data sets
            # TODO: make this more memory efficient
            self.sources.loadIFUCoords()
        
        self.pickSourcesOnIfu() ## check whether source selection from IFU frame
                                ## can be enabled.
        
        self.dataLoaded() ## update status of GUI
    
    
    def loadPositions(self, data=None):
        """
        Read the IFU coordinates from a 3dim. ImageHDU.
        
        Parameters
        ----------
        data : pyfits.ImageHDU instance, optional
            The coordinates of the sources on the IFU data as a function of
            wavelength. NOTE that this method may only be called after a
            catalogue of sources and their spectra have been loaded.
            If no data are provided, it is checked whether IFU coordinates for
            some/all sources are available in the instance of 'Sources' that
            is currently active.
        """
        
        if not "sources" in self.__dict__:
            logging.error("Cannot load positions unless information about sources available.")
            return
        
        # TODO: try to avoid following check
        if self.sources.data == None:
            logging.error("Cannot load positions unless source spectra are available.")
            return
        
        if data != None:
            self.sources.loadPositionsHDU(data)

        if self.sources.isFreePar.any() or self.sources.hasPolyFit.any():
            ## create entries for IFU positions in QComboBox if not existent
            for entry in ["x Positions", "y Positions"]:
                if self.spectrumShowHduComboBox.findText(entry) == -1:
                    self.spectrumShowHduComboBox.addItem(entry)
            self.sourceList["Positions"] = self.sources.providePosIDs()
            
            ## NOTE: currently all IFU coordinates are loaded into memory, this
            ## can be very demanding for large data sets
            # TODO: make this more memory efficient
            self.sources.loadIFUCoords()
            
            self.pickSourcesOnIfu() ## check whether source selection from IFU frame
                                    ## can be enabled.
            
            self.dataLoaded() ## update status of GUI
        

    def loadPsfParameters(self, data):
        """
        Read the parameters of the PSF from a pyfits BinTableHDU.
        
        Parameters
        ----------
        data : a pyfits.BinTableHDU instance
            The parameters of the PSF, given one per line.
        """
        self.psf = PsfAttributes() ## initialize PSF
        self.psf.readFitsHDU(data) ## read data

        ## update QComboBox for newly loaded data
        if self.spectrumShowHduComboBox.findText("PSF") == -1:
            self.spectrumShowHduComboBox.addItem("PSF")
        self.sourceList["PSF"] = self.psf.provideParNames()

        self.dataLoaded() ## update status of GUI


    def dataLoaded(self):
        """
        Update the status of the GUI when new data is loaded.
        """
        ## get current text of QComboBox and emit signal that this entry is selected
        self.on_spectrumShowHduComboBox_activated(self.spectrumShowHduComboBox.currentText())

        ## enable data selection if required
        if not self.spectrumShowGo.isEnabled():
            self.spectrumShowGo.setEnabled(True)
            
        ## initialize mask if needed
        if self.mask == None or self.mask.size != self.sources.ndisp:
            self.mask = np.ones((self.sources.ndisp,), dtype=np.bool)
        

    def openFile(self, fileName):
        """
        Open prm-file and read information from various HDUs. This method
        is called automatically when a new instance is created and the parameter
        'fitsFile' is provided.
        
        Parameters
        ----------
        fileName : string
            The name of the FITS file to be loaded. The method will scan through
            its various HDUs and check whether they contain useful data. Otherwise
            the HDUs are ignored.
        """
        ## to later write the results to this file, we need its absolute path
        self.outfileName = os.path.abspath(fileName)
        
        self.HDUList  = pyfits.open(fileName, mode="update")
    
        ## search headers for HIERARCH PAMPELMUSE keywords.
        if not isinstance(self.namespace, PampelMuseConfig):
            logging.error("Namespace is not a valid PampelMuseConfig instance.")
            logging.error("Cannot read configuration from header of primary HDU.")
        else:
            self.namespace.readHeader(self.HDUList[0])

        ## for logging only:
        hduString = ""

        ## loop over HDUs in file.
        for hdu in self.HDUList:

            hduString += "{0} ".format(hdu.name)
        
            if hdu.name == "SOURCES": self.loadSources(hdu, instrument=self.instrument)
            elif hdu.name == "SPECTRA": self.loadSpectra(hdu)
            elif hdu.name == "POSPARS": self.loadCoordTrans(hdu)
            elif hdu.name == "POSITIONS": self.loadPositions(hdu)
            elif hdu.name == "INPOSITIONS" and not "Coord. Transf." in self.sourceList.keys():
                self.loadPositions(hdu)
            elif hdu.name == "PSFPARS": self.loadPsfParameters(hdu)
   
        self.statusBar.showMessage("Opened file <{0}>, detected HDUs: {1}"\
                                   .format(self.outfileName, hduString),  5000)
                                   
        ## enable saving to new filename
        self.actionSaveAs.setEnabled(True)
    
    
    def openIfu(self, input):
        """
        Open an IFU datacube and display it, using the widget foreseen
        for this purpose.
        
        Parameters
        ----------
        input : string or a pampelmuse.core.cube.Cube instance
            If a string is provided, it must be the name of the FITS file
            containing the data. Note that in agreement with the the other
            routines of PampelMuse, 2dim. data is okay if the instrument
            used in the observation is known (otherwise data must be 3dim.).
        """
        
        ## load cube using instance of DynamicImaCanvas
        self.ifuPlot = DynamicImaCanvas()
        self.ifuPlot.loadCube(input)
        
        ## include this instance in GUI
        ifuFrame = QHBoxLayout(self.ifuViewWidget)
        ifuFrame.addWidget(self.ifuPlot)
        ifuFrame.setAlignment(Qt.AlignCenter)
    
        ## load first layer by default
        self.ifuPlot.loadLayer(0, cmap=plt.cm.Blues, interpolation="nearest")
            
        ## adapt layer range to match spectral dimension of datacube
        self.ifuLayerSelectSlider.setMaximum(self.ifuPlot.cube.ndisp-1)
        
        ## qt designer does not offer a method to define a validator, so we do it here
        self.ifuMinCutLineEdit.setValidator(QDoubleValidator())
        self.ifuMaxCutLineEdit.setValidator(QDoubleValidator())
        
        ## default cut level is 100%
        self.on_ifuCutsComboBox_currentIndexChanged("100%")
            
        ## By default, when a cube is loaded the interaction modus is set to 'grab', meaning
        ## that when one clicks on the cube display, the spectrum of the nearest spaxel is
        ## displayed.
        self.ifuInteractMode = "grab"
        self.cid_ifuGrab_event = self.ifuPlot.figure.canvas.mpl_connect("button_press_event",  self.on_ifuGrab_event)
        
        ## We need two plots, one to hold the currently selected position and one that contains previously
        ## grabbed coordinates (the second one is needed to initialize the coordinate transformation between
        ## the reference and the IFU positions)
        self.ifuCurMarkPlot, = self.ifuPlot.axes.plot([], [], ls="None", marker="x", mew=1.6, ms=12, mec="r", alpha=0.5)
        self.ifuMarkHistPlot, = self.ifuPlot.axes.plot([], [], ls="None", marker="x", mew=1.6, ms=12, mec="r")
        
        self.cubeLoaded = True ## set flag that a cube is available to True

        ## If possible, i.e. when IFU positions are available, the interaction mode is changed to 'grab', meaning
        ## that a list of sources is overplotted on the cube display and when a source is selected, its spectrum
        ## or x-/y- position is displayed as a function of wavelength.
        self.pickSourcesOnIfu()
            
            
    def openReference(self, fileName, **kwargs):
        """
        Open an image as a reference to compare with the IFU data. Note that the idea
        here is that the reference coordinates that are used by pampelMuse are the pixel
        coordinates on that image.
        
        Parameters
        ----------
        fileName : string or nd_array
            Either the name of a FITS file coontaining the image or the image data as
            a 2dim. array must be provided.
        """
        ## initialize an instance of DynamicImaCanvas that will handle the data.
        self.refPlot = DynamicImaCanvas()

        ## either the name of a FITS file or an ndarray can be provided
        # TODO: add kwargs to control how image is displayed. 
        if isinstance(fileName, str):
            self.refPlot.addImage(pyfits.getdata(fileName), cmap=plt.cm.Blues, interpolation="nearest")
        else:
            self.refPlot.addImage(fileName, cmap=plt.cm.Blues, interpolation="nearest")
    
        ## add instance just created to GUI
        refFrame = QHBoxLayout(self.refViewWidget)
        refFrame.addWidget(self.refPlot)
        refFrame.setAlignment(Qt.AlignCenter)

        ## Qt Designer does not offer a method to define a validator, so we do it here
        self.refMinCutLineEdit.setValidator(QDoubleValidator())
        self.refMaxCutLineEdit.setValidator(QDoubleValidator())
        
        ## default cut level is 100%
        self.on_refCutsComboBox_currentIndexChanged("100%")
        
        ## update flag indicating that a referenceimage has been loaded
        self.refLoaded = True
        
        ## initialize picking of sources if possible (source catalogue must be available)
        self.pickSourcesOnRef(**kwargs)


    def on_ifuGrab_event(self, event):
        """
        Handle event when coordinates are grabbed from the IFU viewer.
       
        Parameters
        ----------
        event : a matplotlib.backend_bases.MouseEvent instance
            The event that triggered the method call.
        """
    
        ## do nothing if event happened outside axes
        if event.inaxes != self.ifuPlot.axes: return
        
        ## check if click was inside field of view of IFU
        if not self.ifuPlot.cube.instrument.infov(event.xdata, event.ydata):
            self.statusBar.showMessage("Out of bounds.", 5000)
            return
        
        ## only do something when left mouse button was clicked
        if event.button != 1: return

        ## call handler to show spectrum in nearest spaxel
        self.on_ifuPlot_grabbed(event.xdata, event.ydata)


    def pickSourcesOnIfu(self):
        """
        Initialize the source picking from the IFU display. The idea is that when
        one clicks on a source, its spectrum or coordinates as a function of
        wavelength are displayed. The prerequisite for this mode of interaction
        is that the IFU data as well as the IFU positions are loaded.
        """

        ## check if prerequisites are fulfilled
        if not "sources" in self.__dict__:  return
        if not self.sources.loadedIFUCoords.any(): return
        if not self.cubeLoaded: return

        # TODO: check if next line works if new data is loaded
        # if "dataPlot" in self.ifuPlot.__dict__: del self.ifuPlot.dataPlot

        ## data points will be overplotted as black empty circles
        ## Note that only resolved sources (status in [2,3]) are displayed, source
        ## close to a brighter source (status==1) are displayed as grey empty circles.
        edgecolors = np.asarray(self.sources.resolved.sum()*["k"],  dtype="|S3")
        edgecolors[self.sources.status[self.sources.resolved]==1] = "0.5"
        
        ## if magnitudes are available in the sources catalogue, we use them to scale the
        ## circles representing the sources.
        if self.sources.mag != None:
            sizes = 10.*(10**(-0.2*(self.sources.mag[self.sources.resolved]-max(self.sources.mag[self.sources.resolved]))))
        else:
            sizes = 20*np.ones(self.sources.resolved.sum(), dtype=np.float32)

        self.ifuPlot.addData(self.sources.xdata[0, self.sources.resolved[self.sources.loadedIFUCoords], 0], 
                             self.sources.ydata[0, self.sources.resolved[self.sources.loadedIFUCoords], 0], 
                             s=sizes, ids=self.sources.ID[self.sources.resolved], edgecolors=edgecolors,  
                             picker=3, facecolors="None")
            
        ##  disable source picking, enable spectrum selection
        self.ifuPlot.figure.canvas.mpl_disconnect(self.cid_ifuGrab_event)
            
        ## show spectrum of a source if one is selected
        self.ifuPlot.figure.canvas.mpl_connect("pick_event", self.on_ifuPick_event)
        
        self.ifuInteractMode = "pick" ##update flag to indicate that interaction mode has changed
    
    
    def on_ifuPick_event(self, event):
        """
        Handle action if one or more sources are picked from the IFU canvas.
        
        Parameters
        ----------
        event : a matplotlib.backend_bases.PickEvent instance
            The mouse event that triggered the method call.
        """
    
        if len(event.ind) != 1:
            self.statusBar.showMessage("Multiple sources selected", 5000)
            return
        
        ## get index of source
        i = event.ind[0]
    
        ## call handler to update GUI
        self.on_ifuPlot_picked(i)
    
    
    def pickSourcesOnRef(self,  xOffset=0., yOffset=0., magCut=None, **kwargs):
        """
        Initialize the source picking from the reference image. The idea here is to
        display the spectrum or IFU coordinates of that source (if possible) and to
        highlight the source also on the IFU plot (if possible).
        
        Parameters
        ----------
        xOffset : float, optional
            An offset in x-direction (in pixels) that should be applied to the
            positions in the catalogue before overplotting them on the reference
            image. NEEDS TO BE IMPLEMENTED!
        yOffset : float, optional
            An offset in y-direction (in pixels) that should be applied to the
            positions in the catalogue before overplotting them on the reference
            image. NEEDS TO BE IMPLEMENTED!
        magCut : float, optional
            Can be used to restrict the overplotting to sources brighter than
            the provided magnitude.
        """
        
        ## select sources that have a spectrum & prepare overplotting of sources
        if not "sources" in self.__dict__:  return
        if not self.refLoaded: return
        
        ## determine size of datapoints representing the sources
        ## (as for the IFU plot, the magnitudes from the source
        ## catalogue are used for this purpose if available)
        if self.sources.mag != None:
            if magCut != None: mZero=magCut
            else: mZero=max(self.sources.mag)
            sizes = 10.*(10**(-0.2*(self.sources.mag-mZero)))
            sizes[sizes<=1] = 0
            
            self.refPlotSelected = (self.sources.mag<=mZero)
        else:
            sizes = 20*np.ones(self.sources.nsrc, dtype=np.float32)

            self.refPlotSelected = np.ones((self.sources.nsrc, ), dtype=np.bool)
        self.refPlotSelected[self.sources.ID<0] = False

        ## add the data to the plot.
        self.refPlot.addData(self.sources.x[self.refPlotSelected], self.sources.y[self.refPlotSelected],
                             s=sizes[self.refPlotSelected], marker="o", edgecolor="r", facecolor="None", 
                             picker=5)

        ## initialize event handler
        self.cid_refPick_event = self.refPlot.figure.canvas.mpl_connect("pick_event", self.on_refPick_event)
    
    
    def on_refPick_event(self, event):
        """
        Handle event when a source is picked from the reference viewer.
        
        Parameters
        ----------
        event : a matplotlib.backend_bases.PickEvent instance
            The mouse event that triggered the method call.
        """
        ## do nothing if more than one source selected
        if len(event.ind) != 1:
            self.statusBar.showMessage("Multiple sources selected.")
            return
    
        ## get index of selected source.
        i = event.ind[0]
        
        ## call method that updates the GUI
        if event.mouseevent.button == 1:
            self.on_refPlot_picked(i, True)
        elif event.mouseevent.button == 3:
            self.on_refPlot_picked(i, False)
        

    
## The following methods deal with actions that affect the overall state of the  GUI.
    
    @pyqtSignature("")
    def on_actionQuit_triggered(self):
        """
        If GUI should be closed, check if unsaved changes exist. If so, open dialog and
        ask whether it should  be closed anyway. Otherwise, call the method to  save the
        changes.
        """
        
        if self.unsavedChanges:
            self.dialog = exitUnsavedDialog()
            self.dialog.exec_()

            if self.dialog.status:
                QMainWindow.close(self)
            else:
                return
        else:
            QMainWindow.close(self)
    
    
    @pyqtSignature("")
    def on_actionSave_triggered(self):
        """
        Save changes to the currently opened prm-file.
        """
        
        ## if no file is currently open, open dialog for selecting
        ## a filename
        if self.outfileName == None:
            self.on_actionSaveAs_triggered()
        
        if os.path.exists(self.outfileName):
            HDUList = pyfits.open(str(self.outfileName), mode="update")
        else:
            HDUList = pyfits.HDUList([pyfits.PrimaryHDU(),])
        
        if not isinstance(self.namespace, PampelMuseConfig):
            logging.error("Namespace is not a valid instance of PampelMuseConfig.")
            logging.error("Cannot write current configuration to header of primary HDU of outfile.")
            
        else:
#            if not hasattr(self.namespace, "polyfit"):
#                self.namespace.polyfit = Namespace()
#            self.namespace.polyfit.utc = datetime.datetime.utcnow().strftime("%y-%m-%d %H:%M:%S")
#            self.namespace.polyfit.version = __version__
            self.namespace.makeHeader(hdu=HDUList[0])

        ## if an existing file is opened, exising HDUs will be eventually replaced
        ## for every HDU that should be written, it is checked whether an HDU with
        ## that name exists. If so, it is replaced, otherwise the HDU is appended
        ## to the list

        ## write 'SOURCES' HDU
        try:
            i = HDUList.index_of("SOURCES")
            HDUList[i] = self.sources.makeSourceHDU()
        except KeyError:
            HDUList.append(self.sources.makeSourceHDU())
            
        ## write 'SPECTRA' HDU
        try:
            i = HDUList.index_of("SPECTRA")
            HDUList[i] = self.sources.makeSpecHDU(makeSigma=False)
        except KeyError:
            HDUList.insert(2, self.sources.makeSpecHDU(makeSigma=False))
            ## if no 'SPECTRA' HDU available, it goes to the second place
            ## in the list to make sure it is read prior to the HDUs
            ## dealing with the source positions
            
        ## write 'POSPARS' HDU if a valid coord. transf. exists
        if self.sources.verifyTrans:
            try:
                i = HDUList.index_of("POSPARS")
                HDUList[i] = self.sources.makeTransHDU()
            except KeyError:
                HDUList.append(self.sources.makeTransHDU())
        
        ## write "POSITIONS" HDU containing polynomial fits to source positions.
        if self.sources.hasPolyFit.any():
            try:
                i = HDUList.index_of("POSITIONS")
                HDUList[i] = self.sources.makeCoordinatesHDU(polyFit=True)
            except KeyError:
                HDUList.append(self.sources.makeCoordinatesHDU(polyFit=True))
        
        ## write HDU containing directly fitted coordinates if they exist
        if self.sources.isFreePar.any():
            if self.sources.hasPolyFit.any():
                try:
                    i = HDUList.index_of("INPOSITIONS")
                    HDUList[i] = self.sources.makeCoordinatesHDU(polyFit=False, name="INPOSITIONS")
                except KeyError:
                    HDUList.append(self.sources.makeCoordinatesHDU(polyFit=False, name="INPOSITIONS"))
            else:
                try:
                    i = HDUList.index_of("POSITIONS")
                    HDUList[i] = self.sources.makeCoordinatesHDU(polyFit=False, name="POSITIONS")
                except KeyError:
                    HDUList.append(self.sources.makeCoordinatesHDU(polyFit=False, name="POSITIONS"))
        
        ## write 'PSFPARS' HDU
        if "psf" in self.__dict__ and self.psf.verifyPSF:
            try:
                i = HDUList.index_of("PSFPARS")
                HDUList[i] = self.psf.makePSFHDU()
            except KeyError:
                HDUList.append(self.psf.makePSFHDU())

        ## write the file
        if os.path.exists(self.outfileName):
            HDUList.flush()
            HDUList.close()
        else:
            HDUList.writeto(str(self.outfileName))
            
        ## update status of GUI.
        self.statusBar.showMessage("Saving OK!", 5000)
        self.unsavedChanges = False
        self.actionSave.setEnabled(False)
    
    
    @pyqtSignature("")
    def on_actionSaveAs_triggered(self):
        """
        Save changes to a new file.
        """
        if self.outfileName == None:
            directory="./"
        else:
            directory = self.outfileName
            
        ## Open dialog to select filename
        self.outfileName = QFileDialog.getOpenFileName(self, caption="Please select a filname", directory=directory, filter="*.fits")
        if self.outfileName == "":
            self.statusBar.showMessage("Can't do, no filename selected!")
            return
            
        # TODO: dialog on whether file should be overwritten
            
        ## call method to save data to a file with known filename
        self.on_actionSave_triggered()
    
    
## The following methods deal with actions that affect the state of the spectrum viewer.
    
    def modifyMask(self, limits, mask):
        """
        A region defined by 'limits' is added to or removed from the mask.
        The region is added if 'mask' is True, otherwise it is removed.
        
        Parameters:
        -----------
        limits : (float, float)
            The min./max. values of the range to be masked.
        mask : bool
            Whether the selected region should be included in the mask
            (if True) or excluded from it (if False).
        """
        if self.mask == None:
            return
        
        iMin = np.argmin(abs(self.sources.wave-limits[0]))
        iMax = np.argmin(abs(self.sources.wave-limits[1]))
        
        self.mask[iMin:iMax] = (mask==False)

        self.plotMask()


    def plotMask(self):
        """
        Overplot the mask onto the curve that is currently active in
        the Spectrum Viewer.
        """
        searchMask = np.diff(np.r_[1, self.mask, 1])
        maskStart  = self.sources.wave[np.where(searchMask==-1)[0]]
        maskEnd    = self.sources.wave[np.where(searchMask==1)[0]]

        ## add/update one rectangle for each range that is masked out
        ## the 'addRectangle' method of 'dataPlot' will know from
        ## the provided 'gid' whether an existing rectangle can be
        ## updated or a new one must be created.
        i = 0
        while i<maskStart.size:
            start = maskStart[i]
            end   = maskEnd[i]

            self.dataPlot.addRectangle((start,0), width=end-start, edgecolor="w", facecolor="k", alpha=0.5, gid="mask{0}".format(i))
            i += 1
            
        ## remove rectangles that are not needed any more.
        while "mask{0}".format(i) in self.dataPlot.bar_gid:
            self.dataPlot.delRectangle("mask{0}".format(i))
            i += 1

        self.dataPlot.draw()
    
    
    @pyqtSignature("bool")
    def on_actionFit_all_positions_toggled(self, p0):
        """
        Whether or not all positions should be fitted simultaneously.
        """
        self.fitAll = p0

    
    @pyqtSignature("bool")
    def on_actionForce_common_slope_toggled(self, p0):
        """
        Whether or not a common slope should be used to fit all positions.
        """
        self.commonSlope = p0
    
    
    @pyqtSignature("QString")
    def on_spectrumShowHduComboBox_activated(self, name):
        """
        update the selection in the spectrumShowIdentComboBox
        when a new HDU is selected.
        """
        self.curProp = str(name)

        self.spectrumShowIdentComboBox.clear()

        if name in ["x Positions", "y Positions"]:
            self.spectrumShowIdentComboBox.addItems(self.sourceList["Positions"])
        else:
            self.spectrumShowIdentComboBox.addItems(self.sourceList[str(name)])
    
    
    @pyqtSignature("")
    def on_spectrumShowGo_clicked(self, updateYRange=True):
        """
        Handle event that a new curve should be plotted.
        
        Parameters
        ----------
        updateYRange : bool, optional
            Flag indicating whether the y-limits of the plot canvas should
            be updated to fit the new data.
        """
        self.curRowName = str(self.spectrumShowIdentComboBox.currentText())
        
        ## retrieve current data for selected HDU and property
        if self.curProp == "Spectra":
            self.yFit  = None
            self.yData = self.sources.viewSpectrum(name=self.curRowName)
            yLabel = r"$f_\lambda\,[\mathrm{counts}/\mathrm{layer}]$"

        elif self.curProp == "x Positions":
            self.yData, self.yFit \
                = self.sources.getIFUCoords(coord="x", ID=int(self.curRowName))
            yLabel = r"$x\,[\mathrm{spaxel}]$"

        elif self.curProp == "y Positions":
            self.yData, self.yFit \
                = self.sources.getIFUCoords(coord="y", ID=int(self.curRowName))
            yLabel = r"$y\,[\mathrm{spaxel}]$"

        elif self.curProp == "PSF":
            i = self.psf.parameters.index(str(self.curRowName))
            self.yData = self.psf.data[0,i]
            if self.psf.poly[i]: self.yFit = self.psf.data[1,i]
            else: self.yFit = None
            yLabel=None

        elif self.curProp == "Coord. Transf.":
            i = self.sources.parameters.index(str(self.curRowName))
            self.yData = self.sources.ctData[0,i]
            if self.sources.ctPoly[i]: self.yFit = self.sources.ctData[1,i]
            else: self.yFit = None
            yLabel=None

        ## update plot
        self.dataPlot.updateData(name='data',xData=self.sources.wave,yData=self.yData,
                                 xLabel=r"$\lambda\,[\mathrm{\AA}]$", yLabel=yLabel, 
                                 updateYRange=updateYRange)

        if self.yFit != None:
            self.dataPlot.updateData(\
                name='fit', xData=self.sources.wave, yData=np.copy(self.yFit), 
                updateYRange=updateYRange)
        else:
            self.dataPlot.updateData(name='fit', xData=[], yData=[], 
                                     updateYRange=updateYRange)

        ## show mask
        self.plotMask()

        ## write out some info
        self.statusBar.showMessage("Fetching data successful.", 5000)
    
        ## if IFU and/or reference dataloaded, highlight currently selected
        ## source in the respective plot.
        try:
            ID = int(self.curRowName)
        except ValueError: ## if curRowName is not an ID.
            return

        if self.ifuInteractMode == "pick" and ID in self.ifuPlot.ids:
            self.on_ifuPlot_picked(np.where(self.ifuPlot.ids==ID)[0][0])
        if self.refLoaded and ID in self.sources.ID[self.refPlotSelected]:
            self.on_refPlot_picked(np.where(self.sources.ID[self.refPlotSelected]==ID)[0][0], True)
    
    
    @pyqtSignature("")
    def on_spectrumFitGo_clicked(self):
        """
        Handle event when the currently displayed spectral data should be fitted.
        """
        ## handle case when no data has been selected/plotted:
        if not "yData" in dir(self):
            logging.error("Please select data to fit first.")
            self.statusBar.showMessage("Can't do! Please select data to fit first!", 5000)
            return

        ## get fit function and degree of polynomial
        function = str(self.spectrumFitTypeComboBox.currentText())
        order = int(self.spectrumFitOrderSpinBox.value())
        
        ## earlier versions included a step polynomial consisting of several bins.
##         bins  = int(self.spectrumFitOrderSpinBox.value())
        bins = 1

        if self.curProp == "Spectra":
            self.statusBar.showMessage("Fitting spectra is not supported.",  5000)
##             logging.warning("(and not that useful anyway, ol' goofball!)")
            return

        elif self.curProp == "PSF":
            self.psf.polyFitParameter(name=self.curRowName, order=order,
                                      bins=bins, mask=self.mask,
                                      function=function.lower())

        elif self.curProp in ["x Positions", "y Positions"]:

            if self.fitAll: ids=None
            else: ids=int(self.curRowName)

            if self.curProp == "x Positions": coord="x"
            else: coord="y"

            self.sources.fitIFUCoords(\
                ids=ids, coord=coord, order=order, bins=bins, mask=self.mask,
                function=function.lower(), commonSlope=self.commonSlope, statusBar=self.statusBar)

        elif self.curProp == "Coord. Transf.":
            self.statusBar.showMessage("Fitting coord. transf. parameters is deprecated. Directly fit IFU  x- & y-coordiates instead.",  5000)
            return

        self.unsavedChanges = True
        self.actionSave.setEnabled(True)
        self.on_spectrumShowGo_clicked(updateYRange=False)

    
    @pyqtSignature("bool")
    def on_spectrumZoom_clicked(self, checked):
        """
        Enter/leave zoom mode of spectrum viewer if button is pressed.
        
        Parameters
        ----------
        checked : bool
            Gives the new state of the button.
        """
        self.dataPlot.zoom = checked
        
        ## Leave pan mode if enabled
        if self.dataPlot.pan: self.spectrumPan.click()
    
    
    @pyqtSignature("bool")
    def on_spectrumPan_clicked(self, checked):
        """
        Enter/leave pan mode of spectrum viewer if button is pressed.

        Parameters
        ----------
        checked : bool
            Gives the new state of the button.
        """
        self.dataPlot.pan = checked
        
        ## Leave zoom mode if enabled
        if self.dataPlot.zoom: self.spectrumZoom.click()
    
    
## The following methods handle actions that affect the IFU viewer 
    
    @pyqtSignature("int, int")
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

    
    @pyqtSignature("int")
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
        
        ## synchronize spin box
        self.ifuLayerSelectSpinBox.setValue(position)
        
        ## show new layer if IFU data
        if self.cubeLoaded:
            self.ifuPlot.loadLayer(position)
    
            self.dataPlot.vLine.set_data([self.ifuPlot.wave, self.ifuPlot.wave], self.dataPlot.axes.get_ylim())
            self.dataPlot.draw()
    
    
    @pyqtSignature("int")
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

        ## synchronize slider
        self.ifuLayerSelectSlider.setValue(p0)
        
        ## show new layer of IFU datacube
        if self.cubeLoaded:
            self.ifuPlot.loadLayer(p0)
            
            self.dataPlot.vLine.set_data([self.ifuPlot.wave, self.ifuPlot.wave], self.dataPlot.axes.get_ylim())
            self.dataPlot.draw()
    
    
    @pyqtSignature("QString")
    def on_ifuCutsComboBox_currentIndexChanged(self, p0):
        """
        Handle event that a new cut level for the IFU cube is selected
        from the combo box.
        
        p0 : string
            The text that is currently displayed in the combo box.
        """
        
        ## if new level is 'User', enable the boxes that show the min.
        ## and max. cuts for user input
        if p0 == "User":
            self.ifuMinCutLineEdit.setEnabled(True)
            self.ifuMaxCutLineEdit.setEnabled(True)
            return
        ## otherwise disable them and update cuts using new cut level.
        else:
            self.ifuMinCutLineEdit.setEnabled(False)
            self.ifuMaxCutLineEdit.setEnabled(False)
            self.ifuPlot.changeCuts(vrange=float(p0[:-1]))
    
            self.ifuMinCutLineEdit.setText("{0:.5g}".format(float(self.ifuPlot.cuts[0])))
            self.ifuMaxCutLineEdit.setText("{0:.5g}".format(float(self.ifuPlot.cuts[1])))
    
    
    @pyqtSignature("")
    def on_ifuMinCutLineEdit_returnPressed(self):
        """
        Handle event that minimum cut of the IFU plot is changed via the line
        edit. NOTE that this method is only triggered when <ENTER> is hit.
        """
        self.ifuPlot.changeCuts(locut=float(self.ifuMinCutLineEdit.text()))


    @pyqtSignature("")
    def on_ifuMaxCutLineEdit_returnPressed(self):
        """
        Handle event that maximum cut of the IFU plot is changed via the line
        edit. NOTE that this method is only trigggered when <ENTER> is hit.
        """
        self.ifuPlot.changeCuts(hicut=float(self.ifuMaxCutLineEdit.text()))
        
        
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
        ## show spectrum in spectrum viewer
        self.dataPlot.updateData(name='data',xData=self.ifuPlot.cube.wave,
                                 yData=self.ifuPlot.getNearestSpaxel(x, y),
                                 xLabel=r"$\lambda\,[{0}]$".format(self.ifuPlot.cube.unit),
                                 yLabel=r"$f_\lambda\,[\mathrm{counts}]$")

        ## indicate position where grab event took place in IFU viewer
        self.ifuCurMarkPlot.set_data(x, y)

        ## if the GUI is used to identify sources in the IFU data, add the event to the
        ## selection
        if self.curId != None:
            self.identifiedSources[self.curId] = [x, y, 0., 0.]
            self.updateMarkHistPlot()
        self.ifuPlot.draw()


    def on_ifuPlot_picked(self, index):
        """
        If a source is picked from the datacube image via mouseclick,
        show its spectrum or position.
        
        Parameters
        ----------
        index : int
            The index of the selected source in the IFU viewer plot.
        """
        
        ## recover index of picked source in sources catalog
        i  = np.where(self.sources.ID==self.ifuPlot.ids[index])[0][0]
        
        ## recover ID
        ID = self.sources.ID[i]
        
        ## mark source in datacube image by setting the facecolor of its
        ## marker to the edgecolor of its marker, all other symbols with
        ## transparent face
        facecolors = np.zeros((len(self.ifuPlot.ids), 4), dtype=np.float32)
        facecolors[index] = self.ifuPlot.edgecolors[index]
        self.ifuPlot.dataPlot.set_facecolors(facecolors)
        
        ## check if ID is available in HDU that is currently loaded in the
        ## spectrum viewer. If so, show the corresponding curve.
        indexInComboBox = self.spectrumShowIdentComboBox.findText(str(ID))
        if indexInComboBox >= 0 and self.spectrumShowIdentComboBox.currentText()!=str(ID):
            self.spectrumShowIdentComboBox.setCurrentIndex(indexInComboBox)
            self.spectrumShowGo.click()
    
        self.ifuPlot.figure.canvas.draw()
        
        ## update status bar
        self.statusBar.showMessage("Selected source: {0}".format(ID), 5000)


## The following methods deal with actions that affect the state of the reference view.
    
    @pyqtSignature("QString")
    def on_refCutsComboBox_currentIndexChanged(self, p0):
        """
        Handle event that a new cut level for the reference image is
        selected from the combo box.
        
        p0 : string
            The text that is currently displayed in the combo box.
        """
        if not "refPlot" in self.__dict__: return
        
        ## if new level is 'User', enable the boxes that show the min.
        ## and max. cuts for user input
        if p0 == "User":
            self.refMinCutLineEdit.setEnabled(True)
            self.refMaxCutLineEdit.setEnabled(True)
            return
        ## otherwise disable them and update cuts using new cut level.
        else:
            self.refMinCutLineEdit.setEnabled(False)
            self.refMaxCutLineEdit.setEnabled(False)
            self.refPlot.changeCuts(vrange=float(p0[:-1]))
    
            self.refMinCutLineEdit.setText("{0:.5g}".format(float(self.refPlot.cuts[0])))
            self.refMaxCutLineEdit.setText("{0:.5g}".format(float(self.refPlot.cuts[1])))


    @pyqtSignature("")
    def on_refMinCutLineEdit_returnPressed(self):
        """
        Handle event that minimum cut of the reference plot is changed
        via the line edit. NOTE that this method is only trigggered when
        <ENTER> is hit.
        """
        if not "refPlot" in self.__dict__: return
        
        self.refPlot.changeCuts(locut=float(self.refMinCutLineEdit.text()))
    
    
    @pyqtSignature("")
    def on_refMaxCutLineEdit_returnPressed(self):
        """
        Handle event that maximum cut of the reference plot is changed
        via the line edit. NOTE that this method is only trigggered when
        <ENTER> is hit.
        """
        if not "refPlot" in self.__dict__: return
 
        self.refPlot.changeCuts(hicut=float(self.refMaxCutLineEdit.text()))
    
    
    @pyqtSignature("bool")
    def on_refZoom_clicked(self, checked):
        """
        Handle action when the zoom button for the reference view is enabled
        or disabled.
        
        Parameters
        ----------
        checked : bool
            The new state of the zoom button: True if mode is enabled, False
            otherwise.
        """
        if not "refPlot" in self.__dict__: return
        
        self.refPlot.zoom = checked
        if checked:
            ## Leave pan mode if enabled
            if self.refPlot.pan: self.refPan.click()
            try: ## check if data picking is active, if so, it needs to be stopped while zoom mode is on.
                self.refPlot.figure.canvas.mpl_disconnect(self.cid_refPick_event)
            except:
                pass
        else:
            self.cid_refPick_event = self.refPlot.figure.canvas.mpl_connect("pick_event", self.on_refPick_event)


    @pyqtSignature("bool")
    def on_refPan_clicked(self, checked):
        """
        Handle action when the pan button for the reference view is enabled
        or disabled.
        
        Parameters
        ----------
        checked : bool
            The new state of the pan button: True if mode is enabled, False
            otherwise.
        """
        if not "refPlot" in self.__dict__: return
        
        self.refPlot.pan = checked
        if checked:
            ## Leave zoom mode if enabled
            if self.refPlot.zoom: self.refZoom.click()
            try: ## check if data picking is active, if so, it needs to be stopped while zoom mode is on.
                self.refPlot.figure.canvas.mpl_disconnect(self.cid_refPick_event)
            except:
                pass
        else:
            self.cid_refPick_event = self.refPlot.figure.canvas.mpl_connect("pick_event", self.on_refPick_event)


    @pyqtSignature("")
    def on_refBack_clicked(self):
        """
        Handle action when the 'go back' button for the reference view is clicked.
        """
        if not "refPlot" in self.__dict__: return
        self.refPlot.gotoView(self.refPlot.curViewIndex-1)
        
    
    @pyqtSignature("")
    def on_refForth_clicked(self):
        """
        Handle action when the 'go forth' button for the reference view is clicked.
        """
        if not "refPlot" in self.__dict__: return
        self.refPlot.gotoView(self.refPlot.curViewIndex+1)
    

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
        ## recover index of ID in Sources catalogue
        j = np.where((np.cumsum(self.refPlotSelected)-1)==index)[0][0]

        ## get source ID
        ID = self.sources.ID[j]

        ## source is selected
        if select:
            ## show information on selected source
            statusMessage  = "Picked source #{0}: ".format(int(ID))
            statusMessage += "x={0:.2f}, y={1:.2f}".format(float(self.sources.x[j]), float(self.sources.y[j]))
            if self.sources.mag != None:
                statusMessage += ", mag={0:.2f}".format(float(self.sources.mag[j]))
            self.statusBar.showMessage(statusMessage, 5000)
        
            ## set facecolor of selected marker to its edgecolor, all other sources
            ## are transparent
            facecolors = np.zeros((self.refPlotSelected.sum(), 4), dtype=np.float32)
            facecolors[index] = self.refPlot.edgecolors[index]
            
            if self.ifuInteractMode == "grab":
                self.curId = ID
        
                ## mark previously selected sources
                for ID in self.identifiedSources.keys():
                    i = np.where(self.sources.ID[self.refPlotSelected]==ID)[0][0]
                    facecolors[i] = self.refPlot.edgecolors[i]
                
                ## make facecolor of selected marker 50% transparent
                facecolors[index, -1] /= 2.
            
            elif self.ifuInteractMode == "pick":
                ## check whether selected source is included in the currently loaded dataset in the spectrum viewer.
                ## If so, load the data. same source in IFU viewer; Note that the method called below
                ## will also take care that the source is highlighted in the IFU viewer if possible.
                indexInComboBox = self.spectrumShowIdentComboBox.findText(str(ID))
                if indexInComboBox >= 0 and self.spectrumShowIdentComboBox.currentText()!=str(ID):
                    self.spectrumShowIdentComboBox.setCurrentIndex(indexInComboBox)
                    self.spectrumShowGo.click()

            self.refPlot.dataPlot.set_facecolors(facecolors)
            self.refPlot.draw()

        else:
            if ID in self.identifiedSources.keys():
                del self.identifiedSources[ID]
                self.curId = None

                if self.ifuInteractMode == "grab":
                    self.updateMarkHistPlot()


    def updateMarkHistPlot(self):
        """
        Update the plot containing the identification history, i.e.
        the sources from the reference viewer for which a position was
        identified in the IFU viewer.
        """
        
        ## update data in ifuViewer
        if self.identifiedSources == {}:
            self.ifuMarkHistPlot.set_data([], [])
        else:
            self.ifuMarkHistPlot.set_data(np.asarray(self.identifiedSources.values())[:,0], 
                                          np.asarray(self.identifiedSources.values())[:,1])
        self.ifuPlot.draw()
    
        ## mark sources on reference viewer:
        ## all symbols transparent, except selected, they are colored using the edgecolour
        facecolors = np.zeros((self.refPlotSelected.sum(), 4), dtype=np.float32)
        for ID in self.identifiedSources.keys():
            i = np.where(self.sources.ID[self.refPlotSelected]==ID)[0][0]
            facecolors[i] = self.refPlot.edgecolors[i]
        self.refPlot.dataPlot.set_facecolors(facecolors)
        self.refPlot.draw()
