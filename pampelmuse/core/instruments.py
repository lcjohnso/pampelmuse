"""
instruments.py
==============

Provides
--------
class pampelmuse.core.instruments.Instrument
class pampelmuse.core.instruments.Pmas
class pampelmuse.core.instruments.Argus

Purpose
-------
This module implements classes that provide support in handling instrument-
specific characteristics such dead fibers, holes in FoV, or a non-rectangular
FoV. Additionally, the classes enable support for 2dim. data, with one axis
being the spectral one and the remaining axis indicating fiber number (for
fiber-based IFS).
By design, each class is only valid for data coming from a specific instrument.
Currently, there are dedicated class for PMAS/LArr and FLAMES/Argus. Requests
for support of other instruments will be added on request. Requests should be
made using the E-Mail address below. Note however that if no dedicated class for
an instrument exists, this does NOT mean that data coming from that instrument
cannot be analysed. For such purposes, the class 'Instrument' is used. The only
limitation is that the data must be provided as a 3dim. cube then and that things
like dead fibers or holes in the FoV are not handled.

Latest SVN revision
-------------------
186, 2014/07/08
"""
__author__  = "Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)"
__revision__ = 186
import logging, re
import numpy as np
from astropy.io import fits
from matplotlib import path
from scipy.spatial import distance
from scipy.interpolate import griddata



class Instrument(object):
    """
    The Instrument class can be used with any IFU that has quadratic spaxels
    aranged on a rectangular grid with unity filling factor. It also serves as
    super-class for all instrument specific sub-classes defined.
    """
    name = "Generic IFU"

    _rotation_angle = None ## The rotation angle of the IFS on the sky,  given in degrees(!)
    _spaxel_scale   = None ## The spaxel scaleof the IFS on the sky, given in arcseconds^-1

    _transform = None ## will contain the transformation from spaxels to x&y
    _edgepix = None ## will contain the indices of the spaxels that define the edge of the field of view.
    _edgepath = None ## will be a matplotlib.path around the edge of the FoV

    _sky_fibers = []  ## The indices of the sky fibers
    _cal_fibers = []  ## The indices of the calibration fibers

    spectral_axis = 1 ## The index of the spectral axis in numpy notation;
                      ## NOTE that numpy swaps the axes, so spectral_axis=1 corresponds to the first axis when opening the data
                      ## with fv, ds9 or similar programs.
    shape = None ## will contain the spatial shape of the field of view
    
    def __init__(self, transform=None, edgepix=None, badspax=[], rotation_angle=None, spaxel_scale=None, **kwargs):
        """
        Initialize an instance of the Instrument-class
        
        Parameters
        ----------
        transform : nd_array, optional
            Can be an array of shape (2, n_spaxel) that gives the y- and x-
            coordinates of the spaxels. Note that following the numpy counting,
            the first axis is the y-axis. If transform=None, it can be
            specified after initialization.
        edgepix : nd_array, optional
            Can be an array of integers that contains the indices of the
            spaxels that define the edge of the field of view. NOTE that the
            indices should be provided in such an order that the spaxels follow
            a loop around the field of view.
            If edgepix=None, it can be specified after initialization.
        badspax : array_like, optional
            A list containing the indices of dead spaxels.
        rotation_angle : float, optional
            The rotation angle of the IFS on the sky, given in degrees.
        spaxel_scale : float, optional
            The spaxel scale of the IFS on the sky, given per arcsecond.
        """

        self.set_transform(transform)
        
        self.set_edgepix(edgepix)

        ## Handle bad spaxels (e.g., dead fibers)
        self.badspax = badspax
        for spax in badspax:
            logging.info("Excluding spaxel #{0}.".format(spax))

        if rotation_angle != None:
            self.set_rotation_angle(rotation_angle)

        if spaxel_scale != None:
            self.set_spaxel_scale(spaxel_scale)



    def set_transform(self, value):
        """
        Set the transformation from spaxels to coordinates on the (x,y)-plane.
        Serves as the setter-method for the 'transform'-property

        Parameters
        ----------
        value : nd_array or None
            Unless 'value' is None, an array of shape (2, n_spaxel) that gives
            the y- and x-coordinates of all spaxels. Note that following the
            numpy counting, the first axis is the y-axis.
        """
        if value == None:
            return None ## No transformation specified

        ## Perform sanity checks on provided data
        if value.ndim != 2:
            logging.critical("'transform' must be 2dim., is {0}dim.".format(value.ndim))
        assert value.ndim == 2, "'transform' must be 2dim."

        if value.shape[1] != 2:
            logging.critical("'transform' must contain 2 spatial coordinates, has {0}.".format(value.shape[1]))
        assert value.shape[1]==2, "'transform' must contain 2 spatial coordinates."
        
        self._transform = value


    def get_transform(self):
        """
        The getter method of the 'transform'-property

        Returns
        -------
        transform : nd_array
            The current transformation from spaxels to coordinates on the
            (x,y)-plane as an array of shape (N_spaxels, 2)
        """
        return self._transform


    transform = property(get_transform, set_transform, None, "The mapping of the spaxels on the 2dim. plane.")


    def set_edgepix(self, value):
        """
        Set the spaxels that define the edge of the field of view. Serves as
        the setter-method for the 'edgepix'-property.

        Parameters
        ----------
        value : nd_array or None
            If 'value' is not None, a 1dim. array of integers containing the
            inidces of the spaxels along the edge of the field of view.
            IMPORTANT: The order of the indices matters, it should be such
            that the indices follow a continuous path along the edge of the
            field of view.
        """
        if value == None:
            return None  ## edgepix not defined.

        ## Perform sanity checks on provided data
        if self.transform == None:
            logging.error("Cannot set 'edgepix' unless 'transform' is specified.")
        assert self.transform is not None, "No transformation from spaxels to x- & y-coordinates available."

        if value.ndim != 1:
            logging.critical("'edgepix' must be 1dim., but is {0}dim.".format(value.ndim))
        assert value.ndim == 1, "'edgepix' must be 1dim."

        self._edgepix = value


    def get_edgepix(self):
        """
        The getter-method of the 'edgepix'-property. Note that if the spaxels
        that define edge of FoV have not yet been provided but the 'transform'-
        property  is set, the code tries to infer the spaxels on the edge from
        the provided transformation from spaxels to x- & y-coordinates.
        """
        if self._edgepix == None:
            if self.transform==None:
                logging.warning("No transf. from spaxels to x & y available.")
                return None
            
            ## assume rectangular field of view and fibres aranged row after
            ## row (like PMAS/LArr).
            return np.hstack((np.flatnonzero((self.y==self.y.min())&(self.x>self.x.min())), np.flatnonzero((self.x==self.x.max())&(self.y>self.y.min())),
                              np.flatnonzero((self.y==self.y.max())&(self.x<self.x.max()))[::-1], np.flatnonzero((self.x==self.x.min())&(self.y<self.y.max()))[::-1]))
        else:
            return self._edgepix


    def del_edgepix(self):
        """
        Delete the current selection of spaxels on the edge of the FoV.
        """
        self._edgepix = None


    edgepix = property(get_edgepix, set_edgepix, del_edgepix, "The spaxels that define the edge of the FoV.")


    def get_edgepath(self):
        """
        Create a matplotlib.path.Path instance that follows the edges of
        the field of view. This instance can be used for plotting purposes
        and to check whether a point is inside or outside the FoV.
        
        Returns
        -------
        edgepath:  a matplotlib.path.Path instance
            The edges of the field of view.
        """
        ## can only create Path if transformation and edge spaxels have been defined
        if self.edgepix == None or self.transform == None: return None
        
        if self._edgepath == None:
            ## estimate center of field of view
            center = np.median(self.transform,  axis=0)
        
            ## apply offset of +/-0.5 to spaxels along the egdes of the FoV
            edgepos  = np.asarray(self.transform[np.r_[self.edgepix, self.edgepix[0]]], dtype=np.float32)
            offsets  = 0.5*np.ones(edgepos.shape,  dtype=np.float32)
            offsets[edgepos<center] *= -1
            edgepos += offsets
        
            ## initialize path instance:
            self._edgepath = path.Path(edgepos, closed=True)
        return self._edgepath
    
    
    edgepath = property(get_edgepath, None, None, "A path along the edge of the field of view.")


    def set_rotation_angle(self, value):
        """
        The setter method of the Instrument.rotation_angle property. Sets the
        rotation angle of the IFS on the sky. Note that the angle must be
        provided in degrees.
        
        Parameters
        ----------
        value : float
            The rotation angle in degrees.
        """
        try:
            self._rotation_angle = float(value)
            logging.info("Rotation angle of IFS on the sky set to {0:.1f} degrees.".format(self._rotation_angle))
        except ValueError:
            logging.error("Illegal value for rotation angle: {0}".format(value))

    def get_rotation_angle(self):
        """
        The getter method of the Instrument.rotation_angle property.
        """
        return self._rotation_angle
        
    rotation_angle = property(get_rotation_angle, set_rotation_angle, None, "The rotation angle of the IFS on the sky in decimal degrees.")
        

    def set_spaxel_scale(self, value):
        """
        The setter method of the Instrument.spaxel_scale property. Sets the
        spaxel_scale of the IFS on the sky.
        
        Parameters
        ----------
        value : float
            The spaxel_scale, given per arcsecond.
        """
        try:
            self._spaxel_scale = float(value)
            logging.info("Sampling of IFS set to {0:.2f} spaxel/arcsec".format(self._spaxel_scale))
        except ValueError:
            logging.error("Illegal value for spaxel scale: {0}".format(value))

    def get_spaxel_scale(self):
        """
        The getter method of the Instrument.spaxel_scale property.
        """
        return self._spaxel_scale
        
    spaxel_scale = property(get_spaxel_scale, set_spaxel_scale, None, "The spaxel scale of the IFS on the sky in arcseconds.")


    @property
    def nspaxel(self):
        """
        Returns
        -------
        nspaxel : int
            The number of spaxels of the current instance of 'Instrument'.
        """
        return self.transform.shape[0]


    @property
    def x(self):
        """
        Returns
        -------
        x : nd_array
            The x-coordinates of the spaxels of the current instance of
            'Instrument'.
        """
        return self.transform[:,1]


    @property
    def y(self):
        """
        Returns
        -------
        y : nd_array
            The y-coordinates of the spaxels of the current instance of
            'Instrument'.
        """
        return self.transform[:,0]


    def set_sky_fibers(self, value):
        """
        Set the indices of the fibers used to observe the night sky
        simultaneously with the science data. This method serves a setter for
        the 'sky_fibers' property.

        Parameters
        ----------
        value : array_like
            A list containing the indices of the sky fibers
        """
        self._sky_fibers = value

    def get_sky_fibers(self):
        """
        The getter-method of the 'sky_fibers'-method.
        
        Returns
        -------
        sky_fibers : array_like
            A list containing the indices of the sky fibers.
        """
        return self._sky_fibers


    sky_fibers = property(get_sky_fibers, set_sky_fibers, None, "The numbers of the fibers used as sky fibers.")


    def set_cal_fibers(self, value):
        """
        Set the indices of the fibers used to calibrate the data simultaneously
        with the observations of the science data. This method serves a setter
        for the 'cal_fibers' property.

        Parameters
        ----------
        value : array_like
            A list containing the indices of the calibration fibers.
        """

        self._cal_fibers = value

    def get_cal_fibers(self):
        """
        The getter-method of the 'cal_fibers'-method.
        
        Returns
        -------
        cal_fibers : array_like
            A list containing the indices of the calibration fibers.
        """        
        return self._cal_fibers


    cal_fibers = property(get_cal_fibers, set_cal_fibers, None, "The numbers of the fibers used for calibration.")


    @property
    def aux_fibers(self):
        """
        Returns
        -------
        aux_fibers : array_like
            A list containing the indices of all auxiliary fibers used either
            as sky or calibration fibers.
        """
        return np.asarray(np.sort(np.hstack((self.sky_fibers, self.cal_fibers))), dtype=np.int16)


    @property
    def naux(self):
        """
        Returns
        -------
        naux : int
            The number of auxiliary (= sky or calibration) fibers per datacube.
        """
        return len(self.aux_fibers)
        

    @property
    def nfiber(self):
        """
        Returns
        -------
        nfiber : int
            The total number of fibers (science+sky+calib.) of the instrument.
        """
        return self.naux + self.nspaxel
        

    @property
    def isscience(self):
        """
        Returns
        -------
        isscience : array_like
            A boolean array that is True if a science fibre is present at this
            index and False if an auxiliary fibre is located there.
        """
        _isscience = np.ones((self.nspaxel+self.naux,), dtype=np.bool)
        _isscience[self.aux_fibers] = False

        return _isscience


    @property
    def spatial_axis(self):
        """
        Returns
        -------
        spatial_axis :  array_like
            The index of the spatial axis, either 1 or 2. NOTE that numpy
            swaps the axes, so spatial_axis=2 corresponds to the first axis
            in numpy notation.
        """
        if self.spectral_axis==1: return 0
        else: return 1


    def dist2spax(self, x, y, metric="euclidean"):
        """
        Calculate the distance of one or more points located at positions
        (x_i,y_i) to all spaxels of the current Instrument-instance.

        Parameters
        ----------
        x : float or array_like
            The x-position(s) of the source(s), in units of spaxels.
        y : float or array_like
            The y-position(s) of the source(s), in units of spaxels.
        metric : string
            The metric used to calculate the distances. For the different
            options and their meanings consult the documentation of
            scipy.distance.cdist. The default is 'euclidean'.

        Returns
        -------
        distances : float or array_like
            The distance(s) calculated using the provided metric and in units
            of spaxels.
        """
        return distance.cdist(self.transform, np.vstack((y, x)).T, metric=metric)


    def infov(self, xy):
        """
        Check if the provided (x,y)-positions are inside the field of view of
        the current Instrument-instance. The method uses the 'contains_point'-
        method of matplotlib.path.Path for this purpose. The this reason, the
        provided indices of 'edgepix' should form a polygon along the edge of
        the field of view.

        Parameters
        ----------
        xy : array_like, shape (2,) or (N,2)
            The x-position(s) of the N source(s) on the (x,y) plane, in units
            of spaxels.


        Returns
        -------
        infov : array_like or bool
            A boolean flag for each source indicating whether or not it is
            inside the field of view.
        """
        
        xy = np.asarray(xy)
        
        ## loop over provided coordinates
        if xy.ndim > 1:
            infov = np.zeros(xy.shape[0], dtype=np.bool)
            for i in xrange(infov.size):
                infov[i] = self.edgepath.contains_point((xy[i,1], xy[i,0]))
        else:
            infov = (self.edgepath.contains_point((xy[1], xy[0]))==1)

        return infov


    def distance_to_edge(self, xy, metric="euclidean"):
        """
        Calculate the  distance of one or more point in (x,y)-space to the
        edge of the field of view of the current Instrument instance.

        Parameters
        ----------
        xy : array_like, shape (2,) or (N,2)
            The position(s) of the N source(s) on the (x,y) plane, in units
            of spaxels.
        metric : string
            The metric used to calculate the distances. For the different
            options and their meanings consult the documentation of
            scipy.distance.cdist. The default is 'euclidean'.

        Returns
        -------
        distance_to_edge : array_like or float
            The distance of each provided source to the edge of the field
            of view. Note that sources outside the field of view will have
            negative distances.
        """
        edist = distance.cdist(self.edgepath.to_polygons()[0], np.asarray(xy)[...,[1,0]], metric=metric).min(axis=0)
        edist[self.infov(xy)==False] *= -1 ## sources outside have negative ## values
        
        return edist


    def get_spectrum(self, hdu, i):
        """
        Extract an individual spectrum from the provided HDU. To keep the memory
        usage low, the spectrum is extracted using Section-class provided by
        pyfits.
        
        Parameters
        ----------
        hdu : instance of pyfits.PrimaryHDU or pyfits.ImageHDU
            The HDU from which the spectrum should be extracted
        i : int
            The index of the fiber/spaxel for which the spectrum should be
            extracted. Note that for 3dim. data, the x- & y-position of the
            spaxel is obtained from the attribute 'transform'.
            
        Returns
        -------
        spectrum : nd_array
            The extracted spectrum
        """
        
        
        if i < -self.nspaxel or i >= self.nspaxel:
            logging.error("Spaxel index must be <{0} but is {1}.".format(self.nspaxel, index))
            return None
            
        if hdu.header["NAXIS"] == 2:
            if self.naux>0: i = np.flatnonzero(np.cumsum(self.isscience)==(i+1))[0] ## ignore non-science fibers
            if self.spectral_axis==1: return hdu.section[i]
            else: return hdu.data[:, i] ## hdu.section[:, i]
        elif hdu.header["NAXIS"] == 3:
            j, k = self.transform[i]
            return hdu.data[:, j, k] ## hdu.section[:, j, k]
            

    def get_layer(self, hdu, i, return_grid=False, return_aux=False):
        """
        Extract the monochromatic layer with index i from the provided HDU.
        To keep the memory usage low, the spectrum is extracted using Section
        -class provided by pyfits.
        
        Parameters
        ----------
        hdu : instance of pyfits.PrimaryHDU or pyfits.ImageHDU
            The HDU from which the monochromatic layer should be extracted
        i : int
            The index of the monochromatic layer that should be extracted.
        return_grid : boolean, optional
            Set this flag to return the layer as a 2dim. grid. By default,
            a 1dim. array is returned.
            
        Returns
        -------
        layer : nd_array
            The extracted monochromatic layer.
        """
        if hdu.header["NAXIS"] == 3:
            if not return_grid: return hdu.section[i].flatten()
            else: return hdu.section[i]
        elif self.spectral_axis == 0:
            layer = hdu.section[i]
        else:
            layer = hdu.section[:,i]
        if self.naux>0 and not return_aux: layer = layer[self.isscience]
        if not return_grid: return layer
        else: return self.make_grid(layer)


    def make_grid(self, data, grid=None, shape=None, method='linear', fill_value=np.nan,  **kwargs):
        """
        Calculate a representation of an observed dataset of the current 
        Instrument-instance on a regular grid in the (x,y)-plane. This method
        is basically a wrapper around the 'griddata'-routine of scipy. Check
        the documentation of that routine for further details.

        Parameters
        ----------
        data : nd_array
            The input data. It must have the appropriate number of spaxels
            (either with or without auxiliary fibers). Its spatial and
            spectral axes should coincide with those of the Instrument-
            instance.
        shape : tuple, optional
            The (y,x)-shape of the regular grid to be created. NOTE that
            following the numpy-convention, the first value gives the y-
            size. If no size is provided, the method will use the default
            size of the Instrument-instance.
        method : string, optional
            The interpolation method. Passed on to the call of the 'griddata'
            routine of scipy.
        fill_value : float, optional
            The value given to undefined spaxels in the calculated array.
            Passed on to the call of the 'griddata' routine of scipy.

        Returns
        -------
        out_data : nd_array
            The data representation on the regular grid in the (x,y)-plane.
        """
        logging.debug("Arranging spatial pixels in (x,y) plane...")

        ## extract science signal
        sci_data = self.get_science(data)

        ## Get right dimension for calculation
        if self.spatial_axis == 1 and data.ndim == 1:
            ## add pseudo-spectral axis to handle 1dim. data (single layer):
            calc_data = sci_data.reshape((1,sci_data.shape[0]))
        elif self.spatial_axis == 0 and data.ndim == 1:
            ## add pseudo-spectral axis to handle 1dim. data (single layer):
            calc_data = sci_data.reshape((sci_data.shape[0],1))
        else:
            calc_data = sci_data

        ## check if grid provided
        if grid != None:
            grid_y, grid_x = grid

            shape = grid_y.shape

        ## otherwise, check if shape provided, if not assume default array
        ## size of instrument
        else:
            if shape == None:
                shape = self.shape

            ## prepare regular grid on which to resample the data
            grid_y, grid_x = np.mgrid[0:shape[0]-1:1j*shape[0], 0:shape[1]-1:1j*shape[1]]


        ## run griddata on every layer provided
        if self.spatial_axis==1:
            out_data = np.zeros((calc_data.shape[0],)+shape, dtype=calc_data.dtype)
            for i in xrange(calc_data.shape[0]):
                out_data[i] = griddata((self.y, self.x), calc_data[i], (grid_y, grid_x), method=method, fill_value=fill_value)
        else:
            out_data = np.zeros((calc_data.shape[1],)+shape, dtype=calc_data.dtype)
            for i in xrange(calc_data.shape[1]):
                out_data[i] = griddata((self.y, self.x), calc_data[:,i], (grid_y, grid_x), method=method, fill_value=fill_value)

        logging.debug("...processed {0} spectral layer(s).".format(i+1))
        return np.squeeze(out_data) ## remove pseudo-spectral axis if needed


    def get_science(self, data):
        """
        Extract the science signal from the provided data. Signal of fibers
        marked as sky or calibration (= auxiliary) fibers is neglected and the
        science data is returned.
        
        Parameters
        ----------
        data : array_like
            The provided data, must be 1D or 2D. If it is 2D, the order of
            spatial and spectral axis must match the one of the Instrument
            instance. The data size along the spatial axis must be either
            equal to the number of science and auxiliary fibers or to the
            number of science fibers. In the later case, the data is
            returned without modifications.

        Returns
        -------
        sci_data : nd_array
            The extracted science data.
        """    

        ## Perform sanity checks
        ## - Dimensionality
        if self.spatial_axis == 1 and data.ndim == 1:
            ## add pseudo-spectral axis to handle 1dim. data (single layer):
            calc_data = data.reshape((1,data.shape[0]))
        elif self.spatial_axis == 0 and data.ndim == 1:
            ## add pseudo-spectral axis to handle 1dim. data (single layer):
            calc_data = data.reshape((data.shape[0],1))
        else:
            calc_data = data

        if calc_data.ndim != 2:
            logging.error("Data must be 1D or 2D, but is {0}D.".format(int(calc_data.ndim)))
        assert calc_data.ndim == 2, "Data must be 1D or 2D"

        ## - Number of spaxels
        if calc_data.shape[self.spatial_axis] == self.nspaxel:
            return np.squeeze(calc_data)

        elif calc_data.shape[self.spatial_axis] == (self.nspaxel + self.naux):
            if self.spatial_axis==2:
                return np.squeeze(calc_data[self.isscience])
            else:
                return np.squeeze(calc_data[:,self.isscience])
        else:
            logging.error("Provided data has invalid dimensions.")
            if self.naux>0: logging.error("Expected {0:d} or {1:d} fibres but got {2:d}.".format(self.nspaxel, self.nspaxel+self.naux, calc_data.shape[self.spatial_axis]))
            else: logging.error("Expected {0:d} fibres but got {1:d}.".format(self.nspaxel, calc_data.shape[self.spatial_axis]))
            return None



class Pmas(Instrument):
    """
    Class to handle data observed with the PMAS/LArr instrument
    (Roth et al. 2005, PASP, 117, 620), mounted at the 3.5m telescope at
    Calar Alto observatory.
    """

    name = "pmas"
    shape = (16,16)
    
    rotation_angle = 0 # PMAS cannot bbe rotated on the sky. The rotation angle is fixed to 0 degrees.

    def __init__(self, **kwargs):
        """
        Initialize an instance of the 'PMAS'-class.

        Parameters
        ----------
        kwargs
            Any arguments parsed to this method are passed on to the
            parent-class 'Instrument' of 'PMAS'.
        """
        ## initialize transformation from (x,y)-coords to spaxels
        transform = np.vstack((np.arange(256, dtype=np.int16)//16, np.mod(np.arange(256, dtype=np.int16), 16))).T
        
        ## all pixels at the edge of the fov 
        edgepix = np.asarray([0,16,32,48,64,80,96,112,128,144,160,176,192,208,224,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255,239,223,207,191,175,159,143,127,111,95,79,63,47,31,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1], dtype=np.int16)

        logging.info("Instrument in use in PMAS.")

        super(Pmas, self).__init__(transform, edgepix, **kwargs)



class Argus(Instrument):
    """
    Class to handle data coming from the IFU 'ARGUS' of the FLAMES spectrograph
    (Pasquini et al. 2002, The messenger, 110, 1), mounted on UT2 of the ESO
    Very Large Telescope (VLT).
    """
    name = "argus"
    shape = (14,22)  ## numpy notation - first y, then x
    spectral_axis = 0

    ## The indices of the dedicated sky fibers
    _sky_fibers = np.array([20,41,63,84,105,127,148,169,191,212,233,255,276,297,319], dtype=np.int16)

    ## The indices of the dedicated calibration fibers
    _cal_fibers = np.array([42,106,170,234,298], dtype=np.int16)


    def __init__(self, header=None, **kwargs):
        """
        Initialize an instance of the 'ARGUS' class. NOTE that by default, it
        is assumed that the traces of all fibres (including all 14 sky fibres
        and all 5 calibration fibres) are present in the data. This is the
        format provided by, e.g., p3d (Sandin et al. 2010, A&A, 515, A35). If
        the data have another format, this can be handled by either providing
        the coordinate transformation directly upon initialisation of an
        instance or calling the dedicated 'read_fiber_setup'-method that takes
        a FITS-HDU containing the fiber setup (as provided by the official ESO
        pipeline) as input.
        """

        ## The traces of the last three spaxels are entirely outside the
        ## field of view.
        if not "badspax" in kwargs.keys():
            kwargs["badspax"] = [297,298,299]

        ## initialize transformation from (x,y)-coords to spaxels
        if not "transform" in kwargs.keys():
            # TODO: check if transform is really okay: seems to be mirrored along one axis!
            kwargs["transform"] = np.vstack((np.array([5,4,3,2,1,1,2,3,4,5,6,7,8,9,10,11,12,12,12,13,1,0,0,1,2,3,4,5,6,7,8,9,10,11,11,10,9,8,7,6,6,7,8,9,10,11,12,13,13,12,11,10,9,8,7,6,5,4,3,2,13,12,11,10,9,8,7,6,5,4,3,2,1,0,0,1,2,3,4,5,5,4,3,2,1,0,0,1,2,3,4,5,6,7,8,9,10,11,12,13,2,3,4,5,6,7,8,9,10,11,12,13,13,12,11,10,9,8,7,6,10,11,12,13,13,12,11,10,9,8,7,6,5,4,3,2,1,0,0,1,9,8,7,6,5,4,3,2,1,0,0,1,2,3,4,5,6,7,8,9,1,0,0,1,2,3,4,5,6,7,8,9,10,11,12,13,13,12,11,10,6,7,8,9,10,11,12,13,13,12,11,10,9,8,7,6,5,4,3,2,13,12,11,10,9,8,7,6,5,4,3,2,1,0,0,1,2,3,4,5,5,4,3,2,1,0,0,1,2,3,4,5,6,7,8,9,10,11,12,13,2,3,4,5,6,7,8,9,10,11,12,13,13,12,11,10,9,8,7,6,6,7,8,9,10,11,11,10,9,8,7,6,5,4,3,2,1,0,0,1,13,12,12,12,11,10,9,8,7,6,5,4,3,2,1,1,2,3,4,5],dtype=np.int16),np.array([1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,2,2,3,3,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,5,5,5,5,5,5,5,5,5,5,5,5,5,5,4,4,4,4,4,4,7,7,7,7,7,7,6,6,6,6,6,6,6,6,6,6,6,6,6,6,8,8,8,8,8,8,8,8,8,8,8,8,7,7,7,7,7,7,7,7,10,10,10,10,9,9,9,9,9,9,9,9,9,9,9,9,9,9,8,8,11,11,11,11,11,11,11,11,11,11,10,10,10,10,10,10,10,10,10,10,13,13,12,12,12,12,12,12,12,12,12,12,12,12,12,12,11,11,11,11,14,14,14,14,14,14,14,14,13,13,13,13,13,13,13,13,13,13,13,13,15,15,15,15,15,15,15,15,15,15,15,15,15,15,14,14,14,14,14,14,17,17,17,17,17,17,16,16,16,16,16,16,16,16,16,16,16,16,16,16,18,18,18,18,18,18,18,18,18,18,18,18,17,17,17,17,17,17,17,17,20,20,20,20,20,20,19,19,19,19,19,19,19,19,19,19,19,19,18,18,19,19,20,21,21,21,21,21,21,21,21,21,21,21,21,20,20,20,20,20], dtype=np.int16))).T

        ## all pixels at the edge of the fov
        if not "edgepix" in kwargs.keys():
            kwargs["edgepix"] = np.array([4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,48,47,60,99,112,111,124,123,176,175,188,187,200,239,252,251,280,281,282,283,284,285,286,287,288,289,290,291,292,293,294,295,276,277,278,225,226,213,214,161,162,149,150,137,138,85,86,73,74,21,22,23], dtype=np.int16)

        logging.info("Instrument in use in FLAMES/Argus.")

        if isinstance(header, (fits.header.Header, pyfits.Header)):
            self.set_rotation_angle(header)
            self.set_spaxel_scale(header)

        super(Argus, self).__init__(**kwargs)


        
    def read_fiber_setup(self, hdu):
        """
        A FITS-HDU can be provided that contains the fiber configuration for
        the observation. This HDU is routinely provided by the ESO-pipeline 
        (I guess). The method uses the provided information to update the
        transformation from spaxels to x- & y- coordinates and to find the
        indices of the sky and calibration fibres.

        Parameters
        ----------
        hdu : a FITS binary table HDU instance
            The HDU containing the fibre setup.
        """

        self._sky_fibers = []
        self._cal_fibers = []
        _x          = []
        _y          = []

        ## use 'RECTRACTOR' information from table to identify fibers
        sky_match = re.compile(r"P2\-AR[NC][0-9]+\-([0-9]+|Sky)")
        
        for i in xrange(hdu.header["NAXIS2"]): ## loop over all rows in HDU
        
            match = sky_match.match(hdu.data.field("RETRACTOR")[i])
        
            if match == None: ## calibration fiber
                self._cal_fibers.append(i)
            elif match.group(1) == "Sky":  ## sky fiber
                self._sky_fibers.append(i)
            else: ## science fiber
                _x.append(hdu.data.field("X")[i]-1)
                _y.append(hdu.data.field("Y")[i]-1)

        ## update information
        super(Argus, self).set_transform(np.vstack((_y, _x)).T)
        super(Argus, self).set_sky_fibers(self._sky_fibers)
        super(Argus, self).set_cal_fibers(self._cal_fibers)

        ## some logging on setup that was found.
        logging.info("ARGUS fibre calibration (no. of fibres):")
        logging.info("                            science: {0:3d}".format(self.nspaxel))
        logging.info("                                sky: {0:3d}".format(len(self.sky_fibers)))
        logging.info("                        calibration: {0:3d}".format(len(self.cal_fibers)))

    
    def set_rotation_angle(self, value):
        """
        The setter method of the Argus.rotation_angle property. Note that in
        contrast to the method Instrument.set_rotation_angle, besides a float
        an instance of a FITS header can also be provided, in which case the
        routine will search for the HIERARCH-key 'ESO ADA POSANG'.
        
        Parameters
        ----------
        value : float or instance of astropy.io.fits.header.Header or 
                pyfits.Header, optional
            Either the rotation angle of ARGUS on the sky in degrees or
            a header instance that is searched for information about the
            rotation angle.
        """
        if isinstance(value, (fits.header.Header, pyfits.Header)):
            try:
                value = value["ESO ADA POSANG"]
            except KeyError:
                logging.error("Cannot determine rotation angle, HIERARCH-key 'ESO ADA POSANG' is missing.")
                return
        super(Argus, self).set_rotation_angle(value)

    rotation_angle = property(Instrument.get_rotation_angle, set_rotation_angle, None, "The rotatotion angle of ARGUS on the sky in degrees.")


    def set_spaxel_scale(self, value):
        """
        The setter method of the Argus.spaxel_scale property. Note that in
        contrast to the method Instrument.set_spaxel_scale, besides a float
        an instance of a FITS header can also be provided, in which case the
        routine will search for the HIERARCH-key 'ESO INS1 OPTI1 POS'. The
        value of this key is 'POS_1' for a spaxel scale of 0.52/arcsecond and
        'POS_1_67' for a spaxel_scale of 0.3/arcsecond.
        
        Parameters
        ----------
        value : float or instance of astropy.io.fits.header.Header or 
                pyfits.Header, optional
            Either the spaxel scale of ARGUS on the sky in degrees or
            a header instance that is searched for information about the
            spaxel scale.
        """
        if isinstance(value, (fits.header.Header, pyfits.Header)):
            try:
                scalepos = value["ESO INS1 OPTI1 POS"]
                if scalepos == "POS_1": value = 0.52
                elif scalepos == "POS_1_67": value = 0.3
                else: logging.error("Cannot determine spaxel scale: invalid value '{0}' for HIERARCH-key 'ESO INS1 OPTI1 POS'".format(scalepos))
            except KeyError:
                logging.error("Cannot determine spaxel scale: HIERARCH-key 'ESO INS1 OPTI1 POS' is missing.")
                return
        super(Argus, self).set_spaxel_scale(value)

    spaxel_scale = property(Instrument.get_spaxel_scale, set_spaxel_scale, None, "The spaxel scale of ARGUS on the sky in arcseconds.")



class Muse(Instrument):
    """
    Class to handle data observed with the MUSE instrument
    (Bacon et al. 2010, SPIE, 7735, 7), mounted at UT4 of the
    Very Large Telescope.
    """
    
    name = "muse"
    
    def __init__(self, pixtable=False, shape=None, **kwargs):
        """
        Initialize an instance of the Muse class.
        
        Parameters
        ----------
        pixtable : boolean, optional
            A boolean flag indicating whether the data is available
            in form of a pixtable. THIS IS NOT SUPPORTED YET.
        shape : tuple, optional
            The spatial shape of the data. Note that for MUSE this is
            not a fixed property of the instrument but varies from one
            exposure to the next.
        kwargs
            All other parameters are passed on to the call of the
            super-class.
        """
        logging.info("Instrument in use in MUSE.")

        if pixtable:
            logging.error("The usage of MUSE pixtables is not yet supported.")

        self.shape = shape

        super(Muse, self).__init__(**kwargs)

    
    
    def determine_edge(self, valid):
        """
        The data structure of MUSE is such that the field of view is
        not a fixed property of the instrument, but that each pointing
        comes with a distinct number and arrangement of spaxels. Even
        for a single exposure, the field of view will usually vary
        from one layer to the next. For this reason, the egde of the
        field of view is determined by providing a boolean array that
        contains a flag for each spaxel that indicates whether it con-
        tains valid data or NaN. The routine uses this information to
        determine which spaxels define the edge of the field of view.

        Parameters
        ----------
        valid : nd_array, boolean
            A 1dim. array containing a flag for each spaxel that indi-
            cates whether the spaxel contains valid data.
        """
        if self.transform==None:
            logging.warning("No transf. from spaxels to x & y available.")
            return
        
        if valid.size != self.nspaxel:
            logging.error("Size of boolean array [{0}] must match number of spaxels [{1}].".format(valid.size, self.nspaxel))
            return

        ind1 = []
        ind2 = []
        for yi in xrange(int(self.y.min()), int(self.y.max()), 1):
            
            select = np.flatnonzero(valid&(self.y>=yi)&(self.y<(yi+1)))
            
            if not select.any(): continue
            
            j1 = self.x[select].argmin()
            j2 = self.x[select].argmax()

            ind1.append(select[j1])
            ind2.append(select[j2])

        ind2.reverse()

        self._edgepix = np.asarray(ind1 + ind2, dtype=np.int32)

        
    def set_rotation_angle(self, value):
        """
        The setter method of the Muse.rotation_angle property. Note that in
        contrast to the method Instrument.set_rotation_angle, besides a float
        an instance of a FITS header can also be provided, in which case the
        routine will search for the HIERARCH-key 'ESO INS DROT POSANG'.
        
        Parameters
        ----------
        value : float or instance of astropy.io.fits.header.Header or 
                pyfits.Header, optional
            Either the rotation angle of MUSE on the sky in degrees or
            a header instance that is searched for information about the
            rotation angle.
        """
        if isinstance(value, (fits.header.Header, pyfits.Header)):
            try:
                value = value["ESO INS DROT POSANG"]
            except KeyError:
                logging.error("Cannot determine rotation angle, HIERARCH-key 'ESO INS DROT POSANG' is missing.")
                return
        super(Muse, self).set_rotation_angle(value)

    rotation_angle = property(Instrument.get_rotation_angle, set_rotation_angle, None, "The rotation angle of MUSE on the sky in degrees.")


    def set_spaxel_scale(self, value):
        """
        The setter method of the Muse.spaxel_scale property. Note that in
        contrast to the method Instrument.set_spaxel_scale, besides a float
        an instance of a FITS header can also be provided, in which case the
        routine will search for the HIERARCH-key 'ESO OCS IPS PIXSCALE'.
        
        Parameters
        ----------
        value : float or instance of astropy.io.fits.header.Header or 
                pyfits.Header, optional
            Either the spaxel scale of MUSE on the sky in degrees or
            a header instance that is searched for information about the
            spaxel scale.
        """
        if isinstance(value, (fits.header.Header, pyfits.Header)):
            try:
                value = value["ESO OCS IPS PIXSCALE"]
            except KeyError:
                logging.error("Cannot determine spaxel scale: HIERARCH-key 'ESO OCS IPS PIXSCALE' is missing.")
                return
        super(Muse, self).set_spaxel_scale(value)

    spaxel_scale = property(Instrument.get_spaxel_scale, set_spaxel_scale, None, "The spaxel scale of MUSE on the sky in arcseconds.")



class SimIfu(Instrument):

    name = "simifu"
    shape = (50,50)

    spectral_axis = 0

    rotation_angle = 0
    spaxel_scale = 0.52

    def __init__(self, **kwargs):
        """
        Initialize an instance of the 'SimIfu'-class.

        Parameters
        ----------
        kwargs
            Any arguments parsed to this method are passed on to the
            parent-class 'Instrument' of 'SimIfu'.
        """
        ## initialize transformation from (x,y)-coords to spaxels
        transform = np.vstack((np.arange(2500, dtype=np.int16)//50, np.mod(np.arange(2500, dtype=np.int16), 50))).T

        logging.info("Instrument in use is SimIfu.")

        super(SimIfu, self).__init__(transform, **kwargs)





__all__ = ["Instrument", "Pmas", "Argus", "Muse", "SimIfu"]

available_instruments = {"pmas":Pmas, "giraffe":Argus, "argus":Argus, "muse":Muse, "simifu":SimIfu}
