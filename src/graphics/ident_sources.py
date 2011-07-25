__version__ = 0.4 # last modified: 2011/03/07

import numpy, pylab, sys
from aperture import AperPhoto

## v0.1 - 2010/12/16 - created dedicated file for IdentSources-class.
##                     Removed it from create_prm-file
## v0.2 - 2011/02/08 - forgot to import numpy
## v0.3 - 2011/02/25 - included centre-mode to automatically find cosmic ray
##                     next to a selected pixel
## v0.4 - 2011/03/07 - added option to use bad-pixel mask during aperture
##                     photometry

class IdentSources:
    """
    Compare IFU datacube to high-res image and interactively identify
    sources in the datacube using a list of sources in the high-res
    image. The spectra of the identified sources are extracted using
    aperture photometry. They can be checked visually and be cleaned
    interactively from cosmic ray hits.

    After an instance of the class is initialized, the other method
    defined will be called automatically. As output, the user is
    provided with the IFU pixel positions (<instance>.centers) and
    the spectra (<instance>.spectra) of the identified sources and the
    background spectrum (<instance>.back).
    """
    def __init__(self, ifucube, hrimage, id_all, x_all, y_all, aperrad,
                 ifucuts=100., imacuts=100., wl_start=0., wl_step=1.,
                 extent=None, ifumask=None):
        """
        Initialize a class to identify sources
        
        Parameters
        ----------
        ifucube : nd_array
            The IFU datacube. The spectral axis must be the first one.
        hrimage : nd_array
            The high-resolution image.
        id_all : array_like
            An unique ID for every source detected in the high-res
            image
        x_all : array_like
            The x-coordinates of all sources in id_all. If x_all does
            not contain pixel coordinates, extent should be specified.
        y_all : array_like
            The y-coordinates of all sources in id_all. If y_all does
            not contain pixel coordinates, extent should be specified.
        aperrad : float
            Aperture radius used to extract the spectra
            (in IFU spaxel).
        ifucuts : array_like  or float, optional
            The cuts used to display the datacube. Either as absolute
            ((min,max)) or relative values (in percent).
        imacuts : array_like  or float, optional
            The cuts used to display the high-res image. Either as
            absolute ((min,max)) or relative values (in percent).
        wl_start : float, optional
            Start of wavlength range covered by the datacube (Angstrom)
        wl_step : float, optional
            Dispersion of the datacube (Angstrom/pixel)
        extent : tuple, optional
            If specified coordinates are not pixel coordinates, extent
            might be specified to match the coordinates and the
            high-res image. See matplotlib.pyplot.imshow for further
            information.
        ifumask : ndarray, optional
            A boolean array of same shape as ifucube. if provided, pixels
            set to True in `ifumask` are excluded from aperture photometry
        """

        sys.stdout.write("Preparing GUI for source identification ...")
        sys.stdout.flush()

        self.r_aper = aperrad
        self.len    = ifucube.shape[0]
        
        self.wavl  = wl_step*numpy.arange(ifucube.shape[0])
        self.wavl += wl_start


        ## Calculate median image from IFU datacube
        self.ifucube = ifucube
        self.ifumask = ifumask
        ifudata = numpy.median(ifucube, axis=0)
 

        ## Determine cuts to display images:
        if type(imacuts).__name__ == 'tuple':
            hstlo, hstup = imacuts
        elif type(imacuts).__name__ in ['int', 'float']:
            values = hrimage.copy().reshape(-1)
            values.sort()

            cutlev = imacuts/100.
            npix  = values.shape[0]

            hstlo = values[-int(cutlev*npix)]
            hstup = values[int(cutlev*npix)-1]

        if type(ifucuts).__name__ == 'tuple':
            ifulo, ifuup = ifucuts
        elif type(ifucuts).__name__ in ['int', 'float']:
            values = ifudata.copy().reshape(-1)
            values.sort()

            cutlev = ifucuts/100.
            npix  = values.shape[0]

            ifulo = values[-int(cutlev*npix)]
            ifuup = values[int(cutlev*npix)-1]


        ## initialize figure environment & display images
        self.fig = pylab.figure(figsize=(10,6))

        self.ax1 = self.fig.add_axes([0.1, 0.1, 0.35, 0.8])
        self.ax2 = self.fig.add_axes([0.55, 0.1, 0.35, 0.8])

        self.ax1.imshow(hrimage, cmap=pylab.cm.hot, vmin=hstlo, vmax=hstup,
                        interpolation='nearest', extent=extent)
        self.ax2.imshow(ifudata, cmap=pylab.cm.hot, vmin=ifulo, vmax=ifuup,
                        interpolation='nearest')

        self.fig.suptitle('mouse click to (de)select source, <x> to finish selection')


        ## Create lists that keep track of the selected sources
        ## All available HST sources:
        self.id   = numpy.asarray(id_all)
        self.xhst = numpy.asarray(x_all)
        self.yhst = numpy.asarray(y_all)

        ## Selected HST sources:
        self.selected = numpy.zeros((len(self.id),),dtype='bool')

        ## Selected pixels on IFU:
        self.tmp_coords = {}


        ## Overplot source lists on images
        self.allhst, = self.ax1.plot(self.xhst, self.yhst, marker='o', ms=10,
                                     mew=2, mfc='None', mec='g', ls='None', 
                                     picker=5, alpha=0.5)
        self.selhst, = self.ax1.plot([], [], marker='x', ms=16, mew=2, mec='g',
                                     ls='None')

        self.selifu, = self.ax2.plot([], [], marker='+', ms=8, mew=2, mec='g',
                                     ls='None')

        ## Connect mouse click to source selection
        self.cidhst = self.fig.canvas.mpl_connect('pick_event', self._pickhst)
        
        ## Connect <x>-key to finishing source selection 
        self.cidapp = self.fig.canvas.mpl_connect('key_press_event',
                                                  self._aperphoto)


        self.ax1.set_xlim(extent[0], extent[1])
        self.ax1.set_ylim(extent[2], extent[3])

        self.ax2.set_xlim(-0.5, ifudata.shape[1]-0.5)
        self.ax2.set_ylim(-0.5, ifudata.shape[0]-0.5)

        print "DONE"
        
        print "\nInstructions:"
        print "1) Select source on HST image (mouse click)"
        print "2) Determine its position on IFU image (mouse click)"
        print "3) Carry out 1) and 2) for every useful source"
        print "   (Clicking on an accidentally selected source in HST"
        print "    image will deselect it again.)"
        print "5) Press <x> to finish identification\n"

        pylab.show()



    def _pickhst(self, event):
        """
        Select source on high-res image. Method is connected to
        'pick_event' after a CreateSrc-class is initialized or
        after _pickifu has been called.

        Parameters
        ----------
        event : a matplotlib.backend_bases.Event instance
        """


        if len(event.ind) != 1:
            return True

        self._sel = event.ind[0]

        if not self.selected[self._sel]:
            self.selected[self._sel] = True

            self.fig.canvas.mpl_disconnect(self.cidhst)
            self.cidifu = self.fig.canvas.mpl_connect('button_press_event',
                                                      self._pickifu)

        else:
            self.selected[self._sel] = False

            del self.tmp_coords[self._sel]
            self.ifu_coords = numpy.zeros((2, len(self.tmp_coords)),
                                          dtype='float32')
            j = 0
            for i in xrange(len(self.id)):
                if self.selected[i]:
                    self.ifu_coords[:,j] = self.tmp_coords[i]
                    j += 1
            self.selifu.set_data(self.ifu_coords[0], self.ifu_coords[1])
        
        self.selhst.set_data(self.xhst[self.selected], self.yhst[self.selected])
        self.fig.canvas.draw()



    def _pickifu(self, event):
        """
        Identify source selected in high-res image in IFU datacube.
        Method called connected to 'button_press_event' after _pickhst
        has been called.

        Parameters
        ----------
        event : a matplotlib.backend_bases.Event instance
        """

        if event.inaxes != self.selifu.axes:
            return True

        self.tmp_coords[self._sel] = (event.xdata, event.ydata)
        self.ifu_coords = numpy.zeros((2, len(self.tmp_coords)),dtype='float32')
        j = 0
        for i in xrange(len(self.id)):
            if self.selected[i]:
                self.ifu_coords[:,j] = self.tmp_coords[i]
                j += 1


        self.selifu.set_data(self.ifu_coords[0], self.ifu_coords[1])
        self.fig.canvas.draw()

        self.fig.canvas.mpl_disconnect(self.cidifu)
        self.cidhst = self.fig.canvas.mpl_connect('pick_event', self._pickhst)


        
    def _aperphoto(self, event):
        """
        Finish source identification and perform aperture photometry
        on identified sources. Method is connected to 'key_press_event'.

        Parameters
        ----------
        event : a matplotlib.backend_bases.Event instance
        """
        

        ## Press <x> to finish identification and start aperture photometry
        if event.key != 'x':
            print "Press <x> to finish selection"

        else:
            self.fig.canvas.mpl_disconnect(self.cidhst)
            self.fig.canvas.mpl_disconnect(self.cidapp)

            sys.stdout.write("Identification completed, ")
            sys.stdout.write("doing aperture photometry ...")
            sys.stdout.flush()

            self.Nspc = self.selected.sum()

            photometer = AperPhoto(self.ifucube, self.ifu_coords,
                                   mask=self.ifumask)
            photometer.aper_photo(self.r_aper, calc_moments=False)

            print "DONE"

            self.spec   = photometer.fluxes
            self.back   = photometer.backval


            ## Prepare plotting of spectra
            ## The GUI has two modes:
            ##  1) In the default mode, pixels affected by cosmic rays
            ##     can be replaced using linear interpolation
            ##  2) A zoom mode that is enabled by pressing <z>. Use left
            ##     mouse button to zoom into a region and right mouse
            ##     button to return to full spectrum view. Press <z> again
            ##     to quit the zoom mode and return to default mode.
            sys.stdout.write("Preparing GUI for inspection of spectra ...")

            self.i = 0
            self.c = False ## If enabled, the routine tries to automatically
                           ## centre the pixel selection on the cosmic rays

            pylab.clf()

            self.ax1 = self.fig.add_axes([0.15,0.1,0.8,0.85])

            ## plot first spectrum
            self.allplot, = self.ax1.plot(self.wavl, self.spec[:,self.i],
                                          marker='None', ls='-', color='g')

            ## plot to display selected pixels
            self.ex_plot, = self.ax1.plot([],[], marker='o', mew=0, mfc='r',
                                          ls='None')

            ## plot vertical line to clarify wavelength of current position
            self.vline1 = self.ax1.axvline(self.wavl.min(), ls='--', color='r')

            self.zoom = False ## whether zoom mode is on
            self.rsel = 0     ## Radius for pixel selection; All pixels within
                              ## a distance rsel around the central one will be
                              ## selected

            ## To flag the pixels currently selected:
            self.excluded = numpy.zeros(self.wavl.shape, dtype='bool')
        

            self.xmin, self.xmax = self.ax1.get_xlim()
            self.ymin, self.ymax = self.ax1.get_ylim()

            ## Prepare zoom window:
            self.zwindow = self.ax1.bar([self.xmin], [0.], width=0.,
                                        bottom=None, edgecolor='k', color='w',
                                        linewidth=1.0)

            ## Connect to functions:
            self.bpcid = self.fig.canvas.mpl_connect('button_press_event',
                                                     self._press)
            self.mncid = self.fig.canvas.mpl_connect('motion_notify_event',
                                                     self._motion)
            self.icid = self.fig.canvas.mpl_connect('key_press_event', 
                                                    self._interp)

            
            print "DONE"

            self.fig.canvas.draw()

            print "\nInstructions:"
            print "1) Select bad values in spectrum (mouse clicks)"
            print "   (Selection radius can be changed using keys <0> to <9>)"
            print "2) Press <i> to replace selected values"
            print "3) Repeat 1) and 2) until you the spectrum looks ok"
            print "4) Press <x> to go to the next spectrum\n"
            print "5) Press <z> to enter/leave zoom mode"



    def _press(self, event):
        """
        Select bad pixels using the left mouse button, deselect pixels
        using the right mouse button

        Parameters
        ----------
        event : a matplotlib.backend_bases.Event instance
        """

        jsel = abs(self.wavl-event.xdata).argmin()

        if self.c:
            shift = self.spec[jsel-3:jsel+4,self.i].argmax() - 3
            jsel += shift

        if event.button == 1:
            self.excluded[jsel-self.rsel:jsel+self.rsel+1] = True
        elif event.button == 3:
            self.excluded[jsel-self.rsel:jsel+self.rsel+1] = False

        self._plot_excluded()



    def _motion(self, event):
        """
        Move vertical line to track mouse position

        Parameters
        ----------
        event : a matplotlib.backend_bases.Event instance
        """

        if event.inaxes == self.ax1:

            self.vline1.set_xdata([event.xdata,event.xdata])
            self.fig.canvas.draw()



    def _plot_excluded(self):
        """
        Plot currently selected pixels
        """

        self.ex_plot.set_data(self.wavl[self.excluded],
                              self.spec[self.excluded, self.i])
        
        self.fig.canvas.draw()



    def _interp(self, event):
        """
        Handle 'key_press_events':
        <i> - interpolate currently selected pixels
        <x> - go to next spectrum
        <0> to <9> - change selection radius
        <z> - enter/leave zoom modus
        <c> - enter/leave centre modus

        Parameters
        ----------
        event : a matplotlib.backend_bases.Event instance
        """


        if event.key == 'i': ## Interpolation

            new_spec = numpy.interp(self.wavl, self.wavl[self.excluded==False],
                                    self.spec[self.excluded== False,self.i])
            self.spec[:,self.i] = new_spec

            self.allplot.set_data(self.wavl, new_spec)
            self.ex_plot.set_data([], [])

            self.excluded[:] = False

            self.fig.canvas.draw()


        elif event.key == 'x': ## Load next spectrum, quit after last spectrum

            if self.i < self.Nspc-1:
                self.i += 1

                self.ymin = round(0.9*self.spec[:,self.i].min(), -2)
                self.ymax = round(1.1*self.spec[:,self.i].max(), -2)

                self.allplot.set_data(self.wavl, self.spec[:,self.i])
                self.ex_plot.set_data([], [])
                self.ax1.set_ylim(self.ymin, self.ymax)
                self.excluded[:] = False

                self.fig.canvas.draw()

            else:
                self.fig.canvas.mpl_disconnect(self.bpcid)
                self.fig.canvas.mpl_disconnect(self.mncid)
                self.fig.canvas.mpl_disconnect(self.icid)

                self.centers = {}
                self.spectra = {}
                j = 0
                for i in xrange(len(self.id)):
                    if self.selected[i]:
                        self.centers[self.id[i]] = self.ifu_coords[:,j]
                        self.spectra[self.id[i]] = self.spec[:,j]
                        j += 1


                pylab.close()
                ## centers is a dictionary with the coordinates of the
                ## identified sources if IFU pixels
                ## spectra is a dictionary containing the spectra of the
                ## identified sources


        elif event.key in ['1','2','3','4','5','6','7','8','9','0']:
            self.rsel = int(event.key)     ## Change selection radius


        elif event.key == 'z':  ## Enter/leave zoom modus
            if not self.zoom:
                self.zoom = True
                self.fig.canvas.mpl_disconnect(self.bpcid)
                self.fig.canvas.mpl_disconnect(self.mncid)
                
                self.zscid = self.fig.canvas.mpl_connect('button_press_event',
                                                         self._zstart)
            else:
                self.zoom = False
                self.fig.canvas.mpl_disconnect(self.zscid)

                self.bpcid = self.fig.canvas.mpl_connect('button_press_event',
                                                         self._press)
                self.mncid = self.fig.canvas.mpl_connect('motion_notify_event',
                                                         self._motion)

        elif event.key == 'c':
            self.c = (self.c==False)
            if self.c:
                print "CR centering enabled"
            else:
                print "CR centering disabled"

        else:
            print "<x> to quit, <i> to interpolate, <z> for zoom mode "



    def _zstart(self, event):
        """
        To zoom in, press left mouse button and keep it pressed to
        select the region to zoom into. Press right mouse button to
        reset zoom.

        Parameters
        ----------
        event : a matplotlib.backend_bases.Event instance
        """
        
        if event.button == 1:
            self.xz0 = event.xdata
            self.yz0 = event.ydata

            self.zmcid = self.fig.canvas.mpl_connect('motion_notify_event',
                                                     self._zmotion)
            self.zfcid = self.fig.canvas.mpl_connect('button_release_event',
                                                     self._zfinal)

        elif event.button == 3:
            self.zwindow[0].set_alpha(0)

            self.ax1.set_xlim(self.xmin, self.xmax)
            self.ax1.set_ylim(self.ymin, self.ymax)

            self.fig.canvas.draw()

    
    def _zmotion(self, event):
        """
        Draw rectangle around selected region while a zoom is
        applied (Left mouse button is pressed)
        

        Parameters
        ----------
        event : a matplotlib.backend_bases.Event instance
        """

        xzi = event.xdata
        yzi = event.ydata

        self.zwindow[0].set_alpha(1)
        self.zwindow[0].set_x(min(xzi, self.xz0))
        self.zwindow[0].set_y(min(yzi, self.yz0))
        self.zwindow[0].set_width(abs(self.xz0-xzi))
        self.zwindow[0].set_height(abs(self.yz0-yzi))

        self.fig.canvas.draw()

        
    def _zfinal(self, event):
        """
        If left mouse button is released, change displayed region
        to selected zoom window.

        Parameters
        ----------
        event : a matplotlib.backend_bases.Event instance
        """

        if event.button == 1:
            xlims = [self.xz0, event.xdata]
            ylims = [self.yz0, event.ydata]

            xlims.sort()
            ylims.sort()

            self.ax1.set_xlim(xlims[0], xlims[1])
            self.ax1.set_ylim(ylims[0], ylims[1])

            self.fig.canvas.draw()

            self.fig.canvas.mpl_disconnect(self.zmcid)
            self.fig.canvas.mpl_disconnect(self.zfcid)
