__version__ = 0.5 ## 2011/05/24
import numpy, pyfits, pylab, sys, os, time
from scipy import ndimage

## CHANGELOG
## 2011/02/12 - v0.1
## 2011/02/22 - v0.2, included support for new 'ncoadd-option' of CubeFit
## 2011/03/07 - v0.3, fixed problem that code crashes if polynomials of
##                    different order are used
## 2011/03/21 - v0.4, fixed problem that code didn't work in non-interactive
##                    mode
## 2011/05/24 - v0.5, updated how ndimage is imported


class PolyReplace:
    """
    Replace column(s) in a FITS-table by the best-fit polynomial
    ``p(x) = p[0] * x**n + ... + p[n]`` of order n. A backup of the
    replaced columns is created in an additional extension of the
    FITS-file if requested.
    The columns might be median-filtered prior to the fit. In addition,
    certain regions can be exluded from the fit.
    """
    def __init__(self, fileroot, columns, combine=None, extname='ActiveModel',
                 degree=2, exclude=None, interact=0, verbose=0, flen=None,
                 backup=True):
        """
        Initialize an instance of the PolyFit-class.

        Parameters
        ----------
        fileroot :  string
            Fileroot of the FITS-file to be edited. The routine will look
            for a file `fileroot`.prm.fits
        columns : list
            List containing the names of the columns to be fitted.
        combine : array_like or None, optional
            In case the polynomial fits to some columns should only differ
            by a constant value, provide a list of integers with the same
            length as `columns`. For all columns having the same entry in
            `combine`, the coefficients p[i<n] are forced to have
            the same value. The default is to fit every column independently.
        extname : interger or string, optional
            Either ID or name (header keyword `EXTNAME`) of the extension
            containing the FITS-table.
        degree : integer, optional
            The degree of the fitted polynomial
        exclude: array_like or None, optional
            If certain regions (if wavelength space) should be excluded from
            the fits, provide a list of N*2 entries giving the wavelength at
            the blue and the red edge of the excluded region.
        interact : boolean, optional
            If `interact`>0, the user can interactively select the regions
            that should be excluded from the fit. If `interact`>1, the
            outcome of the fit can be checked visually and the `degree` of
            the polynomial might be modified.
        verbose : integer, optional
            Specifies amount of information printed out on the screen.
        flen : integer or None, optional
            Width of the median filter used to smooth the column data prior
            to the fit. If `None`, no filtering is performed.
            NOT YET IMPLEMENTED
        backup : boolean, optional
            If True, the input columns are stored in an extension named
            'BackupModel'. If this extension does exist, it is modified,
            otherwise it is created.
        """
	self.columns    = list(columns)
        self.ia         = interact

        self.deg     = degree
        self.verbose = max(self.ia, verbose) 
        self.backup  = backup 

        ## open file
        self.infile  = pyfits.open(fileroot+".prm.fits", mode="update")
        self.ext_id  = self.infile.index_of(extname)
  	self.names   = self.infile[self.ext_id].data.names
        self.Nspp    = self.infile[self.ext_id].data.size

        self.xdata   = numpy.arange(self.Nspp, dtype="float32")
	if "WAVELEN" in self.names:
            self.wavl = self.infile[self.ext_id].data.field('WAVELEN')
	else:
	    self.wavl = self.xdata

        try:
            self.ncoadd = self.infile[0].header["NCOADD"]
        except KeyError:
            self.ncoadd = 1
        self.useonly = (numpy.mod(self.xdata, self.ncoadd)==(self.ncoadd>>1))

        ## Exclude certain regions if requested
	self.good = numpy.ones((self.Nspp,), dtype='bool')
	if exclude != None:
            self.exc = numpy.asarray(exclude, dtype="float32")

            for i in xrange(self.exc.shape[0]):
                j_min = abs(self.wavl-self.exc[i,0]).argmin()
                j_max = abs(self.wavl-self.exc[i,1]).argmin()

                self.good[j_min:j_max] = False 
        self.used = numpy.logical_and(self.good, self.useonly)


        ## Check if/which columns are fitted together
        if combine == None: 
            self.Nfits = len(self.columns)
            self.fitno = numpy.arange(self.Nfits, dtype="int16") 
        else:
            cmb = numpy.asarray(combine, dtype="int16")
            if len(self.columns) != cmb.size:
                raise IOError, "'columns' and 'combine' must have equal length"

            self.Nfits = 0
            self.fitno = numpy.zeros(cmb.shape, dtype="int16") - 1
            for i in xrange(len(self.columns)):
                if self.fitno[i] == -1:
                    self.fitno[cmb==cmb[i]] = self.Nfits
                    self.Nfits += 1

        ## Create data arrays
        self.fitdata = numpy.zeros((2, self.Nfits, self.Nspp), dtype="float32")
        self.offsets = numpy.zeros((len(self.columns),), dtype="float32")
        self.pcoeffs = self.Nfits*[[]]


        ## fill fitdata array
        for i in xrange(self.Nfits):
            Ninfit = (self.fitno==i).sum()

            ydata = numpy.zeros((Ninfit, self.Nspp), dtype="float32")
            for jj in xrange(Ninfit):
                kk = ((self.fitno==i).cumsum()==(jj+1)).argmax()

                ydata[jj]=self.infile[self.ext_id].data.field(self.columns[kk])
            
                if Ninfit > 1: 
                    med        = numpy.median(ydata[jj][self.used])
                    ydata[jj] -= med
                    self.offsets[kk] = med

            self.fitdata[0,i] = numpy.average(ydata, axis=0)


        if self.verbose > 0:

            if self.ia > 0:

                print ""
                print "   Instructions:"
                print "   -------------"
                print "   1) Use mouse to (de)select regions of interest."
                print "      *Click and hold mouse button while moving over region."
                print "      *Use left mouse button to select region."
                print "      *Use right mouse button to deselect a selected region."
                print "   2) Press <0>...<9> to change degree of polynomial."
                print "   3) Press <n> to continue with next fit."
                print ""

            self.fig = pylab.figure()
            self.ax  = self.fig.add_axes([0.12,0.12,0.83,0.83])
            self.ii  = 0

            if (self.fitno==self.ii).sum() > 1:
                self.label = "Averaged data"
                if self.verbose > 1:
                    print "Combined fit to columns: ",
                    for j in numpy.arange(self.fitno.size)[self.fitno==self.ii]:
                        print self.columns[j],
                    print ""
            else:
                self.label = self.columns[(self.fitno==self.ii).argmax()]
                if self.verbose > 1:
                    print "Fit to column: %s" %self.label

        self.pcoeffs[self.ii] = numpy.polyfit(self.xdata[self.used],
                                              self.fitdata[0,self.ii,self.used],
                                              self.deg)
        self.fitdata[1,self.ii] = numpy.polyval(self.pcoeffs[self.ii],
                                                self.xdata)
        
        if self.verbose > 2:
            print " Current fit:"
            print numpy.poly1d(self.pcoeffs[self.ii])

        if self.verbose > 0:
            self.curplot, = self.ax.plot(self.wavl[self.useonly],
                                         self.fitdata[0, self.ii, self.useonly],
                                         ls='-', lw=1.2, color='b',
                                         label=self.label)
            self.polyfit, = self.ax.plot(self.wavl,
                                         self.fitdata[1, self.ii].copy(),
                                         ls='--', lw=1.2, color='r',
                                         label="Polynomial fit")

            self.legend = self.ax.legend(loc="upper left")

            yp_min, yp_max = self.ax.get_ylim()

            self.vline1 = self.ax.axvline(self.wavl.min(), ls='--', color='r')
            self.vline2 = self.ax.axvline(self.wavl.min(), ls='--', color='r',
                                            alpha=0)

            self.rect = self.ax.bar(20*[self.wavl.min()],20*[yp_max-yp_min],
                                    width=0, bottom=yp_min, alpha=0.5,
                                    fc='r', lw=0)
            self.plot_excluded(dofit=False)

        if self.ia > 0:

            self.bp = self.fig.canvas.mpl_connect('button_press_event',
                                                  self._press)
            self.br = self.fig.canvas.mpl_connect('button_release_event',
                                                  self._release)
            self.mn = self.fig.canvas.mpl_connect('motion_notify_event',
                                                  self._motion)
            self.kp = self.fig.canvas.mpl_connect('key_press_event',
                                                  self._keypress)

            pylab.show()

        else:
            while self.ii < self.Nfits:
                self.ii += 1
                if self.ii < self.Nfits:
                    self._update_plot(newdata=True)
                else:
                    self._write_results()
            



    def _press(self, event):
        if event.button in [1,3]:
            self.vline1.set_xdata([event.xdata,event.xdata])
            self.vline2.set_xdata([event.xdata,event.xdata])
            self.vline2.set_alpha(1)
            self.fig.canvas.draw()

            self.x1 = event.xdata



    def _motion(self, event):
        if event.inaxes == self.ax:
            self.vline1.set_xdata([event.xdata,event.xdata])
            self.fig.canvas.draw()



    def _release(self, event):
        if event.button in [1,3]:
            self.vline1.set_xdata([event.xdata,event.xdata])
            self.vline2.set_alpha(0)
            self.fig.canvas.draw()

            self.x2 = event.xdata
            if self.x2 > self.x1:
                xmin, xmax = self.x1, self.x2
            else:
                xmin, xmax = self.x2, self.x1

            imin = abs(self.wavl-xmin).argmin()
            imax = abs(self.wavl-xmax).argmin()

            if event.button == 1:
                self.good[imin:imax+1] = False
            else:
                self.good[imin:imax+1] = True
            
            self.used = numpy.logical_and(self.good, self.useonly)

            self.plot_excluded()



    def plot_excluded(self, dofit=True):
        rect = []
        edges = []
        trigger = False
        for i in xrange(self.wavl.shape[0]):
            if self.good[i] == trigger:
                if trigger == False:
                    edges.append(self.wavl[i])
                else:
                    edges.append(self.wavl[i-1])
                trigger = (trigger==False)

        if numpy.mod(len(edges),2): edges.append(self.wavl.max())

        for j in xrange(len(self.rect)):
            if j < (len(edges)>>1):
                self.rect[j].set_x(edges[2*j])
                self.rect[j].set_width(edges[2*j+1]-edges[2*j])
            else:
                self.rect[j].set_x(self.wavl.min())
                self.rect[j].set_width(0)
                
        if dofit:
            self._update_plot()
        else:
            self.fig.canvas.draw()



    def _keypress(self, event):

        if event.key == "n":
            self.ii += 1
            if self.ii < self.Nfits:
                self._update_plot()
            else:
                self._write_results()

        elif event.key in ['0','1','2','3','4','5','6','7','8','9']:
            if self.verbose > 0:
                print "Changed degree of polynomial (old=%i ," %self.deg,
                self.deg = int(event.key)
                print "new=%i)" %self.deg

            self._update_plot(newdata=False)



    def _update_plot(self, newdata=True):
        
        self.pcoeffs[self.ii]=numpy.polyfit(self.xdata[self.used],
                                            self.fitdata[0,self.ii,self.used],
                                            self.deg)
        self.fitdata[1,self.ii]=numpy.polyval(self.pcoeffs[self.ii],self.xdata)

        if newdata and self.verbose > 0:
            if (self.fitno==self.ii).sum() > 1:
                self.label = "Averaged data"
                if self.verbose > 1:
                    print "\nCombined fit for columns: ",
                    for j in numpy.arange(self.fitno.size)[self.fitno==self.ii]:
                        print self.columns[j],
            else:
                self.label = self.columns[(self.fitno==self.ii).argmax()]
                if self.verbose > 1:
                    print "\nFit to column: %s" %self.label

            self.curplot.set_ydata(self.fitdata[0,self.ii, self.useonly])
            self.curplot.set_label(self.label)


        if self.verbose > 0:
            self.polyfit.set_ydata(self.fitdata[1,self.ii].copy())

            if self.verbose > 2:
                print " Current fit:"
                print numpy.poly1d(self.pcoeffs[self.ii])

            ## adjust y limits, update rectangles
            ymin = self.fitdata[0,self.ii].min()
            ymax = self.fitdata[0,self.ii].max()
            dy   = (ymax-ymin)
            ymin -= 0.05*dy
            ymax += 0.05*dy
            self.ax.set_ylim(ymin, ymax)
            for j in xrange(len(self.rect)):
                self.rect[j].set_y(ymin)
                self.rect[j].set_height(ymax-ymin)
                    
            self.legend = self.ax.legend(loc="upper left")
            self.fig.canvas.draw()
        


    def _write_results(self):

        if self.ia:
            self.fig.canvas.mpl_disconnect(self.bp)
            self.fig.canvas.mpl_disconnect(self.mn)
            self.fig.canvas.mpl_disconnect(self.br)
            self.fig.canvas.mpl_disconnect(self.kp)

        if self.verbose > 0:
            pylab.close()

        if self.backup: ## check if backup extension already exists
            try:
                backup_id = self.infile.index_of('BackupModel')
            except KeyError:
                self.infile.append(self.infile[self.ext_id].copy())
                backup_id = len(self.infile)-1
                self.infile[backup_id].header.update('EXTNAME','BackupModel')


        history = [time.ctime() \
                       +' - Processed using PolyReplace, version ' \
                       +str(__version__)]

        if self.verbose > 0:
            print "\nWriting changes to PRM-file...",
            if self.verbose > 2:
                print ""

        for i in xrange(len(self.columns)):
            if self.verbose > 2:
                print "Column '%s'" %self.columns[i],  
                print (13-len(self.columns[i]))*".",

            j = self.fitno[i]

            pcoeffs_i = self.pcoeffs[j]
            pcoeffs_i[-1] += self.offsets[i] 

            history.append(' Replaced %s using polynomial of degree %i' \
                               %(self.columns[i], len(pcoeffs_i)-1))
            coeff_string = '  Coefficients '
            for kk in xrange(len(pcoeffs_i)):
                coeff_string += '%4.2e, ' %pcoeffs_i[kk]
            history.append(coeff_string[:-2])

            if self.backup:
                self.infile[backup_id].data.field(self.columns[i])[:] \
                    = self.infile[self.ext_id].data.field(self.columns[i])

            self.infile[self.ext_id].data.field(self.columns[i])[:] \
                = (self.fitdata[1,j]+self.offsets[i])

            if self.verbose > 2:
                print "OK"

        if self.verbose > 2:
            print "Saving file...........",
            
        for entry in history:
            self.infile[0].header.add_history(entry)

        self.infile[self.ext_id].update()
        if self.backup:
            self.infile[backup_id].update()
        self.infile.flush()

        if self.verbose > 0:
            print "OK"
