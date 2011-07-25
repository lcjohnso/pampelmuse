__version__ = 0.7  ## last modified: 2011/03/01

import  math, numpy, os, pyfits, sys, datetime
import _my_profiles, clip_sigma


## v0.1 - 2010/10/14
## v0.2 - 2010/10/26 - included class IdentSources for interactive
##                     source identification and spectrum extraction 
## v0.3 - 2010/11/01 - renamed CreateSrc-class to CreatePrm. Only
##                     one input file is now generated containing the
##                     the important info in different extensions of
##                     the same FITS-file (see help of CreatePrm for
##                     details)
##      - 2010/11/02 - some minor changes
##      - 2010/11/08 - updated documentation
## v0.4 - 2010/11/09 - included verbose-option and changed _writeout-
##                     method
##      - 2010/11/10 - modified select_single-method. All stars brighter
##                     than the cut whose center is located on a spaxel are
##                     selected in any case.
## v0.5 - 2010/12/16 - Created separate file for IdentSources-class & removed
##                     it from this file.
##                   - Restructured the methods: `clean_field`, `select_single`,
##                     `merge_nearby` and `assign_spectra` have been replaced
##                     by `get_zpoint` & `create_sourcecat`. The method to
##                     determine the coordinate transformation (`trans_coords`)
##                     is not called automatically anymore.
##                   - The selection of the resolved sources is now based on two
##                     criteria: their brightness & the expected S/N. To
##                     estimate the S/N, a PSF model must be provided (will
##                     also be stored in the PRM-file).
##                   - Instead of 'merging' nearby sources, both sources are
##                     kept. The 'AllSources'-extension of the PRM-file now
##                     contains additional columns giving the status of each
##                     source (0-unresolved, 1-resolved, but close to brighter
##                     source, 2-resolved) and the ID of a brighter neighbor (if
##                     present)
## v0.5.1 - 2010/12/20 - bugfix (fileroot->rootname)
## v0.5.2 - 2011/02/08 - renamed method PSF to _PSF
## v0.5.3 - 2011/02/10 - bugfix (i -> sid in in get_zpoint)
## v0.6 - 2011/02/24 - updated trans_coords-method: included option to fix
##                     position parameters A and B and to only calculate
##                     offsets
## v0.7 - 2011/03/01 - modified source selection, included effect of nearby brighter
##                     source on S/N

class CreatePrm:
    """
    Create a prm-file as it is required by CubeFit.
    For every spectral pixel, the src-file contains the informations
    about the positions parameters, the spectra of the resolved stars,
    a background spectrum and a combined spectrum of the unresolved
    stellar component (if requested). All the information is stored in
    a FITS-Table that  is stored as extension 'ActiveModel'.
    Another FITS-table is stored s extension 'AllSources'. It contains
    IDs, positions and broadband-fluxes of all sources in the field.
    """
    def __init__(self, rootname, id_all, x_all, y_all, mag_all, wl_start,
                 wl_step, specpix, spatpix=(16,16), psftype="gauss",
                 psfpars=[3.0,0.,0.], headkeys=[], verbose=2):
        """
        Initialize a class to create a prm-file

        Parameters
        ----------
        rootname: string
            The common root of all files connected to the datacube. The
            PRM-file will be stored under `rootname`.prm.fits.
        id_all : array_like
            A 1D array containing a unique ID (an integer) for every
            source in the field.
        x_all : array_like
            An array of same shape as id_all containing the x coordinate
            of every source
        y_all : array_like
            An array of same shape as id_all containing the y coordinate
            of every source
        mag_all : array_like
            An array of same shape as id_all containing a broadband
            magnitude for every source.
        wl_start : float
            Wavelength of first spectral pixel (in Angstrom)
        wl_step : float
            Wavelength offset between subsequent spectral pixels
            (in Angstrom)
        specpix : integer
            Nummber of spectral pixels in the datacube
        spatpix : tuple, optional
            Number of spatial pixels of the IFU
        psftype : string, optional
            Analytical PSF profile. At the moment `gauss` and `moffat` are
            supported
        psfpars = array_like, optional
            The PSF parameters. 3 for a Gaussian profile `psfpars` and 4 for a
            Moffat profile .
        headkeys : array_like, optional
            List of additional keywords that are written to the FITS-
            header. Syntax for a keyword is (name, value, comment)
        verbose : integer, optional
            The amount of information printed out during the analysis. `0`
            correspond to no information at all, `3` to a lot.
        """
        self.vb = verbose

        self.rootname = rootname

        self._writeout("\n", 1, file=False)
        self._writeout(" "*17+"***   CreatePrm, version %s   ***\n" \
                           %str(__version__), 1, file=False)
        self._writeout("\n", 1, file=False)

        self._writeout("Reading input data...", 2, file=False)


        self.spatpix = spatpix
        self.specpix = specpix

        self.id_in   = numpy.asarray(id_all, dtype="int32").tolist()
        self.x_in    = numpy.asarray(x_all, dtype="float32")
        self.y_in    = numpy.asarray(y_all, dtype="float32")
        self.mag_in  = numpy.asarray(mag_all, dtype="float32")
        self.N_in    = len(self.id_in)

        ## select will contain the selection status of each source:
        ## -1=rejected, 0=unresolved, 1=resolved but brighter source nearby,
        ## 2=resolved
        self.select  = numpy.zeros((self.N_in,), dtype="int16") - 1

        ## create wavelength array
        self.wavelen  = numpy.arange(self.specpix, dtype="float32")
        self.wavelen *= wl_step
        self.wavelen += wl_start

        ## read PSF information
        self.psftype   = psftype
        self.psfpars   = numpy.asarray(psfpars, dtype="float32")
        self.psf_maker = self._PSF()

        self._writeout("OK\n", 2, file=False)

        self.today    = datetime.datetime.now()

        self.srccols = []
        self._add_column("WAVELEN", self.wavelen)

        self.priHDU  = pyfits.PrimaryHDU()
        for i in xrange(len(headkeys)):
            self._update_hdr(headkeys[i][0], headkeys[i][1], headkeys[i][2])
        self._update_hdr('DATE', str(self.today.date()),
                         'date this file was written (yyyy-mm-dd)')
        self.priHDU.header.add_history("Generated "+self.today.ctime()+ \
                                           " using CreatePrm, version " \
                                           +str(__version__))

        ## write PSF information to FITS-header
        self._update_hdr('PSFTYPE', self.psftype, "The analytical PSF profile")

        for i in xrange(self.psfpars.size):
            j = 5-self.psfpars.size+i
            self._update_hdr('PSFPAR%i' %j, self.psfpars[i],
                             "Initial guess for PSF prm. %i" %j)



    def _writeout(self, string, vb_min, higher=True, file=True):
        """
        Short routine to write a string to a file and/or print it
        out on the screen

        Parameters
        ----------
        string : string
            The string to be printed/written
        vb_min : integer
            If verbose level is at least `vb_min`, the string is printed
            out on the screen
        higher : bool, optional
            Flag indicating whether the `string` should also be printed if verbose
            level is higher than `vb_min`
        file : bool, optional
            If true, the string is writen to the logfile
        """
        
        if self.vb == vb_min or self.vb > vb_min and higher:
            sys.stdout.write(string)
            sys.stdout.flush()
        if file:
            self.priHDU.header.add_history(string.rstrip("\n"))


    
    def _add_column(self, name, array, format="E"):
        """
        Routine to add a new column to the output FITS-table.

        Parameters
        ----------
        name : string
            The name of the new column
        array : nd_array
            The 1dim. array containing the data
        format : string, optional
            The format of the new column. Check the pyfits-manual for
            details.
        """
        newcol = pyfits.Column(name=name, format=format, array=array)
        self.srccols.append(newcol)



    def _update_hdr(self, key, value, comment):

        self.priHDU.header.update(key, value, comment)



    def _PSF(self):
        """
        Create a PSF array of type `psftype` and size `spatpix` centered
        on pixel (x, y). `p` contains the shape Parameters of the PSF
        profile.
        """
        if self.psftype == 'moffat':
            self.fwhm = self.psfpars[1]
            return lambda x, y, p: _my_profiles.Moffat(self.spatpix, x, y, 0., p[0], p[1], p[2], p[3]).array
        elif self.psftype == 'gauss':
            self.fwhm = self.psfpars[0]
            return lambda x, y, p: _my_profiles.Gaussian(self.spatpix, x, y, 0., p[0], p[1], p[2]).array



    def trans_coords(self, pos_sel=None, pos_pars=[1.,0.,0.,0.],
                     only_offsets=False):
        """
        Determine transformation from input coordinates to true IFU pixel
        coordinates. The transformation is done using the formulae:
            ( i)  x_ifu = A*x_in + B*y_in + C
            (ii)  y_ifu = A*y_in - B*x_in + D

        Parameters
        ----------
        pos_sel : dictionary, optional
            A dictionary giving the pixel coordinates of at least 4
            sources in the FoV. Syntax is {ID:(xPIX, yPIX)}. If None,
            it is assumed that x_all and y_all are pixel coordinates
            and no transformation is done.
        pos_pars : array_like, optional
            In case the  parameters A, B, C and D of the coordinate
            transformation are known, they can be provided using
            `pos_pars=[A, B, C, D]`.
            Note that `pospars` is only used if `pos_sel` is not provided
            or `only_offsets` is True. In this case, only A and B are fixed.
        only_offsets : boolean, optional
            In case the rotaion and the scale of the transformation are
            known and only offsets (i.e parameters c and D) should be
            calculated using `pos_sel`.
        """

        if pos_sel != None:
            Nsel = len(pos_sel)
            
            incoords  = numpy.zeros((2,Nsel), dtype='float32')
            outcoords = numpy.zeros((2,Nsel), dtype='float32')

            ii = 0
            for id_ii, coords_ii in pos_sel.iteritems():
                jj = self.id_in.index(id_ii)
                incoords[0,ii]  = self.x_in[jj]
                incoords[1,ii]  = self.y_in[jj]
                outcoords[0,ii] = coords_ii[0]
                outcoords[1,ii] = coords_ii[1]

                ii += 1

            self._writeout("Converting input to pixel coordinates...", 2,
                           file=False)

            if not only_offsets:
                self._trans_coords(incoords, outcoords)
                self._writeout("OK\n", 2, file=False)

            else:
                self.para,self.parb = pos_pars[:2]
                self.parc = numpy.mean(outcoords[0] \
                                           -self.para*incoords[0] \
                                           -self.parb*incoords[1])
                self.pard = numpy.mean(outcoords[1] \
                                           +self.parb*incoords[0] \
                                           -self.para*incoords[1])

                
            A, B, C, D = self.para, self.parb, self.parc, self.pard 

            self.x_out = A*self.x_in + B*self.y_in + C
            self.y_out = A*self.y_in - B*self.x_in + D

            self._writeout(" Calculated coordinate conversion:\n", 3)
            self._writeout("  A=%.3f, B=%.3f, C=%.3f, D=%.3f\n"%(A,B,C,D), 3)
            self._writeout("  Pixel scaling  = %.3f\n" %(math.sqrt(A*A+B*B)), 3)
            self._writeout("  Rotation angle = %.2fdeg\n" \
                               %math.degrees(math.atan(B/A)), 3)
            self._writeout("  Offsets of individual sources:\n", 3)
            self._writeout("   ID       DX       DY       \n", 3)

            for id_ii, coords_ii in pos_sel.iteritems():
                jj = self.id_in.index(id_ii)
                dx = coords_ii[0] - self.x_out[jj]
                dy = coords_ii[1] - self.y_out[jj]

                self._writeout("   %9i%9.3f%9.3f\n" %(id_ii,dx, dy), 3)

        else:
            self.x_out = self.x_in
            self.y_out = self.y_in

            self.para,self.parb,self.parc,self.pard = pos_pars
            
            self._writeout("No coordinate conversion done\n", 2)
            self._writeout(" A=1., B=0., C=0., D=0.\n", 3)

        self._add_column("PARA", numpy.resize(self.para, self.specpix))
        self._update_hdr('INITPARA', self.para, 
                         'Initial guess of position parameter A')
        
        self._add_column("PARB", numpy.resize(self.parb, self.specpix))
        self._update_hdr('INITPARB', self.parb,
                         'Initial guess of position parameter B')

        self._add_column("PARC", numpy.resize(self.parc, self.specpix))
        self._update_hdr('INITPARC', self.parc, 
                         'Initial guess of position parameter C')

        self._add_column("PARD", numpy.resize(self.pard, self.specpix))
        self._update_hdr('INITPARD', self.pard, 
                         'Initial guess of position parameter D')

    

    def _trans_coords(self, incoords, outcoords):
        """
        Determine transformation from input to pixel coordinates.
        If pos_sel is specified, method is called automatically

        Parameters
        ----------
        incoords : nd_array
            Array containing the input coordiates, with shape=(2,N)
            N>=4 is the number of datapoints
        outcoords : nd_array
            Array containing the true pixel coordiates for the N
            datapoints. Must have same shape as input array
        """
        
        m = incoords.shape[1]
        M = 2*m
        N = 4

        A = numpy.ones((M,N), dtype='float32')
        A[:m,0] = incoords[0]
        A[m:,0] = incoords[1]    
        A[:m,1] = incoords[1]
        A[m:,1] = -incoords[0]
        A[m:,2] = 0.
        A[:m,3] = 0.

        b = numpy.reshape(outcoords, (-1,))

        self.coord_trans = numpy.linalg.lstsq(A, b)

        self.para  = self.coord_trans[0][0]
        self.parb  = self.coord_trans[0][1]
        self.parc  = self.coord_trans[0][2]
        self.pard  = self.coord_trans[0][3]



    def get_zpoint(self, spectra, apercorr=False, aperrad=3.):
        """
        Method to determine magnitude zeropoint using a sample of input
        spectra, obtained e.g. via aperture spectrophotometry.

        Parameters
        ----------
        spectra : dictionary
            A dictionary containing spectra for the selected sources.
            The keys to the spectra must be the IDs of the respective
            sources.
        apercorr : boolean, optional
            Whether an aperture correction should be performed before
            the zeropoint is determined.
        aperrad : float, optional
            The aperture radius used to obtain the spectra (if any). Only
            used if `apercorr` is True.
        """
        self.spec_sel = spectra
        self.Nselect  = len(spectra)

        if self.vb == 2:
            self._writeout("Calculating magnitude zeropoint...", 2, file=False)

        self._writeout(" IDs of provided input spectra:\n", 3, file=False)
        specstars_string = " "
        i = 0
        for id in self.spec_sel.keys():
            if i>0 and numpy.mod(i,5) == 0: specstars_string += "\n "
            specstars_string += "%8i" %id
            i += 1
            self._update_hdr('IDSRC%0*d' %(3,i), id,
                             'ID of identified source no. %i' %i)
        self._writeout(specstars_string.rstrip("\n ")+"\n", 3, file=False)

        normed_spectra = numpy.zeros((self.Nselect, self.specpix),
                                   dtype="float32")
        mag_zero       = numpy.zeros((self.Nselect,), dtype="float32")
        
        self._writeout("Calculating magnitude zeropoint:\n", 3)
        self._writeout(" ID      FLUX     MAG     ZPOINT\n", 3)

        j = 0
        for sid in self.spec_sel.keys():
            i = self.id_in.index(sid)
     
            if apercorr:
                ## select sources that influence aperture photometry
                nearby=numpy.sqrt((self.x_out-self.x_out[i])**2 + \
                                      (self.y_out-self.y_out[i])**2)<(2*aperrad)
                bright=(self.mag_in<(self.mag_in[i]+2.5))

                chosen=numpy.logical_and(nearby, bright) 
                chosen[i] = False

                ## estimate their contribution & correct for it
                nearpix = numpy.fromfunction(self._select_pix, self.spatpix,
                                             sid=i, rad=aperrad)
                psf_i   = self.psf_maker(self.x_out[i], self.y_out[i],
                                         self.psfpars)
                frac_i  = psf_i[nearpix].sum()
                totflux = 10**(-0.4*(self.mag_in[i]-25.))*frac_i
                for k in xrange(self.N_in):
                    if chosen[k]:
                        psf_k    = self.psf_maker(self.x_out[k], self.y_out[k],
                                                  self.psfpars)
                        frac_k   = psf_k[nearpix].sum()
                        totflux += 10**(-0.4*(self.mag_in[k]-25.))*frac_k
                factor = (10**(-0.4*(self.mag_in[i]-25.)))/totflux
                ## note that `factor` does not depend on `frac_i` because the
                ## term cancels out
                self.spec_sel[sid] *= factor

            spec = self.spec_sel[sid]
                
            med_flux = numpy.median(spec)
            mag_zero[j] = self.mag_in[i]+2.5*numpy.log10(med_flux)
            normed_spectra[j] = spec/med_flux
            
            self._writeout(" %8i%9.1f%8.2f%11.2f\n" \
                               %(sid,med_flux,self.mag_in[i],mag_zero[j]), 3)

            j += 1

        self.avgspec = clip_sigma.clip_sigma(normed_spectra, axis=0)[0]
        self.magz_mean = mag_zero.mean()
        self.magz_std  = mag_zero.std()

        self._writeout(" ZEROPOINT = %6.2f +/- %4.2f\n" %(self.magz_mean,
                                                          self.magz_std), 3)
        self._update_hdr('INITZERO', self.magz_mean,
                         'Estimated zeropoint using input spectra')
        if self.vb == 2: self._writeout("OK\n", 2, file=False)
        
        
     
    def _select_pix(self, x, y, sid, rad):

        r = numpy.sqrt((x-self.y_out[sid])**2 + (y-self.x_out[sid])**2)

        return (r<=rad)



    def create_sourcecat(self, reslim, backspec, snrcut=1., unreslim=None,
                         binrad=0.25, noisecube=None):
        """
        Method to select the 'resolved', 'merged' and 'unresolved'
        sources from the input catalogue. All 'resolved' sources are
        assigned a spectrum which is saved in the PRM-file. Sources
        whose input spectra have been provided in the `get_zpoint`-
        method are assigned their respective spectra while the
        spectra for sources not in spec_sel are estimated by averaging
        over the spectra in spec_sel and scaling the averaged spectrum
        to their broadband magnitude.
        If requested, a combined spectrum for the unresolved stellar
        component is also created.

        Parameters
        ----------
        reslim: float
            Broadband magnitude above which a star is considered
            resolved
        backspec : array_like
            A 1D array containing the spectrum of the background
        snrcut : float, optional
            The cut in S/N below which a source is considered unresolved
        unreslim : float or None, optional
            Unresolved component will include sources with magnitudes
            between `reslim` and `unreslim`. If `unreslim` is set to
            None, an unresolved component is not included.
        binrad : float
            Maximum allowed separation (measured in PSF FWHM) of two 
            stars that are merged
        noisecube : nd_array, optional
            The sigma-cube. If none is provided, the code looks for a file
            `rootname`.err.fits
        """
        self._writeout("Creating catalogue of selected sources:\n", 2,
                       file=False)

        self.reslim = reslim
        self.backspec = backspec
        self.snrcut = snrcut
        self.unreslim = unreslim

        if unreslim == None: ## check if unresolved component should be included
            self.hasunres = False
            self.unreslim = self.reslim
        else:
            self.hasunres = True
            self.unreslim = unreslim

        if self.psftype == "moffat":  ## transform from PSF to pixel scale
            self.binrad = binrad*self.psfpars[1]
        else:
            self.binrad = binrad*self.psfpars[0]

        self.merged = {}
        faintflux   = 0.

        ## open sigma-cube:
        if noisecube == None:
            self._writeout(" Opening sigma-cube...", 2, file=False)
            noisecube = pyfits.getdata("%s.err.fits" %self.rootname, 0)
            self._writeout("OK\n", 2, file=False)
        sigma_data = numpy.median(noisecube, axis=0)

            
        ## Looping over sources in input catalogue
        self._writeout(" Bright sources (possibly) above S/N-cut:\n", 4, file=False)
        self._writeout("      ID| MAG| DST| SNR| FIT|\n", 4, file=False)
        self._writeout(" Selecting sources from input catalogue...", 2, file=False)


        Ndig = len(str(self.N_in)) ## only needed for printing
        j = 0                      ##
        selstars_string = "  "     ##

        i_sort = self.mag_in.argsort()
        for i in i_sort:
            self._writeout("\r Selecting sources from input catalogue...", 3, file=False)
            self._writeout("[%0*d/%i]" %(Ndig, j, self.N_in), 3, file=False)

            id_i = self.id_in[i]

            isunresolved=False
            ismerged    =False
            isresolved  =False
 
            if self.mag_in[i] <= self.unreslim: ## check if above unres. limit

                psf_i = self.psf_maker(self.x_out[i],self.y_out[i],self.psfpars)
                flx_i  = 10**(-0.4*(self.mag_in[i]-self.magz_mean))

                if self.mag_in[i] <= self.reslim: ## check if above res. limit
                    ## estimate S/N of source
                    snr_i = flx_i*numpy.sqrt(((psf_i/sigma_data)**2).sum())
                    if snr_i > self.snrcut: ## check if above SNR cut
                        isresolved=True

                        ## calculate distance to nearest brighter source:
                        rsv = self.select>0
                        dst = numpy.sqrt((self.x_out[i]-self.x_out[rsv])**2 + \
                                             (self.y_out[i]-self.y_out[rsv])**2)
                        if not dst.size == 0:
                            if dst.min() < self.binrad: ## merge with brighter star
                                ismerged=True
                                id_in_p = numpy.asarray(self.id_in)[rsv][dst.argmin()]
                                if self.merged.has_key(id_in_p):
                                    id_in_p = self.merged[id_in_p]
                                self.merged[id_i] = id_in_p
                            else:
                                if dst.min() < 0.75*self.fwhm: ## correct S/N
                                    snr_i *= dst.min()/(0.75*self.fwhm)

                                if snr_i < self.snrcut:
                                    isresolved = False
                                else:
                                    self.merged[id_i] = id_i
                        else:
                            self.merged[id_i] = id_i
                            dst = numpy.array([99.9], dtype="float32")

                        self._writeout("\r%8i|" %id_i, 4, file=False)
                        self._writeout("%4.1f|" %self.mag_in[i], 4, file=False)
                        self._writeout("%4.1f|" %dst.min(), 4, file=False)
                        self._writeout("%4i|" %snr_i, 4, file=False)
                        self._writeout("%4i|" %(2*isresolved-ismerged), 4, file=False)
                        self._writeout((2*Ndig+17)*" "+"\n", 4, file=False)

                if self.x_out[i] > -1.5: ## include all stars above unreslim (almost) on CCD
                    if self.x_out[i] < (self.spatpix[1] + 0.5):
                        if self.y_out[i] > -1.5:
                            if self.y_out[i] < (self.spatpix[0] + 0.5):
                                isunresolved=True

                if not isunresolved: ## include bright stars not on CCD but still contributing
                    m_onccd = self.mag_in[i]-2.5*numpy.log10(psf_i.sum())
                    if m_onccd <= (self.unreslim+2.):
                        isunresolved = True

                if isresolved:
                    if not ismerged: ## add spectrum to FITS-table
                        self.select[i] = 2
                        if self.spec_sel.has_key(id_i):
                            self._add_column("FLUX%i"%id_i, self.spec_sel[id_i])
                        else:
                            flux_i = 10**(-0.4*(self.mag_in[i]-self.magz_mean))
                            spec_i = flux_i*self.avgspec
                            self._add_column("FLUX%i"%id_i, array=spec_i)
                            
                        if numpy.mod(len(selstars_string)-42, 43) == 0:
                            selstars_string += "\n  "
                        selstars_string += "%8i" %id_i
                    else:
                        self.select[i] = 1
                                     
                elif isunresolved:
                    self.select[i] = 0
                    faintflux += flx_i*psf_i.sum()
   
            else:
                break
            j += 1
             
        self._writeout("\r Selecting sources from input catalogue...OK", 2, file=False)
        self._writeout((2*Ndig+1)*" ", 3, file=False)
        self._writeout("\n", 2, file=False)

        self.selstars = numpy.arange(self.N_in)[self.select>0]

        self._writeout(" IDs of 'resolved' sources: \n", 3, file=False)
        self._writeout(selstars_string.rstrip("\n ")+"\n", 3, file=False)

        self._writeout(" Combined stars with distances < %.1f pixel\n" \
                           %self.binrad, 2)
        self._writeout("  Primary (kept)  Secondary (merged)\n", 3)
        for sec, pri in self.merged.iteritems():
            if sec != pri:
                self._writeout("%18i%21i\n" %(pri, sec), 3)


        ## preparing background components
        self._writeout(" Unresolved component included? ", 2, file=False)
        if self.hasunres:

            self._add_column("UNRESOLVED", faintflux*self.avgspec)
            self.backspec -= faintflux*self.avgspec

            self._writeout("True\n", 2, file=False)
            self._writeout("  Magnitude range: (%.1f, %.1f]\n" \
                               %(self.reslim, self.unreslim), 3, file=False)

        else:
            self._writeout("False\n", 2, file=False)
    
        self._add_column("SKY", self.backspec)


        self._update_hdr('HASUNRES', int(self.hasunres),
                         'Whether an unresolved component is included')
        self._update_hdr('RESLIM', self.reslim,
                         'Selected confusion limit')
        if self.hasunres:
            self._update_hdr('UNRESLIM', self.unreslim,
                             'Faintest magnitudes in unresolved comp.')

        self._update_hdr('BINRAD', self.binrad,
                         'Minimum distance of two resolved sources')
                


    def writeto(self):
        """
        Method stores all the results in one FITS-file that has several
        extensions:
        1. Primary     - Empty HDU; header includes important information
                         about generation of prm-file
        2. ActiveModel - FITS-table containing the model parameters prepared
                         model parameters as a function of wavelength
        3. AllSources  - FITS-table containing IDs, positions and broadband
                         fluxes of all sources considered (incl. unresolved)
                         and information about their selection status.
        """
        self._writeout("Writing results to file...", 2, file=False)

        tblHDU = pyfits.new_table(self.srccols)
        tblHDU.header.update('EXTNAME', 'ActiveModel')

        if self.hasunres:
            inall = (self.select>=0)
        else:
            inall = (self.select>=1)

        id_true = numpy.asarray(self.id_in, dtype='float32')[inall]
        id_comb = id_true.copy()
        for true, comb in self.merged.iteritems():
            if true != comb:
                id_comb[id_true.tolist().index(true)] = comb

        bbfluxes = 10**(-0.4*(self.mag_in[inall]-self.magz_mean))

        allcols = [pyfits.Column(name='ID',format='J',array=id_true),
                   pyfits.Column(name='XPOS',format='E',array=self.x_in[inall]),
                   pyfits.Column(name='YPOS',format='E',array=self.y_in[inall]),
                   pyfits.Column(name='FLUX',format='E', array=bbfluxes),
                   pyfits.Column(name='FITOPT',format='J',
                                 array=self.select[inall]),
                   pyfits.Column(name='ID_COMB', format='J', array=id_comb)
                   ]
        allHDU = pyfits.new_table(allcols)
        allHDU.header.update('EXTNAME', 'AllSources')

        HDUlist = pyfits.HDUList([self.priHDU, tblHDU, allHDU])

        overwrite=False
        if os.path.exists("%s.prm.fits" %self.rootname):
            os.system("rm %s.prm.fits" %self.rootname)
            overwrite=True

        HDUlist.writeto("%s.prm.fits" %self.rootname, clobber=True)
        self._writeout("OK\n", 2, file=False)

        if overwrite:
            self._writeout(" Existing file '%s.prm.fits' has been replaced\n" \
                               %self.rootname, 3, file=False)
        self._writeout("\n", 2)
