import math, numpy, os, sys
from scipy import optimize, ndimage ## , special
from copy import deepcopy

## import resample1d
from _my_profiles import Gaussian, Moffat
from clip_sigma import clip_sigma
## from ima_arith import paste_area

## def shift_image(data, dx, dy):
##     
##     tmp_data = resample1d.shift_array(data, dx, axis=1, output='full')
##     
##     return resample1d.shift_array(tmp_data, dy, axis=0, output='full')


## All parameters that can be varied during the fit are stored in 2dim. arrays
## that contain the current values (in row 1) and the previous values (in row 0)
## for recovery and convergence checks
## Last update: 2011/05/24 - updated importing of ndimage

class StarFit:
    def __init__(self, size, mapping, psftype, psffit, fluxfit, posfit, skyfit,
                 unresfit, lowflux_reject=False, verbose=1, makemod=False,
                 makeres=False, clip=5., method=0, fitrad=-1, flux_cv=0.0001,
                 sigma_cv=0.05):
        
 
        self._reject = lowflux_reject
        self.vb      = verbose ## 0-quiet, 1,2-some info, 3-lot of info
        self.makemod = max(makemod, makeres) ## create model array
        self.makeres = makeres               ## and residuals?
        self.clip    = clip  ## efficiency of bad pixel masking

        ###### Prepare arrays ######
        self.size  = size
        self.Nspat = size[0]*size[1]

        self.Nima  = 0 ## Counter for no. of processed images

        self.data   = numpy.zeros(self.Nspat, dtype='float32')
        self.sigma  = numpy.zeros(self.Nspat, dtype='float32')
        self.inmask = numpy.ones(self.Nspat, dtype='bool')


        ###### Source settings ######
        self.map  = numpy.asarray(mapping, dtype="int32")
        self.Ntot = self.map.size       ## number of sources in total
        self.Nrsv = (self.map>-1).sum() ## number of resolved sources
        self.Nind = self.map.max()+1    ## number of independent res. sources
        self.Nflx = self.Nind + 2       ## number of fitted flux components

        if self.Ntot > self.Nrsv: ## check if unresolved component is included
            self.hasunres = True
        else:
            self.hasunres = False


        
        ## sources will contain a normalized PSF for every source
        self.sources = numpy.zeros((self.Nspat, self.Nflx), dtype='float32')

        ###### Fluxes:
        self.fluxfit  = fluxfit

        ## spectra
        self.fluxes   = numpy.zeros((2, self.Nflx), dtype='float32')

        self.method   = method
        self.fitrad   = fitrad
        self.flux_cv  = flux_cv
        self.sigma_cv = sigma_cv

        if self.method == 0:
            self.fit_fluxes = lambda: self.fit_fluxes_smt()
        elif self.method == 1:
            self.fit_fluxes = lambda: self.fit_fluxes_hyb()
        elif self.method == 2:
            self.fit_fluxes = lambda: self.fit_fluxes_itt()
        else:
            raise IOError, "fit_fluxes-option must be 0,1 or 2 (received %i)" \
                %self.method

        ## broadband fluxes
        self.bbfl = numpy.zeros((self.Ntot, ), dtype="float32")
        self.frac = numpy.ones((self.Nrsv, ), dtype="float32")
  

        ###### Positions:
        self.posfit = posfit

        ## xpos, ypos - input coordinates
        self.xpos = numpy.zeros((2, self.Ntot), dtype='float32')
        self.ypos = numpy.zeros((2, self.Ntot), dtype='float32')

        ## xpix, ypix - TRUE pixel coorinates
        self.xpix = numpy.zeros((2, self.Ntot), dtype='float32')
        self.ypix = numpy.zeros((2, self.Ntot), dtype='float32')
            

        ###### PSF settings: ######
        needed = {'moffat':4, 'gauss':3} 
        self.psftype  = psftype ## either 'moffat' or 'gauss'
        self.psffit = numpy.asarray(deepcopy(psffit)) ## boolean fit option
                                                      ## for every PSF par.
        if max(self.psffit) > 0:
            self.psfvar = True ## PSF is variabel
        else:
            self.psfvar = False ## PSF is fixed

        if needed.has_key(psftype):
            self.psfpars  = numpy.zeros((2, needed[psftype]))
        else:
            raise IOError, "PSF type <%s> is not supported" %psftype
           

        ###### Position parameter settings ######
        self.pospars = numpy.zeros((2,4))
        self.pptofit = numpy.zeros((4,), dtype="bool") ## boolean fit option
                                                       ## for every pos. par.
        if posfit == 1:
            self.pptofit[2:] = True
        elif posfit == 2:
            self.pptofit[:] = True


        ###### Background settings: #######
        self.skyfit   = skyfit
        self.unresfit = unresfit


        self.vprint("_" * 68 + '\n', 1)
        self.vprint("|Image  |Iteration  |Current fit  |Chi^2  ", 1)
        self.vprint("|dChi^2/Chi^2  |Deleted  |\n", 1)


        
    def __del__(self):
        self.vprint("\n"+"_" * 68 + '\n', 1)



    ## for later use: more convenient way to call function that creates PSF 
    def PSF(self):
        if self.psftype == 'moffat':
            return lambda x, y, p: Moffat(self.size, x, y, 0.,
                                          p[0], p[1], p[2], p[3]).array
        elif self.psftype == 'gauss':
            return lambda x, y, p: Gaussian(self.size, x, y, 0.,
                                            p[0], p[1], p[2]).array


    
    ## update the data and sigma array (mask array only if provided)
    ## should be called for every image in a data cube
    def update_data(self, data, sigma, inmask=None):

        self.data[:]  = deepcopy(data.reshape(-1))
        self.sigma[:] = deepcopy(sigma.reshape(-1))

        if inmask != None:
            self.inmask[:] = (deepcopy(inmask.reshape(-1))==0)



    ## Methods that update the arrays containing the fit parameters
    ## Note: if the method(s) are not called for a new dataset (image)
    ##       the code will use the best-fit values of the last fit as
    ##       initial guesses.
    ## force can be used to update all parameters (by default, only
    ## the fixed parameters are updated (except if self.Nima=0))
    def update_fluxes(self, fluxes, broadband=None, force=False):
        if self.Nima == 0 or force:
            update = numpy.ones((self.Nflx,), dtype="bool")
        else:
            update = numpy.zeros((self.Nflx,), dtype="bool")
            if self.fluxfit == 0:
                update[:-2] = True
            if self.skyfit == 0:
                update[-2] = True
            if self.unresfit == 0:
                update[-1] = True

        self.fluxes[:,update] = deepcopy(fluxes)

        if broadband != None:
            self.bbfl = broadband
            for i in xrange(self.Nrsv): 
                ## det. relative contribution of merged sources
                total = self.bbfl[self.map==self.map[i]].sum()
                self.frac[i] = self.bbfl[i]/total


    def update_positions(self, xpos, ypos, force=False):
        if self.Nima == 0 or force or self.posfit < 3:
            self.xpos[:] = deepcopy(xpos)
            self.ypos[:] = deepcopy(ypos)



    def update_psfpars(self, psfpars, force=False):
        if self.Nima == 0 or force:
            self.psfpars[:] = deepcopy(psfpars)
        elif self.psffit.min() == 0:
            self.psfpars[:, self.psffit==False] \
                = deepcopy(psfpars[self.psffit==False])



    def update_pospars(self, pospars, force=False):
        if self.Nima == 0 or force:
            self.pospars[:]= deepcopy(pospars)
        elif self.pptofit.min() == 0:
            self.pospars[1, self.pptofit==False] \
                = deepcopy(pospars[self.pptofit==False])



    ###### The routine to perform an (iterative) fit ######
    ## If more than one property (e.g, PSF, positions, fluxes) are fitted, an
    ## iterative process will  be performed
    ## After the fit has converged to a solution, the method will perform an
    ## F-test to check whether all sources are needed in the fit and remove
    ## the ones that are not needed
    def exec_fit(self):    

        self.Nima     += 1
        self.psf_maker = self.PSF()
        self.chisq     = []
        self.infit     = numpy.array([True]*self.Nflx)  ## to keep track of 
                                                        ## fitted fluxes
        self.newpsf    = True


        iteration = 1

        for i in xrange(2): ## Caluclate true pixel positions
            self.xpix[i] =  self.pospars[i,0]*self.xpos[i] \
                + self.pospars[i,1]*self.ypos[i] + self.pospars[i,2]
            self.ypix[i] = -self.pospars[i,1]*self.xpos[i] \
                + self.pospars[i,0]*self.ypos[i] + self.pospars[i,3]

        self.sources[:] = 0.
        for i in xrange(self.Nrsv): ## create PSF for every source
            j = self.map[i]
            self.sources[:,j] += \
                self.frac[i]*self.psf_maker(self.xpix[1,i], self.ypix[1,i],
                                            self.psfpars[1]).reshape(-1)

            if self._reject and self.fluxes[1,j] <= 0.:
                self.fluxes[j] = 0.
                self.infit[j] = False

        self.sources[:,-2:] = 1.

        self._unres_comp()  ## create array with contribution of unresolved
                            ## sources (entries in 'all', not in 'resolved')
        self._unres_norm()  ## add 'resolved' sources w\ flux <= 0 (only if
                            ## self._reject) and normalize

        self.infit[-2] = self.skyfit
        self.infit[-1] = self.unresfit

        ## estimate chi^2 for the initial guesses (& reject bad pixels)
        self._mask(clip=self.clip, chisq_compute=True)

        ## Calculate noise for every source
        self.noise = 1./numpy.sqrt(((self.sources/self.sigma[:,numpy.newaxis])**2)[self.mask].sum(axis=0))

        ## aperture photometry for sources with flux=0
##         if (self.infit==False).sum() > 0:
##             self.aper_photo()


        while True:  ## Iteratively find best solution for all stars
            Nfits = 0
            self.vprint("\r|%7i|%11i" %(self.Nima, iteration), 1)

            if self.infit.sum() > 0:
                Nfits += 1
                self.vprint("|%13s" %"fluxes", 1)
                self.fit_fluxes()
                self.vprint("|%7.1f" %self.chisq[-1] + "\b" * 22, 1)

            if self.posfit > 0:
                Nfits += 1
                self.vprint("|%13s" %"positions", 1)
                self.fit_positions()
                self.vprint("|%7.1f" %self.chisq[-1] + "\b" * 22, 1)

            if self.psfvar > 0:
                Nfits += 1
                self.vprint("|%13s" %"PSF", 1)
                self.fit_psf()
                self.vprint("|%7.1f" %self.chisq[-1]+ "\b" * 22, 1)

            self.deleted = (self.fluxes[1,:-2]==0.).sum()

            ## convergence check
            if Nfits > 1:
                dchisq = (self.chisq[0]-self.chisq[-1])/self.chisq[-1] 
                self.vprint('|'+13*' '+'|%7.1f|%14.1e|%9i|' \
                                %(self.chisq[-1], dchisq, self.deleted), 1)

                if dchisq < 1e-3:
                    Nfits = 0
                else:
                    self.chisq = []

            iteration += 1

            if Nfits < 2:
                if Nfits == 1:
                    self.vprint('|'+13*' '+'|%7.1f|'%self.chisq[-1]+14*' '+\
                                    '|%9i|' %self.deleted, 1)
                break


##         if self.fluxfit > 0:
##             while True: ## Second iteration: Check significance of detections
## 
##                 self.vprint("\r|%7i|%11i" %(self.Nima, iteration), 1)
## 
##                 self.chisq = [self.chisq[-1]]
## 
##                 self.snr = self.fluxes[1]/self.noise
##                 min_snr  = self.snr[self.infit].min()
##                 i_ex     = abs(self.snr-min_snr).argmin()
## 
##                 self.infit[i_ex] = False
##                 
##                 self.vprint("|%13s" %"fluxes", 1)
##                 self.fit_fluxes()
## 
##                 p1 = self.infit.sum()  ## perform F-test
##                 p2 = p1+1
##                 n  = self.mask.sum()
##                 F = ((self.chisq[1]-self.chisq[0]))/(self.chisq[0]/(n-p2))
## 
##                 if special.fdtri(p2-p1, n-p2, 0.95) < F:
##                     self.vprint("|%7.1f|%14s|%9i|" %(self.chisq[0],
##                                                      "F test: exit",
##                                                      self.deleted), 1)
##                     self.fluxes[1, self.infit] = self.fluxes[0, self.infit]
##                     self.infit[i_ex] = True
##                     self.chisq = [self.chisq[0]]
##                     break
##                 else:
##                     self.deleted += 1
##                     self.fluxes[:, i_ex] = 0.
##                     self.vprint('|%7.1f|%14s|%9i|' %(self.chisq[0],
##                                                      "F test: remove",
##                                                      self.deleted), 1)
##                     iteration += 1

        self.outmask = (self.mask.reshape(self.size)==0)
        
        if self.makemod:

            model_1d  = (self.fluxes[1] * self.sources).sum(axis=1)
            self.model = model_1d.reshape(self.size)
            if self.makeres:
                residuals_1d = self.data - model_1d
                self.residuals = residuals_1d.reshape(self.size)

        self.vprint("\n", 4)



    def _unres_comp(self):

        ## only update if unresolved component included and PSF or
        ## positions have changed
        if self.hasunres and self.newpsf:

            self.unres_faint = numpy.zeros(self.size, dtype="float32")

            for i in xrange(self.Ntot):
                if self.map[i] == -1:

                    psf_i = self.psf_maker(self.xpix[1,i], self.ypix[1,i],
                                           self.psfpars[1])
                    self.unres_faint += self.bbfl[i]* psf_i
                        
            self.newpsf=False
            

            

    def _unres_norm(self):
        
        if self.hasunres:
            self.sources[:,-1] = self.unres_faint.reshape((-1,))

            if not self.infit[:-2].min():
                for i in xrange(self.Nrsv):
                    j = self.map[i]
                    if not self.infit[j]:
                        self.sources[:,-1] +=self.bbfl[i]*self.sources[:,j]

            mean_faint = self.sources[:,-1].mean()
            if mean_faint != 0:
                self.sources[:,-1] /= mean_faint


                
    def _mask(self, clip, chisq_compute=False):
        
        ## calculate difference between model and data
        err = self.data - (self.fluxes[1]*self.sources).sum(axis=1)
        err[self.inmask==0] = numpy.median(err) 

        ## find bad pixels via sigma-clipping
        clip_result = clip_sigma((err/self.sigma), clip=clip)

        ## update mask
        self.mask = numpy.logical_and(self.inmask, clip_result[1])

##         print "\n", self.mask.sum()

        ## if requested, add chi^2 value
        if chisq_compute:
            err *= self.mask

            self.chisq.append(((err*err)/(self.sigma*self.sigma)).sum())


            
    def fit_positions(self):

        if self.posfit == 3:
            isfitted = numpy.zeros((self.Ntot,), dtype="bool")
            for i in xrange(self.Nrsv):
                if self.infit[self.map[i]]: isfitted[i] = True
            fitpars = numpy.concatenate((self.xpos[1,isfitted],
                                         self.ypos[1,isfitted])) 
        else:
            fitpars  = self.pospars[1,self.pptofit]
            isfitted = numpy.ones((self.Nrsv,), dtype='bool')

        self.workcopy  = self.data.copy()
        self.workcopy -= (self.fluxes[1,-2:]*self.sources[:,-2:]).sum(axis=1)

        results = optimize.fmin(self._fit_newpsf, fitpars,
                                args=(False,self.xpos[1],self.ypos[1],
                                      self.psfpars[1],self.pospars[1],isfitted),
                                full_output=1, disp=False)

        if self.posfit == 3:
            self.xpos[0, isfitted] = self.xpos[1, isfitted]
            self.ypos[0, isfitted] = self.ypos[1, isfitted]

            self.xpos[1, self.infit[:-2]] = results[0][:isfitted.sum()]
            self.ypos[1, self.infit[:-2]] = results[0][isfitted.sum():]

        else:
            self.pospars[0, self.pptofit] = self.pospars[1, self.pptofit]
            self.pospars[1, self.pptofit] = results[0]

        for i in xrange(2):
            self.xpix[i] =  self.pospars[i,0]*self.xpos[i] \
                + self.pospars[i,1]*self.ypos[i] + self.pospars[i,2]
            self.ypix[i] = -self.pospars[i,1]*self.xpos[i] \
                + self.pospars[i,0]*self.ypos[i] + self.pospars[i,3]

        self._mask(clip=self.clip)
        self.chisq.append(results[1])
        self.newpsf = True



    def fit_psf(self):
        
        fitpars = self.psfpars[1,self.psffit]

        self.workcopy  = self.data.copy()
        self.workcopy -= (self.fluxes[1,-2:]*self.sources[:,-2:]).sum(axis=1)

        isfitted = numpy.ones((self.Nrsv,), dtype='bool')

        results = optimize.fmin(self._fit_newpsf, fitpars,
                                args=(True,self.xpos[1],self.ypos[1],
                                      self.psfpars[1],self.pospars[1],isfitted),
                                full_output=1, disp=False)

        self.psfpars[0,self.psffit] = self.psfpars[1, self.psffit]
        self.psfpars[1,self.psffit] = results[0]

        self._mask(clip=self.clip)
        self.chisq.append(results[1])
        self.newpsf = True


    def _fit_newpsf(self, fitpars, newshape, x, y, psfpars, pospars, isfitted):


        if newshape:
            psfpars[self.psffit] = fitpars
        else:
            if self.posfit == 3:
                x[isfitted] = numpy.asarray(fitpars[:isfitted.sum()],
                                            dtype="float32")
                y[isfitted] = numpy.asarray(fitpars[isfitted.sum():],
                                            dtype="float32")
            else:
                pospars[self.pptofit] = fitpars


        xpix =  pospars[0]*x + pospars[1]*y + pospars[2]
        ypix = -pospars[1]*x + pospars[0]*y + pospars[3]

        self.sources[:,:-2] = 0. 
        for i in xrange(self.Nrsv):
            if isfitted[i]:
                j = self.map[i]
                self.sources[:,j] += self.frac[i]*self.psf_maker(xpix[i], ypix[i], psfpars).reshape(-1)


        err  =self.workcopy-(self.fluxes[1,:-2]*self.sources[:,:-2]).sum(axis=1)
        err *=self.mask

        return ((err*err)/(self.sigma*self.sigma)).sum()


    
    def aper_photo(self, cut=2.0, r_aper=2.5):

        res  = self.data - (self.fluxes[1] * self.sources).sum(axis=1)

        for i in xrange(self.Nind):
            if not self.infit[i]:
                nearpix = numpy.fromfunction(self._select_pix, self.size, cid=i,
                                             r_in=r_aper)
                flx = res[nearpix*self.mask].sum()

                if flx > cut*self.noise[i]:
                    self.fluxes[1,i] = flx
                    self.infit[i] = True


    
    def _select_pix(self, x, y, sid, r_in, r_out=None):

        r = numpy.sqrt((x-self.ypix[1,sid])**2 + (y-self.xpix[1,sid])**2)

        if r_out == None: return (r<=r_in)
        else: return numpy.logical_and(r>r_in, r<=r_out)



    def vprint(self, string, vb_min):
        if self.vb >= vb_min:
            sys.stdout.write(string)
            sys.stdout.flush()



    def fit_fluxes_smt(self):

        self.fluxes[0] = self.fluxes[1]

        self.workcopy  = self.data.copy()
        self.workcopy -=(self.fluxes[1,self.infit==False]*self.sources[:,self.infit==False]).sum(axis=1)

        if self.newpsf:
            self.noise = 1./numpy.sqrt(((self.sources/self.sigma[:,numpy.newaxis])**2).sum(axis=0))
            self._unres_comp()

        deleted = (self.infit[:-2]==False).sum()

        while True:

            self._unres_norm()

            a = (self.sources/self.sigma[:,numpy.newaxis])[self.mask][:,self.infit]
            b = (self.workcopy/self.sigma)[self.mask]

            results = numpy.linalg.lstsq(a,b)

            self.fluxes[1, self.infit] = results[0]

            if self._reject:
                for i in xrange(self.Nind):
                    if self.infit[i] and self.fluxes[1,:-2] < self.noise:
                        self.infit[i] = False
                        self.fluxes[1,i] = 0.
                        ## no need to update workcopy since flux=0


            if (self.infit[:-2]==False).sum() == deleted:
                break
            else:
                deleted = (self.infit[:-2]==False).sum()

        self._mask(clip=self.clip)
        self.chisq.append(results[1])



    def fit_fluxes_hyb(self):

        ## update noise values if PSF/positions have changed
        if self.newpsf:
            self.noise = 1./numpy.sqrt(((self.sources/self.sigma[:,numpy.newaxis])**2)[self.mask].sum(axis=0))
        
        insrcfit = self.infit.copy()

        ## create work copy and subtract all components
        self.workcopy  = self.data.copy()
        self.workcopy -=(self.fluxes[1]*self.sources).sum(axis=1)

        ## Create "valid pixel array" for every source 
        fitapers = numpy.zeros(self.sources.shape, dtype="bool")
        for i in xrange(self.Nrsv):
            j = self.map[i]
            if insrcfit[j]:
                sel_i = numpy.fromfunction(self._select_pix,self.size,
                                           sid=i, r_in=self.fitrad).reshape(-1)
                fitapers[:,j] = numpy.logical_or(fitapers[:,j], sel_i)
        fitapers[:,-2] = insrcfit[-2]
        fitapers[:,-1] = insrcfit[-1]
        
        for j in xrange(self.Nind):
            if not numpy.logical_and(self.mask, fitapers[:,j]).any():
                self.fluxes[:,j] = 0.
                fitapers[:,j]    = False
                insrcfit[j]      = False

        
        iteration = 0
        converged = 0
        while True: ## BEGINNING OF ITERATIVE PART

            self.fluxes[0] = self.fluxes[1]

            ## fitting of sources/background
            fitpix = numpy.logical_and(self.mask, numpy.any(fitapers,axis=1))

            A = ((self.sources*fitapers)[:,insrcfit]/self.sigma[:,numpy.newaxis])[fitpix]
            b = (self.workcopy/self.sigma)[fitpix]
            
            results = numpy.linalg.lstsq(A,b)
            dflux   = results[0]

            self.fluxes[1,insrcfit] += dflux

            ## update workcopy using difference between the two fits
            self.workcopy -= (dflux*self.sources[:,insrcfit]).sum(axis=1)
            

            if iteration >= 4: ## check if sources have converged
                j = 0
                for i in xrange(self.Nind):
                    if insrcfit[i]:
                        cc_flux  = self.flux_cv*self.fluxes[1,i]
                        cc_sigma = self.sigma_cv*self.noise[i]
                        if dflux[j] < max(cc_flux, cc_sigma):
                            ## if so, remove them from list of fitted sources
                            ## and subtract them completely from workcopy
                            converged += 1
                            insrcfit[i] = False
                        j += 1

            if converged > 0 and insrcfit[-2:].any():
                ## fix background to current value
                insrcfit[-2:] = False
                    
            if insrcfit.any() == False or iteration == 10:
                ## max. no of iterations = 10
                break
            else:
                iteration += 1

        self._mask(clip=self.clip, chisq_compute=True)



    def fit_fluxes_itt(self):

        ## update noise values if PSF/positions have changed
        if self.newpsf:
            self.noise = 1./numpy.sqrt(((self.sources/self.sigma[:,numpy.newaxis])**2)[self.mask].sum(axis=0))
        
        insrcfit = self.infit.copy()

        ## create workcopy and subtract all components
        self.workcopy  = self.data.copy()
        self.workcopy -=(self.fluxes[1]*self.sources).sum(axis=1)


        ## Create "valid pixel array" for every source 
        fitapers = numpy.zeros(self.sources.shape, dtype="bool")
        for i in xrange(self.Nrsv):
            j = self.map[i]
            if insrcfit[j]:
                sel_i = numpy.fromfunction(self._select_pix,self.size,
                                           sid=i, r_in=self.fitrad).reshape(-1)
                fitapers[:,j] = numpy.logical_or(fitapers[:,j], sel_i)
        fitapers[:,-2] = insrcfit[-2]
        fitapers[:,-1] = insrcfit[-1]
        
        for j in xrange(self.Nind):
            if not numpy.logical_and(self.mask, fitapers[:,j]).any():
                self.fluxes[:,j] = 0.
                fitapers[:,j]    = False
                insrcfit[j]      = False


        ## sort sources by descending fluxes
        self.fsorted = list(self.fluxes[1,:-2].argsort())
        self.fsorted.reverse()

        ## background components that are fitted:
        bf = numpy.concatenate([numpy.zeros(self.Nind, dtype='bool'),
                                self.infit[-2:]])

        iteration = 0
        converged = 0
        while True: ## BEGINNING OF ITERATIVE PART

            self.fluxes[0] = self.fluxes[1]

            if bf.any() and converged == 0:
                a = (self.sources[:,bf]/self.sigma[:,numpy.newaxis])[self.mask]
                b = (self.workcopy/self.sigma)[self.mask]

                results = numpy.linalg.lstsq(a,b)

                self.fluxes[1,bf] += results[0]
                self.workcopy -= (results[0]*self.sources[:,bf]).sum(axis=1)

            for i in self.fsorted: ## fit the individual sources
                if insrcfit[i]:

                    fitpix = numpy.logical_and(self.mask, fitapers[:,i])

                    a = (self.sources[:,i]/self.sigma)[fitpix]
                    A = a.reshape((-1,1))
                    b = (self.workcopy/self.sigma)[fitpix]

                    results = numpy.linalg.lstsq(A,b)
                    dflux   = results[0][0]
                    
                    self.fluxes[1,i] += dflux
                    self.workcopy    -= (dflux*self.sources[:,i])
            
                    if iteration >= 4: ## check if source has converged
                        cc_flux  = self.flux_cv*self.fluxes[1,i]
                        cc_sigma = self.sigma_cv*self.noise[i]
                        if dflux < max(cc_flux, cc_sigma):
                            self.fsorted.remove(i)
                            converged += 1

            if self.fsorted == [] or iteration == 10: ## max. no of iterations
                break
            else:
                iteration += 1

        self._mask(clip=self.clip, chisq_compute=True)


##     def _fit_samepsf(self, fitpars):
##         last = 0
## 
##         err = self.workcopy.copy()
## 
##         if self.skyfit > 1:
##             last -= 1
##             err -= fitpars[last]*self.unres_array
##         if self.skyfit > 0:
##             last -= 1
##             err -= fitpars[last]
##         if self.fluxfit > 0:
##             err  -= (fitpars[:last, numpy.newaxis, numpy.newaxis] \
##                          *self.sources[self.infit]).sum(axis=0)
## 
##         err *= self.mask
## 
##         return ((err*err)/(self.sigma*self.sigma)).sum()






## class OldFit:
##     def __init__(self, data, sigma, mask=None):
## 
##         self.data  = data
##         self.sigma = sigma  
## 
##         self.size  = self.data.shape
## 
##         self.mask  = numpy.ones(self.size, dtype='int16')
##         if mask != None:
##             self.mask -= (mask>0.)
## 
## 
##         self.pdict   = {}  ## will contain all fit parameters and a boolean fit
##                            ## option for each parameter
## 
##         self.psfvar = False ## is the PSF (parameters, star positions)
##                             ##  variable?
## 
##         self.fluxfit = False
##         
## 
## 
##     def init_psf(self, type, lut=None, data=None, params=[], tofit=[]):
##         """
##         valid PSF types: 'gauss', 'moffat' or 'array'
##         params and tofit are 1D arrays containing the PSF parameters and for 
##         each parameter a (boolean) fit option (used if type is not 'array')
##         NOTE: The flux is automatically set to 1
##         data contains PSF (used if type is 'array')
##         """
## 
##         if sum(tofit) > 0:
##             self.psfvar = True
## 
##         Nparams = {'array':1,'gauss':3, 'moffat':4}
## 
##         p = numpy.asarray(params)
##         f = numpy.asarray(tofit)
## 
##         self.psf = type
## 
##         self.lut = lut
## 
##         if self.psf == 'array':
##             self.pdict['psf'] = data
##             self.Nppsf = 0
## 
##         elif Nparams.has_key(type):
##             self.Nppsf = Nparams[type]
## 
##             if p.shape[0] == self.Nppsf:
##                 parpsf = numpy.zeros((2, Nparams[type]), dtype='float32')
##                 parpsf[0] = params
##                 parpsf[1] = tofit
##                 self.pdict['psf'] = parpsf
##             else:
##                 raise IOError('Invalid number of parameters for PSF-type "'+\
##                                   type+'"! Must be '+str(Nparams[type])+'!')
## 
##         else:
##             raise IOError('Unknown PSF option: %s' %self.psf)
##             
## 
## ## !CHANGED    
##     def init_sources(self, positions, fluxes, posfit, fluxfit):
##         """
## 	<positions> is an array containing the information about the positions
## 	of the sources. To simplify the usage of highres a priori information,
## 	it is constructed in the following way: The first 4 entries contain the 
## 	transformation parameters A, B, C and D, followed by the (not
## 	transformed) x- and y coordinates for every star. The positions on the
## 	fitted array are determined via: x_fit =  A*x + B*y + C
## 	                                 y_fit = -B*x + A*y + D 
## 	<fluxes> is an (Nsources,) shaped array containing the fluxes of every
## 	source. 
## 
##         <posfit> can be either 0, 1, 2 or 3. If 0, the positions are fixed; if
## 	1, only C and D are free; if 2, A to D are free; if 3, individual
## 	positions are fitted.
##         <fluxfit> determines whether the fluxes are fitted (<fluxfit>=1) or not
## 	(<fluxfit>=0).
##         """
## 
##         if posfit > 0:
##             self.psfvar = True
##         if fluxfit > 0:
##             self.fluxfit = True
## 
## ##         self.x_ref = refpos[0]
## ##         self.y_ref = refpos[1]
## 
##         self.ns      = fluxes.shape[0]        
##         self.sources = numpy.arange(self.ns)  ## contains all sources still in
##                                               ## the fit
## 
##         parpos = numpy.zeros((2, 2*self.ns+4), dtype='float32')
##         parflx = numpy.zeros((2, self.ns), dtype='float32')
## 
## 
##         if posfit == 1:
##             parpos[1,2:4] = 1
## 	elif posfit == 2:
## 	    parpos[1,:4] =1
##             
##         parpos[0] = positions
##         parflx[0] = fluxes
## 
##         if posfit == 3:
##             parpos[1,2:] = 1
##         if fluxfit == 1:
##             parflx[1] = 1
## 
##         self.pdict['pos'] = parpos
##         self.pdict['flx'] = parflx
## 
##         self.objects = numpy.zeros((self.ns,self.size[0],self.size[1]),
##                                    dtype='float32')
## 
## 
## 
## 
##     def init_background(self, value=0., tofit=0):
##         """
##         If background should be fitted, set tofit=1. Value than acts as initial
##         guess
##         """
##         parbck = numpy.zeros((2,1), dtype='float32')
##         parbck[0] = value
##         parbck[1] = tofit
##         self.pdict['bck'] = parbck
## 
## 
## ## !CHANGED
##     def exec_fit(self, disp=True, model=False, residuals=False):
## 
##         ## write all fit parameters into one array
##         if self.psf == 'array':
##             self.fitparams = numpy.concatenate((self.pdict['pos'][0],
##                                                 self.pdict['flx'][0],
##                                                 self.pdict['bck'][0]))
##             self.isfitted  = numpy.concatenate((self.pdict['pos'][1],
##                                                 self.pdict['flx'][1],
##                                                 self.pdict['bck'][1]))
##         else:
##             self.fitparams = numpy.concatenate((self.pdict['psf'][0],
##                                                 self.pdict['pos'][0],
##                                                 self.pdict['flx'][0],
##                                                 self.pdict['bck'][0]))
##             self.isfitted  = numpy.concatenate((self.pdict['psf'][1],
##                                                 self.pdict['pos'][1],
##                                                 self.pdict['flx'][1],
##                                                 self.pdict['bck'][1]))
## 
##         self.Npix = self.mask.sum()
## 
##         if not self.fluxfit:
##             select_sources = True
##             stop = True
##         else:
##             select_sources = False
##             stop = False
## 
##         iter = 0
##         self.chisq = 0.
##         while True:
##             
##             if select_sources:
##                 fluxes = self.fitparams[self.Nppsf + 4 + 2*self.ns:-1]
## 
##                 bad    = (fluxes <= 0.)
##                 self.fitparams[self.Nppsf + 4 + 2*self.ns:-1][bad] = 0.
##                 self.isfitted[self.Nppsf + 4 + 2*self.ns:-1][bad] = False
## 
##                 self.sources = numpy.arange(self.ns)[fluxes>0]
## 
##             self.Ndof = self.isfitted.sum()
## 
##             free_params = self.fitparams[self.isfitted==1]
## 
##             if not self.psfvar:
##                 self.make_psf()
## 
##             result = optimize.fmin(self._fit_res, free_params, full_output=1,
##                                    disp=False)
## 
##             if iter > 0: ## using an F test to see whether the fit quality
##                          ## decreased significantly
## 
##                 p2 = self.Ndof+1
##                 p1 = self.Ndof
##                 n  = self.mask.sum()
## 
##                 F = ((result[1]-self.chisq)/(p2-p1))/(self.chisq/(n-p2))
## 
##                 if special.fdtri(p2-p1, n-p2, 0.95) < F: 
##                     ## if so (by more than 2 sigma): stop
##                 
##                     self.fitparams = deepcopy(old_fp)
##                     self.snr       = deepcopy(old_snr)
##                     self.sources   = deepcopy(old_src)
##                     stop = True
## 
##             else:
##                 self.fitparams[self.isfitted==1] = numpy.array(result[0],
##                                                                dtype='float32')
##                 self.chisq = result[1]
## 
## 
##             if stop:
##                 break
## 
##             select_sources = True
##             
##             self.snr  = (self.objects*self.mask).sum(axis=2).sum(axis=1)
##             self.snr *= self.fitparams[self.Nppsf + 4 + 2*self.ns:-1]
##             self.snr /= numpy.sqrt(1./((self.objects/self.sigma)**2).sum(axis=2).sum(axis=1))
## 
##             fluxes = self.fitparams[self.Nppsf + 4 + 2*self.ns:-1]
##             bad    = (fluxes<0).sum()
## 
##             if bad==0:
##                 if len(self.sources) > 0:
##                     iter += 1
##                     old_fp  = deepcopy(self.fitparams)
##                     old_snr = deepcopy(self.snr)
##                     old_src = deepcopy(self.sources)
## 
##                     obj_snrmin=self.sources[self.snr[self.sources].argmin()]
##                     
##                     ind_snrmin = self.Nppsf + 4 + 2*self.ns + obj_snrmin
##                     
##                     self.fitparams[ind_snrmin] = 0.
##                 else:
##                     break
## 
##  
##         self.bestfit_psf = self.fitparams[:self.Nppsf]
##         self.bestfit_pos = self.fitparams[self.Nppsf:self.Nppsf+4+2*self.ns]
##         self.bestfit_flx = self.fitparams[self.Nppsf+4+2*self.ns:-1]
##         self.bestfit_bck = self.fitparams[-1]
##         self.red_chisq   = self.chisq/(self.mask.sum() - self.Ndof)
## 
##         self.deleted = self.ns - self.sources.shape[0]
## 
##         if model or residuals:
##             self.model = self.fitparams[-1]*numpy.ones(self.size,
##                                                        dtype='float32')
##             for i in self.sources:
##                 j = self.Nppsf + 4 + 2*self.ns + i
##                 self.model += self.fitparams[j]*self.objects[i]
## 
##                 if residuals:
##                     self.residuals = numpy.array(self.data - self.model,
##                                                  dtype='float32')
## 
## 
## 
## ## !CHANGED
##     def make_psf(self):
## 
##         A, B, C, D = self.fitparams[self.Nppsf:self.Nppsf+4]
## 
##         for i in self.sources:
## 
##             j = 2*i + self.Nppsf + 4
## 
##             xpos =  A*self.fitparams[j] + B*self.fitparams[j+1] + C
##             ypos = -B*self.fitparams[j] + A*self.fitparams[j+1] + D
## 
## 
##             if self.psf == 'gauss':
##                 PSF_i = Gaussian(self.size, xpos, ypos, 0., self.fitparams[0],
##                                  self.fitparams[1], self.fitparams[2])
##         
##             elif self.psf == 'moffat':
##                 PSF_i = Moffat(self.size, xpos, ypos, 0., self.fitparams[0],
##                                self.fitparams[1], e=self.fitparams[2],
##                                theta=self.fitparams[3])
## 
##             self.objects[i] = PSF_i.array
## 
##             if self.lut != None:
##                 lut_size = (self.lut.shape[0]+1,self.lut.shape[1]+1)
##                 lut_arr  = numpy.zeros(lut_size, dtype='float32')
## 
##                 dx = xpos - int(xpos)
##                 dy = ypos - int(ypos)
## 
##                 if dx >= 0.5:
##                     if dy >= 0.5:
##                         lut_arr[:-1,:-1] = self.lut
##                     else:
##                         lut_arr[1:,:-1] = self.lut
##                 else:
##                     if dy >= 0.5:
##                         lut_arr[:-1,1:] = self.lut
##                     else:
##                         lut_arr[1:,1:] = self.lut
##                             
##                 tmp_lut = ndimage.zoom(lut_arr, 0.5, order=0)
##                 
##                 xcen = (tmp_lut.shape[1]>>1)
##                 ycen = (tmp_lut.shape[0]>>1)
## 
##                 lut = paste_area(tmp_lut, self.size, (xcen, ycen),
##                                  (xpos, ypos))
##                 
##                 self.objects[i] += lut
## 
## 
## 
## 
## ##     def _draw_psf(self):
## ## 
## ##         if self.psf == 'array':
## ##             psf_data = self.pdict['psf']
## ## 
## ##             x_c = psf_data.shape[1]>>1
## ##             y_c = psf_data.shape[0]>>1
## ##             for i in xrange(self.ns):
## ##                 j = 3*i + self.Nppsf
## ##                 dx = self.fitparams[j] - x_c
## ##                 dy = self.fitparams[j+1] - y_c
## ##                 
## ##                 psf_j = shift_image(psf_data, dx, dy)
## ## 
## ##                 max_x = min(psf_j.shape[1], self.size[1])
## ##                 max_y = min(psf_j.shape[0], self.size[0])
## ## 
## ##                 self.objects[i,:max_y,:max_x] = psf_j
## ## not updated yet!!!
## 
## 
## 
## ## !CHANGED
##     def _fit_res(self, pfit):
## 
##         pfit = numpy.asarray(pfit, dtype='float32')
## 
##         self.fitparams[self.isfitted==1] = pfit
## 
##         err = self.data - self.fitparams[-1]
## 
##         if self.psfvar:
##             self.make_psf()
## 
##         for i in self.sources:
##             j = self.Nppsf + 4 + 2*self.ns + i
##             err -= self.fitparams[j]*self.objects[i]
## 
##         err *= self.mask
## 
##         return ((err*err)/(self.sigma*self.sigma)).sum()
## 



