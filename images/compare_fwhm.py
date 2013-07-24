"""
Code to compare the FWHM of a Gaussian and a Moffat profile. The comparison
is performed by creating N realizations of a Moffat profile with varying S/N
and beta and fitting the result with a Gaussian profile. The results of the
simulations are plotted.

LAST CHANGED: 2013/04/26
AUTHOR: Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)
"""
import matplotlib as mpl, matplotlib.pyplot as plt, numpy as np
from scipy import linalg,optimize
from pampelmuse.core.psf.profiles import Moffat, Gaussian
plt.rcParams["axes.linewidth"] = 1.5

####  BASIC SETUP ##############################################################
ARRAY_SIZE = (16,16)  ## The spatial size of the array used in the simulation
NREAL = 1000  ## The number of realizations
cmap = mpl.cm.get_cmap('gist_rainbow') ## The colormap used to plot S/N
norm = mpl.colors.Normalize(vmin=0, vmax=100) ## Normalization of colormap
################################################################################


def fit_array(p, data, sigma, names):
    """
    The function to fit a Moffat with a Gaussian profile, called by
    optimize.leastsq.
    """

    ## Update parameters
    for i in xrange(len(p)):
        try:
            getattr(OUTPSF, "set_{0}".format(names[i]))(p[i])
        except AssertionError:
            raise np.linalg.linalg.LinAlgError("Parameter outside valid range.")
            return 1e6

    ## Calculate residuals
    model = np.zeros(data.shape, dtype=np.float64)

    for psf in OUTPSF:
        if psf.function == "gaussian": ## Check if PSF should be subtracted
            model[psf.used] += psf.f/sigma[psf.used]

    return data/sigma-model



def jacobian(p, data, sigma, names):
    """
    Calculate the Jacobi matrix for the minimization performed in 'fit_array'
    """
    jacobi = np.zeros((len(p),)+data.shape, dtype=np.float64)

    for psf in OUTPSF:
        if psf.function == "gaussian": ## Check if PSF should be subtracted
            for i in xrange(len(p)):
                jacobi[i, psf.used] \
                    -= getattr(psf, "diff_{0}".format(names[i]))/sigma[psf.used]
                    
    return jacobi


## The profile used as input: a Moffat
INPSF = Moffat
INPSF.set_size(ARRAY_SIZE)

## The PSF used to fit the data: a Gaussian
OUTPSF = Gaussian
OUTPSF.set_size(ARRAY_SIZE)

## initialize both PSF instances
inpsf = INPSF(Xc=ARRAY_SIZE[1]/2., Yc=ARRAY_SIZE[0]/2., mag=-10, maxrad=10,
              osradius=2, osfactor=5)
inpsf.fwhm = 4.0

outpsf = OUTPSF(Xc=inpsf.Xc, Yc=inpsf.Yc, mag=0, maxrad=10, osradius=2,
                osfactor=5)

## initialize arrays to hold results, create NREAL random beta values
beta_values = 8.9*np.random.random_sample((NREAL,))+1.1
fwhm_ratio  = np.zeros((NREAL,), dtype=np.float32)
snratios    = np.zeros((NREAL,), dtype=np.float32)

i = 0
for beta in beta_values: ## loop over beta-values
    inpsf.beta = beta

    ## select random magnitude from interval [-7,12)
    inpsf.mag = -7 - 5*np.random.random()
    outpsf.mag = 0. ## normalize Gaussian for initial flux fit

    ## prepare the  data
    data = np.zeros(ARRAY_SIZE, dtype=np.float64).flatten()
    data[inpsf.used] += inpsf.f ## add Moffat

    ## prepare the noise
    sigma = 6.*np.ones(ARRAY_SIZE, dtype=np.float64).flatten()
    sigma += np.sqrt(data)

    ## calculate S/N
    snratios[i] = np.sqrt((inpsf.f**2/sigma[inpsf.used]**2).sum())

    ##  create noisy data used in optimization
    noisy_data = np.random.normal(loc=data, scale=sigma)

    ## initial guess for Gaussian PSF is FWHM of Moffat
    outpsf.fwhm = inpsf.fwhm

    ## perform an initial linear least squares fit to get a flux for the
    ## Gaussian and update the profile
    vec = np.zeros((noisy_data.size,1),  dtype=np.float64)
    vec[outpsf.used,0] += outpsf.f/sigma[outpsf.used]
    
    outpsf.mag = -2.5*np.log10(linalg.lstsq(vec, noisy_data/sigma)[0])[0]

    ## perform non-linear fit and save results
    init_guess = [inpsf.fwhm,]
    names = ["fwhm",]

    bestfit = optimize.leastsq(fit_array, init_guess,
                               args=(noisy_data,sigma,names),
                               Dfun=jacobian, col_deriv=1, full_output=False)

    fwhm_ratio[i] = bestfit[0]/inpsf.fwhm

    i += 1

## plot results, save plot
fig = plt.figure()
ax = fig.add_axes([0.12, 0.12, 0.70, 0.83])
ax.minorticks_on()

ax.axhline(y=1, ls="-", lw=1.5, c="0.2")

ax.scatter(beta_values, fwhm_ratio, marker="o", c=snratios,
           s=25, edgecolors="None", cmap=cmap, norm=norm)

ax.set_xlabel(r"$\beta$", fontsize=20)
ax.set_ylabel(\
    r"$\mathrm{FWHM}_\mathrm{Gaussian}/\mathrm{FWHM}_\mathrm{Moffat}$",
    fontsize=20)

ax.set_xlim(1,10)

cax  = fig.add_axes([0.85, 0.12, 0.02, 0.83])
cb   = mpl.colorbar.ColorbarBase(cax, cmap=cmap,
                                 norm=norm, orientation='vertical')

cb.set_label(r"S/N", fontsize=18)

for axes in [ax, cax]:
    for line in axes.xaxis.get_ticklines() + axes.yaxis.get_ticklines() \
            + axes.xaxis.get_minorticklines() + axes.yaxis.get_minorticklines():
        line.set_markeredgewidth(1.5)

fig.savefig("fwhm_ratio_gauss_moffat.png")

plt.show()
