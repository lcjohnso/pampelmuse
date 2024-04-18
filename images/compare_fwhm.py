"""
Code to compare the FWHM of a Gaussian and a Moffat profile. The comparison
is performed by creating N realizations of a Moffat profile with varying S/N
and beta and fitting the result with a Gaussian profile. The results of the
simulations are plotted.

LAST CHANGED: 2017/01/24
AUTHOR: Sebastian Kamann (skamann@astro.physik.uni-goettingen.de)
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg, optimize

from src.pampelmuse.psf import Moffat, Gaussian

plt.rcParams["axes.linewidth"] = 1.5


# BASIC SETUP ###############################################################
INPSF = Moffat  # The profile used as input: a Moffat
OUTPSF = Gaussian  # The PSF used to fit the data: a Gaussian
ARRAY_SIZE = (16, 16)  # The spatial size of the array used in the simulation
NREAL = 1000  # The number of realizations
cmap = mpl.cm.get_cmap('gist_rainbow')  # The colormap used to plot S/N
norm = mpl.colors.Normalize(vmin=0, vmax=100)  # Normalization of colormap
#############################################################################


def fit_array(p, psf, data, sigma, names):
    """
    Function to fit a Gaussian profile to an image, called by
    optimize.leastsq.

    Parameters
    ----------
    p : array_like
        The values of the fitted parameters
    psf : instance of pampelmuse.psf.Profile
        The PSF instance that is fitted to the data
    data : nd_array
        The data to be fitted.
    sigma : nd_array
        The uncertainties of the data. Note that sigma.shape == data.shape
        is required.
    names : array_like
        The names of the fitted parameters. Note that len(p) == len(names)
        is required.

    Returns
    -------
    residuals : nd_array
        The fit residuals.
    """
    # Update parameters
    for i in xrange(len(p)):
        try:
            setattr(psf, names[i], p[i])
        except AssertionError:
            raise np.linalg.linalg.LinAlgError("Parameter outside valid range.")

    # Calculate residuals
    model = np.zeros(data.shape, dtype=np.float32)

    if psf.function == "gaussian":  # Check if PSF should be subtracted
        model[psf.used] += psf.f/sigma[psf.used]

    return data/sigma - model


def jacobian(p, psf, data, sigma, names):
    """
    Calculate the Jacobi matrix for the minimization performed in 'fit_array'.

    Parameters
    ----------
    Same as for fit_array function

    Returns
    -------
    jacobi : nd_array
       the jacobi matrix for the fitted parameters.
    """
    jacobi = np.zeros((len(p),) + data.shape, dtype=np.float32)

    if psf.function == "gaussian":  # Check if PSF should be subtracted
        for i in xrange(len(p)):
            jacobi[i, psf.used] -= getattr(psf, "diff_{0}".format(names[i]))/sigma[psf.used]
                    
    return jacobi


def main():

    # initialize both PSF instances
    inpsf = INPSF.from_shape(shape=ARRAY_SIZE, uc=ARRAY_SIZE[1]/2., vc=ARRAY_SIZE[0]/2., mag=-10, maxrad=10,
                             osradius=2, osfactor=5)
    inpsf.fwhm = 4.0

    outpsf = OUTPSF.from_shape(shape=ARRAY_SIZE, uc=inpsf.uc, vc=inpsf.vc, mag=0, maxrad=10, osradius=2, osfactor=5)

    # initialize arrays to hold results, create NREAL random beta values
    beta_values = 8.9*np.random.random_sample((NREAL,)) + 1.1
    fwhm_ratio = np.zeros((NREAL,), dtype=np.float32)
    snratios = np.zeros((NREAL,), dtype=np.float32)

    i = 0
    for beta in beta_values:  # loop over beta-values
        inpsf.beta = beta

        # select random magnitude from interval [-7,12)
        inpsf.mag = -7 - 5*np.random.random()
        outpsf.mag = 0.  # normalize Gaussian for initial flux fit

        # prepare the data
        data = np.zeros(ARRAY_SIZE, dtype=np.float64).flatten()
        data[inpsf.used] += inpsf.f  # add Moffat

        # prepare the noise
        sigma = 6.*np.ones(ARRAY_SIZE, dtype=np.float64).flatten()
        sigma += np.sqrt(data)

        # calculate S/N
        snratios[i] = np.sqrt((inpsf.f**2/sigma[inpsf.used]**2).sum())

        #  create noisy data used in optimization
        noisy_data = np.random.normal(loc=data, scale=sigma)

        # initial guess for Gaussian PSF is FWHM of Moffat
        outpsf.fwhm = inpsf.fwhm

        # perform an initial linear least squares fit to get a flux for the Gaussian and update the profile
        vec = np.zeros((noisy_data.size, 1),  dtype=np.float64)
        vec[outpsf.used, 0] += outpsf.f/sigma[outpsf.used]

        outpsf.mag = -2.5*np.log10(linalg.lstsq(vec, noisy_data/sigma)[0])[0]

        # perform non-linear fit and save results
        init_guess = [inpsf.fwhm, ]
        names = ["fwhm", ]

        bestfit = optimize.leastsq(fit_array, init_guess, args=(outpsf, noisy_data, sigma, names),
                                   Dfun=jacobian, col_deriv=1, full_output=False)

        fwhm_ratio[i] = bestfit[0]/inpsf.fwhm

        i += 1

    # plot results, save plot
    fig = plt.figure()
    ax = fig.add_axes([0.12, 0.12, 0.70, 0.83])
    ax.minorticks_on()

    ax.axhline(y=1, ls="-", lw=1.5, c="0.2")

    ax.scatter(beta_values, fwhm_ratio, marker="o", c=snratios, s=25, edgecolors="None", cmap=cmap, norm=norm)

    ax.set_xlabel(r"$\beta$", fontsize=20)
    ax.set_ylabel(r"$\mathrm{FWHM}_\mathrm{Gaussian}/\mathrm{FWHM}_\mathrm{Moffat}$", fontsize=20)

    ax.set_xlim(1, 10)

    cax = fig.add_axes([0.85, 0.12, 0.02, 0.83])
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical')

    cb.set_label(r"S/N", fontsize=18)

    for axes in [ax, cax]:
        axes.tick_params(axis='both', which='both', width=1.5)

    fig.savefig("fwhm_ratio_gauss_moffat.png")

    plt.show()

if __name__ == "__main__":
    main()
