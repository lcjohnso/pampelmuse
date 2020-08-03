
Getting started
===============

Prerequisites
-------------
PampelMuse is written in *python3*. The current release has been tested under version 3.8.2, but may also run with
earlier python installations. In addition, several python-packages are required to run PampelMuse:

    * `numpy <http://www.numpy.org/>`_
    * `scipy <http://scipy.org/>`_
    * `astropy <http://www.astropy.org/>`_
    * `matplotlib <http://matplotlib.org/>`_
    * `pandas <http://pandas.pydata.org/>`_

In order to use the graphical user interface, the following package needs to be available:

    * `PyQt5 <http://www.riverbankcomputing.co.uk/software/pyqt/intro>`_

Finally, in case you plan to use the MAOPPY PSF model developed by [Fetick2019]_ for the analysis of adaptive optics
observations, make sure that the following package and its prerequisites are installed:

    * `MAOPPY <https://gitlab.lam.fr/lam-grd-public/maoppy>`_


Installation
------------

The development of PampelMuse is handled via a GIT repository hosted at https://gitlab.gwdg.de.

To install from the GIT repository, first cd into the folder were you want to install the code::

    cd <path_to_pampelmuse>

Then check out the GIT repository::

    git clone https://gitlab.gwdg.de/skamann/pampelmuse.git

This should download the most recent version of PampelMuse. In order to use the code afterwards, you can install it
using *pip*::

    pip install .

After a successful installation, the following command should succeed::

    PampelMuse --help


Quick-start guide
-----------------

The analysis with PampelMuse is dividing into several stages, each of which is assigned to a different routine. An
overview of the available routines is available :ref:`here <routines>`. In order to execute any of the available
routines, the calling sequence is as follows::

    PampelMuse <user-configuration-file> <ROUTINE>

Besides the name of the routine to be executed, the name of a :ref:`user configuration file <user-configuration>` in
plain JSON format must always be provided. It must at least contain information about the :ref:`integral field data
<integral-field-data>` to be analysed and the :ref:`reference source list <reference-catalogue>` to be used.

In the following, we assume that a properly reduced data cube names `DATACUBE_FINAL.fits` exists and that a valid
reference source list names `sources.csv` is available. In that case, the user configuration file (which we assume
to be stored as `user.json`) may look like this:

.. code-block:: json

    {
        "global": {
            "prefix": "DATACUBE_FINAL",
            },
        "catalog": {
            "name": "sources.csv"
            },
        "psf": {
            "profile": "moffat",
            "fwhm": 3.0
            }
    }

Besides providing information about the integral field data and the reference catalogue, this user configuration also
tells PampelMuse to use a Moffat profile to model :ref:`the PSF <psf>`, with an initial guess for the full width at
half maximum of :math:`3.0\,\text{spaxels}`.

The initialisation stage
^^^^^^^^^^^^^^^^^^^^^^^^

The first stage of the analysis consists in identifying the field of view of the integral field data relative to the
reference catalogue and to select the subset of available sources that will be included in the spectrum extraction
process. These tasks are performed by the :ref:`INITFIT routine <initfit>`. In our case, the calling sequence would
be::

    Pampelmuse user.json INITFIT

If successful, a FITS-file named `DATACUBE_FINAL.prm.fits` will be created, containing all information required by the
routines performing the following steps of the analysis chain. Further information about the PRM-file is available
:ref:`here <output-prm>`.

The fitting stage
^^^^^^^^^^^^^^^^^

In the next stage, PampelMuse tries to optimize the PSF and the parameters determining the :ref:`source coordinates
<coordinates>` as a function of wavelength. To this aim, it will go through the analysis layer-by-layer and optimize
the requested PSF and coordinate transformation parameters, using an iterative approach. These tasks are performed by
the :ref:`CUBEFIT routine <cubefit>`, which in our case would be called as::

    PampelMuse user.json CUBEFIT

.. note::
    A user configuration file still needs to be provided so that PampelMuse can locate the PRM-file. However, the
    current configuration is also stored in the primary header of the PRM-file and read in from there. If a user
    configuration file is provided that is incompatible with the information available in the PRM-file, PampelMuse
    will issue a warning and give preference to the information stored in the FITS header.

As this stage is quite computationally expensive, I recommend to enable spectral binning, e.g. by a factor of
:math:`50`, by including

.. code-block:: json

    {
        "layers": {
            "binning": 50
            }
    }

in the user configuration file. The idea behind this binning is that the parameters linked to the PSF and the
coordinate transformation will vary smoothly with wavelength, so that no crucial information is lost.

All the results obtained during the analysis are written back to the PRM-file.

The inspection stage
^^^^^^^^^^^^^^^^^^^^

In order to visualize the results of the fitting stage, a graphical user interface (GUI) is available that can be
called via::

    PampelMuse user.json POLYFIT

It is embedded in the :ref:`POLYFIT routine <polyfit>` and serves two purposes. First, it enables a visual check of
the PSF and coordinate transformation parameters determined with CUBEFIT. Second, it allows one to model their
wavelength dependencies using polynomials. The purpose of the latter is again twofold. First, a PSF and a coordinate
transformation will become available for each layer, even when spectral binning has been used. Second, high-frequency
variations caused by the finite signal-to-noise of the data will be smoothed out.

.. note::
    I recommend to fit polynomials to the spaxel coordinates of the sources, not to the parameters of the coordinate
    transformation. The reason is that there could be correlated variations of the latter with wavelength, which only
    have a small impact on the resulting spaxel coordinates. If such correlations are missed by the polynomial fits,
    the updated spaxel coordinates may be inaccurate.

After the polynomial fits to the source coordinates and the PSF parameters have been performed, the results can be
written back to the existing PRM-file.

The extraction stage
^^^^^^^^^^^^^^^^^^^^

With a model for the PSF and the spaxel coordinates in hand, we can extract the spectra of the resolvable sources
from the data cube. To do so, we again run the :ref:`CUBEFIT routine <cubefit>`::

    PampelMuse user.json CUBEFIT

If polynomial fits to the spaxel coordinates or the PSF parameters are available, PampelMuse will not try to optimize
them anymore. Hence the optimization happens for the source fluxes only. As this is a linear process, the time
required to process a layer of the data cube will be strongly reduced.

.. important::
    If layer binning has been enabled previously, it should be disabled again for the final `CUBEFIT` run. Otherwise the
    extracted spectra will have a lower resolution than in the data cube. The `layers|binning` parameter belongs to the
    subset of parameters that can be updated during the analysis.

The output stage
^^^^^^^^^^^^^^^^

All extracted spectra are stored as a two-dimensional FITS image in the extension `SPECTRA` of the PRM-file. They can
be saved as individual FITS files using the :ref:`GETSPECTRA routine <getspectra>`::

    Pampelmuse user.json GETSPECTRA

In order to create a filename for each individual output spectrum, a common prefix must be provided in the user
configuration file, via

.. code-block:: json

    {
        "getspectra": {
            "prefix": "<prefix>"
        }
    }

The output filenames are composed of the provided prefix, the source IDs available in the PRM-file, and the observing
date of the integral field data (using Julian dates with four decimal numbers). By default, they are stored in a folder
named `spectra/`. However, the name of the output folder can also be changed via the user configuration file.

