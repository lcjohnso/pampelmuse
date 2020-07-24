
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


