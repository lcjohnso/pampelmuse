
Configuration
=============

The configuration of PampelMuse is controlled via JSON [#f1]_ files. A valid configuration file always consists of
a number of sections, each of which deals with a different aspect of the analysis (e.g. the integral-field data, the
PSF, the reference catalogue ...). Each of the sections then specifies one or more configuration parameters and their
updated values.

Default configuration
---------------------

An overview of all the configuration parameters used by PampelMuse as well as their default values is available
:download:`here <../pampelmuse/config/data/pampelmuse.json>`.

.. note::
    When creating a :ref:`user configuration file <user-configuration>`, it is not required to include all sections or
    parameters listed in the default configuration file. Instead, one only needs to list those parameters that one
    wants to change.

.. note::
    Some parameters are left blank in the default configuration (like the prefix of the IFS data file to be
    analysed) and hence must be specified in the user configuration file.

.. _user-configuration:

Specifying a user configuration
-------------------------------

Each time one of the routines available as part of PampelMuse is called, a user-configuration file *must* be provided.
It tells PampelMuse where to find the integral-field data or the reference catalogue, which PSF profile to use, and so
on. The user-configuration file may only contain sections and keywords available in the default configuration. A
typical user configuration file looks like this:

.. code-block:: json

    {
        "global": {
            "prefix": "DATACUBE_FINAL",
            "cpucount": 4
            },
        "catalog": {
            "name": "sources.csv",
            "passband": "f606w"
            },
        "psf": {
            "profile": "moffat"
            }
    }

Throughout these documentation, we will use the following notation when mentioning a configuration parameter, providing
its section and name, type, and a brief explanation of its purpose.

global|prefix : string
       The prefix of the integral field data to be analysed.


.. rubric:: Footnotes

.. [#f1] See https://en.wikipedia.org/wiki/JSON for a basic introduction.