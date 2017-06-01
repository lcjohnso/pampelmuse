from distutils.core import setup

setup(
    name='PampelMuse',
    version='0.90.0',
    description='A code to extract stellar spectra from integral field spectroscopic data',
    author='Sebastian Kamann',
    author_email='skamann@astro.physik.uni-goettingen.de',
    url='http://www.astro.physik.uni-goettingen.de/~skamann',
    package_dir={},
    packages=['pampelmuse', 'pampelmuse.config', 'pampelmuse.core', 'pampelmuse.filters', 'pampelmuse.graphics',
              'pampelmuse.instruments', 'pampelmuse.psf', 'pampelmuse.routines', 'pampelmuse.utils'],
    scripts=['bin/PampelMuse', ],
    package_data={'pampelmuse.config': ['data/pampelmuse.json', ],
                  'pampelmuse.filters': ['data/*', ]},
    requires=['numpy', 'scipy', 'astropy', 'matplotlib', 'pandas', 'PyQt5']
)
