from distutils.core import setup

setup(
    name='PampelMuse',
    version='0.100.1',
    description='A code to extract stellar spectra from integral field spectroscopic data',
    author='Sebastian Kamann',
    author_email='s.kamann@ljmu.ac.uk',
    url='http://www.astro.physik.uni-goettingen.de/~skamann',
    package_dir={},
    packages=['pampelmuse', 'pampelmuse.config', 'pampelmuse.core', 'pampelmuse.filters', 'pampelmuse.graphics',
              'pampelmuse.instruments', 'pampelmuse.psf', 'pampelmuse.psf.profiles', 'pampelmuse.routines',
              'pampelmuse.utils'],
    scripts=['bin/PampelMuse', ],
    package_data={'pampelmuse.config': ['data/pampelmuse.json', ],
                  'pampelmuse.filters': ['data/*', ]},
    requires=['numpy', 'scipy', 'astropy', 'matplotlib', 'pandas', 'PyQt5']
)
