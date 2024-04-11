from setuptools import setup, find_packages

setup(
    name='PampelMuse',
    version='1.0.6',
    description='A code to extract stellar spectra from integral field spectroscopic data',
    author='Sebastian Kamann',
    author_email='s.kamann@ljmu.ac.uk',
    url='http://www.astro.ljmu.ac.uk/~ariskama',
    package_dir={},
    packages=find_packages(),
    scripts=['bin/PampelMuse', ],
    package_data={'pampelmuse.config': ['data/pampelmuse.json', ], 'pampelmuse.filters': ['data/*', ]},
    install_requires=['numpy', 'scipy', 'astropy>=4.3.1', 'matplotlib>=3.5', 'pandas>1.3', 'tqdm', 'mpdaf'],
    extras_require={'nfm': 'maoppy>=1.5.0', 'gui': 'pyqt'}
)
