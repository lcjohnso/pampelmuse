from setuptools import setup, find_packages

setup(
    name='PampelMuse',
    version='1.0rc1',
    description='A code to extract stellar spectra from integral field spectroscopic data',
    author='Sebastian Kamann',
    author_email='s.kamann@ljmu.ac.uk',
    url='http://www.astro.ljmu.ac.uk/~ariskama',
    package_dir={},
    packages=find_packages(),
    scripts=['bin/PampelMuse', ],
    package_data={'pampelmuse.config': ['data/pampelmuse.json', ], 'pampelmuse.filters': ['data/*', ]},
    install_requires=['numpy', 'scipy', 'astropy', 'matplotlib', 'pandas', 'PyQt5', 'tqdm', 'maoppy>=1.2.0']
)
