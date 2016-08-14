from distutils.core import setup

setup(name='PampelMuse',
      version='0.17.1',
      description='A code to extract stellar spectra from integral field spectroscopic data', 
      author='Sebastian Kamann',
      author_email='skamann@astro.physik.uni-goettingen.de',
      url='http://www.astro.physik.uni-goettingen.de/~skamann',
      package_dir={},
      packages=['pampelmuse', 'pampelmuse.bin', 'pampelmuse.config', 'pampelmuse.core', 'pampelmuse.core.psf',
                'pampelmuse.filters', 'pampelmuse.functions', 'pampelmuse.graphics', 'pampelmuse.routines'],
      scripts=['pampelmuse/bin/PampelMuse', 'pampelmuse/bin/access_configuration.py'],
      package_data={'pampelmuse.config': ['data/PMparams.xml', 'data/PMschema.xsd', 'data/PMstyle.xsl'],
                    'pampelmuse.filters': ['data/*']}
      )
