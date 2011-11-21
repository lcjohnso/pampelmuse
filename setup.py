from distutils.core import setup

setup(name='PampelMuse',
      version='0.1.0',
      author='Sebastian Kamann',
      author_email='skamann@aip.de',
      url='https://subversion.assembla.com/svn/kamann-code.2/trunk/dist',
      package_dir={},
      packages=['pampelmuse', 'pampelmuse.core', 'pampelmuse.config',
                'pampelmuse.functions', 'pampelmuse.graphics',
                'pampelmuse.scripts'],
      scripts=['pampelmuse/scripts/pampelmuse.py'],
      data_files=[('config', ['config/PMparams.xml', 'config/PMschema.xsd',
                              'config/PMstyle.xsl']
                   )]
      )
