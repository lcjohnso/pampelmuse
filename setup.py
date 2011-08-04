from distutils.core import setup

setup(name='PampelMuse',
      version='0.0',
      author='Sebastian Kamann',
      author_email='skamann@aip.de',
      package_dir={'PampelMuse': 'src'},
      packages=['PampelMuse','PampelMuse.core', 'PampelMuse.functions',
                'PampelMuse.graphics', 'PampelMuse.config'],
      scripts=['src/scripts/CubeFit', 'src/scripts/InitCubeFit']
      )
