# -*- coding: utf-8 -*-
"""

"""

requires = """
future
astra
numpy
"""

from future import standard_library
standard_library.install_aliases()
from distutils.core import setup


setup(name='astra',
      # version=__version__,
      version='0.1.0',
      author='Julian Moosmann',
      author_email='moosmann@kth.se',
      url='https://gits-14.sys.kth.se/LCR/astra',
      description='Python bindings for ASTRA Toolbox',
      license='GPLv3',
      install_requires=[requires, 'RL', 'astra'],
      packages=['astra'],
      package_dir={'astra': '.'},
      package_data={'astra': ['*.*']})
