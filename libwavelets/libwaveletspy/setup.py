# -*- coding: utf-8 -*-
"""

"""

from future import standard_library
standard_library.install_aliases()

from distutils.core import setup


setup(name='libwaveletspy',
      # version=__version__,
      version='0.1.0',
      author='Jonas Adler & Julian Moosmann',
      author_email='jonasadl@kth.se',
      url='https://gits-14.sys.kth.se/LCR/libwavelets',
      description='Python bindings for the libwavelets library',
      license='GPLv3',
      packages=['libwaveletspy'],
      package_dir={'libwaveletspy': '.'},
      package_data={'libwaveletspy': ['*.*']}, requires=['scipy', 'numpy',
                                                         'matplotlib', 'numba',
                                                         'future'])
