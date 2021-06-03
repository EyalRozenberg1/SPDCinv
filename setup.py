from distutils.core import setup
from setuptools import find_packages

import os


setup(name='spdc_inv',
      version='1.0',
      author='Eyal Rozenberg',
      author_email='eyalr@campus.technion.ac.il',
      url='https://github.com/EyalRozenberg1/spdc_inv',
      license='',
      classifiers=[
          'Programming Language :: Python :: 3.7',
      ],
      package_dir={'spdc_inv': os.path.join(os.getcwd(), 'src')},
      packages=find_packages(exclude=[]),
      install_requires=[
            'numpy',
            'scipy',
            'matplotlib',
      ],
      )
