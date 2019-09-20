import os
from distutils.core import setup
from setuptools import find_packages

setup(name='profit',
      version='0.1dev',
      description='Protein Fitness: Determine protein evolutionary fitness',
      author = 'Ayush Karnawat',
      author_email='ayushkarnawat@gmail.com',
      packages=find_packages(),
      license='MIT',
      url='https://github.com/ayushkarnawat/profit',
      long_description=open('README.md', encoding='utf-8').read(),
      long_description_content_type='text/markdown'
    )