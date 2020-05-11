from distutils.core import setup
from setuptools import find_packages

setup(name="profit",
      version="0.1.0",
      description="Profit: Determine evolutionary protein fitness",
      long_description=open("README.md", encoding="utf-8").read(),
      author="Ayush Karnawat",
      author_email="ayushkarnawat@gmail.com",
      url="https://github.com/ayushkarnawat/profit",
      packages=find_packages(),
      license="MIT",
      long_description_content_type="text/markdown")
