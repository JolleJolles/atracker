#! /usr/bin/env python

from setuptools import setup, find_packages
import sys

exec(open("atracker/__version__.py").read())

DESCRIPTION="""A python module for automated animal tracking"""
DISTNAME="atracker"
MAINTAINER="Jolle Jolles"
MAINTAINER_EMAIL="j.w.jolles@gmail.com"
URL="https://github.com/JolleJolles"
DOWNLOAD_URL=""

with open("README.md") as f:
    readme = f.read()

if __name__ == "__main__":

    setup(name=DISTNAME,
          author=MAINTAINER,
          author_email=MAINTAINER_EMAIL,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          long_description=readme,
          long_description_content_type="text/markdown",
          url=URL,
          install_requires=["pythutils",
                            "pyyaml",
                            "future",
                            "numpy",#==1.2.6"
                            "pandas",#==1.3.5",
                            "openpyxl",
                            "xlrd",
                            "xlwt",
                            "scipy",
                            "pathos",
                            "PyQt5",
                            "python_box",
                            "screeninfo",
                            "shapely",
                            "Pillow==8.4.0",
                            "scikit-learn",
                            "opencv-contrib-python",#==3.4.11.45",
                            "localconfig==0.4.2; python_version>='2' and python_version<'3'",
                            "localconfig==1.1.1; python_version>='3'",
                            "pirecorder"],
          entry_points={"console_scripts": [],},
          download_url=DOWNLOAD_URL,
          version=__version__,
          license="License :: OSI Approved :: Apache Software License",
          platforms=["Windows", "Linux", "Mac OS-X"],
          packages=find_packages(),
          include_package_data=True,
          classifiers=[
                     "Intended Audience :: Science/Research",
                     "Programming Language :: Python :: 3",
                     "License :: OSI Approved :: Apache Software License",
                     "Topic :: Scientific/Engineering :: Visualization",
                     "Topic :: Scientific/Engineering :: Image Recognition",
                     "Topic :: Scientific/Engineering :: Information Analysis",
                     "Topic :: Multimedia :: Video",
                     "Operating System :: POSIX",
                     "Operating System :: Unix",
                     "Operating System :: MacOS"],
          )
