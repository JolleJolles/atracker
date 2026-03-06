#! /usr/bin/env python

from setuptools import setup, find_packages

# Read version info
version = {}
with open("atracker/__version__.py") as f:
    exec(f.read(), version)

# Project metadata
DESCRIPTION = "A flexible Python toolkit for animal tracking"
DISTNAME = "atracker"
MAINTAINER = "Jolle Jolles"
MAINTAINER_EMAIL = "j.w.jolles@gmail.com"
URL = "https://github.com/JolleJolles"
DOWNLOAD_URL = ""

# Read long description from README
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

# Setup
if __name__ == "__main__":
    setup(
        name=DISTNAME,
        version=version["__version__"],
        author=MAINTAINER,
        author_email=MAINTAINER_EMAIL,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        long_description=long_description,
        long_description_content_type="text/markdown",
        url=URL,
        download_url=DOWNLOAD_URL,
        license="Apache-2.0",
        platforms=["Windows", "Linux", "Mac OS-X"],
        packages=find_packages(),
        include_package_data=True,
        install_requires=[
            # Core scientific stack
            "numpy",
            "scipy",
            "pandas",
            "opencv-contrib-python",
            "imageio[ffmpeg]",
            "shapely",

            # Machine learning utilities
            "scikit-learn",  # needed for pairwise_distances

            # Plotting
            "matplotlib",    # needed for barcode visualization
            
            # File handling
            "openpyxl",      # read/write .xlsx files
            "screeninfo",    # screen resolution detection

            # UI
            "PyQt5",

            # Utility packages
            "python_box",    # dot-access for dict-like configs
            "localconfig>=1.1.4,<2",  # INI-style config support
            "pythutils",     # Jolle's utility library
        ],
        entry_points={
            "console_scripts": [],
        },
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
            "Operating System :: MacOS",
        ],
    )
