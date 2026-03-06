# ATracker

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18889019.svg)](https://doi.org/10.5281/zenodo.18889019)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9-blue.svg)]()

**A flexible Python toolkit for animal tracking.**

## Citation

If you use ATracker in your research, please cite:

Jolles JW (2026). *ATracker: A flexible Python toolkit for animal tracking*. Zenodo. https://doi.org/10.5281/zenodo.18889019


## Overview

**Atracker** provides tools for the full workflow of behavioural tracking experiments, from preparing videos and defining tracking regions to running automated or manual tracking and generating trajectory data for analysis.

ATracker is particularly well suited for behavioural experiments where flexible annotation, quality control, and integration with custom analysis pipelines are essential. Relying on `opencv`, it is optimized for cross-platform use and runs on **macOS**, **Windows**, and **Linux**.

ATracker is developed and actively maintained by **Dr Jolle Jolles**, senior researcher at **CEAB-CSIC**.

## Key Features

- **Interactive annotation tools** for defining ROIs, masks, zones, and walls  
- **Flexible thresholding system** supporting grayscale and HSV colour tracking  
- **Automated contour-based tracking** with centroid, orientation, skeleton and shape features  
- **Manual tracking and correction interface** for efficient annotation and quality control  
- **Batch processing framework** for large collections of videos  
- **Project organisation tools** for managing tracking experiments and outputs  
- **Config-based workflows** for reproducible tracking pipelines  
- Support for **multi-object tracking**, **colour tracking**, and experimental **barcode detection**

## When is ATracker useful?

**ATracker** is particularly useful when:

- experiments require custom masks, regions or zones
- tracking parameters must be tuned interactively
- manual correction of trajectories is needed
- multiple tracking modes are used within the same experiment
- large video datasets must be processed in a structured workflow

## Project Structure

- The **`documentation/` folder** provides full instructions for:
  - Installing prerequisites and setting up your Python environment  
  - Platform-specific guidance for **macOS**, **Windows**, and **Linux**  
  - Using ATracker to set up a project workspace, define tracking settings, and run analysis  
  - Processing and interpreting tracking results  

- The **`tutorial/` folder** contains a curated set of example videos to learn ATracker's capabilities. It is accompanied by a compact Jupyter notebook (`atracker_tutorial_compact.ipynb`) that demonstrates the key steps and options for:
  - Setting up the workspace  
  - Drawing masks, ROIs, zones, and walls  
  - Setting thresholding parameters  
  - Running tracking in various modes  

## Installation

ATracker can be installed directly from the repository:

`pip install git+https://github.com/JolleJolles/atracker.git`

and then be run with:

```
import atracker as AT

AT.set_interactive(...)
```

## Compatibility

ATracker relies on a stable set of packages and is designed to work out of the box with **Python 3.9**. Due to the ongoing updates in upstream libraries (e.g., `opencv`, `PyQt5`, etc.), we recommend using the versions specified in `setup.py`.