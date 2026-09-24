# ATracker

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18889019.svg)](https://doi.org/10.5281/zenodo.18889019)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)

**A flexible Python toolkit for animal tracking.**

## Citation

If you use ATracker in your research, please cite:

Jolles JW (2026). *ATracker: A flexible Python toolkit for animal tracking*. Zenodo. https://doi.org/10.5281/zenodo.18889019


## Overview

**ATracker** provides tools for the full workflow of behavioural tracking experiments, from preparing videos and defining tracking regions to running automated or manual tracking and generating trajectory data for analysis.

ATracker is particularly well suited for behavioural experiments where flexible annotation, quality control, and integration with custom analysis pipelines are essential. Relying on `opencv`, it is optimized for cross-platform use and runs on **macOS**, **Windows**, and **Linux**.

ATracker is developed and actively maintained by **Dr Jolle Jolles**, senior researcher at **CEAB-CSIC**.

## Key Features

- **Interactive annotation tools** for defining ROIs, masks, zones, and walls
- **Flexible thresholding system** supporting grayscale and HSV colour tracking
- **Automated contour-based tracking** with centroid, orientation, skeleton, and shape features
- **Manual tracking and correction interface** for efficient annotation and quality control
- **Post-processing pipeline** for trajectory smoothing, interpolation, and distance calculations
- **Batch measurement tools** for manual measurement of points and distances across image/video collections
- **Batch processing framework** for large collections of videos
- **Project organisation tools** for managing tracking experiments and outputs
- **Config-based workflows** for reproducible tracking pipelines
- Support for **multi-object tracking**, **colour tracking**, and experimental **barcode detection**

## When is ATracker useful?

**ATracker** is particularly useful when:

- experiments require custom masks, regions, or zones
- tracking parameters must be tuned interactively per video
- manual correction or annotation of trajectories is needed
- multiple tracking modes are used within the same experiment
- large video datasets must be processed in a structured, reproducible workflow

## Project Structure

- The **`documentation/` folder** provides full instructions for:
  - Installing prerequisites and setting up your Python environment
  - Platform-specific guidance for **macOS**, **Windows**, and **Linux**
  - Using ATracker to set up a project workspace, define tracking settings, and run analysis
  - [Tuning automated tracking](documentation/3-tracking.md), covering thresholding, gamma, area/aspect-ratio filters, flicker rejection and the rolling size filter
  - [Processing and interpreting tracking results](documentation/4-processing_data.md), including how to choose processing parameters
  - [Manual tracking and correction](documentation/5-manual_tracking.md), with project and standalone examples

- The **`tutorial/` folder** contains a curated set of example videos to learn ATracker's capabilities. It is accompanied by a compact Jupyter notebook (`atracker_tutorial_compact.ipynb`) that demonstrates the key steps for:
  - Setting up the workspace
  - Drawing masks, ROIs, zones, and walls
  - Setting thresholding parameters
  - Running tracking in various modes

## Installation

Install ATracker directly from GitHub:

```bash
pip install git+https://github.com/JolleJolles/atracker.git
```

All dependencies (including `pythutils`) are installed automatically.

## Usage

The core workflow uses the `ATracker` class, which manages a project folder:

```python
from atracker import ATracker

AT = ATracker("/path/to/project")   # initialise project
AT.setup_files()                    # register videos and extract metadata
AT.set_interactive()                # interactively set ROI, mask, thresholds
AT.track()                          # run automated tracking
```

Post-process tracked data with `Processor`:

```python
from atracker import Processor

P = Processor("/path/to/project")
P.process()
```

For manual annotation or visual quality control, use `manual_tracker` and `batch_measure`:

```python
from atracker import manual_tracker, batch_measure

manual_tracker("/path/to/video.mp4")           # interactive manual tracking
batch_measure("/path/to/images", output="measures.csv")  # batch measurement
```

## Compatibility

ATracker relies on a stable set of packages and is designed to work out of the box with **Python 3.9+**. Due to ongoing updates in upstream libraries (e.g., `opencv`, `PyQt5`), we recommend using the versions specified in `setup.py` for a fully tested environment.
