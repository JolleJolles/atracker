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
- **Image measurement tools** for manual measurement of points and distances across image/video collections
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

## Documentation

The current guides use Markdown, with practical examples for each workflow:

1. [Installing ATracker](documentation/1-installing_atracker.md)
2. [Using ATracker: capabilities and workflow overview](documentation/2-using_atracker.md)
3. [Folder tracking](documentation/3-folder_tracking.md)
4. [Standalone tracking, processing and visualisation](documentation/4-standalone_tracking.md)
5. [Tracking editor: setup, correction and event review](documentation/5-tracking_editor.md)
6. [Processing tracking data](documentation/6-processing_data.md)
7. [Measuring images](documentation/7-measuring_images.md)

Older notebooks are preserved in [documentation/archive](documentation/archive/README.md)
as historical examples; use the guides above for current APIs.

## Installation

Install ATracker directly from GitHub:

```bash
pip install git+https://github.com/JolleJolles/atracker.git
```

All dependencies (including `pythutils`) are installed automatically.

## Usage

For a structured video project:

```python
from atracker import ATracker

AT = ATracker("/path/to/project")
AT.setup_files(fname_extract=False)
AT.get_bgfiles()
AT.editor(inds=[0], purpose="thresholding", threshtypes=["bw"])
AT.track(folder="originals", threshtype="bw", overwrite=False)
AT.process()
```

Configure ROI, masks, object counts and calibration as needed before tracking.
The [folder guide](documentation/3-folder_tracking.md) walks through setup.

For one video without a project:

```python
from atracker import track_video, process_video

tracked = track_video("/path/to/video.mp4", save_settings=True)
processed = process_video(tracked, videofile="/path/to/video.mp4", conv=0.25)
```

Here `conv` is the experiment's calibration in mm/pixel. See the
[standalone guide](documentation/4-standalone_tracking.md) for ROI/mask handling,
reusing setup and rendering an overlay video.

For image measurements:

```python
from atracker import measure_images

measure_images("/path/to/images", px_per_mm=10,
    variables=["length"], out_csv="measurements.csv")
```

## Compatibility

ATracker relies on a stable set of packages and is designed to work out of the box with **Python 3.9+**. Dependencies are declared in `setup.py`; keep a record of your working environment when reproducing an analysis.
