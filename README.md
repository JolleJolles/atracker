# ATracker

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18889019.svg)](https://doi.org/10.5281/zenodo.18889019)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)

**A flexible Python toolkit for animal tracking.**

## Citation

Jolles JW (2026). *ATracker: A flexible Python toolkit for animal tracking*. Zenodo.
https://doi.org/10.5281/zenodo.18889019

## Overview

ATracker combines video setup, automated and manual tracking, interactive
correction, event review and trajectory processing. Use a project overview for
collections of recordings, or track and process individual videos independently.
It is developed by **Dr Jolle Jolles**, senior researcher at **CEAB-CSIC**.

Version **2.0** brings together **over 70 changes since 1.0**, spanning tracking,
the interactive interface, processing, visualisation and documentation. The editor
now brings setup and coordinate correction into one workflow, individual videos
can be tracked without a project, and batch tracking is more reliable. The release
also adds event review, configurable body measurements and support for tracking
incoming recordings during experiments. See the [changelog](CHANGELOG.md) for the
release summary and [Upgrading across breaking changes](documentation/8-upgrading.md) for changes
to existing scripts.

## Features

- **Flexible animal tracking:** background subtraction or HSV colour detection,
  single or multiple animals, and a choice of blob-centre tracking or centreline,
  head/tail and body-shape measurements.
- **Interactive setup and annotation:** define tracking regions, masks, zones and
  calibration; tune detection, track manually, correct coordinates and review events.
- **Trajectory processing and visualisation:** filter and smooth tracks, interpolate
  gaps, calculate movement and spatial measures, and inspect tracking overlays.
- **Individual videos or whole experiments:** use standalone functions or organised
  projects, run batches in parallel, reuse settings and watch for incoming recordings.
- **Measurements and analysis outputs:** measure lengths and outlines on images,
  and export coordinates, movement measures and geometry for downstream analysis.

Barcode handling remains experimental; modular detector backends and the
instance-based project redesign are planned, not released capabilities.

## Installation

Install from GitHub in a virtual environment:

```bash
python -m pip install git+https://github.com/JolleJolles/atracker.git
```

See [installation](documentation/1-installing_atracker.md) for environment setup,
notebooks and troubleshooting. Python 3.9 or later is required. The GUI uses
PyQt5 and OpenCV; runtime dependencies are declared in `setup.py`.

## Quick start

For a video project:

```python
from atracker import ATracker

AT = ATracker("/path/to/project")
AT.setup_files(fname_extract=False)
AT.get_bgfiles()
AT.editor(inds=[0], threshtypes=["bw"])
AT.set_config(body_mode="centroid", show_tracking=False)
AT.track(folder="originals", threshtype="bw", overwrite=False)
AT.process()
```

Initialisation moves supported videos in the project root into `0originals`.
Use a dedicated project folder. Configure ROI, masks, object counts and calibration
as needed; [folder tracking](documentation/3-folder_tracking.md) covers the workflow.

For centreline measurements, independently of B/W or colour detection:

```python
AT.set_config(body_mode="skeleton", head_rule="near_widest",
    skeleton_points=10, save_body=True)
```

In centroid mode `cx/cy` is the blob centroid; in skeleton mode it is the centreline
midpoint. Head/tail estimates use a configurable heuristic. See the
[output reference](documentation/6-processing_data.md#7-output-columns-units-and-position-provenance)
for coordinate conventions, units, validity and raw geometry files.

For a video without a project:

```python
from atracker import track_video, process_video

tracked = track_video("/path/to/video.mp4", body_mode="centroid", save_settings=True)
processed = process_video(tracked, videofile="/path/to/video.mp4", conv=0.25)
```

Here `conv` is calibration in mm/pixel. See [standalone tracking](documentation/4-standalone_tracking.md)
for setup reuse and visualisation.

For candidate review in a project:

```python
AT.event_checker("jump_candidates.csv", modifier="complete", unreviewed_only=True)
```

For new recordings sharing a tested setup:

```python
AT.watch_folder("/path/to/incoming", template_ind=0)
```

For image measurements:

```python
from atracker import measure_images

measure_images("/path/to/images", px_per_mm=10,
    variables=["length"], out_csv="measurements.csv")
```

## Documentation

1. [Installing ATracker](documentation/1-installing_atracker.md)
2. [Capabilities and workflow overview](documentation/2-using_atracker.md)
3. [Folder tracking, batches and incoming recordings](documentation/3-folder_tracking.md)
4. [Standalone tracking, processing and visualisation](documentation/4-standalone_tracking.md)
5. [Tracking editor and event review](documentation/5-tracking_editor.md)
6. [Processing, output columns and units](documentation/6-processing_data.md)
7. [Measuring images](documentation/7-measuring_images.md)
8. [Upgrading across breaking changes](documentation/8-upgrading.md)

See the [changelog](CHANGELOG.md) for release changes and the roadmap.
Archived notebooks are [historical examples](documentation/archive/README.md),
not current tutorials. Report problems through [GitHub Issues](https://github.com/JolleJolles/atracker/issues).
