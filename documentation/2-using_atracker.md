# Using ATracker

ATracker tracks animals in video, provides an editor for setup and correction,
processes trajectories, and produces tracking overlays. You can work with a
structured project or use standalone functions on individual files.

## 1. Choose the guide for your task

| Guide | What you will do |
| --- | --- |
| [1. Installation](1-installing_atracker.md) | Install Python, ATracker and a notebook kernel. |
| [2. Using ATracker](2-using_atracker.md) | Choose a workflow and understand its capabilities and outputs. |
| [3. Folder tracking](3-folder_tracking.md) | Register videos, configure a project, tune detection and track batches. |
| [4. Standalone tracking](4-standalone_tracking.md) | Track, process and visualise individual videos without a project. |
| [5. Tracking editor](5-tracking_editor.md) | Draw setup geometry, correct coordinates and review candidate events. |
| [6. Processing](6-processing_data.md) | Filter trajectories, interpolate gaps and calculate movement measures. |
| [7. Image measurements](7-measuring_images.md) | Draw calibrated lengths and areas and save measurements to CSV. |

The Markdown guides are the current documentation. Older notebooks are preserved
in [archive](archive/README.md) as historical examples; their APIs and outputs may
be outdated.

## 2. Track videos in a project

A project keeps video metadata and settings together and is useful when files
need different ROIs, masks, object counts or threshold configurations.

```python
from atracker import ATracker

AT = ATracker("/path/to/existing/project")
AT.setup_files(fname_extract=False)
AT.get_files()
```

Initialising a project creates its folders and moves supported videos found in
the project root into `0originals`. Use a dedicated data folder. Setup registers
the videos in an overview spreadsheet; reopening `ATracker` loads saved settings.
Continue with [folder tracking](3-folder_tracking.md) to create backgrounds, draw
geometry and tune detection before running the batch.

## 3. Track an individual video

```python
from atracker import track_video

tracked_csv = track_video("/path/to/trial.mp4", save_settings=True)
```

The background is extracted automatically, the full frame is used by default,
and threshold calibration opens interactively. Optional ROI/mask drawing and a
saved `.atrk` setup bundle let you tailor or reuse the setup. See the
[standalone tutorial](4-standalone_tracking.md) for processing and visualisation.

## 4. Edit, review and measure

The [tracking editor](5-tracking_editor.md) supports project setup and correction:

```python
AT.editor(names="your_video", purpose="timepoints")
AT.event_checker("events.csv", modifier="complete", unreviewed_only=True)
```

Coordinate editing changes tracked points. Event checking instead reviews named
intervals and saves their bounds and decisions to the supplied event CSV.

For measurements on a folder of images:

```python
from atracker import measure_images

measurements = measure_images("/path/to/images", px_per_mm=10,
    variables=["length"], out_csv="measurements.csv")
```

Draw each measurement in the interface; results include pixel and calibrated
lengths, plus areas when applicable. See [image measurements](7-measuring_images.md).

## 5. Process and visualise

After tracking and any coordinate correction:

```python
AT.process()
AT.visualise(folder="processed")
```

Processing creates cleaned trajectories and movement measurements; visualisation
renders the results over the source video. Inspect representative outputs before
choosing interpolation and smoothing settings for the full experiment.

## What is saved?

| Workflow | Main records and outputs |
| --- | --- |
| Project | Overview spreadsheet, configuration file, named threshold YAML, setup images, tracked/processed CSVs and optional videos. |
| Standalone tracking | Tracked CSV and optional overlay video; `.atrk` setup bundle when requested. Keep run and processing parameters in your script. |
| Event review | Original event CSV with adjusted frames, `review` decisions and an optional modifier column. |
| Image measurements | Measurement CSV, with one row per image, plus a returned DataFrame. |

Detection uses contours from background subtraction or colour thresholding. It
can track multiple objects and extract optional shape/orientation features, but
identity continuity and detection quality still need inspection, especially during
occlusion. Processing cleans recorded trajectories; it cannot reconstruct
behaviour that was consistently detected incorrectly.
