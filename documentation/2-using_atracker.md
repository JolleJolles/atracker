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

Body detail is separate from detection method: use `body_mode="centroid"` for a
blob centre or `body_mode="skeleton"` for a centreline midpoint and tips, with a
configurable head rule. Optional geometry files retain sampled centrelines and
contours; see [body settings](3-folder_tracking.md#9-other-tracking-checks-and-outputs).


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

## 6. Export geometry for later analysis

Save portable geometry alongside selected original videos:

```python
geometry_files = AT.export_geometry(names=["your_video"])
```

This writes `0originals/your_video.geometry.json`. Multi-region videos use
`your_video_R2.geometry.json`, matching their tracking output names. You can also
select with `inds`, `query` or `cats`; omit selectors to export all project rows not marked excluded. Existing exports are retained unless `overwrite=True`.

```python
geometry_files = AT.export_geometry(inds=[0, 1],
    zone_names={1: "refuge", 2: "open_water"}, overwrite=True)
```

`zone_names` maps the numeric colour-palette IDs used by processing to your labels.
Unmapped zones use names such as `zone_1`. Use `output="/path/to/geometry"` to choose
another directory.

Each JSON is a GeoJSON FeatureCollection with additional metadata:

| Contents | Convention |
| --- | --- |
| ROI, masks, walls, mask zones and zones | Ordered polygon outlines, including separate parts and holes. |
| Coordinates | Full-frame pixels: origin top left, x rightward, y downward. ROI offsets have not been subtracted. |
| Calibration | `calibration.mm_per_pixel`; null if no calibration is set. |
| Metadata | Video/region, image dimensions, FPS, frame limits and animal count. |
| Source and associated files | Relative references to setup images and expected tracked/processed CSVs and tracking video. These output paths need not exist yet. |
| Provenance | Outlines extracted from the saved raster images; not the editor's original drawing points. |

By default, pixel contours are retained without extra simplification. Set
`epsilon=1` to simplify outlines with a one-pixel tolerance. Thin or single-pixel
features with no polygon area are omitted. Mask/wall regions follow dark pixels
at grayscale <=10; zones use the processing colour-palette thresholds. No mask
expansion/erosion from later processing is applied. ROI polygons represent crop
bounds (right/bottom exclusive); image-derived outlines follow boundary pixel
centres, so their areas are approximations to the raster regions.

### Read the export in R

```r
info <- jsonlite::fromJSON("your_video.geometry.json")
regions <- sf::st_read("your_video.geometry.json", quiet = TRUE)
regions <- sf::st_set_crs(regions, NA)
```

The coordinates are pixels, not longitude/latitude; remove any geographic CRS
assigned by the GeoJSON reader before spatial calculations. Pixel distances can
be multiplied by `info$calibration$mm_per_pixel` to obtain millimetres. Points must
use the same full-frame coordinate convention. For a point inside a polygon,
distance to that polygon is zero; distance to its boundary measures how far
inside it lies. Choose the boundary for wall/edge distance questions.

### Images or vector geometry?

Tracking, processing and the editor currently continue to read the saved images.
The JSON is an analysis snapshot: regenerate it with `overwrite=True` after
changing setup images or calibration. Editing JSON does not update tracking.

A future vector-first workflow could store the original drawing coordinates as
the authoritative geometry and generate mask/zone images from them. That would
retain precise shapes, names and overlaps while keeping raster images convenient
for detection. It requires coordinated editor and loader changes; this export
adds portable geometry without changing the existing image workflow.

Tracking and processing outputs record centroid/head/tail provenance separately.
See the [output-column and units reference](6-processing_data.md#7-output-columns-units-and-position-provenance)
for observed, manual, interpolated, missing and legacy-unknown positions.

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
