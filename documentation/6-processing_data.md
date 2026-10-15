# Processing tracking data

[Guide overview](2-using_atracker.md)

Use `AT.process()` to turn tracked CSV files into cleaned trajectories and movement measurements. This guide assumes you have already created your project instance, `AT`, and run tracking.

First inspect representative tracking videos in `AT.dirs["tracked"]`. If detection fails throughout a video, improve its background, masks or tracking thresholds and track it again. For local errors, missing positions or identity corrections, follow [Manual tracking and correction](5-tracking_editor.md) before processing.

## 1. Select files and run processing

Use `AT.get_files()` to select tracked CSVs, then pass the returned filenames to `AT.process()`:

```python
# Select one video by its overview index.
inds, files = AT.get_files(
    cdir="tracked",
    inds=[0],
    filetype=".csv",
    existonly=True,
)
AT.process(names=files)
```

Replace `0` with the overview index you want. `existonly=True` selects only CSVs that exist. `AT.process()` takes filenames through `names`; it does not take `inds`, `query` or `cats` directly.

To select by date instead, use the same workflow:

```python
inds, files = AT.get_files(
    cdir="tracked",
    query="date in ['260225', '260226']",
    filetype=".csv",
    existonly=True,
)
AT.process(names=files)
```

Use the date format stored in your overview. Add `cats=["setup", "date"]` to `get_files()` only if you want one representative per setup/date combination for testing, rather than every matching video.

The call above uses all default processing settings and skips existing processed outputs. Use `AT.process(names=files, overwrite=True)` when you want to regenerate those outputs. Input tracking CSVs are not overwritten.

Failed files are reported while the remaining files continue. `AT.process()` and
`AT.visualise()` return `False` if any task fails or the batch is interrupted.
Rerun processing with `overwrite=False` to skip completed outputs and retry failed
files. Processed CSVs replace existing outputs only after a successful write.
With threaded workers, Ctrl+C cancels pending tasks and waits for active files to
finish. A forcibly killed worker may leave a `.tmp` file; it is not treated as a
completed CSV.


Outputs are saved in `AT.dirs["processed"]` as `<video>_F.csv`. To process all tracked CSVs with defaults, use `AT.process()`; to keep an overview-based selection, keep passing `names=files`.

## 2. Copy and adjust the complete default settings

This dictionary contains **every parameter accepted by `AT.process()`**, with its current default value. Copy it into your notebook once:

```python
settings = dict(
    # Files and execution
    names=None,                  # None = all tracked CSVs
    pools=1,
    overwrite=False,

    # Frame range and calibration
    fulldata=True,
    convert=True,
    changefps=None,

    # Masks and trajectory filtering
    mask_margin=15,
    max_traj_gap=50,
    min_traj_len=10,
    roi_edge_margin=10,

    # Gap filling and smoothing
    interpolate=True,
    interp_gap_com=500,
    interp_gap_orient=100,
    smoothwin=10,
    orient_min_speed=1,

    # Measurements to compute
    compute_movement=True,
    compute_distances=True,
)
```

Then use your selection from step 1 and run processing:

```python
settings["names"] = files
AT.process(**settings)
```

`**settings` passes each dictionary entry as a named argument to `AT.process()`. For example, `settings["smoothwin"] = 11` has the same effect as specifying `smoothwin=11` directly in the call. It does not change the package's defaults or save settings in the project configuration; keep this dictionary in your notebook or script to record your choices.

To change a setting and rerun the selected files:

```python
settings["smoothwin"] = 11
settings["overwrite"] = True
AT.process(**settings)
```

When using the complete dictionary, change its entries instead of repeating the same parameter in the function call: `AT.process(smoothwin=11, **settings)` would supply `smoothwin` twice and raise an error.

### Existing edited files

The file selector currently prefers an `_E.csv` variant when present, but the processor's loader marks these files as empty. Use ordinary tracked CSVs for this workflow and check for old `_E.csv` files before processing; do not create that suffix for new manual corrections.

## 3. What processing does

The pipeline prepares the selected frame range and animal IDs, removes short isolated detections, and then:

1. Adds time and optionally samples at a lower frame rate.
2. Fills eligible centroid gaps, excluding gaps near masks or ROI exits.
3. Removes mask-adjacent detections and separates trajectories.
4. Cleans head/tail data where available and smooths coordinates.
5. Converts coordinates using the overview's `conv` factor.
6. Computes movement, orientation and distance variables when enabled.
7. Organises and saves the output, leaving unavailable measurements as missing values.

Interpolation estimates movement between observations; it cannot recover unseen turns, identity swaps or movement under cover. `interpolate=False` disables gap filling, but filtering and smoothing still run unless their settings are changed too.

`fulldata=True` uses `frame_start` and `frame_stop` from the overview, falling back to the video limits. `fulldata=False` uses the first and last recorded tracking frames. Both construct a frame grid within that range, using the detected tracking frame step; neither guarantees an observed position at every frame.

## 4. All parameters explained

The tables below cover every entry in the default dictionary above. You can also run `help(AT.process)` for the callable documentation.

### Files and execution

| Parameter | Default | Meaning |
| --- | --- | --- |
| `names` | `None` | Process all tracked CSVs, or pass the filenames returned by `AT.get_files()`. An empty list selects no files. Names may also be basenames or full paths. |
| `pools` | `1` | Number of workers. Start with one while tuning; the current implementation uses threads, including from notebooks. |
| `overwrite` | `False` | Skip existing processed outputs. Set `True` to regenerate them after changing settings. |

### Processing behaviour

| Parameter | Default | What to choose |
| --- | --- | --- |
| `fulldata` | `True` | Keep the planned trial window, including missing intervals. Use `False` to limit it to the observed tracking range. |
| `convert` | `True` | Use calibrated coordinates. Check each overview row's `conv` first; missing calibration falls back to 1. |
| `changefps` | `None` | Keep the original sampling initially. If reducing it, use a positive integer no higher than the video FPS and reassess temporal settings. |
| `interpolate` | `True` | Fill short, defensible gaps. Disable to examine missing detections without filling them. |
| `interp_gap_com` | `500` | Maximum centroid gap length in frame units. Reduce for animals whose paths can change quickly. |
| `interp_gap_orient` | `100` | Maximum head/tail and orientation gap length. Usually choose a shorter interval than for centroids. |
| `smoothwin` | `10` | Coordinate/orientation smoothing window in samples. Set `1` to disable. Odd windows of at least 7 are useful trials; short sections may remain unsmoothed. |
| `max_traj_gap` | `50` | Gap in frame units that separates trajectories. Choose how long a disappearance should break continuity. |
| `min_traj_len` | `10` | Minimum trajectory length; also controls removal of short isolated detection bursts. Lower if valid visits are very brief. |
| `mask_margin` | `15` | Pixel margin used to remove near-mask detections and prevent interpolation near the mask. |
| `roi_edge_margin` | `10` | Pixel margin for rejecting head/tail positions near the ROI boundary and avoiding interpolation across exits. |
| `orient_min_speed` | `1` | Minimum computed speed for using movement heading when head-derived orientation is missing. |
| `compute_movement` | `True` | Compute displacement, speed, acceleration, heading, orientation and turning variables. |
| `compute_distances` | `True` | Compute distances using available ROI, mask, wall, zone and point annotations. Disable if these are unnecessary. |

### Units and frame ranges

Spatial margins are always in **pixels**, even with `convert=True`. With a calibration in mm/pixel, converted coordinates and distances are in mm, while `speed` is in cm/s. `orient_min_speed` is compared with that speed column. The current speed calculation still divides by 10 when `convert=False`, so do not interpret that output directly as pixels/s. Angular differences such as `turnspeed` are per sampling step, rather than automatically scaled to degrees/s.

Frame-gap parameters use frame indices, while `smoothwin` uses retained samples. At full-rate tracking, 25 frames at 25 FPS represent one second. After frame skipping or `changefps`, do not assume every temporal parameter has the same conversion to seconds.

## 5. Choose settings for your experiment

Start with a few representative files: one with clean tracking, one with gaps, and one with refuge entries or ROI exits if relevant. Keep the input CSVs unchanged and change one setting at a time.

| What you observe | First adjustment | What to check afterward |
| --- | --- | --- |
| Long straight bridges across unobserved movement | Reduce `interp_gap_com`, or compare with `interpolate=False`. | Genuine turns and stops are not replaced by estimated paths. |
| Small coordinate jitter and exaggerated speed spikes | Try increasing `smoothwin` gradually, for example 7 → 11 → 15. | Real short movements and turns remain visible. |
| Turns become rounded or peak speed is suppressed | Reduce `smoothwin`; compare with `1`. | The trajectory follows the video without excessive jitter. |
| Tracking flickers at a refuge boundary | Inspect the mask, then adjust `mask_margin` in pixels. | A larger margin removes noise without removing valid entry/exit observations. |
| Too many separate trajectories | Inspect remaining gaps before increasing `max_traj_gap`. | Separate visits or disappearances are not incorrectly joined. |
| Valid brief visits disappear | Reduce `min_traj_len`. | Isolated noise bursts do not become accepted trajectories. |
| Orientation is unreliable while stationary | Raise `orient_min_speed` or shorten `interp_gap_orient`. | Heading is not treated as body orientation during pauses or backward movement. |

For example, for full-rate **25 FPS** tracking, a 12-frame centroid gap is about 0.48 seconds and an 11-sample smoothing window about 0.44 seconds. These are trial values, not universal defaults:

```python
# Adjust the complete settings dictionary from step 2.
settings.update(
    names=files,
    overwrite=True,
    changefps=None,
    interp_gap_com=12,
    interp_gap_orient=5,
    smoothwin=11,
    max_traj_gap=25,
)
AT.process(**settings)
```

Compare the result with the video and with a run using `interpolate=False, smoothwin=1`. Check missing positions, the number of trajectories per animal, and speed around gaps, turns and mask crossings. Some missing values should remain; a complete trajectory is not necessarily a more accurate one. Copy outputs you want to compare before another `overwrite=True` run.

Once satisfied, save the chosen `settings` in your analysis notebook or script and apply them consistently:

```python
settings["names"] = None  # all tracked CSVs, not just the pilot selection
AT.process(**settings)
```

Record any justified differences between experimental setups. Increase `pools` only after the single-file results are satisfactory.


## 6. Visualise the processed tracking

```python
AT.visualise(
    folder="processed",
    names=["your_video_name_F"],  # processed CSV basename, without .csv
    overwrite=True,
    startframe=1000, stopframe=1500,
    resize=1,                  # preserve resolution; 0.5 halves both dimensions
    trajlength=50,
    draw_centroid=True, draw_id=True, draw_orient=True,
    draw_scene=True,           # show all pieces of each zone
    writevideo=True, showvideo=False,
    pools=1,
)
```

The output is saved beside the processed CSV as `<name>_V.mp4`. If OpenCV cannot open the writer, ATracker uses FFmpeg H.264 at CRF 18. Resizing down and then enlarging the player window reduces apparent sharpness regardless of the encoder.

Heading arrows automatically detect whether the stored headings use upward- or downward-increasing y coordinates from the trajectory's pixel motion. For a very short or stationary selection where this cannot be inferred, pass `heading_y_up=True` for y-up headings or `heading_y_up=False` for image-coordinate headings. This does not change stored tracking data.

Each disconnected piece of a zone colour is displayed and included when calculating zone distances. Reprocess older CSVs to update their zone distances.

## 7. Output columns, units and position provenance

### Tracked CSVs

Raw `<video>.csv` files contain detections, not necessarily a row for every frame
and animal. Frame numbers are one-based, refer to the source video, and retain
their original values when tracking starts later or skips frames.

| Column | Meaning and units |
| --- | --- |
| `frame` | Source-video frame number. |
| `id` | Tracker identity label; colour methods can use named labels. |
| `cx`, `cy` | Centroid, in pixels relative to the ROI's top-left; x rightward, y downward. |
| `hx`, `hy`, `tx`, `ty` | Optional head/tail positions in the same pixel convention. |
| `angle` | Optional detector orientation in degrees; axial B/W angles can be ambiguous by 180 degrees. |
| `area` | Detected contour area in square pixels. |
| `aspect_ratio` | Detected contour aspect ratio, dimensionless. |
| `consmerged` | Optional internal contour merge indicator. |
| `position_source`, `head_source`, `tail_source` | Origins of the centroid, head and tail, respectively; see below. |
| `position_valid`, `head_valid`, `tail_valid` | Whether the corresponding x/y pair is finite in this file. |

Manual coordinate CSVs use the same pixel convention, but may omit detector
metadata such as area. Editor saves preserve source labels for unchanged points
and mark added, moved or deleted point pairs `manual`. Deleted rows are retained
as empty coordinate rows so deletion provenance is not lost. Changing only the
head does not label the centroid as manually corrected.

### Processed CSVs

Processed `<video>_F.csv` files retain pixel positions for overlays and add the
following fields when their calculations are enabled. Which rows are included
depends on `fulldata`, frame limits and resampling settings.

| Column | Meaning and units |
| --- | --- |
| `frame`, `time` | Original frame number and `frame / original_fps` in seconds; frame 1 is therefore not time zero. |
| `ID` | Overview animal identity if assigned, otherwise the tracker label. |
| `traj` | Valid trajectory segment within an individual; missing when no valid trajectory is retained. |
| `inmask` | 1 when no valid trajectory is retained, including missing/filtered data; not proof of physical occupancy inside a mask. |
| `cx`, `cy` | Processed centroid pixels, ROI-relative with y downward; may be interpolated/smoothed. |
| `fx`, `fy`, `tx`, `ty` | Processed head/tail pixels; raw `hx`, `hy` become `fx`, `fy`. |
| `cx_c`, `cy_c`, `fx_c`, `fy_c` | Converted centroid/head positions in mm with mm/pixel calibration; x ROI-relative, y flipped upward by default (`datflip=True`). |
| `displ`, `cumdispl` | Displacement per retained step and cumulative displacement per individual, in mm with calibration. |
| `speed`, `accel` | Linear speed in cm/s and acceleration in cm/s² with calibration, using the processing FPS. |
| `heading`, `orient` | Movement heading and estimated body orientation in degrees; heading uses `atan2(dx, dy)`. |
| `turnspeed`, `turnaccel` | Wrapped angular change per retained step and change of that angular change; not degrees/s or degrees/s². |
| `cumturn`, `abscumturn` | Cumulative signed and absolute angular changes in degrees, per individual. |
| `rdist` | Signed ROI-edge distance in mm with calibration; positive inside the ROI. |
| `mdist`, `wdist`, `mzdist` | Mask, wall and mask-zone distances when retained; polygon distances can be signed. |
| `mx`, `my`, `wx`, `wy` | Nearest saved mask/wall reference vertices in ROI-relative pixels. |
| `z1dist`, `z2dist`, ... | Distance to each palette zone, in mm with calibration; zero inside a zone. |
| `pt1dist`, `pt2dist`, ... | Distance to saved reference points, in mm with calibration. |

Optional fields can be absent. In particular, head coordinates are omitted when
B/W angle is used instead. Current output ordering does not retain every
intermediate processing column. Without calibration, physical units cannot be
inferred: displacement remains in pixel units and the legacy speed formula still
divides by 10. Frame skipping/resampling also affects how step differences relate
to time; inspect the retained frame interval before interpreting derivatives.

### Source labels and validity

| Source value | Meaning |
| --- | --- |
| `observed` | A coordinate supplied by this version's automated tracker. |
| `manual` | A point added, moved or deleted in the coordinate editor. |
| `interpolated` | A missing point filled by coordinate interpolation during processing. |
| `missing` | No coordinate was supplied for that point type. |
| `unknown` | A coordinate exists in an older/imported file without recorded provenance. |

Sources describe how positions were obtained; they are not quality/confidence
scores. A filtered observed position can remain `position_source="observed"` with
`position_valid=False`. A manually deleted point is `manual` with validity false.
An interpolated replacement is labelled `interpolated`. Smoothing can change
existing values without changing their source label. Head/tail origins are
tracked separately; orientation estimates and individual IDs do not receive a
provenance label in this update.

Legacy files cannot reveal whether their points were originally automated or
manual, so they use `unknown` until a new edit or interpolation establishes the
source. Validity is recalculated from the final saved coordinate pairs.

## What is saved?

Processed data are written as `<video>_F.csv`; input tracking CSVs are retained.
Coordinates in pixels remain available for overlays, with converted coordinates
and movement measures added when enabled. `overwrite=True` regenerates existing
processed outputs after you change settings or correct coordinates.

For processing an individual CSV without a project, use `process_video()` as shown
in the [standalone guide](4-standalone_tracking.md). Return to the
[guide overview](2-using_atracker.md) to choose another workflow.
