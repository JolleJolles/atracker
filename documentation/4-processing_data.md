# Processing tracking data

Use `AT.process()` to turn tracked CSV files into cleaned trajectories and movement measurements. This guide assumes you have already created your project instance, `AT`, and run tracking.

First inspect representative tracking videos in `AT.dirs["tracked"]`. If detection fails throughout a video, improve its background, masks or tracking thresholds and track it again. For local errors, missing positions or identity corrections, follow [Manual tracking and correction](5-manual_tracking.md) before processing.

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

## Existing edited files

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
