# Processing tracking data

Use `AT.process()` to turn tracked CSV files into cleaned trajectories and movement measurements. This guide assumes you have already created your project instance, `AT`, and run tracking.

First inspect representative tracking videos in `AT.dirs["tracked"]`. If detection fails throughout a video, improve its background, masks or tracking thresholds and track it again. For local errors, missing positions or identity corrections, follow [Manual tracking and correction](5-manual_tracking.md) before processing.

## Start with one file

```python
from pathlib import Path

tracked_files = sorted(Path(AT.dirs["tracked"]).glob("*.csv"))
print([p.name for p in tracked_files])

# Replace this with a tracked CSV name from the list above.
selected = ["your_video.csv"]
AT.process(names=selected, pools=1, overwrite=True)
```

`names` accepts tracked CSV names, full paths or basenames without `.csv`. It does not select processed output files. Omitting `names` processes all available tracked CSVs. `overwrite=True` replaces existing processed results; it does not overwrite the input tracking CSVs.

Outputs are written to `AT.dirs["processed"]` as `<video>_F.csv`:

```python
import pandas as pd

result = pd.read_csv(Path(AT.dirs["processed"]) / "your_video_F.csv")
print(result.head())
print(result.columns.tolist())
```

The file selector currently prefers an `_E.csv` variant when present, but the processor's loader marks these files as empty. Use ordinary tracked CSVs for this workflow and check for old `_E.csv` files before processing; do not create that suffix for new manual corrections.

## What processing does

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

## Parameters you will use most

Defaults below match the current `AT.process()` signature. For the full callable documentation, run `help(AT.process)`.

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
| `pools` | `1` | Number of workers. Start with one while tuning; the current implementation uses threads, including from notebooks. |

Spatial margins are always in **pixels**, even with `convert=True`. With a calibration in mm/pixel, converted coordinates and distances are in mm, while `speed` is in cm/s. `orient_min_speed` is compared with that speed column. The current speed calculation still divides by 10 when `convert=False`, so do not interpret that output directly as pixels/s. Angular differences such as `turnspeed` are per sampling step, rather than automatically scaled to degrees/s.

Frame-gap parameters use frame indices, while `smoothwin` uses retained samples. At full-rate tracking, 25 frames at 25 FPS represent one second. After frame skipping or `changefps`, do not assume every temporal parameter has the same conversion to seconds.

## Choose settings for your experiment

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
settings = dict(
    fulldata=True,
    convert=True,          # overview conv must be calibrated in mm/pixel
    changefps=None,
    interpolate=True,
    interp_gap_com=12,
    interp_gap_orient=5,
    smoothwin=11,
    max_traj_gap=25,
    min_traj_len=10,
    mask_margin=15,        # inspect this distance on your own video
    roi_edge_margin=10,
)

AT.process(names=selected, pools=1, overwrite=True, **settings)
```

Compare the result with the video and with a run using `interpolate=False, smoothwin=1`. Check missing positions, the number of trajectories per animal, and speed around gaps, turns and mask crossings. Some missing values should remain; a complete trajectory is not necessarily a more accurate one. Copy outputs you want to compare before another `overwrite=True` run.

Once satisfied, save the chosen `settings` in your analysis notebook or script and apply them consistently:

```python
AT.process(pools=1, overwrite=True, **settings)
```

Record any justified differences between experimental setups. Increase `pools` only after the single-file results are satisfactory.
