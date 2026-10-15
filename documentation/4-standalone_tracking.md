# Standalone tracking, processing and visualisation

[Guide overview](2-using_atracker.md)

Track individual videos without creating an ATracker project, overview spreadsheet
or folder structure. Run these examples in a Python script or notebook after
[installing ATracker](1-installing_atracker.md).

## 1. Track a video

```python
from pathlib import Path
import cv2
from atracker import track_video, process_video, load_atcache, Visualiser

video = Path("/path/to/trial.mp4")
tracked_csv = track_video(str(video), n_objects=1, draw_roi=True,
    draw_mask=False, save_settings=True, create_vid=True)
```

ATracker extracts a background automatically, opens an interface to draw the ROI,
then opens threshold calibration. Save your choices in each interface to continue.
Use `draw_roi=False` for the full frame, or `draw_mask=True` to draw an exclusion
mask. There is no mask by default. See [tracking settings](3-folder_tracking.md) for help
with detection and threshold calibration.

Outputs beside the original video:

| File | Contents |
| --- | --- |
| `trial.csv` | Tracked coordinates. |
| `trial_TR.mp4` | Tracking overlay video (`create_vid=False` skips it). |
| `trial.atrk` | Saved background, ROI, mask and threshold settings. |

Use `output="/path/to/results"` to choose an output directory for the CSV and
tracking video. `save_settings=True` still saves the `.atrk` beside the original;
provide a filename instead to choose its location. `show_tracking=True` enables a
live preview. `frame_start` and `frame_stop` restrict tracking to an inclusive,
one-based frame interval.

## 2. Process the coordinates

Load the saved ROI so processing uses the same region as tracking:

```python
settings = load_atcache(str(video.with_suffix(".atrk")))
processed_csv = process_video(tracked_csv, videofile=str(video),
    roi=settings["roi"], conv=0.25)
```

Replace `0.25` with your calibration in **millimetres per pixel**. The original
video supplies FPS, resolution and frame count. Processing filters trajectories,
interpolates eligible gaps, smooths coordinates and calculates movement measures.
It writes `trial_F.csv` beside the tracked CSV; use `outdir` to choose another
location. Existing processed outputs are skipped unless `overwrite=True`.

The saved mask is not automatically passed to processing. To apply it there too:

```python
mask_file = str(video.with_name(video.stem + "_mask.png"))
cv2.imwrite(mask_file, settings["img_mask"])
processed_csv = process_video(tracked_csv, videofile=str(video),
    roi=settings["roi"], conv=0.25, maskfile=mask_file, overwrite=True)
```

Run that optional block only when you drew a mask. Wall and zone images can also
be supplied through `wallfile` and `zonefile`. See
[processing settings](6-processing_data.md) before changing interpolation or
smoothing for your experiment.

## 3. Visualise the processed tracking

Create an overlay video from the processed CSV, using the original FPS and ROI:

```python
capture = cv2.VideoCapture(str(video))
fps = capture.get(cv2.CAP_PROP_FPS)
capture.release()

visualiser = Visualiser(objects=1)
visualiser.render(processed_csv, videofile=str(video),
    outfile=str(video.with_name(video.stem + "_processed.mp4")),
    roi=settings["roi"], fps=fps, trajlength=50,
    draw_contours=False, draw_head=False, draw_tail=False, draw_orient=False)
```

The overlay uses pixel coordinates retained in the processed CSV. The converted
`cx_c` and `cy_c` columns are available for analysis. `trajlength` controls the
trajectory trail in frames. Add `showvideo=True` for a live preview; use
`writevideo=False, showvideo=True` to preview without writing a video.

For manual coordinate corrections, use the standalone tools described in
[manual tracking](5-tracking_editor.md), then rerun processing with
`overwrite=True`.

## 4. Reuse settings and automate a batch

To rerun one video without setup dialogs:

```python
tracked_csv = track_video(str(video), load_settings=True,
    set_threshold=False, create_vid=False, overwrite=False)
```

`load_settings=True` loads that video's `.atrk`. Set `set_threshold=False`
explicitly: loading settings alone does not disable threshold calibration.
`overwrite=False` keeps existing tracking outputs.

To use one setup across several videos:

```python
setup_file = "/path/to/trial.atrk"
settings = load_atcache(setup_file)

for video in Path("/path/to/videos").glob("*.mp4"):
    tracked_csv = track_video(str(video), load_settings=setup_file,
        set_threshold=False, create_vid=False, overwrite=False)
    processed_csv = process_video(tracked_csv, videofile=str(video),
        roi=settings["roi"], conv=0.25)
```

Use a directory containing source videos so the loop does not pick up generated
overlays. A shared setup also reuses its **background image**, ROI, mask and
thresholds; use it only with matching camera and arena setups. If backgrounds
differ, run setup for each video and save its own `.atrk` instead. Add the
visualisation call inside the loop if you also want processed overlay videos.

## What is saved?

The `.atrk` file is an optional setup bundle. It does not contain every tracking
or processing parameter: number of objects, frame limits, FPS overrides,
calibration and processing options remain in your script or notebook. Keep that
code alongside the outputs to record how they were produced. Without
`save_settings`, background and other setup files are temporary and cleaned up
after tracking. The original video is preserved.
