# Manual tracking and correction

Use the interactive editor to correct local tracking errors or annotate positions from scratch. For widespread detection failures, improve automated tracking first. Run [data processing](4-processing_data.md) after checking and saving the coordinates.

Use `AT.editor(purpose="timepoints")` for project work. `AT.check_interactive()` remains available for older notebooks, and the standalone `manual_tracker()` function works without a project. Older examples using `AT.process(manfix=True, manonly=True)`, `Tracker_man`, `man_types` or `statevar` do not apply to the current API.

## Correct files in an ATracker project

With your project already loaded as `AT`, open selected videos in the multi-file editor:

```python
# Replace with a video basename from AT.overview["video"].
AT.editor(names=["your_video"], purpose="timepoints")
```

You can also select overview rows or use a query:

```python
AT.editor(inds=[0, 1], purpose="timepoints")

AT.editor(
    query="date in ['260225', '260226']",
    purpose="timepoints",
)
```

This opens the original video and loads its coordinate file from `AT.dirs["tracked"]/<video>.csv`, if present. When the overview has multiple regions for a video, the Editor uses `<video>_R<region>.csv`, matching the tracker. Select a specific region with its overview index or `names="your_video_R2"`. Without an existing CSV, you can start annotating from scratch. Avoid `cats` when you intend to correct every video: it selects representatives of category combinations.

**Save current** (or **S**) writes the current task immediately and stays on the same video. For coordinates, this replaces the tracked CSV. **Next** and **Prev** navigate separately and retain pending work. **Close** checks for unsaved edits across all visited videos and purposes; it does not save them automatically.

You can complete several tasks without leaving the window:

```python
AT.editor(names="your_video", purpose="roi")
```

Draw the ROI and press **S**, choose **Mask** and draw/save it, then choose **Coordinate data** to load and edit the matching tracked CSV. Switching purposes retains unfinished drawings and coordinate edits separately. Return to a purpose to resume its pending work; on first use, its existing project data is loaded. ROI and frame-limit saves update the overview spreadsheet immediately. Masks and zones are saved as full-frame images; coordinates are edited against the saved ROI.

**Measurement** saves calibration: draw a polyline, press **S**, and enter its known length in millimetres. **Thresholding** and **Thresholding color** load the video's configured threshold settings; when several configurations are available, choose which one to edit. Saving updates that named configuration.

For dark footage, open `AT.editor(names="your_video", purpose="thresholding")` and adjust **Tracking gamma** in the threshold controls. The default **1.00** leaves detection unchanged; higher values brighten shadows before background subtraction and blur. Both the video frame and background receive the same correction. Retune the threshold while inspecting the detection preview, then press **S** to save `gamma` with the selected threshold configuration. Subsequent tracking uses that setting for every video using the configuration. Colour thresholding also supports tracking gamma. Existing configurations without `gamma` use 1.0.

The separate **Gamma** control under Drawing functions only changes the displayed image and has no effect on tracking. Tracking gamma does not alter source videos or saved backgrounds.

To edit or create a named threshold configuration independently of the overview assignments:

```python
AT.editor(inds=inds, threshtypes=["bwdark"])
# Also accepted: purpose="thresholding", threshtypes=["bwdark"]
```

An existing `bwdark` entry is loaded; a new entry starts from defaults. **S / Save current** adds or updates it in `AT.threshinfo` and writes `AT.cfiles["threshinfo"]`, preserving other entries. With several names, the Editor asks which configuration to edit. Without an explicit purpose, the first name selects B/W thresholding for a `bw` prefix, otherwise colour thresholding. This does not assign the configuration to videos: tracking uses each overview row's `thresh_types` field, as with `set_interactive(threshtypes=...)`.

The editor retains existing coordinate ID labels when saving, while showing numeric IDs for editing. It still writes coordinate columns rather than preserving every automated-tracker metadata column.

For the older, one-window-per-video workflow:

```python
AT.check_interactive(inds=[0, 1], fileaction="overwrite")
```

This also edits tracked CSVs while displaying original videos, but uses the older per-video window and filename handling. Prefer `AT.editor()` for region-aware, multi-purpose work.

## Add, move and delete positions

In the editor, choose the animal ID and point type (**Centroid**, **Head** or **Tail**) before editing.

1. Navigate to the frame you want to correct using the frame control or video slider.
2. Select **(re)Draw point at current frame** and click the desired body position to add or replace a point.
3. To adjust an existing point, select **Move point nearest to mouse** and drag it.
4. To remove several points, choose a rectangle or polygon, draw around them, and use **In shape**. Set the visible frame range and **Current ID** scope first to limit the selection.
5. Review the corrected interval, then press **S** to save it.

The **Last**, **In shape**, **Visible** and **Undo** buttons provide deletion controls. Check the selected point type, ID scope and visible range before deleting. The **Change angle of nearest point** mode edits an angle associated with an existing point; it is separate from moving its head or tail.

| Control | Action |
| --- | --- |
| Space | Play/pause video. |
| Q / Y | First/last video frame. |
| E / R | Previous/next frame. |
| W / T | Jump backward/forward by the displayed FPS step. |
| I / O | Previous/next animal ID when multiple IDs are configured. |
| P | Cycle Centroid, Head and Tail point types. |
| S | Save current and stay in the project editor; save and close in the standalone/older interface. |
| Escape twice quickly | Close the window, prompting about pending project edits; already saved files remain saved. |

Choose the number of IDs with the editor's ID-count control when starting a new annotation. Use the point display, frame labels and visible-range controls to inspect a manageable interval rather than the entire trajectory at once.

## Track a standalone video

No ATracker project is required:

```python
from atracker import manual_tracker

manual_tracker(
    media_file="/path/to/video.mp4",
    data_file="/path/to/video_coordinates.csv",
    fileaction="overwrite",
)
```

An existing coordinate CSV is loaded; otherwise the editor starts empty. Choose the number of animals and point type in the interface, add positions, then press **S** to save and close. For a new file, add at least one point before saving.

If `data_file` is omitted, the CSV is placed beside the video with the same basename. `fileaction="newfile"` saves to a numbered alternative when the target exists. Use an explicit `data_file` to keep annotation output separate from other CSVs.

To annotate several standalone videos in sequence:

```python
from pathlib import Path
from atracker import manual_tracker

for video in sorted(Path("/path/to/videos").glob("*.mp4")):
    result = manual_tracker(
        media_file=str(video),
        data_file=str(video.with_name(video.stem + "_coordinates.csv")),
        fileaction="overwrite",
    )
    if result == "exit":
        break
```

Standalone CSVs are not automatically registered with an ATracker project. To use `AT.process()`, the video needs a matching overview row with frame limits, FPS, ROI and calibration, and its coordinate CSV must use the expected basename in `AT.dirs["tracked"]`.

## Decide what needs manual correction

Place points consistently on the same part of the animal. Prioritise identity swaps, turns, rapid changes in speed, and visible refuge entry/exit frames. Do not invent positions while the animal is hidden: leave gaps and choose appropriate processing settings instead.

Sparse manual annotation needs special care during processing: isolated points may be removed by `min_traj_len`, and long gaps may remain unfilled. Pilot a short annotated interval and inspect its output before scoring an entire experiment. The old animal-state shortcuts and `statevar` argument are not supported by the current interface; keep separate event annotations if your analysis needs explicit refuge states.

The editor remembers its drawing tool, window geometry and display preferences in `~/.atracker/editor_prefs.json`, outside your package. Return to [Processing tracking data](4-processing_data.md#5-choose-settings-for-your-experiment) when the saved coordinates are ready.
