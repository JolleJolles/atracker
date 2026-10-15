# Tracking editor: setup, correction and event review

[Guide overview](2-using_atracker.md)

Use the interactive editor to correct tracked coordinates, annotate positions from
scratch, or review candidate events. The examples below cover project and
standalone workflows. For widespread detection failures, improve
[automated tracking](3-folder_tracking.md) before correcting individual frames.

| Task | Function |
| --- | --- |
| Correct coordinates in a project | `AT.editor(..., purpose="timepoints")` |
| Draw an ROI, mask or zones; adjust tracking settings | `AT.editor(..., purpose=...)` |
| Annotate coordinates without a project | `manual_tracker(...)` |
| Confirm candidate event intervals | `AT.event_checker(...)` or standalone `event_checker(...)` |

## 1. Open project videos for correction

With your project loaded as `AT`, select a video by its overview basename:

```python
AT.editor(names="your_video", purpose="timepoints")
```

The editor opens the original video and its tracked CSV. If no coordinate file
exists, it starts empty. Select several videos by name, overview index or query:

```python
AT.editor(names=["video_1", "video_2"], purpose="timepoints")
AT.editor(inds=[0, 1], purpose="timepoints")
AT.editor(query="date in ['260225', '260226']", purpose="timepoints")
```

Replace these names and dates with values from your overview. Avoid `cats` when
you want every matching video: that option selects representatives of category
combinations.

**Next / Prev** move between files and retain pending edits. **S / Save current**
saves the current task and stays on that video. Closing prompts before discarding
unsaved work across visited files and purposes.

Tracked coordinates are loaded from `AT.dirs["tracked"]/<video>.csv`. For videos
with multiple overview regions, the editor uses `<video>_R<region>.csv`, matching
the tracker. Select a region through its overview index or
`names="your_video_R2"`.

## 2. Add, move and delete positions

Choose the animal ID and point type (**Centroid**, **Head** or **Tail**) first.
For a new annotation, set the number of animals using the ID-count control.

1. Navigate to the frame using the frame box, video slider or keyboard.
2. Select **(re)Draw point at current frame** and click the animal's position.
3. To adjust a point, select **Move point nearest to mouse** and drag it.
4. To delete several points, draw a rectangle or polygon and use **In shape**.
   Set the visible frame range and **Current ID** scope before deleting.
5. Inspect the interval, then press **S** to save.

**Last**, **In shape**, **Visible** and **Undo** control deletion. The
**Change angle of nearest point** tool edits a point's associated angle; it does
not move its head or tail. Use point visibility, frame labels and visible-range
controls to focus on a manageable interval.

| Control | Action |
| --- | --- |
| Space | Play/pause. |
| Q / Y | First/last video frame. |
| E / R | Previous/next frame. |
| W / T | Backward/forward by the displayed FPS step. |
| I / O | Previous/next animal ID. |
| P | Cycle Centroid, Head and Tail. |
| S | Save current in the project editor; save and close in standalone manual tracking. |
| Escape twice quickly | Close, checking for pending project edits. |

Place points consistently on the same part of the animal. Prioritise identity
swaps, turns, abrupt speed changes and refuge entry/exit. Leave gaps when the
animal is hidden rather than inventing positions.

Coordinate saves preserve the origins of unchanged points and mark added, moved
or deleted points `manual`, separately for centroid/head/tail. See the
[output and provenance reference](6-processing_data.md#7-output-columns-units-and-position-provenance).

After saving, rerun [processing](6-processing_data.md) with `overwrite=True` to
update existing processed outputs. Sparse annotation needs care: isolated points
may be removed by `min_traj_len`, and long gaps may remain unfilled. Pilot a short
interval before annotating an entire experiment.

## 3. Edit setup and threshold settings

Open an editor purpose directly:

```python
AT.editor(names="your_video", purpose="roi")
```

You can switch purposes within the window. Draw and save an ROI, choose **Mask**
and draw/save it, then choose **Coordinate data** to correct tracking. Each purpose
retains its own pending work; returning to it restores that draft.

| Purpose | What to do and save |
| --- | --- |
| `"roi"` | Draw the ROI; saving updates the project overview. |
| `"mask"` | Draw an exclusion mask; saved as a full-frame image. |
| `"zones"` | Draw zones; saved as a full-frame image. |
| `"framelimits"` | Set the tracking interval; saving updates the overview. |
| `"measure"` | Draw a polyline, press S and enter its known length in millimetres to save calibration. |
| `"thresholding"` | Edit a named B/W threshold configuration. |
| `"thresholding color"` | Edit a named colour threshold configuration. |

Coordinates are edited against the saved ROI. Save setup changes before using
them for later tasks.

### Tune a named threshold configuration

To edit or create a configuration independently of overview assignments:

```python
AT.editor(names="your_video", threshtypes=["bwdark"])
```

An existing configuration is loaded; a new one starts from defaults. Without an
explicit purpose, a `bw` prefix selects B/W thresholding; other names select
colour thresholding. If several names are supplied, choose which one to edit.

**S / Save current** updates that entry in `AT.threshinfo` and its YAML file,
preserving other configurations. It does not assign the configuration to videos:
tracking still uses each overview row's `thresh_types` field.

**Next / Prev** retain the current named configuration and its unsaved settings
across videos. Revisiting a video uses the latest draft for that configuration.
Different configuration names retain separate drafts; navigation does not save
them.

### Brighten dark footage for detection

```python
AT.editor(names="your_video", purpose="thresholding")
```

Increase **Tracking gamma** in the threshold controls, inspect the detection
preview, retune the threshold and press **S**. The default `1.00` leaves detection
unchanged; higher values brighten shadows. The video frame and background both
receive this correction before background subtraction and blur.

Tracking gamma is saved with the named threshold configuration and applies to all
videos using it. Colour thresholding also supports it; configurations without a
`gamma` entry use `1.0`. Source videos and saved backgrounds are unchanged.
The separate **Gamma** control under Drawing functions changes only the display.
See [tracking settings](3-folder_tracking.md) for the detection pipeline and parameter
choices.

## 4. Annotate a standalone video

No project or overview is required:

```python
from atracker import manual_tracker

manual_tracker(media_file="/path/to/video.mp4",
    data_file="/path/to/video_coordinates.csv", fileaction="overwrite")
```

An existing coordinate CSV is loaded; otherwise annotation starts empty. Choose
the number of animals and point type, add positions, then press **S** to save and
close. A new file needs at least one point before saving.

If `data_file` is omitted, the CSV is placed beside the video with the same
basename. `fileaction="newfile"` saves to a numbered alternative when the target
exists. An explicit filename keeps manual annotations separate from other CSVs.

To annotate several videos in sequence:

```python
from pathlib import Path
from atracker import manual_tracker

for video in sorted(Path("/path/to/videos").glob("*.mp4")):
    result = manual_tracker(media_file=str(video),
        data_file=str(video.with_name(video.stem + "_coordinates.csv")),
        fileaction="overwrite")
    if result == "exit":
        break
```

Follow the [standalone tutorial](4-standalone_tracking.md) to process and visualise
these files with explicit video, ROI and calibration settings. Standalone CSVs
are not automatically registered with a project. To use `AT.process()`, provide a
matching overview row and put the coordinate CSV under the expected basename in
`AT.dirs["tracked"]`.

## 5. Review candidate events

Open a MoveR event-window CSV against your project's original videos:

```python
AT.event_checker("processed/barrier_jump_frames.csv", modifier="complete")
```

The CSV needs `video`, `start_frame`, `end_frame` and `state` columns. The `state`
value labels the candidate, for example `state_jump` or `state_slow`. A `review`
column is added if absent. Extra columns and row order are preserved.

Video names match files in the originals directory. Region suffixes such as
`_R2` can resolve to the corresponding full original video. Event review updates
the candidate CSV, leaving tracking coordinates and overview frame limits alone.

### Review only unfinished events

```python
AT.event_checker("processed/barrier_jump_frames.csv",
    modifier="complete", unreviewed_only=True)
```

Only rows with blank `review` values are selected when opening the checker.
Saving still preserves all reviewed rows in the original CSV. Omit this option
to browse every event.

### Navigate, adjust and confirm

| Control | Action |
| --- | --- |
| Prev / Next; left/right keys with canvas focus | Select the previous/next candidate in CSV order, seeking to its start. |
| Video arrows | Skip to the previous/next different video. |
| Frame box; ±1-frame / ±1-second buttons | Step through the current video. |
| Candidate start/end | Edit inclusive bounds, or set a bound to the current frame. |
| Confirm | Save `confirmed` in `review`. |
| Deny | Save `denied` in `review`; retain the row. |
| Unreviewed | Clear `review` and save. |
| Advance after review | Move to the next candidate after a review button; enabled by default. |
| S / Save current | Save all pending CSV edits without advancing. |

Playback resumes on navigation if it was playing. The bottom timeline shows
candidate intervals for the current video, with the selected interval outlined.
Click an interval to select it, click/drag elsewhere to seek, or scroll to step
through frames. Shift-scroll uses the displayed FPS step. Drag the divider
between sidebar and video to resize the sidebar.

Navigation retains pending edits without saving. Closing offers Save, Discard or
Cancel. If the CSV was changed externally, reopen it before saving.

### Add an event modifier

`modifier="complete"` adds a **Complete** checkbox, for example for a jump spanning
the whole barrier. Its value is saved in a `complete` column as `true` or `false`;
blank means not yet assessed. Existing values are restored per event.

Confirm/Deny records the current checkbox value. Changing the checkbox alone
remains pending until Save or a review button is pressed. Use another name, such
as `modifier="contact"`, to record a different yes/no property.

### Frame numbering and standalone review

CSV bounds are **inclusive and one-based by default**, matching the editor.
For zero-based exports:

```python
AT.event_checker("processed/barrier_jump_frames.csv", frame_base=0)
```

The interface still displays one-based frames and saves back using the input
convention. Candidate bounds are checked against video lengths before opening.

Without a project:

```python
from atracker import event_checker

event_checker("barrier_jump_frames.csv", video_dir="/path/to/originals",
    modifier="complete", unreviewed_only=True)
```

The equivalent project editor call is
`AT.editor(purpose="event_checker", candidates="path/to/events.csv")`.
When analysing the reviewed file, select `review == "confirmed"` for accepted
events; distinguish denied rows from blank, unreviewed ones.

## What is saved?

| Workflow | Saved output |
| --- | --- |
| Project coordinate editing | Replaces the matching tracked CSV on Save. Existing ID labels are retained; not all automated-tracker metadata columns are preserved. |
| Project setup editing | Updates the overview, named threshold YAML or full-frame setup images, depending on purpose. |
| Standalone manual tracking | Writes the specified coordinate CSV on Save, then closes. |
| Event checking | Writes edited bounds, review decisions and optional modifier values to the original event CSV. |

Display preferences and drawing-tool choices are stored outside the package in
`~/.atracker/editor_prefs.json`.

Use `AT.editor(purpose=...)` for project setup and
`AT.editor(purpose="timepoints")` for coordinate correction. Save current writes
edits directly to the associated project files.
