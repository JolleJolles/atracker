# Manual tracking and correction

Use the interactive editor to correct local tracking errors or annotate positions from scratch. For widespread detection failures, improve automated tracking first. Run [data processing](4-processing_data.md) after checking and saving the coordinates.

The current interfaces are `AT.editor(purpose="timepoints")`, `AT.check_interactive()` and the standalone `manual_tracker()` function. Older examples using `AT.process(manfix=True, manonly=True)`, `Tracker_man`, `man_types` or `statevar` do not apply to the current API.

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

This opens the original video and loads its coordinate file from `AT.dirs["tracked"]/<video>.csv`, if present. Without an existing CSV, you can start annotating from scratch. Avoid `cats` when you intend to correct every video: it selects representatives of category combinations.

**Store** (or **S**) writes the current coordinate data to that tracked CSV, replacing its contents. Keep a backup before editing. Use **Next** and **Prev** to move between videos, and store each corrected file. **Save** closes the editor after your Store operations; it does not itself write pending coordinate edits. Press **Store** or **S** before leaving each file.

The editor writes coordinate columns rather than preserving every automated-tracker metadata column. It also uses numeric editing IDs, so verify the saved identity mapping against the overview before analysing multi-animal data.

For the older, one-window-per-video workflow:

```python
AT.check_interactive(inds=[0, 1], fileaction="overwrite")
```

This also edits tracked CSVs while displaying original videos. For region-specific filenames, check the CSV selected by this interface before saving; the newer `AT.editor()` currently uses the plain video basename.

## Add, move and delete positions

In the editor, choose the animal ID and point type (**Centroid**, **Head** or **Tail**) before editing.

1. Navigate to the frame you want to correct using the frame control or video slider.
2. Select **(re)Draw point at current frame** and click the desired body position to add or replace a point.
3. To adjust an existing point, select **Move point nearest to mouse** and drag it.
4. To remove several points, choose a rectangle or polygon, draw around them, and use **In shape**. Set the visible frame range and **Current ID** scope first to limit the selection.
5. Review the corrected interval, then press **S** to store it.

The **Last**, **In shape**, **Visible** and **Undo** buttons provide deletion controls. Check the selected point type, ID scope and visible range before deleting. The **Change angle of nearest point** mode edits an angle associated with an existing point; it is separate from moving its head or tail.

| Control | Action |
| --- | --- |
| Space | Play/pause video. |
| Q / Y | First/last video frame. |
| E / R | Previous/next frame. |
| W / T | Jump backward/forward by the displayed FPS step. |
| I / O | Previous/next animal ID when multiple IDs are configured. |
| P | Cycle Centroid, Head and Tail point types. |
| S | Store in the multi-file editor; save and close in the standalone/older interface. |
| Escape twice quickly | Exit the window; already stored files remain saved. |

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

The editor remembers its drawing tool, window geometry and display preferences in `~/.atracker/editor_prefs.json`, outside your package. Return to [Processing tracking data](4-processing_data.md#choose-settings-for-your-experiment) when the saved coordinates are ready.
