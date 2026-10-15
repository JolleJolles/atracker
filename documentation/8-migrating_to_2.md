# Migrating to ATracker 2.0

[Guide overview](2-using_atracker.md)

Version 2.0 retains the numbered project folders and overview spreadsheet, but
simplifies public entry points and makes body and orientation outputs explicit.
Keep a copy of settings and outputs you need for comparisons before reprocessing.
The planned instance-based folder redesign is a future update, not part of 2.0.

## 1. Update calls and imports

| Previous call | Current call |
| --- | --- |
| `AT.set_interactive(roi=True, inds=[0])` | `AT.editor(inds=[0], purpose="roi")` |
| `AT.set_interactive(mask=True)` | `AT.editor(purpose="mask")` |
| `AT.set_interactive(zones=True)` | `AT.editor(purpose="zones")` |
| `AT.set_interactive(framelimits=True)` | `AT.editor(purpose="framelimits")` |
| `AT.set_interactive(conv=True, ...)` | `AT.editor(purpose="measure")`; enter the known length in the editor. |
| `AT.set_interactive(threshtypes=["bw"])` | `AT.editor(threshtypes=["bw"])` |
| `AT.check_interactive(inds=[0, 1])` | `AT.editor(inds=[0, 1], purpose="timepoints")` |
| `batch_measure(...)` | `measure_images(...)` |
| `state_checker(...)` | `event_checker(...)` |
| `AT.set_config(advanced=True)` | `AT.set_config(body_mode="skeleton")` |
| `track_video(..., simple=False)` | `track_video(..., body_mode="skeleton")` |

Use top-level imports such as `from atracker import ATracker, track_video,
process_video, editor, manual_tracker, event_checker, measure_images`.
Historical internal module names are not a supported compatibility layer.
The project editor's Save current button writes directly; the old `fileaction`
argument is not used. See [the editor guide](5-tracking_editor.md) for current
purposes; unsupported legacy drawing purposes do not have automatic replacements.

## 2. Choose body detail independently of detection

```python
AT.set_config(body_mode="centroid")
# Or estimate a centreline and endpoints:
AT.set_config(body_mode="skeleton", head_rule="near_widest",
    save_body=True, skeleton_points=10)
```

Both modes work with B/W background subtraction and HSV colour contours.
`cx/cy` is the blob centroid or centreline midpoint according to mode;
`center_source` records the actual definition, including extraction fallbacks.
Existing saved boolean settings migrate when opening a project, but scripts must
use the current arguments. Area remains in pixels² and body length in pixels;
`length_method` distinguishes bounding-box length from centreline arc length.

The default head rule selects the endpoint nearest the widest body region.
Alternative centroid-distance rules and unassigned endpoints are described in
[body settings](3-folder_tracking.md#9-other-tracking-checks-and-outputs).
Optional geometry files contain raw per-frame/object contours and centrelines;
processing or manual coordinate edits do not reshape that raw geometry.

## 3. Update downstream column names and orientation assumptions

Processed head columns now match tracked columns: `hx/hy` and converted
`hx_c/hy_c`, replacing processed `fx/fy` and `fx_c/fy_c`. Older files remain readable.
`heading` describes movement; `orient` describes estimated facing direction.
To explicitly restore movement-based estimates of missing orientation:

```python
AT.process(orientation_fallback="heading", orient_min_speed=1,
    overwrite=True)
```

The default `orientation_fallback="none"` keeps these concepts separate.
`orientation_source` identifies heading-derived estimates, interpolation and other
origins. Isolated head/tail swap correction is enabled by default; disable it with
`correct_head_tail=False` when retaining those endpoint assignments is preferred.
Manual endpoint labels and sustained reversals are not automatically flipped.

New per-point source and validity fields survive editing and processing. Existing
files without provenance receive `unknown`, rather than being assumed observed.
See [the output reference](6-processing_data.md#7-output-columns-units-and-position-provenance).

## 4. Rerun deliberately

`overwrite=False` skips existing output files; it does not compare settings.
Use `overwrite=True` to apply changed settings or corrected zone-distance
calculations. Tracking/processing replace CSVs only after completing the write.
Existing outputs are not modified merely by upgrading the package.
The old `v1.0.1` Git tag is not presented as an additional published release in
this changelog; 2.0.0 collects the development since the 1.0.0 release.
