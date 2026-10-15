# Upgrading across breaking changes

[Guide overview](2-using_atracker.md)

This guide records changes that require updates to scripts, settings or saved
files. New breaking releases should add their instructions here. ATracker expects
current settings and coordinate formats; it does not rewrite older files on load.
Use the explicit conversion examples below on copies, inspect the results and
keep the originals until you are satisfied.

## Version 2.0

The numbered project folders and overview spreadsheet remain. The planned
instance-based project layout is a future update.

### Update function calls

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

### Convert overview headers when needed

Current projects use `exclude`, not `Exclude` or `skip`. To combine old exclusion
flags and write a converted spreadsheet alongside the original:

```python
from pathlib import Path
import pandas as pd

path = Path("/path/to/project/project_overview.xlsx")
overview = pd.read_excel(path, dtype=object)
overview["exclude"] = (overview.reindex(columns=["exclude", "Exclude", "skip"])
    .apply(pd.to_numeric, errors="coerce").eq(1).any(axis=1).astype(int))
overview = overview.drop(columns=["Exclude", "skip"], errors="ignore")
overview.to_excel(path.with_name(path.stem + "_2.xlsx"), index=False)
```

Review the new spreadsheet, keep a backup of the original, then replace the
project's overview with the converted file before opening ATracker.

### Replace old configurations explicitly

Automatic `_migrate_config()` has been removed. Old body-mode and tracking-setting
aliases are rejected rather than silently interpreted. Create a current config,
then copy the tracking settings you intend to keep:

```python
from pathlib import Path
from localconfig import LocalConfig
from atracker import ATracker

project = Path("/path/to/project")
path = project / f"{project.name}_config.conf"
old = dict(LocalConfig(str(path), compact_form=True).items("track"))
backup = path.with_suffix(".conf.v1")
if backup.exists():
    raise FileExistsError(backup)
path.rename(backup)
AT = ATracker(str(project))  # creates a current config; keeps the overview

renamed = {"linkdisthreshold": "link_dist", "mergedmindist": "merge_dist",
    "shape_area_tol": "size_filter_tol", "shape_history_len": "size_filter_memory"}
keep = {"overwrite", "create_vid", "create_dat", "link_dist", "merge_dist",
    "size_filter", "size_filter_tol", "size_filter_memory", "min_aspect_ratio",
    "max_aspect_ratio", "skip_frames", "max_framedist", "track_merges",
    "check_flicker", "video_codec", "video_preset", "video_crf", "video_resize"}
settings = {renamed.get(key, key): value for key, value in old.items()
    if renamed.get(key, key) in keep}
settings["body_mode"] = old.get("body_mode", "skeleton" if
    old.get("advanced", not old.get("simple", True)) else "centroid")
if "size_filter" not in settings and "contour_mode" in old:
    settings["size_filter"] = old["contour_mode"] == "dynamic"
AT.set_config(**settings)
```

This deliberately starts other options from current defaults, including display
settings. Review them with the current guides. Frame limits belong in the
overview/editor, not old configuration fields. Detection configurations remain in
the project's threshold YAML file. To revert, close ATracker and restore the
saved config backup. Run this conversion once; subsequent runs use the new file.

### Choose body detail and head assignment

```python
AT.set_config(body_mode="centroid")
# Or estimate the centreline midpoint and endpoints:
AT.set_config(body_mode="skeleton", head_rule="near_widest",
    save_body=True, skeleton_points=10)
```

Both modes work with background subtraction and HSV colour contours. `cx/cy` is
the blob centroid or centreline midpoint according to mode; `center_source`
records the actual definition, including extraction fallbacks. Area remains in
pixels² and length in pixels, with `length_method` identifying the calculation.
See [body settings](3-folder_tracking.md#9-other-tracking-checks-and-outputs) for
head rules. Optional geometry files contain raw shapes, not processed or edited
coordinate estimates.

### Convert older coordinate CSVs

Current coordinate files use separate `cx/cy`, `hx/hy`, `tx/ty` columns, with
`frame` and `id` (or `ID` for processed output). Head columns no longer use
`fx/fy` or `fx_c/fy_c`, and tuple-valued `com`, `head` and `tail` columns are no
longer unpacked automatically. Convert them explicitly:

```python
from ast import literal_eval
from pathlib import Path
import numpy as np
import pandas as pd

path = Path("/path/to/old_tracking.csv")
tracking = pd.read_csv(path)
if "id" not in tracking:
    tracking = tracking.rename(columns={"ID": "id"})
tracking = tracking.rename(columns={"fx": "hx", "fy": "hy", "fx_c": "hx_c", "fy_c": "hy_c"})

def unpack(value):
    return (np.nan, np.nan) if pd.isna(value) else literal_eval(value.replace("nan", "None"))

for column, coordinates in {"com": ["cx", "cy"], "head": ["hx", "hy"], "tail": ["tx", "ty"]}.items():
    if column in tracking:
        tracking[coordinates] = pd.DataFrame(tracking.pop(column).map(unpack).tolist(),
            index=tracking.index, columns=coordinates).astype(float)
tracking.to_csv(path.with_name(path.stem + "_2.csv"), index=False)
```

Use the converted tracking file after checking it against the original. Keep it
in the same folder if it contains relative geometry references. This conversion
only changes representation; it does not improve detection or reconstruct missing
body information. To apply current processing, regenerate processed outputs from
the converted tracked data rather than relabelling processed values as raw data.
Coordinate files without origin fields receive `unknown` origins, not an assumed
`observed` label.

### Review orientation assumptions and reprocessing

Processed head columns retain `hx/hy` and `hx_c/hy_c`. `heading` describes movement;
`orient` describes estimated facing direction. Missing orientation no longer
silently becomes heading. To enable that estimate explicitly:

```python
AT.process(orientation_fallback="heading", orient_min_speed=1, overwrite=True)
```

The default is `orientation_fallback="none"`. `orientation_source` identifies
heading-derived estimates, interpolation and other origins. Isolated endpoint
swap correction is enabled by default; use `correct_head_tail=False` to disable
it. Manual labels and sustained reversals are not automatically flipped.
See the [output reference](6-processing_data.md#7-output-columns-units-and-position-provenance).

`overwrite=False` checks for existing outputs, not changed settings. Use
`overwrite=True` to apply new choices or corrected zone-distance calculations.
Upgrading the package alone does not reprocess saved results.
