"""Project overview/file handling for the editor; independent of Qt widgets."""
import os
from ast import literal_eval

import cv2
import numpy as np
import pandas as pd
import yaml

from atracker.helpers.logging import lineprint
from atracker.helpers.naming import output_basename
from atracker.helpers.output import write_csv_atomic


def editor_files(project, resolved_inds, threshtypes, default_aspect):
    """Build editor inputs using the same output names as automated tracking."""
    file_infos = []
    for ind in resolved_inds:
        row = project.overview.loc[ind]
        vid_name = str(row.get("video", ""))
        basename = output_basename(project.overview, ind)
        vid_path = os.path.join(project.dirs["originals"], vid_name + ".mp4")

        bgimg = row.get("bgimg")
        bgpath = os.path.join(project.dirs["originals"], bgimg) if isinstance(bgimg, str) else None

        maskimg = row.get("maskimg")
        maskpath = os.path.join(project.dirs["originals"], maskimg) if isinstance(maskimg, str) else None

        zoneimg = row.get("zoneimg")
        zonespath = os.path.join(project.dirs["originals"], zoneimg) if isinstance(zoneimg, str) else None

        roi_val = row.get("roi")
        roi = None
        if isinstance(roi_val, str):
            try:
                roi = literal_eval(roi_val)
            except (ValueError, SyntaxError) as exc:
                raise ValueError(f"Invalid ROI for overview row {ind}: {roi_val!r}") from exc

        frame_start = row.get("frame_start")
        frame_start = None if pd.isna(frame_start) or str(frame_start).strip() == "" else int(frame_start)
        frame_stop = row.get("frame_stop")
        frame_stop = None if pd.isna(frame_stop) or str(frame_stop).strip() == "" else int(frame_stop)

        tracked_csv = os.path.join(project.dirs["tracked"], basename + ".csv")

        thresh_types = row.get("thresh_types")
        requested_types = threshtypes if threshtypes is not None else (
            [t.strip() for t in thresh_types.split(",") if t.strip()] if isinstance(thresh_types, str) else [])
        thresh_dict = {}
        if requested_types:
            thresh_dict = project.threshinfo.get(requested_types[0], {})

        file_infos.append({
            "ind": ind,
            "video_name": vid_name,
            "output_basename": basename,
            "video_path": vid_path if os.path.isfile(vid_path) else None,
            "background_path": bgpath if (bgpath and os.path.isfile(bgpath)) else None,
            "mask_path": maskpath if (maskpath and os.path.isfile(maskpath)) else None,
            "zones_path": zonespath if (zonespath and os.path.isfile(zonespath)) else None,
            "roi": roi,
            "frame_start": frame_start,
            "frame_stop": frame_stop,
            "tracked_csv": tracked_csv,
            "threshold_dict": thresh_dict,
            "threshold_defaults": {"min_aspect_ratio": default_aspect[0], "max_aspect_ratio": default_aspect[1]},
            "threshold_options": {t: project.threshinfo.get(t, {}) for t in requested_types},
            "dirs": project.dirs,
        })

    return file_infos


def save_editor_data(project, file_infos, file_idx, ind, purpose_key, data):
    """Save one editor purpose and persist changed overview fields immediately."""
    overview_dirty = False
    fi = file_infos[file_idx]
    vid_name = fi["output_basename"]

    if purpose_key == "mask":
        if isinstance(data, np.ndarray):
            outname = f"{vid_name}_mask.jpg"
            outpath = os.path.join(project.dirs["originals"], outname)
            if not cv2.imwrite(outpath, data):
                raise OSError(f"Could not write {outpath}")
            fi["mask_path"] = outpath
            project.overview.loc[ind, "maskimg"] = outname
            overview_dirty = True
            lineprint(f"Stored mask: {outname}")

    elif purpose_key == "roi":
        if data:
            project.overview.loc[ind, "roi"] = str(data)
            fi["roi"] = data
            overview_dirty = True
            lineprint(f"Stored ROI: {data}")

    elif purpose_key == "zones":
        if isinstance(data, np.ndarray):
            outname = f"{vid_name}_zone.jpg"
            outpath = os.path.join(project.dirs["originals"], outname)
            if not cv2.imwrite(outpath, data):
                raise OSError(f"Could not write {outpath}")
            fi["zones_path"] = outpath
            if "zoneimg" not in project.overview.columns:
                project.overview["zoneimg"] = pd.Series(dtype=object)
            project.overview.loc[ind, "zoneimg"] = outname
            overview_dirty = True
            lineprint(f"Stored zones: {outname}")

    elif purpose_key == "framelimits":
        if data:
            project.overview.loc[ind, "frame_start"] = data[0]
            project.overview.loc[ind, "frame_stop"] = data[1]
            fi["frame_start"], fi["frame_stop"] = data
            overview_dirty = True
            lineprint(f"Stored frame limits: start={data[0]}, stop={data[1]}")

    elif purpose_key == "timepoints":
        if data is not None and hasattr(data, "to_csv"):
            csv_path = fi["tracked_csv"]
            data = data.copy()
            labels = fi.get("id_labels", {})
            for editor_id in data["id"].unique():
                if editor_id not in labels:
                    candidate = int(editor_id)
                    while str(candidate) in labels.values():
                        candidate += 1
                    labels[editor_id] = str(candidate)
            fi["id_labels"] = labels
            data["id"] = data["id"].map(labels)
            from atracker.helpers.provenance import mark_manual_edits
            from atracker.helpers.data import load_and_convert_tracking_dataframe
            original = load_and_convert_tracking_dataframe(csv_path) if os.path.exists(csv_path) else None
            data = mark_manual_edits(data, original)
            write_csv_atomic(data, csv_path, index=False)
            lineprint(f"Stored coordinate data: {os.path.basename(csv_path)}")

    elif purpose_key == "measure":
        project.overview.loc[ind, "conv"] = float(data)
        overview_dirty = True

    elif purpose_key in ("thresholding", "thresholding color"):
        if isinstance(data, dict):
            tt = fi.get("threshold_type")
            if tt:
                project.threshinfo[tt] = data
                with open(project.cfiles["threshinfo"], "w") as f:
                    yaml.safe_dump(project.threshinfo, f, default_flow_style=False)
                for target in file_infos:
                    if tt in target["threshold_options"]:
                        target["threshold_options"][tt] = dict(data)
                lineprint(f"Stored thresholding for {tt}")
            else:
                raise ValueError("No threshold configuration selected for this video.")
    else:
        raise ValueError(f"Saving is not implemented for {purpose_key}")

    if overview_dirty:
        project.save()
