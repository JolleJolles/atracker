"""Position origin is independent of whether processing retains its coordinates."""
import ast

import numpy as np
import pandas as pd


SOURCE_COLUMNS = ("position_source", "head_source", "tail_source")
RAW_BODY_COLUMNS = ("area", "aspect_ratio", "length", "length_method", "body_mode", "center_source",
                    "body_valid", "body_straightness", "head_assignment", "head_asymmetry",
                    "body_file", "tracking_run")



def point_columns(data):
    return (("position", "cx", "cy", "com"),
            ("head", "hx" if "hx" in data else "fx", "hy" if "hy" in data else "fy", "head"),
            ("tail", "tx", "ty", "tail"))


def point_values(data, x, y, packed=None):
    if x in data and y in data:
        return data[[x, y]].apply(pd.to_numeric, errors="coerce")
    if packed and packed in data:
        def unpack(value):
            try:
                value = ast.literal_eval(value) if isinstance(value, str) else value
                return float(value[0]), float(value[1])
            except (TypeError, ValueError, IndexError, SyntaxError):
                return np.nan, np.nan
        return pd.DataFrame(data[packed].map(unpack).tolist(), index=data.index, columns=[x, y])
    return pd.DataFrame(np.nan, index=data.index, columns=[x, y])


def ensure_position_sources(data, default="unknown"):
    """Add provenance without replacing known origins when coordinates are filtered."""
    data = data.copy()
    for name, x, y, packed in point_columns(data):
        valid = np.isfinite(point_values(data, x, y, packed)).all(axis=1)
        column = name + "_source"
        if column not in data:
            data[column] = np.where(valid, default, "missing")
        else:
            blank = data[column].isna() | data[column].astype(str).str.strip().eq("")
            data.loc[blank, column] = np.where(valid[blank], default, "missing")
        data[name + "_valid"] = valid
    return data


def mark_interpolated(data, before, columns):
    """Called by gap filling, before smoothing, for an individual coordinate pair."""
    first = columns[0]
    name = {"cx": "position", "hx": "head", "fx": "head", "tx": "tail"}.get(first)
    if name is not None and name + "_source" in data:
        valid = np.isfinite(data[columns].apply(pd.to_numeric, errors="coerce")).all(axis=1)
        data.loc[~before & valid, name + "_source"] = "interpolated"
        data[name + "_valid"] = valid
    return data


def mark_manual_edits(edited, original=None):
    """Compare editor points with their loaded originals; keep unchanged origins.

    Original points are compared at the editor's integer-pixel precision. Fully
    deleted rows are retained as missing coordinate rows with manual origins.
    """
    edited = edited.copy()
    if original is None:
        return ensure_position_sources(edited, default="manual")
    original = original.copy()
    if "IDstr" in original:
        original["id"] = original["IDstr"]
    original["id"] = original["id"].astype(str)
    edited["id"] = edited["id"].astype(str)
    # Keep deletion records, including rows whose last point was removed.
    keys = ["id", "frame"]
    removed = original.loc[~pd.MultiIndex.from_frame(original[keys]).isin(
        pd.MultiIndex.from_frame(edited[keys])), keys]
    edited = pd.concat([edited, removed], ignore_index=True)
    original = ensure_position_sources(original).set_index(keys)
    baseline = original.reindex(pd.MultiIndex.from_frame(edited[keys])).reset_index()
    result = ensure_position_sources(edited, default="manual")
    for name, x, y, packed in point_columns(result):
        values = point_values(result, x, y, packed)
        bx = "hx" if name == "head" and "hx" in baseline else x
        by = "hy" if name == "head" and "hy" in baseline else y
        prior = point_values(baseline, bx, by, packed)
        # The editor represents points using QPoint integer coordinates.
        prior = np.trunc(prior.to_numpy(dtype=float))
        now = values.to_numpy(dtype=float)
        unchanged = ((now == prior) | (np.isnan(now) & np.isnan(prior))).all(axis=1)
        sources = baseline[name + "_source"].fillna("missing").to_numpy()
        result[name + "_source"] = np.where(unchanged, sources, "manual")
        # New frames with no point have no source for that point type.
        new_missing = ~np.isfinite(now).all(axis=1) & baseline[name + "_source"].isna().to_numpy()
        result.loc[new_missing, name + "_source"] = "missing"
    for column in RAW_BODY_COLUMNS:
        if column in baseline and column not in result:
            result[column] = baseline[column].to_numpy()
    if "head_assignment" in result:
        manual_head = result.head_source.eq("manual") & result.head_valid
        result.loc[manual_head, "head_assignment"] = "manual"
    return result.sort_values(keys).reset_index(drop=True)
