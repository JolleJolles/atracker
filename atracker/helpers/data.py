#! /usr/bin/env python

import os
import glob
import signal
from ast import literal_eval
from collections import defaultdict

import numpy as np
import pandas as pd

from pythutils.fileutils import listfiles


def get_filepart(dir, sep="_", part=2, remove_ext=True, ext=None):
    """Extract a specific part of filenames in a directory."""
    files = listfiles(dir)
    out = []
    for f in files:
        if ext is not None:
            if isinstance(ext, str):
                if not f.endswith(ext):
                    continue
            else:
                if not f.endswith(tuple(ext)):
                    continue
        name = os.path.splitext(f)[0] if remove_ext else f
        parts = name.split(sep)
        if len(parts) >= part:
            out.append(parts[part - 1])
        else:
            out.append(None)
    return out


def require_current_tracking_columns(data):
    """Validate external tracking files; conversions belong in the upgrade guide."""
    obsolete = {"com", "head", "tail", "fx", "fy", "fx_c", "fy_c"} & set(data.columns)
    if obsolete or not {"frame", "cx", "cy"}.issubset(data.columns):
        raise ValueError("Expected flat frame/cx/cy coordinates and hx/hy for head points; "
                         "convert older CSVs using documentation/8-upgrading.md")
    if not ({"id", "ID"} & set(data.columns)):
        raise ValueError("Tracking CSV must contain an id or ID column")


def load_and_convert_tracking_dataframe(data_file, firstframe=None, lastframe=None):
    """Load current coordinates and map identity labels to numeric editor IDs."""
    from .provenance import ensure_position_sources
    df = pd.read_csv(data_file)
    require_current_tracking_columns(df)
    df = df.rename(columns={"ID": "id"}) if "id" not in df else df
    df['IDstr'] = df['id'].astype(str)
    for column in ('hx', 'hy', 'tx', 'ty', 'angle'):
        if column not in df:
            df[column] = np.nan
    df = ensure_position_sources(df)
    if firstframe is not None:
        df = df[df.frame >= firstframe]
    if lastframe is not None:
        df = df[df.frame <= lastframe]
    labels = sorted(df.IDstr.unique(), key=str)
    df['ID'] = df.IDstr.map({name: i + 1 for i, name in enumerate(labels)})
    return df.drop(columns='id').reset_index(drop=True)


def ensure_columns(df, columns_with_defaults):
    for col, default in columns_with_defaults.items():
        if col not in df.columns:
            df[col] = default
    return df


def subdic(dic, inds):
    return {k: [j for i, j in enumerate(dic[k]) if i in inds] for k, v in dic.items()}


def duplicate_row(df, index, number):
    row_to_duplicate = pd.DataFrame([df.loc[index]] * (number - 1), columns=df.columns)
    row_to_duplicate['region'] = range(2, number + 1)
    df = pd.concat([df, row_to_duplicate], ignore_index=True)
    return df


def safe_literal_eval(val):
    try:
        if isinstance(val, str) and val.startswith("("):
            return literal_eval(val)
        elif pd.isna(val):
            return (np.nan, np.nan)
        else:
            return val
    except (ValueError, SyntaxError):
        return (np.nan, np.nan)


def eval_func_tuple(f_args):
    """Takes a tuple of a function and args, evaluates and returns result."""
    return f_args[0](*f_args[1:])


def initializer():
    """Ignore CTRL+C in the worker process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True
