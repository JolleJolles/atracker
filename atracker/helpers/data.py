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


def load_and_convert_tracking_dataframe(data_file, firstframe=None, lastframe=None):
    """
    Loads a tracking data CSV, converts old formats to new, filters frames,
    and returns a standardized DataFrame with columns:
    frame, id, IDstr, cx, cy, hx, hy, tx, ty, angle
    """
    df = pd.read_csv(data_file)
    df.columns = [c.lower() for c in df.columns]

    new_cols = {'frame', 'id', 'cx', 'cy', 'hx', 'hy', 'tx', 'ty', 'angle'}
    if not new_cols.issubset(df.columns):
        if {'com', 'angle'}.issubset(df.columns):
            def parse_point(s):
                try:
                    return literal_eval(str(s))
                except Exception:
                    return (np.nan, np.nan)
            new_df = pd.DataFrame()
            new_df['frame'] = df['frame']
            new_df['IDstr'] = df['id'].astype(str)
            new_df[['cx', 'cy']] = df['com'].apply(parse_point).apply(pd.Series)
            new_df['hx'] = np.nan
            new_df['hy'] = np.nan
            new_df['tx'] = np.nan
            new_df['ty'] = np.nan
            new_df['angle'] = pd.to_numeric(df['angle'], errors='coerce')
            df = new_df
        elif {'cx', 'cy'}.issubset(df.columns):
            new_df = pd.DataFrame()
            new_df['frame'] = df['frame']
            new_df['IDstr'] = df['id'].astype(str)
            new_df['cx'] = df['cx']
            new_df['cy'] = df['cy']
            new_df['hx'] = np.nan
            new_df['hy'] = np.nan
            new_df['tx'] = np.nan
            new_df['ty'] = np.nan
            new_df['angle'] = np.nan
            df = new_df
        else:
            raise ValueError(f"Unrecognized tracking file format: {data_file}")
    else:
        df['IDstr'] = df['id'].astype(str)

    if firstframe is not None:
        df = df[df['frame'] >= firstframe]
    if lastframe is not None:
        df = df[df['frame'] <= lastframe]

    unique_ids = sorted(df['IDstr'].unique(), key=str)
    id_map = {name: i + 1 for i, name in enumerate(unique_ids)}
    df['ID'] = df['IDstr'].map(id_map)

    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    return df.reset_index(drop=True)


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


def lit_converter(val):
    if pd.isnull(val):
        return (np.nan, np.nan)
    if isinstance(val, (tuple, list)):
        return val
    try:
        return literal_eval(val)
    except Exception:
        return (np.nan, np.nan)


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
