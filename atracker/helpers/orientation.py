"""Body orientation stays separate from movement heading and keeps its origin."""
import numpy as np
import pandas as pd

from pythutils.mathutils import points_to_angle
from .geometry import get_anglediff
from .trajectory import fillmissing, smooth


def correct_isolated_swaps(data):
    """Correct isolated endpoint transpositions supported by both adjacent frames.

    Never use movement heading to force facing direction. Sustained reversals and
    manually labelled tips are left alone. This is deliberately conservative.
    """
    data = data.copy()
    data['head_tail_swapped'] = False
    if not {'fx', 'fy', 'tx', 'ty'}.issubset(data.columns):
        return data
    heads = data[['fx', 'fy']].to_numpy(dtype=float, copy=True)
    tails = data[['tx', 'ty']].to_numpy(dtype=float, copy=True)
    for i in range(1, len(data) - 1):
        if not np.isfinite(np.r_[heads[i-1:i+2].ravel(), tails[i-1:i+2].ravel()]).all():
            continue
        if any(data.iloc[i].get(col) == 'manual' for col in ('head_source', 'tail_source')):
            continue
        if data.iloc[i].get('head_assignment') == 'unassigned':
            continue
        # Compare each endpoint with both neighbours; a real sustained turn does not qualify.
        same = sum(np.linalg.norm(heads[i]-heads[j]) + np.linalg.norm(tails[i]-tails[j]) for j in (i-1, i+1))
        swapped = sum(np.linalg.norm(tails[i]-heads[j]) + np.linalg.norm(heads[i]-tails[j]) for j in (i-1, i+1))
        if same > 0 and swapped < .25 * same:
            label = data.index[i]
            data.loc[label, ['fx', 'fy', 'tx', 'ty']] = np.r_[tails[i], heads[i]]
            data.loc[label, 'head_tail_swapped'] = True
            for col in ('head_source', 'tail_source'):
                if col in data:
                    data.loc[label, col] = 'corrected'
    return data


def process_orientation(final, settings):
    """Compute and smooth separately within each animal/trajectory."""
    final['orient'] = np.nan
    final['orientation_source'] = 'missing'
    final['orientation_valid'] = False
    for _, indices in final.dropna(subset=['ID', 'traj']).groupby(['ID', 'traj'], sort=False).groups.items():
        part = final.loc[indices].copy()
        angles = pd.Series(np.nan, index=part.index)
        source = pd.Series('missing', index=part.index)
        if (settings.orientfrombw or not {'fx', 'fy'}.issubset(part.columns)) and 'angle' in part:
            angles = pd.to_numeric(part['angle'], errors='coerce')
            source.loc[angles.notna()] = 'unknown'
            if 'head_source' in part:
                source.loc[angles.notna()] = part.loc[angles.notna(), 'head_source'].replace('missing', 'unknown')
        elif {'fx', 'fy', 'cx', 'cy'}.issubset(part.columns):
            valid = np.isfinite(part[['fx', 'fy', 'cx', 'cy']]).all(axis=1)
            valid &= (np.hypot(part.fx-part.cx, part.fy-part.cy) > 1e-9)
            if 'head_assignment' in part:
                valid &= ~part.head_assignment.isin(['unassigned', 'unresolved'])
            angles.loc[valid] = points_to_angle(
                (part.loc[valid, 'cx'], part.loc[valid, 'cy']),
                (part.loc[valid, 'fx'], part.loc[valid, 'fy']), flip=True)
            source.loc[valid] = part.loc[valid, 'head_source'] if 'head_source' in part else 'unknown'
            if 'head_tail_swapped' in part:
                source.loc[valid & part.head_tail_swapped.fillna(False)] = 'corrected'
        if 'head_assignment' in part:
            angles.loc[part.head_assignment.eq('unassigned')] = np.nan
            source.loc[part.head_assignment.eq('unassigned')] = 'missing'
        if settings.orientation_fallback == 'heading':
            eligible = angles.isna() & (part.speed >= settings.orient_min_speed) & part.heading.notna()
            if 'head_assignment' in part:
                eligible &= ~part.head_assignment.eq('unassigned')
            angles.loc[eligible] = part.loc[eligible, 'heading']
            source.loc[eligible] = 'heading'
        part['vx'], part['vy'] = np.nan, np.nan
        valid = angles.notna()
        if valid.any():
            radians = np.radians(angles.loc[valid].to_numpy(dtype=float))
            part.loc[valid, ['vx', 'vy']] = np.column_stack((np.sin(radians), np.cos(radians)))
        if settings.interpolate:
            part, _ = fillmissing(part, ['vx', 'vy'], None, None,
                                  lenthresh=settings.interp_gap_orient)
            filled = ~valid & np.isfinite(part[['vx', 'vy']]).all(axis=1)
            # Unassigned biological endpoints must not acquire a facing direction by filling.
            if 'head_assignment' in part:
                filled &= ~part.head_assignment.eq('unassigned')
                part.loc[part.head_assignment.eq('unassigned'), ['vx', 'vy']] = np.nan
            source.loc[filled] = 'interpolated'
        if settings.smoothwin > 1:
            part = smooth(part, part.traj.unique(), ['vx', 'vy'], settings.smoothwin)
        usable = np.isfinite(part[['vx', 'vy']]).all(axis=1) & (np.hypot(part.vx, part.vy) > 1e-8)
        part['orient'] = np.nan
        part.loc[usable, 'orient'] = points_to_angle(
            (part.loc[usable, 'vx'], part.loc[usable, 'vy']), flip=True)
        source.loc[~usable] = 'missing'
        final.loc[indices, 'orient'] = part.orient
        final.loc[indices, 'orientation_source'] = source
        final.loc[indices, 'orientation_valid'] = usable
        final.loc[indices, 'turnspeed'] = get_anglediff(part.orient)
        final.loc[indices, 'turnaccel'] = get_anglediff(final.loc[indices, 'turnspeed'])
    return final
