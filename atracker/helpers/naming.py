"""Shared names for outputs associated with overview tracking rows."""
import pandas as pd


def output_basename(overview, ind):
    """Append an integer region only when multiple rows share the video."""
    row = overview.loc[ind]
    video = str(row['video'])
    if (overview.video == video).sum() <= 1:
        return video
    region = row.get('region')
    if region is None or pd.isna(region):
        raise ValueError(f'Missing region for overview row {ind}')
    number = float(region)
    if not number.is_integer() or number < 1:
        raise ValueError(f'Invalid region for overview row {ind}: {region!r}')
    return f'{video}_R{int(number)}'
