"""Track provenance through point edits, interpolation, filtering and CSV saves."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from atracker.helpers.provenance import ensure_position_sources, mark_manual_edits
from atracker.helpers.data import load_and_convert_tracking_dataframe
from atracker.helpers.trajectory import fillmissing, process_trajectories
from atracker import process_video


class PositionProvenanceTest(unittest.TestCase):
    def test_new_and_legacy_origins(self):
        data = pd.DataFrame(dict(cx=[10, np.nan], cy=[20, np.nan]))
        observed = ensure_position_sources(data, default="observed")
        self.assertEqual(list(observed.position_source), ["observed", "missing"])
        self.assertEqual(list(ensure_position_sources(data).position_source), ["unknown", "missing"])
        observed.loc[0, ["cx", "cy"]] = np.nan
        filtered = ensure_position_sources(observed)
        self.assertEqual(filtered.loc[0, "position_source"], "observed")
        self.assertFalse(filtered.loc[0, "position_valid"])

    def test_separate_manual_points_and_deletions(self):
        original = ensure_position_sources(pd.DataFrame(dict(frame=[1, 2], id=["F01"]*2,
            cx=[10.8,20], cy=[10,20], hx=[12,22], hy=[10,20], tx=[8,18], ty=[10,20])), "observed")
        edited = pd.DataFrame(dict(frame=[1], id=["F01"], cx=[10], cy=[10], hx=[13], hy=[10],tx=[8],ty=[10]))
        result = mark_manual_edits(edited, original)
        self.assertEqual(result.loc[0, "position_source"], "observed")
        self.assertEqual(result.loc[0, "head_source"], "manual")
        self.assertEqual(result.loc[0, "tail_source"], "observed")
        self.assertEqual(result.loc[1, "position_source"], "manual")
        self.assertFalse(result.loc[1, "position_valid"])
        again = mark_manual_edits(result, result)
        self.assertEqual(list(again.head_source), list(result.head_source))

    def test_new_manual_positions_and_empty_editor(self):
        edited = pd.DataFrame(dict(frame=[3], id=["F01"], cx=[20], cy=[30]))
        self.assertEqual(mark_manual_edits(edited).position_source.iloc[0], "manual")
        empty = edited.iloc[:0]
        self.assertFalse(mark_manual_edits(empty, edited).position_valid.iloc[0])

    def test_interpolation_and_trajectory_filling(self):
        data = ensure_position_sources(pd.DataFrame(dict(cx=[20,21,np.nan,23,24],cy=[20]*5, inroi=1)), "observed")
        filled, _ = fillmissing(data.copy(), ["cx","cy"], None, ((0,0),(100,100)), lenthresh=2)
        self.assertEqual(filled.loc[2, "position_source"], "interpolated")
        self.assertEqual(filled.loc[1, "position_source"], "observed")
        filled, _, _ = process_trajectories(data.copy(), cover=False, mintrajlength=1)
        self.assertEqual(filled.loc[2, "position_source"], "interpolated")

    def test_loader_keeps_sources_and_head_coordinates(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'raw.csv'
            pd.DataFrame(dict(frame=[1],id=['F01'],cx=[20],cy=[20],hx=[21],hy=[20],
                position_source=['observed'],head_source=['manual'])).to_csv(f,index=False)
            loaded=load_and_convert_tracking_dataframe(f)
            self.assertEqual(loaded.position_source.iloc[0],'observed')
            self.assertEqual(loaded.head_source.iloc[0],'manual')
            self.assertEqual(loaded.hx.iloc[0],21)

    def test_packed_or_old_head_columns_require_explicit_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'old.csv'
            pd.DataFrame(dict(frame=[1],id=['F01'],cx=[20],cy=[20],fx=[21],fy=[20])).to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError, '8-upgrading.md'):
                load_and_convert_tracking_dataframe(path)
            with self.assertRaisesRegex(ValueError, '8-upgrading.md'):
                process_video(str(path),roi=((0,0),(100,100)),conv=.25,overwrite=True)

    def test_head_interpolation_retains_centroid_and_tail_origins(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d)/'raw.csv'
            raw = pd.DataFrame(dict(frame=range(1,7),id=['F01']*6,cx=range(20,26),cy=[30]*6,
                hx=[22,23,np.nan,25,26,27],hy=[30]*6,tx=range(18,24),ty=[30]*6))
            ensure_position_sources(raw, "observed").to_csv(f,index=False)
            output = process_video(str(f),roi=((0,0),(100,100)),conv=.25,min_traj_len=1,
                smoothwin=1,compute_distances=False,compute_movement=False,overwrite=True)
            result = pd.read_csv(output)
            self.assertEqual(result.loc[2,'head_source'],'interpolated')
            self.assertEqual(result.loc[2,'position_source'],'observed')
            self.assertEqual(result.loc[2,'tail_source'],'observed')
            self.assertTrue(result.loc[2,'head_valid'])
            self.assertIn('tx',result.columns)

    def test_processing_roundtrip_and_disabled_interpolation(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'raw.csv'
            pd.DataFrame(dict(frame=range(1,7),id=['F01']*6,cx=[20,21,np.nan,23,24,25],cy=[20]*6,
                position_source=['observed','manual','missing','observed','observed','observed'])).to_csv(f,index=False)
            options=dict(roi=((0,0),(100,100)),conv=.25,min_traj_len=1,smoothwin=1,
                compute_distances=False,compute_movement=False,overwrite=True)
            output=process_video(str(f),**options)
            result=pd.read_csv(output)
            self.assertEqual(list(result.position_source), ['observed','manual','interpolated','observed','observed','observed'])
            self.assertTrue(result.position_valid.all())
            output=process_video(str(f),interpolate=False,**options)
            result=pd.read_csv(output)
            self.assertEqual(result.loc[2,'position_source'],'missing')
            self.assertFalse(result.loc[2,'position_valid'])


if __name__ == '__main__':
    unittest.main()
