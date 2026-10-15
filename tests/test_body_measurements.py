import gzip
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
from localconfig import LocalConfig

from atracker import ATracker, process_video, track_video, save_atcache
from atracker.helpers.body import measure_body, resample_line
from atracker.helpers.body_output import BodyWriter
from atracker.helpers.detection import ProcessImage
from atracker.helpers.orientation import correct_isolated_swaps, process_orientation


class BodyMeasurementsTest(unittest.TestCase):
    def fish(self):
        return np.array([(15, 30), (25, 20), (42, 23), (75, 30), (42, 37), (25, 40)], np.int32).reshape(-1, 1, 2)

    def test_centroid_and_midline_modes_and_head_rules(self):
        contour = self.fish()
        centroid = (35, 30)
        basic = measure_body(contour, centroid, 'centroid', 'near_widest', 10)
        body = measure_body(contour, centroid, 'skeleton', 'near_widest', 10)
        self.assertEqual(basic['com'], centroid)
        self.assertTrue(body['body_valid'])
        self.assertEqual(len(body['skeleton']), 10)
        self.assertLess(body['head'][0], body['tail'][0])
        self.assertGreater(body['com'][0], centroid[0])
        self.assertGreater(body['body_straightness'], .9)
        near = measure_body(contour, centroid, 'skeleton', 'near_centroid', 10)
        far = measure_body(contour, centroid, 'skeleton', 'far_centroid', 10)
        self.assertEqual(near['head'], far['tail'])
        np.testing.assert_allclose(np.array(body['skeleton'])[0], body['head'])

    def test_arc_length_resampling_follows_bend(self):
        points = resample_line([(0,0), (10,0), (10,20)], 4)
        np.testing.assert_allclose(points, [(0,0), (10,0), (10,10), (10,20)])

    def test_colour_and_background_allow_same_body_detail(self):
        image = np.full((60,100,3), 255, np.uint8)
        cv2.drawContours(image, [self.fish()], -1, (0,0,0), -1)
        common = dict(blur=1, erode=1, blur2=1, min_area=10, max_area=5000,
                      min_aspect_ratio=0, body_mode='skeleton')
        _, _, bw = ProcessImage(image, np.full_like(image,255), **common).process()
        _, _, colour = ProcessImage(image, threshtype='dark', colmin=(0,0,0), colmax=(180,255,80), **common).process()
        self.assertTrue(bw['body_valid'][0])
        self.assertTrue(colour['body_valid'][0])
        self.assertEqual(len(colour['skeleton'][0]), 10)
        from atracker.visualise import Visualiser
        bw['id'] = [1] * len(bw['com'])
        preview = image.copy()
        vis = Visualiser(objects=1, thresh_types=['bw'])
        vis.draw_thresh_pass(preview, image, np.zeros(image.shape[:2],np.uint8), bw,
                             [self.fish()], [1], bw['com'], {}, set(), 'bw')
        self.assertFalse(np.array_equal(preview, image))


    def test_geometry_output_stream_and_abort(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory)/'fish.body.jsonl.gz')
            writer = BodyWriter(path, dict(frame_base=1))
            writer.write(dict(frame=1, id='F01', centerline=[[1,2],[3,4]]))
            writer.commit()
            writer.abort()
            with gzip.open(path, 'rt') as stream:
                records = [json.loads(line) for line in stream]
            self.assertEqual(records[0]['tracking_run'], writer.run)
            self.assertEqual(records[1]['id'], 'F01')
            replacement = BodyWriter(path, {})
            replacement.abort()
            with gzip.open(path, 'rt') as stream:
                self.assertEqual(json.loads(next(stream))['tracking_run'], writer.run)

    def test_outdated_config_is_rejected_without_rewriting_it(self):
        with tempfile.TemporaryDirectory() as directory:
            project = object.__new__(ATracker)
            project.cfiles = {'config':str(Path(directory)/'project.conf')}
            project.config = LocalConfig(project.cfiles['config'], compact_form=True)
            project.config.add_section('track')
            project.config.track.advanced = True
            project.config.save()
            before = Path(project.cfiles['config']).read_bytes()
            with self.assertRaisesRegex(ValueError, '8-upgrading.md'):
                project._validate_config()
            self.assertEqual(Path(project.cfiles['config']).read_bytes(), before)

    def test_only_isolated_swaps_are_corrected(self):
        data = pd.DataFrame(dict(fx=[1.,-1,1], fy=0., tx=[-1.,1,-1], ty=0., head_source='observed', tail_source='observed'))
        corrected = correct_isolated_swaps(data)
        self.assertEqual(corrected.loc[1, 'fx'], 1)
        self.assertEqual(corrected.loc[1, 'head_source'], 'corrected')
        data.loc[1,'head_source'] = 'manual'
        self.assertEqual(correct_isolated_swaps(data).loc[1, 'fx'], -1)
        data.fx = [1.,-1,-1]; data.tx = [-1.,1,1]
        self.assertFalse(correct_isolated_swaps(data).head_tail_swapped.any())

    def test_orientation_is_separate_from_heading_and_trajectory_boundaries(self):
        settings = SimpleNamespace(orientfrombw=True, orientation_fallback='none', orient_min_speed=1,
                                   interpolate=False, interp_gap_orient=5, smoothwin=1)
        data = pd.DataFrame(dict(ID=['a','a','b'],traj=[1,1,2],angle=[179,-179,45],heading=[0,0,0],speed=[5,5,5]))
        result = process_orientation(data, settings)
        self.assertAlmostEqual(abs(result.loc[1,'turnspeed']), 2)
        self.assertTrue(pd.isna(result.loc[2,'turnspeed']))
        self.assertTrue(result.orientation_valid.all())
        settings.orientfrombw = False
        data = data.drop(columns='angle')
        result = process_orientation(data, settings)
        self.assertTrue(result.orient.isna().all())
        settings.orientation_fallback = 'heading'
        result = process_orientation(data, settings)
        self.assertTrue((result.orientation_source == 'heading').all())

    def test_processing_preserves_body_measures_and_consistent_tip_names(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'fish.csv'
            raw = pd.DataFrame(dict(frame=range(1,7), id='F01', cx=range(30,36), cy=30,
                hx=range(35,41),hy=30,tx=range(25,31),ty=30,area=100,length=10,
                length_method='centerline',body_mode='skeleton',head_assignment='near_widest',
                center_source='centerline_midpoint',body_straightness=.95,body_file='fish.body.jsonl.gz'))
            raw.to_csv(path,index=False)
            output = process_video(str(path), roi=((0,0),(100,100)),conv=.25,min_traj_len=1,
                smoothwin=1,interpolate=False,compute_distances=False,overwrite=True)
            result = pd.read_csv(output)
            self.assertIn('hx', result)
            self.assertNotIn('fx', result)
            self.assertEqual(result.loc[0,'length'], 10)
            self.assertEqual(result.loc[0,'body_straightness'], .95)
            self.assertTrue(result.orientation_valid.all())

    def test_unassigned_tips_do_not_become_facing_direction(self):
        settings = SimpleNamespace(orientfrombw=False, orientation_fallback='heading', orient_min_speed=1,
                                   interpolate=True, interp_gap_orient=5, smoothwin=1)
        data = pd.DataFrame(dict(ID='a',traj=1,cx=[10,11,12],cy=10,fx=[12,13,14],fy=10,
                                 head_assignment='unassigned',speed=5,heading=90))
        result = process_orientation(data, settings)
        self.assertTrue(result.orient.isna().all())

    def test_tracking_saves_geometry_matching_csv_in_both_modes(self):
        # In-memory decoded frames exercise the tracker without OS video codecs.
        class Capture:
            def __init__(self, frames):
                self.frames=frames; self.position=0; self.open=True
            def set(self, prop, value):
                if prop == cv2.CAP_PROP_POS_FRAMES: self.position=int(value)
            def get(self, prop):
                return self.position if prop == cv2.CAP_PROP_POS_FRAMES else 0
            def isOpened(self): return self.open
            def release(self): self.open=False
            def read(self):
                if self.position >= len(self.frames): return False, None
                frame=self.frames[self.position].copy(); self.position += 1
                return True, frame
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            video=folder/'fish.mp4'; video.write_bytes(b'video placeholder')
            frame=np.full((60,100,3),255,np.uint8)
            cv2.fillPoly(frame,[self.fish()],(0,0,0))
            setup=folder/'fish.atrk'
            save_atcache(str(setup),np.full_like(frame,255),None,
                {'bw':dict(blur=1,blur2=1,erode=1,threshold=40,min_area=10,max_area=5000)},((0,0),(100,60)))
            for mode in ('centroid','skeleton'):
                output=folder/mode; output.mkdir()
                with patch('atracker.standalone.get_vid_params', return_value=(25,100,60,6)), \
                        patch('atracker.track.check_media', return_value=True), \
                        patch('atracker.track.cv2.VideoCapture', side_effect=lambda _: Capture([frame]*6)):
                    csv=track_video(str(video),body_mode=mode,load_settings=str(setup),set_threshold=False,
                        create_vid=False,save_body=True,output=str(output))
                raw=pd.read_csv(csv)
                self.assertEqual(len(raw),6)
                with gzip.open(output/raw.body_file.iloc[0],'rt') as stream:
                    records=[json.loads(line) for line in stream]
                self.assertEqual(records[0]['tracking_run'],raw.tracking_run.iloc[0])
                self.assertEqual(len(records),7)
                self.assertEqual(raw.body_mode.iloc[0],mode)
                if mode=='skeleton':
                    self.assertEqual(len(records[1]['centerline']),10)
                    self.assertTrue(raw.hx.notna().all())
                self.assertTrue(video.exists())

    def test_editor_preserves_raw_body_metadata(self):
        from atracker.helpers.data import load_and_convert_tracking_dataframe
        from atracker.helpers.provenance import mark_manual_edits
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'fish.csv'
            pd.DataFrame(dict(frame=[1],id=['F01'],cx=[10],cy=[20],area=[100],length=[20],
                              body_mode=['centroid'],body_file=['fish.body.jsonl.gz'])).to_csv(path,index=False)
            loaded=load_and_convert_tracking_dataframe(path)
            self.assertEqual(loaded.loc[0,'area'],100)
            original=loaded.copy(); original['id']=original.IDstr
            edited=pd.DataFrame(dict(frame=[1],id=['F01'],cx=[11],cy=[20]))
            saved=mark_manual_edits(edited,original)
            self.assertEqual(saved.loc[0,'body_file'],'fish.body.jsonl.gz')
            self.assertEqual(saved.loc[0,'area'],100)
