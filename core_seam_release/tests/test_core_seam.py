"""Check source geometry, ownership and conservative duplicate handling."""
from pathlib import Path
import contextlib
import io
import json
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import build_core_seam_dataset as build
import merge_core_seam_results as merge
from core_seam_io import read_image, write_image


class CoreSeamTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)/"sample \u9ea6\u82d7"
        self.root.mkdir()
        self.images=self.root/'keyframes_images';self.images.mkdir()
        world=np.random.default_rng(16).integers(20,220,(400,200,3),dtype=np.uint8)
        write_image(self.images/'frame_1.png',world[:300])
        write_image(self.images/'frame_2.png',world[100:400])
        self.keyframes=self.root/'keyframes.csv'
        pd.DataFrame([dict(k=0,file='frame_1.png',x_global_px=0.,y_global_px=0.,img_w=200,img_h=300),
                      dict(k=1,file='frame_2.png',x_global_px=0.,y_global_px=100.,img_w=200,img_h=300)]).to_csv(self.keyframes,index=False)
        with contextlib.redirect_stdout(io.StringIO()):
            self.dataset=build.run(dict(KEYFRAME_CSV=str(self.keyframes),OUT_DIR=str(self.root/'dataset')))
        self.core=pd.read_csv(self.dataset/'core_crops_meta.csv')
        self.seam=pd.read_csv(self.dataset/'seam_crops_meta.csv')
        self.columns=['file','x1','y1','x2','y2','score']

    def tearDown(self):
        self.temp.cleanup()

    def detection(self,meta,x,y,score=.9):
        x-=float(meta.crop_global_x1);y-=float(meta.crop_global_y1)
        return dict(file=meta.crop_file,x1=x-10,y1=y-10,x2=x+10,y2=y+10,score=score)

    def fuse(self,core,seam,**options):
        c=self.root/'core.csv';s=self.root/'seam.csv'
        pd.DataFrame(core,columns=self.columns).to_csv(c,index=False)
        pd.DataFrame(seam,columns=self.columns).to_csv(s,index=False)
        cfg=dict(KEYFRAME_CSV=str(self.keyframes),CORE_META_CSV=str(self.dataset/'core_crops_meta.csv'),
                 SEAM_META_CSV=str(self.dataset/'seam_crops_meta.csv'),STRIP_META_JSON=str(self.dataset/'strip_meta.json'),
                 CORE_DET_CSV=str(c),SEAM_DET_CSV=str(s),OUT_DIR=str(self.root/'merged'))
        cfg.update(options)
        with contextlib.redirect_stdout(io.StringIO()):out=merge.run(cfg)
        return pd.read_csv(out/'dedup_keep_global_core_seam_safe_light_clean.csv'),out

    def test_core_pixels_and_global_crop_origin(self):
        for row in self.core.itertuples():
            image=read_image(self.dataset/'core_crops'/row.crop_file)
            source=read_image(self.images/row.src_file)
            expected=source[row.crop_y1_local:row.crop_y1_local+row.crop_h,row.crop_x1_local:row.crop_x1_local+row.crop_w]
            np.testing.assert_array_equal(image,expected)
            self.assertEqual(row.crop_global_x1,row.frame_fx1+row.crop_x1_local)
            self.assertEqual(row.crop_global_y1,row.frame_fy1+row.crop_y1_local)

    def test_detection_boxes_return_to_global_coordinates(self):
        result,_=self.fuse([self.detection(self.core.iloc[0],30,80)],[])
        self.assertEqual(len(result),1)
        self.assertEqual(result.anchor_x_global.iloc[0],30)
        self.assertEqual(result.anchor_y_global.iloc[0],80)

    def test_missing_seam_detection_retains_eligible_core(self):
        row,y=self.overlapping_owner_point()
        result,_=self.fuse([self.detection(row,30,y)],[])
        self.assertEqual(len(result),1)
        self.assertEqual(result.keep_reason.iloc[0],'core_fallback_no_seam_dets')

    def test_matching_seam_replaces_core(self):
        row,y=self.overlapping_owner_point()
        result,_=self.fuse([self.detection(row,30,y)],[self.detection(self.seam.iloc[0],30,y)])
        self.assertEqual(len(result),1)
        self.assertEqual(result.source.iloc[0],'seam')

    def overlapping_owner_point(self):
        # Exercise fallback with an explicit overlap in the ownership metadata.
        s=float(self.seam.iloc[0].seam_left_s)+.5
        self.core.loc[0,'core_right_s']=s+.5
        self.core.to_csv(self.dataset/'core_crops_meta.csv',index=False)
        reference=json.loads((self.dataset/'strip_meta.json').read_text())
        y=reference['origin'][1]+s/reference['e_along'][1]
        return self.core.iloc[0],y

    def test_context_only_detection_does_not_gain_ownership(self):
        result,_=self.fuse([self.detection(self.core.iloc[0],30,80),self.detection(self.core.iloc[0],30,200)],[])
        self.assertEqual(len(result),1)
        self.assertEqual(result.anchor_y_global.iloc[0],80)

    def test_seam_only_detection_input(self):
        result,_=self.fuse([],[self.detection(self.seam.iloc[0],100,200)])
        self.assertEqual(len(result),1)
        self.assertEqual(result.source.iloc[0],'seam')

    def test_unknown_crop_name_is_rejected(self):
        det=self.detection(self.core.iloc[0],30,80);det['file']='unknown.png'
        with self.assertRaisesRegex(ValueError,'do not match'):
            self.fuse([det],[])

    def test_empty_result_is_explicit_failure(self):
        with self.assertRaisesRegex(RuntimeError,'No final detections'):
            self.fuse([],[])

    def test_residual_duplicate_uses_confidence(self):
        result,out=self.fuse([self.detection(self.core.iloc[0],30,80,.9),self.detection(self.core.iloc[0],30.5,80.5,.7)],[],EXPECTED_ROWS_PER_STRIP=1)
        self.assertEqual(len(result),1)
        self.assertEqual(result.score.iloc[0],.9)
        self.assertEqual(json.loads((out/'run_status.json').read_text())['residual_duplicates_removed'],1)

    def test_cleanup_can_be_disabled(self):
        result,_=self.fuse([self.detection(self.core.iloc[0],30,80,.9),self.detection(self.core.iloc[0],30.5,80.5,.7)],[],DO_FINAL_LIGHT_CLEAN=False,EXPECTED_ROWS_PER_STRIP=1)
        self.assertEqual(len(result),2)

    def test_mixed_frame_sizes_are_rejected(self):
        write_image(self.images/'frame_2.png',np.zeros((301,200,3),dtype=np.uint8))
        with self.assertRaisesRegex(ValueError,'dimensions'):
            with contextlib.redirect_stdout(io.StringIO()):
                build.run(dict(KEYFRAME_CSV=str(self.keyframes),OUT_DIR=str(self.root/'bad')))


if __name__=='__main__':
    unittest.main()
