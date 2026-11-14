"""Check coordinate conventions, source-image export and execution boundaries."""
from pathlib import Path
import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import pandas as pd
import strip_registration as registration


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        registration.CONFIG.clear()
        registration.CONFIG.update(registration.DEFAULT_CONFIG)

    def tearDown(self):
        self.temp.cleanup()

    def frames(self, count=8, rotation=0):
        directory = self.root / "frames"
        directory.mkdir()
        rng = np.random.default_rng(5614)
        world = rng.integers(30, 220, (384 + 32 * count, 512, 3), dtype=np.uint8)
        world = cv2.GaussianBlur(world, (3, 3), .6)
        for i in range(count):
            image = world[32 * i:32 * i + 384].copy()
            image = registration.rotate_img(image, (-rotation) % 360)
            registration.write_image(directory / f"frame_{i + 1}.png", image)
        return directory

    def execute(self, frames, output="results", **settings):
        options = dict(ROTATE_DEG=0, RENDER_PREVIEW=False)
        options.update(settings)
        with contextlib.redirect_stdout(io.StringIO()):
            return registration.run(frames, self.root / output, options)

    def test_synthetic_translation_uses_pixel_coordinates(self):
        out = self.execute(self.frames())
        poses = pd.read_csv(out / "poses_all_corrected.csv")
        expected_y = np.arange(len(poses)) * 32
        np.testing.assert_allclose(poses.y_corrected, expected_y, atol=5, rtol=0)
        self.assertTrue(np.all(np.diff(poses.y_corrected) > 0))
        self.assertEqual(json.loads((out / "meta.json").read_text())["sign_y"], 1)

    def test_rotated_exports_are_source_pixels(self):
        frames = self.frames(rotation=90)
        original = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in frames.iterdir()}
        out = self.execute(frames, ROTATE_DEG=90)
        selected = pd.read_csv(out / "keyframes.csv")
        for row in selected.itertuples():
            source = registration.read_image(frames / row.file)
            expected = cv2.rotate(source, cv2.ROTATE_90_CLOCKWISE)
            actual = registration.read_image(out / "keyframes_images" / row.file)
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual((row.img_h, row.img_w), actual.shape[:2])
        self.assertEqual(original, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in frames.iterdir()})

    def test_unicode_image_paths_round_trip(self):
        directory = self.root / "source images \u9ea6\u82d7"
        directory.mkdir()
        image = np.random.default_rng(5).integers(0, 255, (60, 80, 3), dtype=np.uint8)
        path = directory / "frame \u7a7a\u683c.png"
        registration.write_image(path, image)
        np.testing.assert_array_equal(registration.read_image(path), image)

    def test_unreadable_image_fails_explicitly(self):
        path = self.root / "corrupt.png"
        path.write_bytes(b"not an image")
        with self.assertRaises(OSError):
            registration.read_image(path)

    def test_inconsistent_dimensions_record_failure(self):
        frames = self.frames(count=2)
        registration.write_image(frames / "frame_2.png", np.zeros((400, 512, 3), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "same dimensions"):
            self.execute(frames)
        status = json.loads((self.root / "results" / "run_status.json").read_text())
        self.assertEqual(status["status"], "failed")

    def test_fallbacks_are_retained_and_reported(self):
        frames = self.frames(count=2)
        with patch.object(registration, "estimate_shift", return_value=(0., 0., .9, "phase")):
            out = self.execute(frames)
        poses = pd.read_csv(out / "poses_all_corrected.csv")
        anchors = pd.read_csv(out / "anchors_refined.csv")
        self.assertEqual(poses.method.iloc[1], "pred")
        self.assertTrue(anchors.method.str.startswith("fallback_coarse").all())
        status = json.loads((out / "run_status.json").read_text())
        self.assertEqual(status["status"], "completed_needs_review")
        self.assertEqual(status["predicted_frame_steps"], 1)

    def test_nonempty_output_keeps_previous_files(self):
        frames = self.frames(count=2)
        prior = self.root / "results"
        prior.mkdir()
        marker = prior / "existing.txt"
        marker.write_text("keep")
        out = self.execute(frames)
        self.assertEqual(out.parent, prior)
        self.assertTrue(out.name.startswith("run_"))
        self.assertEqual(marker.read_text(), "keep")

    def test_ncc_fallback_matches_known_shift(self):
        frames = self.frames()
        a = registration.read_image(frames / "frame_1.png")
        b = registration.read_image(frames / "frame_2.png")
        with patch.object(registration, "phase_shift", return_value=(0., 0., 0.)):
            dx, dy, score, method = registration.estimate_shift(a, b, .25, pred_sy_full=32, sign_y=1, H_full=384)
        self.assertEqual(method, "ncc")
        self.assertAlmostEqual(dy, -32, delta=4)
        self.assertAlmostEqual(dx, 0, delta=4)

    def test_unrotated_exports_cannot_mismatch_coordinate_frame(self):
        with self.assertRaisesRegex(ValueError, "images match"):
            registration.validate_config(dict(registration.DEFAULT_CONFIG, ROTATE_DEG=90, SAVE_ROTATED_KEYFRAMES=False))


if __name__ == "__main__":
    unittest.main()
