# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os
import unittest
from pathlib import Path

import torch

import monai.transforms as mt
from monai.data.dataloader import DataLoader
from monai.data.video_dataset import CameraDataset, VideoDataset, VideoFileDataset
from monai.utils.module import optional_import
from tests.util import assert_allclose, download_url_or_skip_test, testing_data_config

cv2, has_cv2 = optional_import("cv2")

NUM_CAPTURE_DEVICES = CameraDataset.get_num_devices()
TRANSFORMS = mt.Compose(
    [mt.EnsureChannelFirst(True, "no_channel"), mt.DivisiblePad(16), mt.ScaleIntensity(), mt.CastToType(torch.float32)]
)


class Base:
    class TestVideoDataset(unittest.TestCase):
        video_source: int | str
        ds: type[VideoDataset]

        def get_video_source(self):
            return self.video_source

        def get_ds(self, *args, **kwargs) -> VideoDataset:
            return self.ds(*args, video_source=self.get_video_source(), transform=TRANSFORMS, **kwargs)  # type: ignore

        @unittest.skipIf(has_cv2, "Only tested when OpenCV not installed.")
        def test_no_opencv_raises(self):
            with self.assertRaises(RuntimeError):
                _ = self.get_ds(max_num_frames=10)

        @unittest.skipUnless(has_cv2, "OpenCV required.")
        def test_multiprocessing(self):
            for num_workers in (0, 2):
                multiprocessing = num_workers > 0
                ds = self.get_ds(max_num_frames=100, multiprocessing=multiprocessing)
                dl = DataLoader(ds, num_workers=num_workers, batch_size=2)
                _ = next(iter(dl))

        @unittest.skipUnless(has_cv2, "OpenCV required.")
        def test_multiple_sources(self, should_match: bool = True):
            ds1 = self.get_ds()
            ds2 = self.get_ds()
            if should_match:
                assert_allclose(ds1.get_frame(), ds2.get_frame())

        @unittest.skipUnless(has_cv2, "OpenCV required.")
        def test_dataset(self, known_num_frames=None, known_fps=None):
            num_frames = (10,) if known_num_frames is None else (10, None)
            for max_num_frames in num_frames:
                ds = self.get_ds(max_num_frames=max_num_frames)
                if known_fps is not None:
                    self.assertEqual(ds.get_fps(), known_fps)
                frames = list(ds)
                if max_num_frames is not None:
                    self.assertEqual(len(frames), max_num_frames)
                elif known_num_frames is not None:
                    self.assertEqual(len(frames), len(ds))
                for f in frames:
                    self.assertTupleEqual(f.shape, frames[0].shape)


@unittest.skipIf(NUM_CAPTURE_DEVICES == 0, "At least one capture device required.")
class TestCameraDataset(Base.TestVideoDataset):
    video_source = 0
    ds = CameraDataset

    @unittest.skipUnless(has_cv2, "OpenCV required.")
    def test_multiple_sources(self):
        super().test_multiple_sources(should_match=False)

    @unittest.skipUnless(has_cv2, "OpenCV required.")
    def test_device_out_of_range(self):
        capture_device = NUM_CAPTURE_DEVICES + 1
        with self.assertRaises(RuntimeError):
            _ = CameraDataset(capture_device, TRANSFORMS, 0)


class TestVideoFileDataset(Base.TestVideoDataset):
    ds = VideoFileDataset

    @classmethod
    def setUpClass(cls):
        super(__class__, cls).setUpClass()
        codecs = VideoFileDataset.get_available_codecs()
        if ".mp4" in codecs.values():
            fname = "endo.mp4"
            config = testing_data_config("videos", "endovis")
            cls.known_fps = 2.0
            cls.known_num_frames = 23
        elif ".avi" in codecs.values():
            fname = "ultrasound.avi"
            config = testing_data_config("videos", "ultrasound")
            cls.known_fps = 2.0
            cls.known_num_frames = 523
        else:
            cls.known_fps = None
            cls.known_num_frames = None
            cls.video_source = None
            return
        tests_path = Path(__file__).parents[1].as_posix()
        cls.video_source = os.path.join(tests_path, "testing_data", fname)
        download_url_or_skip_test(
            url=config["url"],
            filepath=cls.video_source,
            hash_val=config.get("hash_val"),
            hash_type=config.get("hash_type", "sha256"),
        )

    @unittest.skipUnless(has_cv2, "OpenCV required.")
    def test_dataset(self):
        super().test_dataset(self.known_num_frames, self.known_fps)
        self.assertEqual(self.get_ds().get_num_frames(), self.known_num_frames)

    def test_available_codecs(self):
        codecs = VideoFileDataset.get_available_codecs()
        if not has_cv2:
            self.assertEqual(codecs, {})
        else:
            self.assertGreaterEqual(len(codecs), 0)

    def get_video_source(self):
        if self.video_source is None:
            raise unittest.SkipTest("missing required codecs")
        return super().get_video_source()


if __name__ == "__main__":
    unittest.main()
