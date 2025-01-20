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

import unittest

import numpy as np
import scipy.ndimage
import torch
from parameterized import parameterized

from monai.config import USE_COMPILED
from monai.data import MetaTensor, set_track_meta
from monai.transforms import Rotate
from tests.lazy_transforms_utils import test_resampler_lazy
from tests.util import HAS_CUPY, TEST_NDARRAYS_ALL, NumpyImageTestCase2D, NumpyImageTestCase3D, test_local_inversion

TEST_CASES_2D: list[tuple] = []
for p in TEST_NDARRAYS_ALL:
    TEST_CASES_2D.append((p, np.pi / 6, False, "bilinear", "border", False))
    TEST_CASES_2D.append((p, np.pi / 4, True, "bilinear", "border", False))
    TEST_CASES_2D.append((p, -np.pi / 4.5, True, "nearest", "border" if USE_COMPILED else "reflection", False))
    TEST_CASES_2D.append((p, np.pi, False, "nearest", "zeros", False))
    TEST_CASES_2D.append((p, -np.pi / 2, False, "bilinear", "zeros", True))
    if HAS_CUPY:  # 1 and cuda image requires cupy
        TEST_CASES_2D.append((p, -np.pi / 2, False, 1, "constant", True))

TEST_CASES_3D: list[tuple] = []
for p in TEST_NDARRAYS_ALL:
    TEST_CASES_3D.append((p, -np.pi / 2, True, "nearest", "border", False))
    TEST_CASES_3D.append((p, np.pi / 4, True, "bilinear", "border", False))
    TEST_CASES_3D.append((p, -np.pi / 4.5, True, "nearest", "border" if USE_COMPILED else "reflection", False))
    TEST_CASES_3D.append((p, np.pi, False, "nearest", "zeros", False))
    TEST_CASES_3D.append((p, -np.pi / 2, False, "bilinear", "zeros", False))
    if HAS_CUPY:
        TEST_CASES_3D.append((p, -np.pi / 2, False, 1, "zeros", False))

TEST_CASES_SHAPE_3D: list[tuple] = []
for p in TEST_NDARRAYS_ALL:
    TEST_CASES_SHAPE_3D.append((p, [-np.pi / 2, 1.0, 2.0], "nearest", "border", False))
    TEST_CASES_SHAPE_3D.append((p, [np.pi / 4, 0, 0], "bilinear", "border", False))
    TEST_CASES_SHAPE_3D.append((p, [-np.pi / 4.5, -20, 20], "nearest", "reflection", False))


class TestRotate2D(NumpyImageTestCase2D):
    @parameterized.expand(TEST_CASES_2D)
    def test_correct_results(self, im_type, angle, keep_size, mode, padding_mode, align_corners):
        init_param = {
            "angle": angle,
            "keep_size": keep_size,
            "mode": mode,
            "padding_mode": padding_mode,
            "align_corners": align_corners,
            "dtype": np.float64,
        }
        rotate_fn = Rotate(**init_param)
        call_param = {"img": im_type(self.imt[0])}
        rotated = rotate_fn(**call_param)
        test_resampler_lazy(rotate_fn, rotated, init_param, call_param, atol=1e-4 if USE_COMPILED else 1e-6)
        if keep_size:
            np.testing.assert_allclose(self.imt[0].shape, rotated.shape)
        _order = 0 if mode == "nearest" else 1
        if padding_mode == "border":
            _mode = "nearest"
        elif padding_mode == "reflection":
            _mode = "reflect"
        else:
            _mode = "constant"

        expected = []
        for channel in self.imt[0]:
            expected.append(
                scipy.ndimage.rotate(
                    channel, -np.rad2deg(angle), (0, 1), not keep_size, order=_order, mode=_mode, prefilter=False
                )
            )
        expected = np.stack(expected).astype(np.float32)
        rotated = rotated.cpu() if isinstance(rotated, torch.Tensor) else rotated
        good = np.sum(np.isclose(expected, rotated, atol=1e-3))
        self.assertLessEqual(np.abs(good - expected.size), 5, "diff at most 5 pixels")


class TestRotate3D(NumpyImageTestCase3D):
    @parameterized.expand(TEST_CASES_3D)
    def test_correct_results(self, im_type, angle, keep_size, mode, padding_mode, align_corners):
        init_param = {
            "angle": [angle, 0, 0],
            "keep_size": keep_size,
            "mode": mode,
            "padding_mode": padding_mode,
            "align_corners": align_corners,
            "dtype": np.float64,
        }
        rotate_fn = Rotate(**init_param)
        call_param = {"img": im_type(self.imt[0])}
        rotated = rotate_fn(**call_param)
        test_resampler_lazy(rotate_fn, rotated, init_param, call_param)
        if keep_size:
            np.testing.assert_allclose(self.imt[0].shape, rotated.shape)
        _order = 0 if mode == "nearest" else 1
        if padding_mode == "border":
            _mode = "nearest"
        elif padding_mode == "reflection":
            _mode = "reflect"
        else:
            _mode = "constant"

        expected = []
        for channel in self.imt[0]:
            expected.append(
                scipy.ndimage.rotate(
                    channel, -np.rad2deg(angle), (1, 2), not keep_size, order=_order, mode=_mode, prefilter=False
                )
            )
        expected = np.stack(expected).astype(np.float32)
        rotated = rotated.cpu() if isinstance(rotated, torch.Tensor) else rotated
        n_good = np.sum(np.isclose(expected, rotated, atol=1e-3))
        self.assertLessEqual(expected.size - n_good, 5, "diff at most 5 pixels")

    @parameterized.expand(TEST_CASES_SHAPE_3D)
    def test_correct_shape(self, im_type, angle, mode, padding_mode, align_corners):
        rotate_fn = Rotate(angle, True, align_corners=align_corners, dtype=np.float64)
        im = im_type(self.imt[0])
        set_track_meta(False)
        rotated = rotate_fn(im, mode=mode, padding_mode=padding_mode)
        self.assertNotIsInstance(rotated, MetaTensor)
        np.testing.assert_allclose(self.imt[0].shape, rotated.shape)
        set_track_meta(True)
        rotated = rotate_fn(im, mode=mode, padding_mode=padding_mode)
        np.testing.assert_allclose(self.imt[0].shape, rotated.shape)
        test_local_inversion(rotate_fn, rotated, im)

    def test_ill_case(self):
        for p in TEST_NDARRAYS_ALL:
            rotate_fn = Rotate(10, True)
            with self.assertRaises(ValueError):  # wrong shape
                rotate_fn(p(self.imt))

            rotate_fn = Rotate(10, keep_size=False)
            with self.assertRaises(ValueError):  # wrong mode
                rotate_fn(p(self.imt[0]), mode="trilinear_spell_error")


if __name__ == "__main__":
    unittest.main()
