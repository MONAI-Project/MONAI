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
from monai.data import MetaTensor
from monai.transforms import Rotated
from tests.lazy_transforms_utils import test_resampler_lazy
from tests.util import TEST_NDARRAYS_ALL, NumpyImageTestCase2D, NumpyImageTestCase3D, test_local_inversion

TEST_CASES_2D: list[tuple] = []
for p in TEST_NDARRAYS_ALL:
    TEST_CASES_2D.append((p, -np.pi / 6, False, "bilinear", "border", False))
    TEST_CASES_2D.append((p, -np.pi / 4, True, "bilinear", "border", False))
    TEST_CASES_2D.append((p, np.pi / 4.5, True, "nearest", "reflection", False))
    TEST_CASES_2D.append((p, -np.pi, False, "nearest", "zeros", False))
    TEST_CASES_2D.append((p, np.pi / 2, False, "bilinear", "zeros", True))

TEST_CASES_3D: list[tuple] = []
for p in TEST_NDARRAYS_ALL:
    TEST_CASES_3D.append((p, -np.pi / 6, False, "bilinear", "border", False))
    TEST_CASES_3D.append((p, -np.pi / 4, True, "bilinear", "border", False))
    TEST_CASES_3D.append((p, np.pi / 4.5, True, "nearest", "reflection", False))
    TEST_CASES_3D.append((p, -np.pi, False, "nearest", "zeros", False))
    TEST_CASES_3D.append((p, np.pi / 2, False, "bilinear", "zeros", True))


@unittest.skipIf(USE_COMPILED, "unittests are not designed for both USE_COMPILED=True/False")
class TestRotated2D(NumpyImageTestCase2D):
    @parameterized.expand(TEST_CASES_2D)
    def test_correct_results(self, im_type, angle, keep_size, mode, padding_mode, align_corners):
        init_param = {
            "keys": ("img", "seg"),
            "angle": angle,
            "keep_size": keep_size,
            "mode": (mode, "nearest"),
            "padding_mode": padding_mode,
            "align_corners": align_corners,
            "dtype": np.float64,
        }
        rotate_fn = Rotated(**init_param)
        im = im_type(self.imt[0])
        call_param = {"data": {"img": im, "seg": im_type(self.segn[0])}}
        rotated = rotate_fn(**call_param)
        # test lazy
        lazy_init_param = init_param.copy()
        for k, m in zip(init_param["keys"], init_param["mode"]):
            lazy_init_param["keys"], lazy_init_param["mode"] = k, m
            test_resampler_lazy(
                rotate_fn, rotated, lazy_init_param, call_param, output_key=k, atol=1e-4 if USE_COMPILED else 1e-6
            )
        if keep_size:
            np.testing.assert_allclose(self.imt[0].shape, rotated["img"].shape)
        _order = 0 if mode == "nearest" else 1
        if padding_mode == "border":
            _mode = "nearest"
        elif padding_mode == "reflection":
            _mode = "reflect"
        else:
            _mode = "constant"
        expected = scipy.ndimage.rotate(
            self.imt[0, 0], -np.rad2deg(angle), (0, 1), not keep_size, order=_order, mode=_mode, prefilter=False
        )
        for k, v in rotated.items():
            rotated[k] = v.cpu() if isinstance(v, torch.Tensor) else v
        good = np.sum(np.isclose(expected, rotated["img"][0], atol=1e-3))
        self.assertLessEqual(np.abs(good - expected.size), 5, "diff at most 5 pixels")
        test_local_inversion(rotate_fn, rotated, {"img": im}, "img")

        expected = scipy.ndimage.rotate(
            self.segn[0, 0], -np.rad2deg(angle), (0, 1), not keep_size, order=0, mode=_mode, prefilter=False
        )
        expected = np.stack(expected).astype(int)
        if isinstance(rotated["seg"], MetaTensor):
            rotated["seg"] = rotated["seg"].as_tensor()  # pytorch 1.7 compatible
        self.assertLessEqual(np.count_nonzero(expected != rotated["seg"][0]), 30)


@unittest.skipIf(USE_COMPILED, "unittests are not designed for both USE_COMPILED=True/False")
class TestRotated3D(NumpyImageTestCase3D):
    @parameterized.expand(TEST_CASES_3D)
    def test_correct_results(self, im_type, angle, keep_size, mode, padding_mode, align_corners):
        init_param = {
            "keys": ("img", "seg"),
            "angle": [0, angle, 0],
            "keep_size": keep_size,
            "mode": (mode, "nearest"),
            "padding_mode": padding_mode,
            "align_corners": align_corners,
            "dtype": np.float64,
        }
        rotate_fn = Rotated(**init_param)
        call_param = {"data": {"img": im_type(self.imt[0]), "seg": im_type(self.segn[0])}}
        rotated = rotate_fn(**call_param)
        # test lazy
        lazy_init_param = init_param.copy()
        for k, m in zip(init_param["keys"], init_param["mode"]):
            lazy_init_param["keys"], lazy_init_param["mode"] = k, m
            test_resampler_lazy(
                rotate_fn, rotated, lazy_init_param, call_param, output_key=k, atol=1e-4 if USE_COMPILED else 1e-6
            )
        if keep_size:
            np.testing.assert_allclose(self.imt[0].shape, rotated["img"].shape)
        _order = 0 if mode == "nearest" else 1
        if padding_mode == "border":
            _mode = "nearest"
        elif padding_mode == "reflection":
            _mode = "reflect"
        else:
            _mode = "constant"
        expected = scipy.ndimage.rotate(
            self.imt[0, 0], np.rad2deg(angle), (0, 2), not keep_size, order=_order, mode=_mode, prefilter=False
        )
        for k, v in rotated.items():
            rotated[k] = v.cpu() if isinstance(v, torch.Tensor) else v
        good = np.sum(np.isclose(expected.astype(np.float32), rotated["img"][0], atol=1e-3))
        self.assertLessEqual(np.abs(good - expected.size), 5, "diff at most 5 voxels.")

        expected = scipy.ndimage.rotate(
            self.segn[0, 0], np.rad2deg(angle), (0, 2), not keep_size, order=0, mode=_mode, prefilter=False
        )
        expected = np.stack(expected).astype(int)
        if isinstance(rotated["seg"], MetaTensor):
            rotated["seg"] = rotated["seg"].as_tensor()  # pytorch 1.7 compatible
        self.assertLessEqual(np.count_nonzero(expected != rotated["seg"][0]), 160)


@unittest.skipIf(USE_COMPILED, "unittests are not designed for both USE_COMPILED=True/False")
class TestRotated3DXY(NumpyImageTestCase3D):
    @parameterized.expand(TEST_CASES_3D)
    def test_correct_results(self, im_type, angle, keep_size, mode, padding_mode, align_corners):
        rotate_fn = Rotated(
            ("img", "seg"), [0, 0, angle], keep_size, (mode, "nearest"), padding_mode, align_corners, dtype=np.float64
        )
        rotated = rotate_fn({"img": im_type(self.imt[0]), "seg": im_type(self.segn[0])})
        if keep_size:
            np.testing.assert_allclose(self.imt[0].shape, rotated["img"].shape)
        _order = 0 if mode == "nearest" else 1
        if padding_mode == "border":
            _mode = "nearest"
        elif padding_mode == "reflection":
            _mode = "reflect"
        else:
            _mode = "constant"
        expected = scipy.ndimage.rotate(
            self.imt[0, 0], -np.rad2deg(angle), (0, 1), not keep_size, order=_order, mode=_mode, prefilter=False
        )
        for k, v in rotated.items():
            rotated[k] = v.cpu() if isinstance(v, torch.Tensor) else v
        good = np.sum(np.isclose(expected, rotated["img"][0], atol=1e-3))
        self.assertLessEqual(np.abs(good - expected.size), 5, "diff at most 5 voxels")

        expected = scipy.ndimage.rotate(
            self.segn[0, 0], -np.rad2deg(angle), (0, 1), not keep_size, order=0, mode=_mode, prefilter=False
        )
        expected = np.stack(expected).astype(int)
        if isinstance(rotated["seg"], MetaTensor):
            rotated["seg"] = rotated["seg"].as_tensor()  # pytorch 1.7 compatible
        self.assertLessEqual(np.count_nonzero(expected != rotated["seg"][0]), 160)


if __name__ == "__main__":
    unittest.main()
