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
from monai.transforms import RandRotate
from tests.lazy_transforms_utils import test_resampler_lazy
from tests.util import (
    TEST_NDARRAYS_ALL,
    NumpyImageTestCase2D,
    NumpyImageTestCase3D,
    assert_allclose,
    test_local_inversion,
)

TEST_CASES_2D: list[tuple] = []
for p in TEST_NDARRAYS_ALL:
    TEST_CASES_2D.append((p, np.pi / 2, True, "bilinear", "border", False))
    TEST_CASES_2D.append((p, np.pi / 4, True, "nearest", "border", False))
    TEST_CASES_2D.append((p, np.pi, False, "nearest", "zeros", True))
    if not USE_COMPILED:
        TEST_CASES_2D.append((p, (-np.pi / 4, 0), False, "nearest", "zeros", True))

TEST_CASES_3D: list[tuple] = []
for p in TEST_NDARRAYS_ALL:
    TEST_CASES_3D.append(
        (p, np.pi / 2, -np.pi / 6, (0.0, np.pi), False, "bilinear", "border", False, (1, 81, 110, 112))
    )
    TEST_CASES_3D.append(
        (
            p,
            np.pi / 4,
            (-np.pi / 9, np.pi / 4.5),
            (np.pi / 9, np.pi / 6),
            False,
            "nearest",
            "border",
            True,
            (1, 97, 100, 97),
        )
    )
    TEST_CASES_3D.append(
        (
            p,
            0.0,
            (2 * np.pi, 2.06 * np.pi),
            (-np.pi / 180, np.pi / 180),
            True,
            "nearest",
            "zeros",
            True,
            (1, 64, 48, 80),
        )
    )
    TEST_CASES_3D.append((p, (-np.pi / 4, 0), 0, 0, False, "nearest", "zeros", False, (1, 64, 61, 87)))


class TestRandRotate2D(NumpyImageTestCase2D):
    @parameterized.expand(TEST_CASES_2D)
    def test_correct_results(self, im_type, degrees, keep_size, mode, padding_mode, align_corners):
        init_param = {
            "range_x": degrees,
            "prob": 1.0,
            "keep_size": keep_size,
            "mode": mode,
            "padding_mode": padding_mode,
            "align_corners": align_corners,
            "dtype": np.float64,
        }
        rotate_fn = RandRotate(**init_param)
        rotate_fn.set_random_state(243)
        call_param = {"img": im_type(self.imt[0])}
        rotated = rotate_fn(**call_param)

        # test lazy
        test_resampler_lazy(rotate_fn, rotated, init_param=init_param, call_param=call_param, seed=243)
        rotate_fn.lazy = False

        _order = 0 if mode == "nearest" else 1
        if mode == "border":
            _mode = "nearest"
        elif mode == "reflection":
            _mode = "reflect"
        else:
            _mode = "constant"
        angle = rotate_fn.x
        expected = scipy.ndimage.rotate(
            self.imt[0, 0], -np.rad2deg(angle), (0, 1), not keep_size, order=_order, mode=_mode, prefilter=False
        )
        expected = np.stack(expected).astype(np.float32)
        rotated = rotated.cpu() if isinstance(rotated, torch.Tensor) else rotated
        good = np.sum(np.isclose(expected, rotated[0], atol=1e-3))
        self.assertLessEqual(np.abs(good - expected.size), 40, "diff at most 40 pixels")


@unittest.skipIf(USE_COMPILED, "unit tests not for compiled version.")
class TestRandRotate3D(NumpyImageTestCase3D):
    @parameterized.expand(TEST_CASES_3D)
    def test_correct_results(self, im_type, x, y, z, keep_size, mode, padding_mode, align_corners, expected):
        init_param = {
            "range_x": x,
            "range_y": y,
            "range_z": z,
            "prob": 1.0,
            "keep_size": keep_size,
            "mode": mode,
            "padding_mode": padding_mode,
            "align_corners": align_corners,
            "dtype": np.float64,
        }
        rotate_fn = RandRotate(**init_param)
        rotate_fn.set_random_state(243)
        im = im_type(self.imt[0])
        call_param = {"img": im}
        rotated = rotate_fn(**call_param)

        # test lazy
        test_resampler_lazy(rotate_fn, rotated, init_param=init_param, call_param=call_param, seed=243)
        rotate_fn.lazy = False

        assert_allclose(rotated.shape, expected, rtol=1e-7, atol=0)
        test_local_inversion(rotate_fn, rotated, im)

        set_track_meta(False)
        rotated = rotate_fn(im)
        self.assertNotIsInstance(rotated, MetaTensor)
        self.assertIsInstance(rotated, torch.Tensor)
        set_track_meta(True)


class TestRandRotateDtype(NumpyImageTestCase2D):
    @parameterized.expand(TEST_CASES_2D)
    def test_correct_results(self, im_type, degrees, keep_size, mode, padding_mode, align_corners):
        rotate_fn = RandRotate(
            range_x=1.0,
            prob=0.5,
            keep_size=keep_size,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
            dtype=np.float64,
        )
        im = im_type(self.imt[0])
        rotated = rotate_fn(im)
        self.assertEqual(rotated.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
