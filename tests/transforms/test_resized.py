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
import skimage.transform
import torch
from parameterized import parameterized

from monai.data import MetaTensor, set_track_meta
from monai.transforms import Invertd, Resize, Resized
from tests.lazy_transforms_utils import test_resampler_lazy
from tests.util import (
    TEST_NDARRAYS_ALL,
    NumpyImageTestCase2D,
    SkipIfAtLeastPyTorchVersion,
    assert_allclose,
    test_local_inversion,
)

TEST_CASE_0 = [{"keys": "img", "spatial_size": 15}, (6, 10, 15)]

TEST_CASE_1 = [
    {"keys": "img", "spatial_size": 15, "mode": "area", "anti_aliasing": True, "anti_aliasing_sigma": None},
    (6, 10, 15),
]

TEST_CASE_2 = [
    {"keys": "img", "spatial_size": 6, "mode": "trilinear", "align_corners": True, "anti_aliasing_sigma": 2.0},
    (2, 4, 6),
]

TEST_CASE_3 = [
    {
        "keys": ["img", "label"],
        "spatial_size": 6,
        "mode": ["trilinear", "nearest"],
        "align_corners": [True, None],
        "anti_aliasing": [False, True],
        "anti_aliasing_sigma": (None, 2.0),
    },
    (2, 4, 6),
]

TEST_CORRECT_CASES = [
    ((32, -1), "area", False),
    ((64, 64), "area", True),
    ((32, 32, 32), "area", True),
    ((256, 256), "bilinear", False),
    ((256, 256), "bilinear", True),
    ((128, 128), "nearest", False),
    ((128, 128), "nearest", True),
]


@SkipIfAtLeastPyTorchVersion((2, 2, 0))  # https://github.com/Project-MONAI/MONAI/issues/7445
class TestResized(NumpyImageTestCase2D):
    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            resize = Resized(keys="img", spatial_size=(128, 128, 3), mode="order")
            resize({"img": self.imt[0]})

        with self.assertRaises(ValueError):
            resize = Resized(keys="img", spatial_size=(128,), mode="order")
            resize({"img": self.imt[0]})

    def test_unchange(self):
        resize = Resized(keys="img", spatial_size=(128, 64), mode="bilinear")
        set_track_meta(False)
        for p in TEST_NDARRAYS_ALL:
            im = p(self.imt[0])
            result = resize({"img": im})["img"]
            assert_allclose(im, result, type_test=False)
        set_track_meta(True)

    @parameterized.expand(TEST_CORRECT_CASES)
    def test_correct_results(self, spatial_size, mode, anti_aliasing):
        init_param = {
            "keys": "img",
            "spatial_size": spatial_size,
            "mode": mode,
            "anti_aliasing": anti_aliasing,
            "dtype": np.float32,
        }
        resize = Resized(**init_param)
        _order = 0
        if mode.endswith("linear"):
            _order = 1
        if spatial_size == (32, -1):
            spatial_size = (32, 64)
        expected = [
            skimage.transform.resize(
                channel, spatial_size, order=_order, clip=False, preserve_range=False, anti_aliasing=anti_aliasing
            )
            for channel in self.imt[0]
        ]

        expected = np.stack(expected).astype(np.float32)
        for p in TEST_NDARRAYS_ALL:
            im = p(self.imt[0])
            call_param = {"data": {"img": im}}
            out = resize(**call_param)
            lazy_resize = Resized(**init_param)
            if init_param["mode"] in ("bilinear", "nearest"):
                test_resampler_lazy(lazy_resize, out, init_param, call_param, output_key="img", atol=1e-5)
            test_local_inversion(resize, out, {"img": im}, "img")
            assert_allclose(out["img"], expected, type_test=False, atol=1.0)

    @parameterized.expand([TEST_CASE_0, TEST_CASE_1, TEST_CASE_2, TEST_CASE_3])
    def test_longest_shape(self, input_param, expected_shape):
        input_data = {
            "img": np.random.randint(0, 2, size=[3, 4, 7, 10]),
            "label": np.random.randint(0, 2, size=[3, 4, 7, 10]),
        }
        input_param["size_mode"] = "longest"
        rescaler = Resized(**input_param)
        result = rescaler(input_data)
        for k in rescaler.keys:
            np.testing.assert_allclose(result[k].shape[1:], expected_shape)
        set_track_meta(False)
        result = Resized(**input_param)(input_data)
        self.assertNotIsInstance(result["img"], MetaTensor)
        np.testing.assert_allclose(result["img"].shape[1:], expected_shape)
        set_track_meta(True)

    def test_identical_spatial(self):
        test_input = {"X": np.ones((1, 10, 16, 17))}
        xform = Resized("X", (-1, 16, 17))
        out = xform(test_input)
        out["Y"] = 2 * out["X"]
        transform_inverse = Invertd(keys="Y", transform=xform, orig_keys="X")
        assert_allclose(transform_inverse(out)["Y"].array, np.ones((1, 10, 16, 17)) * 2)

    def test_consistent_resize(self):
        spatial_size = (16, 16, 16)
        rescaler_1 = Resize(spatial_size=spatial_size, anti_aliasing=True, anti_aliasing_sigma=(0.5, 1.0, 2.0))
        rescaler_2 = Resize(spatial_size=spatial_size, anti_aliasing=True, anti_aliasing_sigma=None)
        rescaler_dict = Resized(
            keys=["img1", "img2"],
            spatial_size=spatial_size,
            anti_aliasing=(True, True),
            anti_aliasing_sigma=[(0.5, 1.0, 2.0), None],
        )
        test_input_1 = torch.randn([3, 32, 32, 32])
        test_input_2 = torch.randn([3, 32, 32, 32])
        test_input_dict = {"img1": test_input_1, "img2": test_input_2}
        assert_allclose(rescaler_1(test_input_1), rescaler_dict(test_input_dict)["img1"])
        assert_allclose(rescaler_2(test_input_2), rescaler_dict(test_input_dict)["img2"])


if __name__ == "__main__":
    unittest.main()
