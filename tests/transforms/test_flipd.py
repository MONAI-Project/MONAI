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
import torch
from parameterized import parameterized

from monai import config
from monai.data.meta_obj import set_track_meta
from monai.data.meta_tensor import MetaTensor
from monai.transforms import Flipd
from tests.lazy_transforms_utils import test_resampler_lazy
from tests.util import TEST_DEVICES, TEST_NDARRAYS_ALL, NumpyImageTestCase2D, assert_allclose, test_local_inversion

INVALID_CASES = [("wrong_axis", ["s", 1], TypeError), ("not_numbers", "s", TypeError)]

VALID_CASES = [("no_axis", None), ("one_axis", 1), ("many_axis", [0, 1])]

TORCH_CASES = []
for track_meta in (False, True):
    for device in TEST_DEVICES:
        TORCH_CASES.append([[0, 1], torch.zeros((1, 3, 2)), track_meta, *device])


class TestFlipd(NumpyImageTestCase2D):
    @parameterized.expand(INVALID_CASES)
    def test_invalid_cases(self, _, spatial_axis, raises):
        with self.assertRaises(raises):
            flip = Flipd(keys="img", spatial_axis=spatial_axis)
            flip({"img": self.imt[0]})

    @parameterized.expand(VALID_CASES)
    def test_correct_results(self, _, spatial_axis):
        for p in TEST_NDARRAYS_ALL:
            init_param = {"keys": "img", "spatial_axis": spatial_axis}
            flip = Flipd(**init_param)
            expected = [np.flip(channel, spatial_axis) for channel in self.imt[0]]
            expected = np.stack(expected)
            im = p(self.imt[0])
            call_param = {"data": {"img": im}}
            result = flip(**call_param)
            test_resampler_lazy(flip, result, init_param, call_param, output_key="img")
            assert_allclose(result["img"], p(expected), type_test="tensor")
            test_local_inversion(flip, {"img": result["img"]}, {"img": im}, "img")

    @parameterized.expand(TORCH_CASES)
    def test_torch(self, spatial_axis, img: torch.Tensor, track_meta: bool, device):
        set_track_meta(track_meta)
        img = img.to(device)
        init_param = {"keys": "image", "spatial_axis": spatial_axis}
        xform = Flipd(**init_param)
        call_param = {"data": {"image": img}}
        res = xform(**call_param)  # type: ignore
        self.assertEqual(img.shape, res["image"].shape)
        if track_meta:
            test_resampler_lazy(xform, res, init_param, call_param, output_key="image")
            self.assertIsInstance(res["image"], MetaTensor)
        else:
            self.assertNotIsInstance(res["image"], MetaTensor)
            self.assertIsInstance(res["image"], torch.Tensor)
            with self.assertRaisesRegex(ValueError, "MetaTensor"):
                xform.inverse(res)

    @unittest.skipIf(not config.USE_META_DICT, "not using meta dict")
    def test_meta_dict(self):
        xform = Flipd("image", [0, 1])
        res = xform({"image": torch.zeros(1, 3, 4)})
        self.assertEqual(res["image"].applied_operations, res["image_transforms"])


if __name__ == "__main__":
    unittest.main()
