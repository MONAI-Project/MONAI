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

from monai.data.meta_obj import set_track_meta
from monai.data.meta_tensor import MetaTensor
from monai.transforms import Flip
from tests.lazy_transforms_utils import test_resampler_lazy
from tests.util import TEST_DEVICES, TEST_NDARRAYS_ALL, NumpyImageTestCase2D, assert_allclose, test_local_inversion

INVALID_CASES = [("wrong_axis", ["s", 1], TypeError), ("not_numbers", "s", TypeError)]

VALID_CASES = [("no_axis", None), ("one_axis", 1), ("many_axis", [0, 1]), ("negative_axis", [0, -1])]

TORCH_CASES = []
for track_meta in (False, True):
    for device in TEST_DEVICES:
        TORCH_CASES.append([[0, 1], torch.zeros((1, 3, 2)), track_meta, *device])


class TestFlip(NumpyImageTestCase2D):
    @parameterized.expand(INVALID_CASES)
    def test_invalid_inputs(self, _, spatial_axis, raises):
        with self.assertRaises(raises):
            flip = Flip(spatial_axis)
            flip(self.imt[0])

    @parameterized.expand(VALID_CASES)
    def test_correct_results(self, _, spatial_axis):
        for p in TEST_NDARRAYS_ALL:
            im = p(self.imt[0])
            init_param = {"spatial_axis": spatial_axis}
            flip = Flip(**init_param)
            expected = [np.flip(channel, spatial_axis) for channel in self.imt[0]]
            expected = np.stack(expected)
            call_param = {"img": im}
            result = flip(**call_param)
            test_resampler_lazy(flip, result, init_param, call_param)
            assert_allclose(result, p(expected), type_test="tensor")
            test_local_inversion(flip, result, im)

    @parameterized.expand(TORCH_CASES)
    def test_torch(self, spatial_axis, img: torch.Tensor, track_meta: bool, device):
        set_track_meta(track_meta)
        img = img.to(device)
        init_param = {"spatial_axis": spatial_axis}
        xform = Flip(**init_param)
        call_param = {"img": img}
        res = xform(**call_param)  # type: ignore[arg-type]
        self.assertEqual(img.shape, res.shape)
        if track_meta:
            test_resampler_lazy(xform, res, init_param, call_param)
            self.assertIsInstance(res, MetaTensor)
        else:
            self.assertNotIsInstance(res, MetaTensor)
            self.assertIsInstance(res, torch.Tensor)
            with self.assertRaisesRegex(ValueError, "MetaTensor"):
                xform.inverse(res)


if __name__ == "__main__":
    unittest.main()
