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

from monai.data import MetaTensor, set_track_meta
from monai.transforms import RandFlip
from tests.lazy_transforms_utils import test_resampler_lazy
from tests.util import TEST_NDARRAYS_ALL, NumpyImageTestCase2D, assert_allclose, test_local_inversion

INVALID_CASES = [("wrong_axis", ["s", 1], TypeError), ("not_numbers", "s", TypeError)]

VALID_CASES = [("no_axis", None), ("one_axis", 1), ("many_axis", [0, 1])]


class TestRandFlip(NumpyImageTestCase2D):
    @parameterized.expand(INVALID_CASES)
    def test_invalid_inputs(self, _, spatial_axis, raises):
        with self.assertRaises(raises):
            flip = RandFlip(prob=1.0, spatial_axis=spatial_axis)
            flip(self.imt[0])

    @parameterized.expand(VALID_CASES)
    def test_correct_results(self, _, spatial_axis):
        for p in TEST_NDARRAYS_ALL:
            im = p(self.imt[0])
            init_param = {"prob": 1.0, "spatial_axis": spatial_axis}
            flip = RandFlip(**init_param)
            set_track_meta(False)
            result = flip(im)
            self.assertNotIsInstance(result, MetaTensor)
            self.assertIsInstance(result, torch.Tensor)
            set_track_meta(True)
            expected = [np.flip(channel, spatial_axis) for channel in self.imt[0]]
            expected = np.stack(expected)
            call_param = {"img": im}
            result = flip(**call_param)
            assert_allclose(result, p(expected), type_test="tensor")
            test_local_inversion(flip, result, im)

            # test lazy
            test_resampler_lazy(flip, result, init_param, call_param)


if __name__ == "__main__":
    unittest.main()
