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

from monai.data import MetaTensor, set_track_meta
from monai.transforms import RandRotate90
from tests.lazy_transforms_utils import test_resampler_lazy
from tests.util import TEST_NDARRAYS_ALL, NumpyImageTestCase2D, assert_allclose, test_local_inversion


class TestRandRotate90(NumpyImageTestCase2D):
    def test_default(self):
        rotate = RandRotate90()
        for p in TEST_NDARRAYS_ALL:
            rotate.set_random_state(123)
            im = p(self.imt[0])
            call_param = {"img": im}
            rotated = rotate(**call_param)
            test_local_inversion(rotate, rotated, im)
            expected = [np.rot90(channel, 0, (0, 1)) for channel in self.imt[0]]
            expected = np.stack(expected)
            assert_allclose(rotated, p(expected), rtol=1.0e-5, atol=1.0e-8, type_test="tensor")

            # test lazy
            test_resampler_lazy(rotate, rotated, call_param=call_param, seed=123)
            rotate.lazy = False

    def test_k(self):
        init_param = {"max_k": 2}
        rotate = RandRotate90(**init_param)
        for p in TEST_NDARRAYS_ALL:
            im = p(self.imt[0])
            set_track_meta(False)
            rotated = rotate(im)
            self.assertNotIsInstance(rotated, MetaTensor)
            self.assertIsInstance(rotated, torch.Tensor)

            set_track_meta(True)
            rotate.set_random_state(123)
            call_param = {"img": im}
            rotated = rotate(**call_param)
            test_local_inversion(rotate, rotated, im)
            expected = [np.rot90(channel, 0, (0, 1)) for channel in self.imt[0]]
            expected = np.stack(expected)
            assert_allclose(rotated, p(expected), rtol=1.0e-5, atol=1.0e-8, type_test="tensor")

            # test lazy
            test_resampler_lazy(rotate, rotated, call_param=call_param, seed=123)
            rotate.lazy = False

    def test_spatial_axes(self):
        rotate = RandRotate90(spatial_axes=(0, 1), prob=1.0)
        for p in TEST_NDARRAYS_ALL:
            rotate.set_random_state(1234)
            im = p(self.imt[0])
            call_param = {"img": im}
            rotated = rotate(**call_param)
            # test lazy
            test_resampler_lazy(rotate, rotated, call_param=call_param, seed=1234)
            rotate.lazy = False

            self.assertEqual(len(rotated.applied_operations), 1)
            expected = [np.rot90(channel, rotate._rand_k, (0, 1)) for channel in self.imt[0]]
            expected = np.stack(expected)
            assert_allclose(rotated, p(expected), rtol=1.0e-5, atol=1.0e-8, type_test="tensor")
            test_local_inversion(rotate, rotated, im)

    def test_prob_k_spatial_axes(self):
        rotate = RandRotate90(prob=1.0, max_k=2, spatial_axes=(0, 1))
        for p in TEST_NDARRAYS_ALL:
            rotate.set_random_state(234)
            im = p(self.imt[0])
            call_param = {"img": im}
            rotated = rotate(**call_param)
            # test lazy
            test_resampler_lazy(rotate, rotated, call_param=call_param, seed=234)
            rotate.lazy = False

            test_local_inversion(rotate, rotated, im)
            expected = [np.rot90(channel, 1, (0, 1)) for channel in self.imt[0]]
            expected = np.stack(expected)
            assert_allclose(rotated, p(expected), rtol=1.0e-5, atol=1.0e-8, type_test="tensor")


if __name__ == "__main__":
    unittest.main()
