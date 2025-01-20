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
from itertools import product

import numpy as np
import torch
from parameterized import parameterized

from monai.networks.utils import meshgrid_xy
from monai.transforms import RandSmoothDeformd, RandSmoothFieldAdjustContrastd, RandSmoothFieldAdjustIntensityd
from tests.util import TEST_NDARRAYS, assert_allclose, is_tf32_env

_rtol = 5e-3 if is_tf32_env() else 1e-4

x, y = meshgrid_xy(torch.linspace(-1, 2, 11), torch.linspace(-2.1, 1.2, 8))
pattern2d = x.pow(2).add_(y.pow(2)).sqrt_()

x, y, z = meshgrid_xy(torch.linspace(-1, 2, 11), torch.linspace(-2.1, 1.2, 8), torch.linspace(-0.1, 10.2, 6))
pattern3d = x.pow(2).add_(y.pow(2)).add_(z.pow(2)).sqrt_()

INPUT_SHAPES = ((1, 8, 8), (1, 12, 7), (2, 8, 8), (2, 13, 8), (1, 8, 8, 8), (3, 7, 4, 5))

TESTS_CONTRAST = []
TESTS_INTENSITY = []
TESTS_DEFORM = []

KEY = "test"

for arr_type, shape in product(TEST_NDARRAYS, INPUT_SHAPES):
    in_arr = arr_type(np.ones(shape, np.float32))
    exp_arr = arr_type(np.ones(shape, np.float32))
    rand_size = (4,) * (len(shape) - 1)

    device = torch.device("cpu")

    if isinstance(in_arr, torch.Tensor) and in_arr.get_device() >= 0:
        device = torch.device(in_arr.get_device())

    TESTS_CONTRAST.append(
        (
            {"keys": (KEY,), "spatial_size": shape[1:], "rand_size": rand_size, "prob": 1.0, "device": device},
            {KEY: in_arr},
            {KEY: exp_arr},
        )
    )

    TESTS_INTENSITY.append(
        (
            {
                "keys": (KEY,),
                "spatial_size": shape[1:],
                "rand_size": rand_size,
                "prob": 1.0,
                "device": device,
                "gamma": (0.9, 1),
            },
            {KEY: in_arr},
            {KEY: exp_arr},
        )
    )

    TESTS_DEFORM.append(
        (
            {
                "keys": (KEY,),
                "spatial_size": shape[1:],
                "rand_size": rand_size,
                "prob": 1.0,
                "device": device,
                "def_range": 0.1,
            },
            {KEY: in_arr},
            {KEY: exp_arr},
        )
    )


class TestSmoothField(unittest.TestCase):
    @parameterized.expand(TESTS_CONTRAST)
    def test_rand_smooth_field_adjust_contrastd(self, input_param, input_data, expected_val):
        g = RandSmoothFieldAdjustContrastd(**input_param)
        g.set_random_state(123)

        res = g(input_data)
        for key, result in res.items():
            expected = expected_val[key]
            assert_allclose(result, expected, rtol=_rtol, atol=1e-1, type_test="tensor")

    def test_rand_smooth_field_adjust_contrastd_pad(self):
        input_param, input_data, expected_val = TESTS_CONTRAST[0]

        g = RandSmoothFieldAdjustContrastd(pad=1, **input_param)
        g.set_random_state(123)

        res = g(input_data)
        for key, result in res.items():
            expected = expected_val[key]
            assert_allclose(result, expected, rtol=_rtol, atol=1e-1, type_test="tensor")

    @parameterized.expand(TESTS_INTENSITY)
    def test_rand_smooth_field_adjust_intensityd(self, input_param, input_data, expected_val):
        g = RandSmoothFieldAdjustIntensityd(**input_param)
        g.set_random_state(123)

        res = g(input_data)
        for key, result in res.items():
            expected = expected_val[key]
            assert_allclose(result, expected, rtol=_rtol, atol=1e-1, type_test="tensor")

    def test_rand_smooth_field_adjust_intensityd_pad(self):
        input_param, input_data, expected_val = TESTS_INTENSITY[0]

        g = RandSmoothFieldAdjustIntensityd(pad=1, **input_param)
        g.set_random_state(123)

        res = g(input_data)
        for key, result in res.items():
            expected = expected_val[key]
            assert_allclose(result, expected, rtol=_rtol, atol=1e-1, type_test="tensor")

    @parameterized.expand(TESTS_DEFORM)
    def test_rand_smooth_deformd(self, input_param, input_data, expected_val):
        g = RandSmoothDeformd(**input_param)
        g.set_random_state(123)

        res = g(input_data)
        for key, result in res.items():
            expected = expected_val[key]
            assert_allclose(result, expected, rtol=_rtol, atol=1e-1, type_test="tensor")

    def test_rand_smooth_nodeformd(self):
        """Test input is very close to output when deformation is very low, verifies there's no transposition."""

        for label, im in zip(("2D", "3D"), (pattern2d, pattern3d)):
            with self.subTest(f"Testing {label} case with shape {im.shape}"):
                rsize = (3,) * len(im.shape)
                g = RandSmoothDeformd(
                    keys=(KEY,), spatial_size=im.shape, rand_size=rsize, prob=1.0, device=device, def_range=1e-20
                )
                g.set_random_state(123)

                expected_val = {KEY: im[None]}

                res = g(expected_val)
                for key, result in res.items():
                    expected = expected_val[key]

                    self.assertSequenceEqual(tuple(result.shape), tuple(expected.shape))

                    assert_allclose(result, expected, rtol=_rtol, atol=1e-1, type_test="tensor")

    def test_rand_smooth_deformd_pad(self):
        input_param, input_data, expected_val = TESTS_DEFORM[0]

        g = RandSmoothDeformd(pad=1, **input_param)
        g.set_random_state(123)

        res = g(input_data)
        for key, result in res.items():
            expected = expected_val[key]
            assert_allclose(result, expected, rtol=_rtol, atol=1e-1, type_test="tensor")


if __name__ == "__main__":
    unittest.main()
