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

from monai.transforms.intensity.array import ScaleIntensityRangePercentiles
from tests.util import TEST_NDARRAYS, NumpyImageTestCase2D, assert_allclose


class TestScaleIntensityRangePercentiles(NumpyImageTestCase2D):
    def test_scaling(self):
        img = self.imt[0]
        lower = 10
        upper = 99
        b_min = 0
        b_max = 255

        a_min = np.percentile(img, lower)
        a_max = np.percentile(img, upper)
        expected = (((img - a_min) / (a_max - a_min)) * (b_max - b_min) + b_min).astype(np.uint8)
        scaler = ScaleIntensityRangePercentiles(lower=lower, upper=upper, b_min=b_min, b_max=b_max, dtype=np.uint8)
        for p in TEST_NDARRAYS:
            result = scaler(p(img))
            self.assertEqual(result.dtype, torch.uint8)
            assert_allclose(result, p(expected), type_test="tensor", rtol=1e-4)

    def test_relative_scaling(self):
        img = self.imt[0]
        lower = 10
        upper = 99
        b_min = 100
        b_max = 300
        scaler = ScaleIntensityRangePercentiles(lower=lower, upper=upper, b_min=b_min, b_max=b_max, relative=True)

        expected_a_min = np.percentile(img, lower)
        expected_a_max = np.percentile(img, upper)
        expected_b_min = ((b_max - b_min) * (lower / 100.0)) + b_min
        expected_b_max = ((b_max - b_min) * (upper / 100.0)) + b_min
        expected_img = (img - expected_a_min) / (expected_a_max - expected_a_min)
        expected_img = (expected_img * (expected_b_max - expected_b_min)) + expected_b_min

        for p in TEST_NDARRAYS:
            result = scaler(p(img))
            assert_allclose(result, p(expected_img), type_test="tensor", rtol=1e-3)

        scaler = ScaleIntensityRangePercentiles(
            lower=lower, upper=upper, b_min=b_min, b_max=b_max, relative=True, clip=True
        )
        for p in TEST_NDARRAYS:
            result = scaler(p(img))
            assert_allclose(
                result, p(np.clip(expected_img, expected_b_min, expected_b_max)), type_test="tensor", rtol=0.1
            )

    def test_invalid_instantiation(self):
        self.assertRaises(ValueError, ScaleIntensityRangePercentiles, lower=-10, upper=99, b_min=0, b_max=255)
        self.assertRaises(ValueError, ScaleIntensityRangePercentiles, lower=101, upper=99, b_min=0, b_max=255)
        self.assertRaises(ValueError, ScaleIntensityRangePercentiles, lower=30, upper=-20, b_min=0, b_max=255)
        self.assertRaises(ValueError, ScaleIntensityRangePercentiles, lower=30, upper=900, b_min=0, b_max=255)

    def test_channel_wise(self):
        img = np.tile(self.imt, (3, 1, 1, 1))
        lower = 10
        upper = 99
        b_min = 0
        b_max = 255
        scaler = ScaleIntensityRangePercentiles(
            lower=lower, upper=upper, b_min=b_min, b_max=b_max, channel_wise=True, dtype=np.uint8
        )
        expected = []
        for c in img:
            a_min = np.percentile(c, lower)
            a_max = np.percentile(c, upper)
            expected.append((((c - a_min) / (a_max - a_min)) * (b_max - b_min) + b_min).astype(np.uint8))
        expected = np.stack(expected)

        for p in TEST_NDARRAYS:
            result = scaler(p(img))
            assert_allclose(result, p(expected), type_test="tensor", rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
