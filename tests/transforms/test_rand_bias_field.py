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

from monai.transforms import RandBiasField
from tests.util import TEST_NDARRAYS

TEST_CASES_2D = [{"prob": 1.0}, (3, 32, 32)]
TEST_CASES_3D = [{"prob": 1.0}, (3, 32, 32, 32)]
TEST_CASES_2D_ZERO_RANGE = [{"prob": 1.0, "coeff_range": (0.0, 0.0)}, (2, 3, 3)]
TEST_CASES_2D_ONES = [
    {"prob": 1.0, "coeff_range": (1.0, 1.0)},
    np.asarray([[[7.389056, 0.1353353], [7.389056, 22026.46]]]),
]


class TestRandBiasField(unittest.TestCase):
    @parameterized.expand([TEST_CASES_2D, TEST_CASES_3D])
    def test_output_shape(self, class_args, img_shape):
        for p in TEST_NDARRAYS:
            for degree in [1, 2, 3]:
                bias_field = RandBiasField(degree=degree, **class_args)
                img = p(np.random.rand(*img_shape))
                output = bias_field(img)
                np.testing.assert_equal(output.shape, img_shape)
                self.assertIn(output.dtype, (np.float32, torch.float32))

                img_zero = np.zeros([*img_shape])
                output_zero = bias_field(img_zero)
                np.testing.assert_equal(output_zero, img_zero)

    @parameterized.expand([TEST_CASES_2D_ZERO_RANGE])
    def test_zero_range(self, class_args, img_shape):
        bias_field = RandBiasField(**class_args)
        img = np.ones(img_shape)
        output = bias_field(img)
        np.testing.assert_allclose(output, np.ones(img_shape), rtol=1e-3)

    @parameterized.expand([TEST_CASES_2D_ONES])
    def test_one_range_input(self, class_args, expected):
        bias_field = RandBiasField(**class_args)
        img = np.ones([1, 2, 2])
        output = bias_field(img)
        np.testing.assert_allclose(output, expected.astype(bias_field.dtype), rtol=1e-3)

    def test_zero_prob(self):
        bias_field = RandBiasField(prob=0.0)
        img = np.random.rand(3, 32, 32)
        output = bias_field(img)
        np.testing.assert_equal(output, img)


if __name__ == "__main__":
    unittest.main()
