# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np
from parameterized import parameterized

from monai.transforms import RandBiasFieldd

TEST_CASES_2D = [{}, (3, 32, 32)]
TEST_CASES_3D = [{}, (3, 32, 32, 32)]
TEST_CASES_2D_ZERO_RANGE = [{"coeff_range": (0.0, 0.0)}, (3, 32, 32)]
TEST_CASES_2D_ONES = [{"coeff_range": (1.0, 1.0)}, np.asarray([[[2, -2], [2, 10]]])]


class TestRandBiasFieldd(unittest.TestCase):
    @parameterized.expand(
        [
            TEST_CASES_2D,
            TEST_CASES_3D,
        ]
    )
    def test_output_shape(self, class_args, img_shape):
        key = "img"
        bias_field = RandBiasFieldd(keys=[key], **class_args)
        img = np.random.rand(*img_shape)
        output = bias_field({key: img})
        np.testing.assert_equal(output[key].shape, img_shape)
        np.testing.assert_equal(output[key].dtype, bias_field.rand_bias_field.dtype)

    @parameterized.expand([TEST_CASES_2D_ZERO_RANGE])
    def test_zero_range(self, class_args, img_shape):
        key = "img"
        bias_field = RandBiasFieldd(keys=[key], **class_args)
        img = np.random.rand(*img_shape)
        output = bias_field({key: img})
        np.testing.assert_equal(output[key], np.zeros(img_shape))

    @parameterized.expand([TEST_CASES_2D_ONES])
    def test_one_range_input(self, class_args, expected):
        key = "img"
        bias_field = RandBiasFieldd(keys=[key], **class_args)
        img = np.ones([1, 2, 2])
        output = bias_field({key: img})
        np.testing.assert_equal(output[key], expected.astype(bias_field.rand_bias_field.dtype))


if __name__ == "__main__":
    unittest.main()
