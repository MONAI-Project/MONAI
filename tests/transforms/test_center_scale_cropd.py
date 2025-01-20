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
from parameterized import parameterized

from monai.transforms import CenterScaleCropd
from tests.croppers import CropTest

TESTS = [
    [{"keys": "img", "roi_scale": [0.6, 0.3, -1]}, (3, 3, 3, 3), (3, 2, 1, 3), False],
    [{"keys": "img", "roi_scale": [0.6, 0.3, -1]}, (3, 3, 1, 3), (3, 2, 1, 3), True],
    [{"keys": "img", "roi_scale": [0.6, 0.3, -1]}, (3, 3, 4, 3), (3, 2, 2, 3), True],
    [{"keys": "img", "roi_scale": 0.6}, (3, 3, 3, 3), (3, 2, 2, 2), True],
    [{"keys": "img", "roi_scale": 0.5}, (3, 3, 3, 3), (3, 2, 2, 2), True],
]

TEST_VALUES = [
    [
        {"keys": "img", "roi_scale": [0.4, 0.4]},
        np.array([[[0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 2, 3, 2, 0], [0, 1, 2, 1, 0], [0, 0, 0, 0, 0]]]),
        np.array([[[1, 2], [2, 3]]]),
    ]
]


class TestCenterScaleCropd(CropTest):
    Cropper = CenterScaleCropd

    @parameterized.expand(TESTS)
    def test_shape(self, input_param, input_shape, expected_shape, _):
        self.crop_test(input_param, input_shape, expected_shape)

    @parameterized.expand(TEST_VALUES)
    def test_value(self, input_param, input_arr, expected_arr):
        self.crop_test_value(input_param, input_arr, expected_arr)

    @parameterized.expand(TESTS)
    def test_pending_ops(self, input_param, input_shape, _, align_corners):
        self.crop_test_pending_ops(input_param, input_shape, align_corners)


if __name__ == "__main__":
    unittest.main()
