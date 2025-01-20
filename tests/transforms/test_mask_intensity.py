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

from monai.transforms import MaskIntensity
from tests.util import TEST_NDARRAYS, assert_allclose

TEST_CASE_1 = [
    {"mask_data": np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]]])},
    np.array([[[1, 1, 1], [2, 2, 2], [3, 3, 3]], [[4, 4, 4], [5, 5, 5], [6, 6, 6]]]),
    np.array([[[0, 0, 0], [0, 2, 0], [0, 0, 0]], [[0, 0, 0], [0, 5, 0], [0, 0, 0]]]),
]

TEST_CASE_2 = [
    {"mask_data": np.array([[[0, 0, 0], [0, 5, 0], [0, 0, 0]]])},
    np.array([[[1, 1, 1], [2, 2, 2], [3, 3, 3]], [[4, 4, 4], [5, 5, 5], [6, 6, 6]]]),
    np.array([[[0, 0, 0], [0, 2, 0], [0, 0, 0]], [[0, 0, 0], [0, 5, 0], [0, 0, 0]]]),
]

TEST_CASE_3 = [
    {"mask_data": np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 1, 0], [0, 1, 0], [0, 1, 0]]])},
    np.array([[[1, 1, 1], [2, 2, 2], [3, 3, 3]], [[4, 4, 4], [5, 5, 5], [6, 6, 6]]]),
    np.array([[[0, 0, 0], [0, 2, 0], [0, 0, 0]], [[0, 4, 0], [0, 5, 0], [0, 6, 0]]]),
]

TEST_CASE_4 = [
    {
        "mask_data": np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]),
        "select_fn": lambda x: np.where((x > 3) & (x < 7), True, False),
    },
    np.array([[[1, 1, 1], [2, 2, 2], [3, 3, 3]], [[4, 4, 4], [5, 5, 5], [6, 6, 6]]]),
    np.array([[[0, 0, 0], [2, 2, 2], [0, 0, 0]], [[0, 0, 0], [5, 5, 5], [0, 0, 0]]]),
]

TEST_CASE_5 = [
    {"mask_data": np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 1, 0], [0, 1, 0], [0, 1, 0]]])},
    torch.as_tensor([[[1, 1, 1], [2, 2, 2], [3, 3, 3]], [[4, 4, 4], [5, 5, 5], [6, 6, 6]]]),
    torch.as_tensor([[[0, 0, 0], [0, 2, 0], [0, 0, 0]], [[0, 4, 0], [0, 5, 0], [0, 6, 0]]]),
]


class TestMaskIntensity(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4, TEST_CASE_5])
    def test_value(self, arguments, image, expected_data):
        for p in TEST_NDARRAYS:
            result = MaskIntensity(**arguments)(p(image))
            assert_allclose(result, p(expected_data), type_test="tensor")

    def test_runtime_mask(self):
        mask_data = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]]])
        img = np.array([[[1, 1, 1], [2, 2, 2], [3, 3, 3]], [[4, 4, 4], [5, 5, 5], [6, 6, 6]]])
        expected = np.array([[[0, 0, 0], [0, 2, 0], [0, 0, 0]], [[0, 0, 0], [0, 5, 0], [0, 0, 0]]])

        result = MaskIntensity()(img=img, mask_data=mask_data)
        assert_allclose(result, expected, type_test="tensor")


if __name__ == "__main__":
    unittest.main()
