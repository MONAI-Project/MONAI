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

from monai.transforms import MaskIntensityd

TEST_CASE_1 = [
    {"keys": "img", "mask_data": np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]]])},
    {"img": np.array([[[1, 1, 1], [2, 2, 2], [3, 3, 3]], [[4, 4, 4], [5, 5, 5], [6, 6, 6]]])},
    np.array([[[0, 0, 0], [0, 2, 0], [0, 0, 0]], [[0, 0, 0], [0, 5, 0], [0, 0, 0]]]),
]

TEST_CASE_2 = [
    {"keys": "img", "mask_data": np.array([[[0, 0, 0], [0, 5, 0], [0, 0, 0]]])},
    {"img": np.array([[[1, 1, 1], [2, 2, 2], [3, 3, 3]], [[4, 4, 4], [5, 5, 5], [6, 6, 6]]])},
    np.array([[[0, 0, 0], [0, 2, 0], [0, 0, 0]], [[0, 0, 0], [0, 5, 0], [0, 0, 0]]]),
]

TEST_CASE_3 = [
    {"keys": "img", "mask_data": np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 1, 0], [0, 1, 0], [0, 1, 0]]])},
    {"img": np.array([[[1, 1, 1], [2, 2, 2], [3, 3, 3]], [[4, 4, 4], [5, 5, 5], [6, 6, 6]]])},
    np.array([[[0, 0, 0], [0, 2, 0], [0, 0, 0]], [[0, 4, 0], [0, 5, 0], [0, 6, 0]]]),
]

TEST_CASE_4 = [
    {"keys": "img", "mask_key": "mask"},
    {
        "img": np.array([[[1, 1, 1], [2, 2, 2], [3, 3, 3]], [[4, 4, 4], [5, 5, 5], [6, 6, 6]]]),
        "mask": np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 1, 0], [0, 1, 0], [0, 1, 0]]]),
    },
    np.array([[[0, 0, 0], [0, 2, 0], [0, 0, 0]], [[0, 4, 0], [0, 5, 0], [0, 6, 0]]]),
]

TEST_CASE_5 = [
    {
        "keys": "img",
        "mask_data": np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]),
        "select_fn": lambda x: np.where((x > 3) & (x < 7), True, False),
    },
    {"img": np.array([[[1, 1, 1], [2, 2, 2], [3, 3, 3]], [[4, 4, 4], [5, 5, 5], [6, 6, 6]]])},
    np.array([[[0, 0, 0], [2, 2, 2], [0, 0, 0]], [[0, 0, 0], [5, 5, 5], [0, 0, 0]]]),
]


class TestMaskIntensityd(unittest.TestCase):

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4, TEST_CASE_5])
    def test_value(self, arguments, image, expected_data):
        result = MaskIntensityd(**arguments)(image)
        np.testing.assert_allclose(result["img"], expected_data)


if __name__ == "__main__":
    unittest.main()
