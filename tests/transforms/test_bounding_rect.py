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

import monai
from monai.transforms import BoundingRect
from tests.util import TEST_NDARRAYS

TEST_CASE_1 = [(2, 3), [[0, 0], [1, 2]]]

TEST_CASE_2 = [(1, 8, 10), [[0, 7, 1, 9]]]

TEST_CASE_3 = [(2, 16, 20, 18), [[0, 16, 0, 20, 0, 18], [0, 16, 0, 20, 0, 18]]]


class TestBoundingRect(unittest.TestCase):
    def setUp(self):
        monai.utils.set_determinism(1)

    def tearDown(self):
        monai.utils.set_determinism(None)

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3])
    def test_shape(self, input_shape, expected):
        test_data = np.random.randint(0, 8, size=input_shape)
        test_data = test_data == 7
        for p in TEST_NDARRAYS:
            result = BoundingRect()(p(test_data))
            np.testing.assert_allclose(result, expected)

    def test_select_fn(self):
        test_data = np.random.randint(0, 8, size=(2, 3))
        test_data = test_data == 7
        for p in TEST_NDARRAYS:
            bbox = BoundingRect(select_fn=lambda x: x < 1)(p(test_data))
            np.testing.assert_allclose(bbox, [[0, 3], [0, 3]])


if __name__ == "__main__":
    unittest.main()
