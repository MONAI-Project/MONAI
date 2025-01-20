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

from monai.apps.pathology.transforms.post.dictionary import GenerateSuccinctContourd

y, x = np.ogrid[0:5, 0:5]
TEST_CASE_1 = [
    [
        np.array([[1.5, 0.0], [1.0, 0.5], [0.5, 1.0], [0.0, 1.5]]),
        np.array([[0.0, 2.5], [0.5, 3.0], [1.0, 3.5], [1.5, 4.0]]),
        np.array([[4.0, 1.5], [3.5, 1.0], [3.0, 0.5], [2.5, 0.0]]),
        np.array([[2.5, 4.0], [3.0, 3.5], [3.5, 3.0], [4.0, 2.5]]),
    ],
    5,
    5,
    [[2, 0], [0, 2], [2, 4], [4, 2]],
]

TEST_CASE_2 = [
    [
        np.array([[1.5, 0.0], [1.0, 0.5], [0.5, 1.0], [0.5, 2.0], [0.0, 2.5]]),
        np.array([[0.0, 3.5], [0.5, 4.0], [0.5, 5.0], [1.0, 5.5], [1.5, 6.0]]),
        np.array([[4.0, 2.5], [3.5, 2.0], [3.5, 1.0], [3.0, 0.5], [2.5, 0.0]]),
        np.array([[2.5, 6.0], [3.0, 5.5], [3.5, 5.0], [3.5, 4.0], [4.0, 3.5]]),
    ],
    5,
    7,
    [[3, 0], [2, 1], [1, 1], [0, 2], [1, 3], [2, 3], [3, 4], [4, 3], [5, 3], [6, 2], [5, 1], [4, 1]],
]


class TestGenerateSuccinctContour(unittest.TestCase):

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    def test_shape(self, data, height, width, expected):
        test_data = {"contour": data}
        result = GenerateSuccinctContourd(keys="contour", height=height, width=width)(test_data)
        np.testing.assert_allclose(result["contour"], expected)


if __name__ == "__main__":
    unittest.main()
