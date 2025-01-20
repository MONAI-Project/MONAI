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

from parameterized import parameterized

from monai.transforms import FgBgToIndices
from tests.util import TEST_NDARRAYS, assert_allclose

TESTS_CASES = []
for p in TEST_NDARRAYS:
    TESTS_CASES.append(
        [
            {"image_threshold": 0.0, "output_shape": None},
            p([[[0, 1, 1], [1, 0, 1], [1, 1, 0]]]),
            None,
            p([1, 2, 3, 5, 6, 7]),
            p([0, 4, 8]),
        ]
    )

    TESTS_CASES.append(
        [
            {"image_threshold": 0.0, "output_shape": None},
            p([[[0, 1, 1], [1, 0, 1], [1, 1, 0]]]),
            p([[[1, 1, 1], [1, 0, 1], [1, 1, 1]]]),
            p([1, 2, 3, 5, 6, 7]),
            p([0, 8]),
        ]
    )

    TESTS_CASES.append(
        [
            {"image_threshold": 1.0, "output_shape": None},
            p([[[0, 1, 1], [1, 0, 1], [1, 1, 0]]]),
            p([[[3, 3, 3], [3, 1, 3], [3, 3, 3]]]),
            p([1, 2, 3, 5, 6, 7]),
            p([0, 8]),
        ]
    )

    TESTS_CASES.append(
        [
            {"image_threshold": 1.0, "output_shape": None},
            p([[[0, 1, 2], [3, 0, 4], [5, 6, 0]]]),
            p([[[3, 3, 3], [3, 1, 3], [3, 3, 3]]]),
            p([1, 2, 3, 5, 6, 7]),
            p([0, 8]),
        ]
    )

    TESTS_CASES.append(
        [
            {"image_threshold": 1.0, "output_shape": [3, 3]},
            p([[[0, 1, 2], [3, 0, 4], [5, 6, 0]]]),
            p([[[3, 3, 3], [3, 1, 3], [3, 3, 3]]]),
            p([[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]]),
            p([[0, 0], [2, 2]]),
        ]
    )


class TestFgBgToIndices(unittest.TestCase):
    @parameterized.expand(TESTS_CASES)
    def test_type_shape(self, input_data, label, image, expected_fg, expected_bg):
        fg_indices, bg_indices = FgBgToIndices(**input_data)(label, image)
        assert_allclose(fg_indices, expected_fg)
        assert_allclose(bg_indices, expected_bg)


if __name__ == "__main__":
    unittest.main()
