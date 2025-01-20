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

from monai.transforms import map_binary_to_indices
from tests.util import TEST_NDARRAYS, assert_allclose

TESTS = []
for p in TEST_NDARRAYS:
    TESTS.append(
        [
            {"label": p(np.array([[[0, 1, 1], [1, 0, 1], [1, 1, 0]]])), "image": None, "image_threshold": 0.0},
            np.array([1, 2, 3, 5, 6, 7]),
            np.array([0, 4, 8]),
        ]
    )
    TESTS.append(
        [
            {
                "label": p(np.array([[[0, 1, 1], [1, 0, 1], [1, 1, 0]]])),
                "image": p(np.array([[[1, 1, 1], [1, 0, 1], [1, 1, 1]]])),
                "image_threshold": 0.0,
            },
            np.array([1, 2, 3, 5, 6, 7]),
            np.array([0, 8]),
        ]
    )
    TESTS.append(
        [
            {
                "label": p(np.array([[[0, 1, 1], [1, 0, 1], [1, 1, 0]]])),
                "image": p(np.array([[[3, 3, 3], [3, 1, 3], [3, 3, 3]]])),
                "image_threshold": 1.0,
            },
            np.array([1, 2, 3, 5, 6, 7]),
            np.array([0, 8]),
        ]
    )
    TESTS.append(
        [
            {
                "label": p(np.array([[[0, 1, 2], [3, 0, 4], [5, 6, 0]]])),
                "image": p(np.array([[[3, 3, 3], [3, 1, 3], [3, 3, 3]]])),
                "image_threshold": 1.0,
            },
            np.array([1, 2, 3, 5, 6, 7]),
            np.array([0, 8]),
        ]
    )


class TestMapBinaryToIndices(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_type_shape(self, input_data, expected_fg, expected_bg):
        fg_indices, bg_indices = map_binary_to_indices(**input_data)
        assert_allclose(fg_indices, expected_fg, type_test=False)
        assert_allclose(bg_indices, expected_bg, type_test=False)


if __name__ == "__main__":
    unittest.main()
