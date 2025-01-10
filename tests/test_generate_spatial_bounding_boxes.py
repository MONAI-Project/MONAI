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

from monai.transforms import generate_spatial_bounding_boxes
from monai.utils import min_version, optional_import
from tests.utils import TEST_NDARRAYS

skimage, has_cucim = optional_import("cucim.skimage")
measure, has_measure = optional_import("skimage.measure", "0.14.2", min_version)
cp, has_cp = optional_import("cupy")

TESTS = []
for p in TEST_NDARRAYS:
    TESTS.append(
        [
            {
                "img": p(
                    np.array([[[0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 2, 3, 2, 0], [0, 1, 2, 1, 0], [0, 0, 0, 0, 0]]])
                ),
                "select_fn": lambda x: x > 0,
                "channel_indices": None,
                "margin": 0,
            },
            ([[1, 1]], [[4, 4]]),
        ]
    )
    TESTS.append(
        [
            {
                "img": p(
                    np.array([[[0, 0, 0, 0, 0], [0, 1, 1, 1, 0], [0, 1, 3, 1, 0], [0, 1, 1, 1, 0], [0, 0, 0, 0, 0]]])
                ),
                "select_fn": lambda x: x > 1,
                "channel_indices": None,
                "margin": 0,
            },
            ([[2, 2]], [[3, 3]]),
        ]
    )
    TESTS.append(
        [
            {
                "img": p(
                    np.array([[[0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 2, 3, 2, 0], [0, 1, 2, 1, 0], [0, 0, 0, 0, 0]]])
                ),
                "select_fn": lambda x: x > 0,
                "channel_indices": 0,
                "margin": 0,
            },
            ([[1, 1]], [[4, 4]]),
        ]
    )
    TESTS.append(
        [
            {
                "img": p(
                    np.array([[[0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 2, 3, 2, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]])
                ),
                "select_fn": lambda x: x > 0,
                "channel_indices": None,
                "margin": 1,
            },
            ([[0, 0]], [[4, 5]]),
        ]
    )
    TESTS.append(
        [
            {
                "img": p(
                    np.array([[[0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 2, 3, 2, 0], [0, 1, 2, 1, 0], [0, 0, 0, 0, 0]]])
                ),
                "select_fn": lambda x: x > 0,
                "channel_indices": None,
                "margin": [2, 1],
                "allow_smaller": False,
            },
            ([[-1, 0]], [[6, 5]]),
        ]
    )
    TESTS.append(
        [
            {
                "img": p(
                    np.array([[[0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 2, 3, 2, 0], [0, 1, 2, 1, 0], [0, 0, 0, 0, 0]]])
                ),
                "select_fn": lambda x: x > 0,
                "channel_indices": None,
                "margin": [2, 1],
                "allow_smaller": True,
            },
            ([[0, 0]], [[5, 5]]),
        ]
    )
    TESTS.append(
        [
            {
                "img": p(
                    np.array([[[1, 0, 0, 0, 0], [0, 0, 0, 0, 1], [1, 0, 0, 0, 0], [0, 0, 0, 0, 1], [1, 0, 0, 0, 0]]])
                ),
                "select_fn": lambda x: x > 0,
                "channel_indices": None,
                "margin": 0,
            },
            ([[0, 0], [1, 4], [2, 0], [3, 4], [4, 0]], [[1, 1], [2, 5], [3, 1], [4, 5], [5, 1]]),
        ]
    )
    TESTS.append(
        [
            {
                "img": p(
                    np.array([[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]])
                ),
                "select_fn": lambda x: x > 0,
                "channel_indices": None,
                "margin": 0,
            },
            ([[2, 2]], [[5, 5]]),
        ]
    )
    TESTS.append(
        [
            {
                "img": p(
                    np.array([[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]])
                ),
                "select_fn": lambda x: x > 0,
                "channel_indices": None,
                "margin": 0,
            },
            ([], []),
        ]
    )


@unittest.skipUnless(has_measure, "Skimage.measure required.")
class TestGenerateSpatialBoundingBox(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_value(self, input_data, expected_box):
        result = generate_spatial_bounding_boxes(**input_data)
        self.assertTupleEqual(result, expected_box)


if __name__ == "__main__":
    unittest.main()
