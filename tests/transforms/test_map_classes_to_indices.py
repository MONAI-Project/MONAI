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

from monai.transforms import map_classes_to_indices
from tests.util import TEST_NDARRAYS, assert_allclose

TESTS = []
for p in TEST_NDARRAYS:
    TESTS.append(
        [
            # test Argmax data
            {
                "label": p(np.array([[[0, 1, 2], [2, 0, 1], [1, 2, 0]]])),
                "num_classes": 3,
                "image": None,
                "image_threshold": 0.0,
            },
            [np.array([0, 4, 8]), np.array([1, 5, 6]), np.array([2, 3, 7])],
        ]
    )

    TESTS.append(
        [
            {
                "label": p(np.array([[[0, 1, 2], [2, 0, 1], [1, 2, 0]]])),
                "num_classes": 3,
                "image": p(np.array([[[132, 1434, 51], [61, 0, 133], [523, 44, 232]]])),
                "image_threshold": 60,
            },
            [np.array([0, 8]), np.array([1, 5, 6]), np.array([3])],
        ]
    )

    TESTS.append(
        [
            # test One-Hot data
            {
                "label": p(
                    np.array(
                        [
                            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                            [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
                            [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
                        ]
                    )
                ),
                "image": None,
                "image_threshold": 0.0,
            },
            [np.array([0, 4, 8]), np.array([1, 5, 6]), np.array([2, 3, 7])],
        ]
    )

    TESTS.append(
        [
            {
                "label": p(
                    np.array(
                        [
                            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                            [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
                            [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
                        ]
                    )
                ),
                "num_classes": None,
                "image": p(np.array([[[132, 1434, 51], [61, 0, 133], [523, 44, 232]]])),
                "image_threshold": 60,
            },
            [np.array([0, 8]), np.array([1, 5, 6]), np.array([3])],
        ]
    )

    TESTS.append(
        [
            # test empty class
            {
                "label": p(np.array([[[0, 1, 2], [2, 0, 1], [1, 2, 0]]])),
                "num_classes": 5,
                "image": None,
                "image_threshold": 0.0,
            },
            [np.array([0, 4, 8]), np.array([1, 5, 6]), np.array([2, 3, 7]), np.array([]), np.array([])],
        ]
    )

    TESTS.append(
        [
            # test empty class
            {
                "label": p(
                    np.array(
                        [
                            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                            [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
                            [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
                            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                        ]
                    )
                ),
                "image": None,
                "image_threshold": 0.0,
                "max_samples_per_class": 2,
            },
            [np.array([0, 8]), np.array([1, 6]), np.array([2, 7]), np.array([]), np.array([])],
        ]
    )


class TestMapClassesToIndices(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_value(self, input_data, expected_indices):
        indices = map_classes_to_indices(**input_data)
        for i, e in zip(indices, expected_indices):
            assert_allclose(i, e, type_test=False)


if __name__ == "__main__":
    unittest.main()
