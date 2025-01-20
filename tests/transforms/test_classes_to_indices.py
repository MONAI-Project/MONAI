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

from monai.transforms import ClassesToIndices
from tests.util import TEST_NDARRAYS, assert_allclose

TESTS_CASES = []
for p in TEST_NDARRAYS:
    TESTS_CASES.append(
        [
            # test Argmax data
            {"num_classes": 3, "image_threshold": 0.0},
            p([[[0, 1, 2], [2, 0, 1], [1, 2, 0]]]),
            None,
            [p([0, 4, 8]), p([1, 5, 6]), p([2, 3, 7])],
        ]
    )

    TESTS_CASES.append(
        [
            {"num_classes": 3, "image_threshold": 60},
            p([[[0, 1, 2], [2, 0, 1], [1, 2, 0]]]),
            p([[[132, 1434, 51], [61, 0, 133], [523, 44, 232]]]),
            [p([0, 8]), p([1, 5, 6]), p([3])],
        ]
    )

    TESTS_CASES.append(
        [
            # test One-Hot data
            {"image_threshold": 0.0},
            p(
                [
                    [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
                    [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
                ]
            ),
            None,
            [p([0, 4, 8]), p([1, 5, 6]), p([2, 3, 7])],
        ]
    )

    TESTS_CASES.append(
        [
            {"num_classes": None, "image_threshold": 60},
            p(
                [
                    [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
                    [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
                ]
            ),
            p([[[132, 1434, 51], [61, 0, 133], [523, 44, 232]]]),
            [p([0, 8]), p([1, 5, 6]), p([3])],
        ]
    )

    TESTS_CASES.append(
        [
            # test output_shape
            {"num_classes": 3, "image_threshold": 0.0, "output_shape": [3, 3]},
            p([[[0, 1, 2], [2, 0, 1], [1, 2, 0]]]),
            None,
            [p([[0, 0], [1, 1], [2, 2]]), p([[0, 1], [1, 2], [2, 0]]), p([[0, 2], [1, 0], [2, 1]])],
        ]
    )


class TestClassesToIndices(unittest.TestCase):
    @parameterized.expand(TESTS_CASES)
    def test_value(self, input_args, label, image, expected_indices):
        indices = ClassesToIndices(**input_args)(label, image)
        for i, e in zip(indices, expected_indices):
            assert_allclose(i, e)


if __name__ == "__main__":
    unittest.main()
