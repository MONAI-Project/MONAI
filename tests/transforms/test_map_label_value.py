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

from monai.transforms import MapLabelValue
from tests.util import TEST_NDARRAYS, assert_allclose

TESTS = []
for p in TEST_NDARRAYS:
    TESTS.extend(
        [
            [{"orig_labels": [3, 2, 1], "target_labels": [0, 1, 2]}, p([[3, 1], [1, 2]]), p([[0.0, 2.0], [2.0, 1.0]])],
            [
                {"orig_labels": [3, 5, 8], "target_labels": [0, 1, 2]},
                p([[[3], [5], [5], [8]]]),
                p([[[0.0], [1.0], [1.0], [2.0]]]),
            ],
            [{"orig_labels": [1, 2, 3], "target_labels": [0, 1, 2]}, p([3, 1, 1, 2]), p([2.0, 0.0, 0.0, 1.0])],
            [{"orig_labels": [1, 2, 3], "target_labels": [0.5, 1.5, 2.5]}, p([3, 1, 1, 2]), p([2.5, 0.5, 0.5, 1.5])],
        ]
    )
    # note: PyTorch 1.5.1 doesn't support rich dtypes
    TESTS.append(
        [
            {"orig_labels": [1.5, 2.5, 3.5], "target_labels": [0, 1, 2], "dtype": np.int8},
            p([3.5, 1.5, 1.5, 2.5]),
            p([2, 0, 0, 1]),
        ]
    )
    TESTS.append(
        [
            {"orig_labels": [1.5, 2.5, 3.5], "target_labels": [0, 1, 2], "dtype": torch.int8},
            p([3.5, 1.5, 1.5, 2.5]),
            p([2, 0, 0, 1]),
        ]
    )
TESTS.extend(
    [
        [
            {"orig_labels": ["label3", "label2", "label1"], "target_labels": [0, 1, 2]},
            np.array([["label3", "label1"], ["label1", "label2"]]),
            np.array([[0, 2], [2, 1]]),
        ],
        [
            {"orig_labels": [3.5, 2.5, 1.5], "target_labels": ["label0", "label1", "label2"], "dtype": "str"},
            np.array([[3.5, 1.5], [1.5, 2.5]]),
            np.array([["label0", "label2"], ["label2", "label1"]]),
        ],
        [
            {
                "orig_labels": ["label3", "label2", "label1"],
                "target_labels": ["label1", "label2", "label3"],
                "dtype": "str",
            },
            np.array([["label3", "label1"], ["label1", "label2"]]),
            np.array([["label1", "label3"], ["label3", "label2"]]),
        ],
    ]
)


class TestMapLabelValue(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_shape(self, input_param, input_data, expected_value):
        result = MapLabelValue(**input_param)(input_data)
        if isinstance(expected_value, torch.Tensor):
            assert_allclose(result, expected_value)
        else:
            np.testing.assert_equal(result, expected_value)
        self.assertTupleEqual(result.shape, expected_value.shape)


if __name__ == "__main__":
    unittest.main()
