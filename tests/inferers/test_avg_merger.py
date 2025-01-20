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

import torch
from parameterized import parameterized
from torch.nn.functional import pad

from monai.inferers import AvgMerger
from tests.util import assert_allclose

TENSOR_4x4 = torch.randint(low=0, high=255, size=(2, 3, 4, 4), dtype=torch.float32)
TENSOR_4x4_WITH_NAN = TENSOR_4x4.clone()
TENSOR_4x4_WITH_NAN[..., 2:, 2:] = float("nan")

# no-overlapping 2x2
TEST_CASE_0_DEFAULT_DTYPE = [
    dict(merged_shape=TENSOR_4x4.shape),
    [
        (TENSOR_4x4[..., :2, :2], (0, 0)),
        (TENSOR_4x4[..., :2, 2:], (0, 2)),
        (TENSOR_4x4[..., 2:, :2], (2, 0)),
        (TENSOR_4x4[..., 2:, 2:], (2, 2)),
    ],
    TENSOR_4x4,
]

# overlapping 2x2
TEST_CASE_1_DEFAULT_DTYPE = [
    dict(merged_shape=TENSOR_4x4.shape),
    [
        (TENSOR_4x4[..., 0:2, 0:2], (0, 0)),
        (TENSOR_4x4[..., 0:2, 1:3], (0, 1)),
        (TENSOR_4x4[..., 0:2, 2:4], (0, 2)),
        (TENSOR_4x4[..., 1:3, 0:2], (1, 0)),
        (TENSOR_4x4[..., 1:3, 1:3], (1, 1)),
        (TENSOR_4x4[..., 1:3, 2:4], (1, 2)),
        (TENSOR_4x4[..., 2:4, 0:2], (2, 0)),
        (TENSOR_4x4[..., 2:4, 1:3], (2, 1)),
        (TENSOR_4x4[..., 2:4, 2:4], (2, 2)),
    ],
    TENSOR_4x4,
]

# overlapping 3x3 (non-divisible)
TEST_CASE_2_DEFAULT_DTYPE = [
    dict(merged_shape=TENSOR_4x4.shape),
    [
        (TENSOR_4x4[..., :3, :3], (0, 0)),
        (TENSOR_4x4[..., :3, 1:], (0, 1)),
        (TENSOR_4x4[..., 1:, :3], (1, 0)),
        (TENSOR_4x4[..., 1:, 1:], (1, 1)),
    ],
    TENSOR_4x4,
]

#  overlapping 2x2 with NaN values
TEST_CASE_3_DEFAULT_DTYPE = [
    dict(merged_shape=TENSOR_4x4_WITH_NAN.shape),
    [
        (TENSOR_4x4_WITH_NAN[..., 0:2, 0:2], (0, 0)),
        (TENSOR_4x4_WITH_NAN[..., 0:2, 1:3], (0, 1)),
        (TENSOR_4x4_WITH_NAN[..., 0:2, 2:4], (0, 2)),
        (TENSOR_4x4_WITH_NAN[..., 1:3, 0:2], (1, 0)),
        (TENSOR_4x4_WITH_NAN[..., 1:3, 1:3], (1, 1)),
        (TENSOR_4x4_WITH_NAN[..., 1:3, 2:4], (1, 2)),
        (TENSOR_4x4_WITH_NAN[..., 2:4, 0:2], (2, 0)),
        (TENSOR_4x4_WITH_NAN[..., 2:4, 1:3], (2, 1)),
        (TENSOR_4x4_WITH_NAN[..., 2:4, 2:4], (2, 2)),
    ],
    TENSOR_4x4_WITH_NAN,
]

# non-overlapping 2x2 with missing patch
TEST_CASE_4_DEFAULT_DTYPE = [
    dict(merged_shape=TENSOR_4x4.shape),
    [(TENSOR_4x4[..., :2, :2], (0, 0)), (TENSOR_4x4[..., :2, 2:], (0, 2)), (TENSOR_4x4[..., 2:, :2], (2, 0))],
    TENSOR_4x4_WITH_NAN,
]

# with value_dtype set to half precision
TEST_CASE_5_VALUE_DTYPE = [
    dict(merged_shape=TENSOR_4x4.shape, value_dtype=torch.float16),
    [
        (TENSOR_4x4[..., :2, :2], (0, 0)),
        (TENSOR_4x4[..., :2, 2:], (0, 2)),
        (TENSOR_4x4[..., 2:, :2], (2, 0)),
        (TENSOR_4x4[..., 2:, 2:], (2, 2)),
    ],
    TENSOR_4x4,
]
# with count_dtype set to int32
TEST_CASE_6_COUNT_DTYPE = [
    dict(merged_shape=TENSOR_4x4.shape, count_dtype=torch.int32),
    [
        (TENSOR_4x4[..., :2, :2], (0, 0)),
        (TENSOR_4x4[..., :2, 2:], (0, 2)),
        (TENSOR_4x4[..., 2:, :2], (2, 0)),
        (TENSOR_4x4[..., 2:, 2:], (2, 2)),
    ],
    TENSOR_4x4,
]
# with both value_dtype, count_dtype set to double precision
TEST_CASE_7_COUNT_VALUE_DTYPE = [
    dict(merged_shape=TENSOR_4x4.shape, value_dtype=torch.float64, count_dtype=torch.float64),
    [
        (TENSOR_4x4[..., :2, :2], (0, 0)),
        (TENSOR_4x4[..., :2, 2:], (0, 2)),
        (TENSOR_4x4[..., 2:, :2], (2, 0)),
        (TENSOR_4x4[..., 2:, 2:], (2, 2)),
    ],
    TENSOR_4x4,
]

# shape larger than what is covered by patches
TEST_CASE_8_LARGER_SHAPE = [
    dict(merged_shape=(2, 3, 4, 6)),
    [
        (TENSOR_4x4[..., :2, :2], (0, 0)),
        (TENSOR_4x4[..., :2, 2:], (0, 2)),
        (TENSOR_4x4[..., 2:, :2], (2, 0)),
        (TENSOR_4x4[..., 2:, 2:], (2, 2)),
    ],
    pad(TENSOR_4x4, (0, 2), value=float("nan")),
]


class AvgMergerTests(unittest.TestCase):
    @parameterized.expand(
        [
            TEST_CASE_0_DEFAULT_DTYPE,
            TEST_CASE_1_DEFAULT_DTYPE,
            TEST_CASE_2_DEFAULT_DTYPE,
            TEST_CASE_3_DEFAULT_DTYPE,
            TEST_CASE_4_DEFAULT_DTYPE,
            TEST_CASE_5_VALUE_DTYPE,
            TEST_CASE_6_COUNT_DTYPE,
            TEST_CASE_7_COUNT_VALUE_DTYPE,
            TEST_CASE_8_LARGER_SHAPE,
        ]
    )
    def test_avg_merger_patches(self, arguments, patch_locations, expected):
        merger = AvgMerger(**arguments)
        for pl in patch_locations:
            merger.aggregate(pl[0], pl[1])
        output = merger.finalize()
        if "value_dtype" in arguments:
            self.assertTrue(merger.get_values().dtype, arguments["value_dtype"])
        if "count_dtype" in arguments:
            self.assertTrue(merger.get_counts().dtype, arguments["count_dtype"])
        # check for multiple call of finalize
        self.assertIs(output, merger.finalize())
        # check if the result is matching the expectation
        assert_allclose(output, expected)

    def test_avg_merger_finalized_error(self):
        with self.assertRaises(ValueError):
            merger = AvgMerger(merged_shape=(1, 3, 2, 3))
            merger.finalize()
            merger.aggregate(torch.zeros(1, 3, 2, 2), (3, 3))

    def test_avg_merge_none_merged_shape_error(self):
        with self.assertRaises(ValueError):
            AvgMerger(merged_shape=None)


if __name__ == "__main__":
    unittest.main()
