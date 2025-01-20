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

from monai.inferers import SlidingWindowSplitter
from tests.util import assert_allclose

# ----------------------------------------------------------------------------
# Tensor test cases
# ----------------------------------------------------------------------------
# random int tensor (0, 255)
TENSOR_4x4 = torch.randint(low=0, high=255, size=(2, 3, 4, 4), dtype=torch.float32)

# random int tensor (0, 255) with artifacts at [..., :2, 2:]
TENSOR_4x4_artifact = TENSOR_4x4.clone()
TENSOR_4x4_artifact[..., :2, 2:] = 512.0

# no-overlapping 2x2
TEST_CASE_TENSOR_0 = [
    TENSOR_4x4,
    {"patch_size": (2, 2), "overlap": 0.0},
    [
        (TENSOR_4x4[..., :2, :2], (0, 0)),
        (TENSOR_4x4[..., :2, 2:], (0, 2)),
        (TENSOR_4x4[..., 2:, :2], (2, 0)),
        (TENSOR_4x4[..., 2:, 2:], (2, 2)),
    ],
]

# no-overlapping 3x3 with pad
TEST_CASE_TENSOR_1 = [
    TENSOR_4x4,
    {"patch_size": (3, 3), "overlap": 0.0, "pad_mode": "constant"},
    [
        (TENSOR_4x4[..., :3, :3], (0, 0)),
        (pad(TENSOR_4x4[..., :3, 3:], (0, 2)), (0, 3)),
        (pad(TENSOR_4x4[..., 3:, :3], (0, 0, 0, 2)), (3, 0)),
        (pad(TENSOR_4x4[..., 3:, 3:], (0, 2, 0, 2)), (3, 3)),
    ],
]

# overlapping 2x2 with fraction
TEST_CASE_TENSOR_2 = [
    TENSOR_4x4,
    {"patch_size": (2, 2), "overlap": (0.5, 0.5)},
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
]

# overlapping 3x3 with fraction (non-divisible)
TEST_CASE_TENSOR_3 = [
    TENSOR_4x4,
    {"patch_size": (3, 3), "overlap": 2.0 / 3.0},
    [
        (TENSOR_4x4[..., :3, :3], (0, 0)),
        (TENSOR_4x4[..., :3, 1:], (0, 1)),
        (TENSOR_4x4[..., 1:, :3], (1, 0)),
        (TENSOR_4x4[..., 1:, 1:], (1, 1)),
    ],
]

# overlapping 2x2 with number of pixels
TEST_CASE_TENSOR_4 = [
    TENSOR_4x4,
    {"patch_size": (2, 2), "overlap": (1, 1)},
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
]

# overlapping 3x3 with number of pixels (non-divisible)
TEST_CASE_TENSOR_5 = [
    TENSOR_4x4,
    {"patch_size": (3, 3), "overlap": 2},
    [
        (TENSOR_4x4[..., :3, :3], (0, 0)),
        (TENSOR_4x4[..., :3, 1:], (0, 1)),
        (TENSOR_4x4[..., 1:, :3], (1, 0)),
        (TENSOR_4x4[..., 1:, 1:], (1, 1)),
    ],
]
# non-overlapping 2x2 with positive offset
TEST_CASE_TENSOR_6 = [
    TENSOR_4x4,
    {"patch_size": (2, 2), "offset": 1},
    [
        (TENSOR_4x4[..., 1:3, 1:3], (1, 1)),
        (pad(TENSOR_4x4[..., 1:3, 3:], (0, 1)), (1, 3)),
        (pad(TENSOR_4x4[..., 3:, 1:3], (0, 0, 0, 1)), (3, 1)),
        (pad(TENSOR_4x4[..., 3:, 3:], (0, 1, 0, 1)), (3, 3)),
    ],
]

# non-overlapping 2x2 with negative offset
TEST_CASE_TENSOR_7 = [
    TENSOR_4x4,
    {"patch_size": (2, 2), "offset": -1},
    [
        (pad(TENSOR_4x4[..., :1, :1], (1, 0, 1, 0)), (-1, -1)),
        (pad(TENSOR_4x4[..., :1, 1:3], (0, 0, 1, 0)), (-1, 1)),
        (pad(TENSOR_4x4[..., :1, 3:], (0, 1, 1, 0)), (-1, 3)),
        (pad(TENSOR_4x4[..., 1:3, :1], (1, 0)), (1, -1)),
        (TENSOR_4x4[..., 1:3, 1:3], (1, 1)),
        (pad(TENSOR_4x4[..., 1:3, 3:], (0, 1)), (1, 3)),
        (pad(TENSOR_4x4[..., 3:, :1], (1, 0, 0, 1)), (3, -1)),
        (pad(TENSOR_4x4[..., 3:, 1:3], (0, 0, 0, 1)), (3, 1)),
        (pad(TENSOR_4x4[..., 3:, 3:], (0, 1, 0, 1)), (3, 3)),
    ],
]

# non-overlapping 2x2 with positive offset and no padding
TEST_CASE_TENSOR_8 = [
    TENSOR_4x4,
    {"patch_size": (2, 2), "offset": 1, "pad_mode": None},
    [(TENSOR_4x4[..., 1:3, 1:3], (1, 1))],
]


# ----------------------------------------------------------------------------
# Filtering function test cases
# ----------------------------------------------------------------------------
def gen_filter(filter_type, value=None):
    """ "Generate patch filtering function for testing"""
    if filter_type.lower() == "high":

        def my_filter(patch, location):
            if torch.any(patch > value):
                return True
            return False

    elif filter_type.lower() == "low":

        def my_filter(patch, location):
            if torch.any(patch < value):
                return True
            return False

    elif filter_type.lower() == "location":

        def my_filter(patch, location):
            if location in value:
                return True
            return False

    return my_filter


TEST_CASE_FILTER_FN_0 = [
    TENSOR_4x4_artifact,
    {"patch_size": (2, 2), "filter_fn": gen_filter("low", 256)},
    [
        (TENSOR_4x4_artifact[..., :2, :2], (0, 0)),
        (TENSOR_4x4_artifact[..., 2:, :2], (2, 0)),
        (TENSOR_4x4_artifact[..., 2:, 2:], (2, 2)),
    ],
]

TEST_CASE_FILTER_FN_1 = [
    TENSOR_4x4_artifact,
    {"patch_size": (2, 2), "filter_fn": gen_filter("high", 256)},
    [(TENSOR_4x4_artifact[..., :2, 2:], (0, 2))],
]

TEST_CASE_FILTER_FN_2 = [
    TENSOR_4x4_artifact,
    {"patch_size": (2, 2), "filter_fn": gen_filter("location", [(2, 2), (2, 0)])},
    [(TENSOR_4x4_artifact[..., 2:, :2], (2, 0)), (TENSOR_4x4_artifact[..., 2:, 2:], (2, 2))],
]


# ----------------------------------------------------------------------------
# Error test cases
# ----------------------------------------------------------------------------
def extra_parameter_filter(patch, location, extra):
    return


def missing_parameter_filter(patch):
    return


# invalid overlap: float 1.0
TEST_CASE_ERROR_0 = [TENSOR_4x4, {"patch_size": (2, 2), "overlap": 1.0}, ValueError]
# invalid overlap: negative float
TEST_CASE_ERROR_1 = [TENSOR_4x4, {"patch_size": (2, 2), "overlap": -0.1}, ValueError]
# invalid overlap: negative integer
TEST_CASE_ERROR_2 = [TENSOR_4x4, {"patch_size": (2, 2), "overlap": -1}, ValueError]
# invalid overlap: integer larger than patch size
TEST_CASE_ERROR_3 = [TENSOR_4x4, {"patch_size": (2, 2), "overlap": 3}, ValueError]

# invalid offset: positive and larger than image size
TEST_CASE_ERROR_4 = [TENSOR_4x4, {"patch_size": (2, 2), "offset": 4}, ValueError]
# invalid offset: negative and larger than patch size (in magnitude)
TEST_CASE_ERROR_5 = [TENSOR_4x4, {"patch_size": (2, 2), "offset": -3, "pad_mode": "constant"}, ValueError]
# invalid offset: negative and no padding
TEST_CASE_ERROR_6 = [TENSOR_4x4, {"patch_size": (2, 2), "offset": -1, "pad_mode": None}, ValueError]

# invalid filter function: with more than two positional parameters
TEST_CASE_ERROR_7 = [TENSOR_4x4, {"patch_size": (2, 2), "filter_fn": extra_parameter_filter}, ValueError]
# invalid filter function: with less than two positional parameters
TEST_CASE_ERROR_8 = [TENSOR_4x4, {"patch_size": (2, 2), "filter_fn": missing_parameter_filter}, ValueError]
# invalid filter function: non-callable
TEST_CASE_ERROR_9 = [TENSOR_4x4, {"patch_size": (2, 2), "filter_fn": 1}, ValueError]


class SlidingWindowSplitterTests(unittest.TestCase):
    @parameterized.expand(
        [
            TEST_CASE_TENSOR_0,
            TEST_CASE_TENSOR_1,
            TEST_CASE_TENSOR_2,
            TEST_CASE_TENSOR_3,
            TEST_CASE_TENSOR_4,
            TEST_CASE_TENSOR_5,
            TEST_CASE_TENSOR_6,
            TEST_CASE_TENSOR_7,
            TEST_CASE_TENSOR_8,
            TEST_CASE_FILTER_FN_0,
            TEST_CASE_FILTER_FN_1,
            TEST_CASE_FILTER_FN_2,
        ]
    )
    def test_split_patches_tensor(self, image, arguments, expected):
        patches = SlidingWindowSplitter(**arguments)(image)
        patches = list(patches)
        self.assertEqual(len(patches), len(expected))
        for p, e in zip(patches, expected):
            assert_allclose(p[0], e[0])
            self.assertTupleEqual(p[1], e[1])

    @parameterized.expand(
        [
            TEST_CASE_ERROR_0,
            TEST_CASE_ERROR_1,
            TEST_CASE_ERROR_2,
            TEST_CASE_ERROR_3,
            TEST_CASE_ERROR_4,
            TEST_CASE_ERROR_5,
            TEST_CASE_ERROR_6,
            TEST_CASE_ERROR_7,
            TEST_CASE_ERROR_8,
            TEST_CASE_ERROR_9,
        ]
    )
    def test_split_patches_errors(self, image, arguments, expected_error):
        with self.assertRaises(expected_error):
            patches = SlidingWindowSplitter(**arguments)(image)
            patches = list(patches)


if __name__ == "__main__":
    unittest.main()
