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
from torch.nn.functional import pad

from monai.inferers import ZarrAvgMerger
from monai.utils import get_package_version, optional_import, version_geq
from tests.utils import assert_allclose

np.seterr(divide="ignore", invalid="ignore")
zarr, has_zarr = optional_import("zarr")
if has_zarr:
    if version_geq(get_package_version("zarr"), "3.0.0"):
        directory_store = zarr.storage.LocalStore("test.zarr")
    else:
        directory_store = zarr.storage.DirectoryStore("test.zarr")
else:
    directory_store = None
numcodecs, has_numcodecs = optional_import("numcodecs")

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
    dict(merged_shape=TENSOR_4x4.shape, value_dtype=np.float16),
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
    dict(merged_shape=TENSOR_4x4.shape, count_dtype=np.int32),
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
    dict(merged_shape=TENSOR_4x4.shape, value_dtype=np.float64, count_dtype=np.float64),
    [
        (TENSOR_4x4[..., :2, :2], (0, 0)),
        (TENSOR_4x4[..., :2, 2:], (0, 2)),
        (TENSOR_4x4[..., 2:, :2], (2, 0)),
        (TENSOR_4x4[..., 2:, 2:], (2, 2)),
    ],
    TENSOR_4x4,
]
# with both value_dtype, count_dtype set to double precision
TEST_CASE_8_DTYPE = [
    dict(merged_shape=TENSOR_4x4.shape, dtype=np.float64),
    [
        (TENSOR_4x4[..., :2, :2], (0, 0)),
        (TENSOR_4x4[..., :2, 2:], (0, 2)),
        (TENSOR_4x4[..., 2:, :2], (2, 0)),
        (TENSOR_4x4[..., 2:, 2:], (2, 2)),
    ],
    TENSOR_4x4,
]

# shape larger than what is covered by patches
TEST_CASE_9_LARGER_SHAPE = [
    dict(merged_shape=(2, 3, 4, 6)),
    [
        (TENSOR_4x4[..., :2, :2], (0, 0)),
        (TENSOR_4x4[..., :2, 2:], (0, 2)),
        (TENSOR_4x4[..., 2:, :2], (2, 0)),
        (TENSOR_4x4[..., 2:, 2:], (2, 2)),
    ],
    pad(TENSOR_4x4, (0, 2), value=float("nan")),
]

# explicit directory store
TEST_CASE_10_DIRECTORY_STORE = [
    dict(merged_shape=TENSOR_4x4.shape, store=directory_store),
    [
        (TENSOR_4x4[..., :2, :2], (0, 0)),
        (TENSOR_4x4[..., :2, 2:], (0, 2)),
        (TENSOR_4x4[..., 2:, :2], (2, 0)),
        (TENSOR_4x4[..., 2:, 2:], (2, 2)),
    ],
    TENSOR_4x4,
]

# memory store for all arrays
TEST_CASE_11_MEMORY_STORE = [
    dict(
        merged_shape=TENSOR_4x4.shape,
        store=zarr.storage.MemoryStore(),
        value_store=zarr.storage.MemoryStore(),
        count_store=zarr.storage.MemoryStore(),
    ),
    [
        (TENSOR_4x4[..., :2, :2], (0, 0)),
        (TENSOR_4x4[..., :2, 2:], (0, 2)),
        (TENSOR_4x4[..., 2:, :2], (2, 0)),
        (TENSOR_4x4[..., 2:, 2:], (2, 2)),
    ],
    TENSOR_4x4,
]

# explicit chunk size
TEST_CASE_12_CHUNKS = [
    dict(merged_shape=TENSOR_4x4.shape, chunks=(1, 1, 2, 2)),
    [
        (TENSOR_4x4[..., :2, :2], (0, 0)),
        (TENSOR_4x4[..., :2, 2:], (0, 2)),
        (TENSOR_4x4[..., 2:, :2], (2, 0)),
        (TENSOR_4x4[..., 2:, 2:], (2, 2)),
    ],
    TENSOR_4x4,
]

# test for LZ4 compressor
TEST_CASE_13_COMPRESSOR_LZ4 = [
    dict(merged_shape=TENSOR_4x4.shape, compressor="LZ4"),
    [
        (TENSOR_4x4[..., :2, :2], (0, 0)),
        (TENSOR_4x4[..., :2, 2:], (0, 2)),
        (TENSOR_4x4[..., 2:, :2], (2, 0)),
        (TENSOR_4x4[..., 2:, 2:], (2, 2)),
    ],
    TENSOR_4x4,
]

# test for pickle compressor
TEST_CASE_14_COMPRESSOR_PICKLE = [
    dict(merged_shape=TENSOR_4x4.shape, compressor="Pickle"),
    [
        (TENSOR_4x4[..., :2, :2], (0, 0)),
        (TENSOR_4x4[..., :2, 2:], (0, 2)),
        (TENSOR_4x4[..., 2:, :2], (2, 0)),
        (TENSOR_4x4[..., 2:, 2:], (2, 2)),
    ],
    TENSOR_4x4,
]

# test for LZMA compressor
TEST_CASE_15_COMPRESSOR_LZMA = [
    dict(merged_shape=TENSOR_4x4.shape, compressor="LZMA"),
    [
        (TENSOR_4x4[..., :2, :2], (0, 0)),
        (TENSOR_4x4[..., :2, 2:], (0, 2)),
        (TENSOR_4x4[..., 2:, :2], (2, 0)),
        (TENSOR_4x4[..., 2:, 2:], (2, 2)),
    ],
    TENSOR_4x4,
]

# test with thread locking
TEST_CASE_16_WITH_LOCK = [
    dict(merged_shape=TENSOR_4x4.shape, thread_locking=True),
    [
        (TENSOR_4x4[..., :2, :2], (0, 0)),
        (TENSOR_4x4[..., :2, 2:], (0, 2)),
        (TENSOR_4x4[..., 2:, :2], (2, 0)),
        (TENSOR_4x4[..., 2:, 2:], (2, 2)),
    ],
    TENSOR_4x4,
]

# test without thread locking
TEST_CASE_17_WITHOUT_LOCK = [
    dict(merged_shape=TENSOR_4x4.shape, thread_locking=False),
    [
        (TENSOR_4x4[..., :2, :2], (0, 0)),
        (TENSOR_4x4[..., :2, 2:], (0, 2)),
        (TENSOR_4x4[..., 2:, :2], (2, 0)),
        (TENSOR_4x4[..., 2:, 2:], (2, 2)),
    ],
    TENSOR_4x4,
]


@unittest.skipUnless(has_zarr and has_numcodecs, "Requires zarr (and numcodecs) packages.)")
class ZarrAvgMergerTests(unittest.TestCase):
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
            TEST_CASE_8_DTYPE,
            TEST_CASE_9_LARGER_SHAPE,
            TEST_CASE_10_DIRECTORY_STORE,
            TEST_CASE_11_MEMORY_STORE,
            TEST_CASE_12_CHUNKS,
            TEST_CASE_13_COMPRESSOR_LZ4,
            TEST_CASE_14_COMPRESSOR_PICKLE,
            TEST_CASE_15_COMPRESSOR_LZMA,
            TEST_CASE_16_WITH_LOCK,
            TEST_CASE_17_WITHOUT_LOCK,
        ]
    )
    def test_zarr_avg_merger_patches(self, arguments, patch_locations, expected):
        if "compressor" in arguments:
            if arguments["compressor"] != "default":
                arguments["compressor"] = zarr.codec_registry[arguments["compressor"].lower()]()
        if "value_compressor" in arguments:
            if arguments["value_compressor"] != "default":
                arguments["value_compressor"] = zarr.codec_registry[arguments["value_compressor"].lower()]()
        if "count_compressor" in arguments:
            if arguments["count_compressor"] != "default":
                arguments["count_compressor"] = zarr.codec_registry[arguments["count_compressor"].lower()]()
        merger = ZarrAvgMerger(**arguments)
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
        assert_allclose(output[:], expected.numpy())

    def test_zarr_avg_merger_finalized_error(self):
        with self.assertRaises(ValueError):
            merger = ZarrAvgMerger(merged_shape=(1, 3, 2, 3))
            merger.finalize()
            merger.aggregate(torch.zeros(1, 3, 2, 2), (3, 3))

    def test_zarr_avg_merge_none_merged_shape_error(self):
        with self.assertRaises(ValueError):
            ZarrAvgMerger(merged_shape=None)


if __name__ == "__main__":
    unittest.main()
