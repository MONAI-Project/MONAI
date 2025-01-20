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

from monai.data import MetaTensor, set_track_meta
from monai.transforms.spatial.array import GridPatch
from tests.util import TEST_NDARRAYS, SkipIfBeforePyTorchVersion, assert_allclose

A = np.arange(16).repeat(3).reshape(4, 4, 3).transpose(2, 0, 1)
A11 = A[:, :2, :2]
A12 = A[:, :2, 2:]
A21 = A[:, 2:, :2]
A22 = A[:, 2:, 2:]

TEST_CASE_0 = [{"patch_size": (2, 2)}, A, [A11, A12, A21, A22]]
TEST_CASE_1 = [{"patch_size": (2, 2), "num_patches": 3}, A, [A11, A12, A21]]
TEST_CASE_2 = [{"patch_size": (2, 2), "num_patches": 5}, A, [A11, A12, A21, A22, np.zeros((3, 2, 2))]]
TEST_CASE_3 = [{"patch_size": (2, 2), "offset": (0, 0)}, A, [A11, A12, A21, A22]]
TEST_CASE_4 = [{"patch_size": (2, 2), "offset": (0, 0)}, A, [A11, A12, A21, A22]]
TEST_CASE_5 = [{"patch_size": (2, 2), "offset": (2, 2)}, A, [A22]]
TEST_CASE_6 = [{"patch_size": (2, 2), "offset": (0, 2)}, A, [A12, A22]]
TEST_CASE_7 = [{"patch_size": (2, 2), "offset": (2, 0)}, A, [A21, A22]]
TEST_CASE_8 = [{"patch_size": (2, 2), "num_patches": 3, "sort_fn": "max"}, A, [A22, A21, A12]]
TEST_CASE_9 = [{"patch_size": (2, 2), "num_patches": 4, "sort_fn": "min"}, A, [A11, A12, A21, A22]]
TEST_CASE_10 = [{"patch_size": (2, 2), "overlap": 0.5, "num_patches": 3}, A, [A11, A[:, :2, 1:3], A12]]
TEST_CASE_11 = [
    {"patch_size": (3, 3), "num_patches": 2, "constant_values": 255, "pad_mode": "constant"},
    A,
    [A[:, :3, :3], np.pad(A[:, :3, 3:], ((0, 0), (0, 0), (0, 2)), mode="constant", constant_values=255)],
]
TEST_CASE_12 = [
    {"patch_size": (3, 3), "offset": (-2, -2), "num_patches": 2, "pad_mode": "constant"},
    A,
    [np.zeros((3, 3, 3)), np.pad(A[:, :1, 1:4], ((0, 0), (2, 0), (0, 0)), mode="constant")],
]
# Only threshold filtering
TEST_CASE_13 = [{"patch_size": (2, 2), "threshold": 50.0}, A, [A11]]
TEST_CASE_14 = [{"patch_size": (2, 2), "threshold": 150.0}, A, [A11, A12, A21]]
# threshold filtering with num_patches more than available patches (no effect)
TEST_CASE_15 = [{"patch_size": (2, 2), "num_patches": 3, "threshold": 50.0}, A, [A11]]
# threshold filtering with num_patches less than available patches (count filtering)
TEST_CASE_16 = [{"patch_size": (2, 2), "num_patches": 2, "threshold": 150.0}, A, [A11, A12]]

TEST_CASE_META_0 = [
    {"patch_size": (2, 2)},
    A,
    [A11, A12, A21, A22],
    [{"location": [0, 0]}, {"location": [0, 2]}, {"location": [2, 0]}, {"location": [2, 2]}],
]

TEST_CASE_META_1 = [
    {"patch_size": (2, 2)},
    MetaTensor(x=A, meta={"path": "path/to/file"}),
    [A11, A12, A21, A22],
    [
        {"location": [0, 0], "path": "path/to/file"},
        {"location": [0, 2], "path": "path/to/file"},
        {"location": [2, 0], "path": "path/to/file"},
        {"location": [2, 2], "path": "path/to/file"},
    ],
]

TEST_CASES = []
for p in TEST_NDARRAYS:
    TEST_CASES.append([p, *TEST_CASE_0])
    TEST_CASES.append([p, *TEST_CASE_1])
    TEST_CASES.append([p, *TEST_CASE_2])
    TEST_CASES.append([p, *TEST_CASE_3])
    TEST_CASES.append([p, *TEST_CASE_4])
    TEST_CASES.append([p, *TEST_CASE_5])
    TEST_CASES.append([p, *TEST_CASE_6])
    TEST_CASES.append([p, *TEST_CASE_7])
    TEST_CASES.append([p, *TEST_CASE_8])
    TEST_CASES.append([p, *TEST_CASE_9])
    TEST_CASES.append([p, *TEST_CASE_10])
    TEST_CASES.append([p, *TEST_CASE_11])
    TEST_CASES.append([p, *TEST_CASE_12])
    TEST_CASES.append([p, *TEST_CASE_13])
    TEST_CASES.append([p, *TEST_CASE_14])
    TEST_CASES.append([p, *TEST_CASE_15])
    TEST_CASES.append([p, *TEST_CASE_16])


class TestGridPatch(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    @SkipIfBeforePyTorchVersion((1, 11, 1))
    def test_grid_patch(self, in_type, input_parameters, image, expected):
        input_image = in_type(image)
        splitter = GridPatch(**input_parameters)
        output = splitter(input_image)
        self.assertEqual(len(output), len(expected))
        for output_patch, expected_patch in zip(output, expected):
            assert_allclose(
                output_patch,
                in_type(expected_patch),
                type_test=False,
                device_test=bool(isinstance(in_type(expected_patch), torch.Tensor)),
            )

    @parameterized.expand([TEST_CASE_META_0, TEST_CASE_META_1])
    @SkipIfBeforePyTorchVersion((1, 9, 1))
    def test_grid_patch_meta(self, input_parameters, image, expected, expected_meta):
        set_track_meta(True)
        splitter = GridPatch(**input_parameters)
        output = splitter(image)
        self.assertEqual(len(output), len(expected))
        if "path" in expected_meta[0]:
            self.assertTrue(output.meta["path"] == expected_meta[0]["path"])
        for output_patch, expected_patch, expected_patch_meta in zip(output, expected, expected_meta):
            assert_allclose(output_patch, expected_patch, type_test=False)
            self.assertIsInstance(output_patch, MetaTensor)
            self.assertEqual(output_patch.meta["location"], expected_patch_meta["location"])
            self.assertTrue(output_patch.meta["spatial_shape"], list(output_patch.shape[1:]))
            if "path" in expected_meta[0]:
                self.assertEqual(output_patch.meta["path"], expected_patch_meta["path"])


if __name__ == "__main__":
    unittest.main()
