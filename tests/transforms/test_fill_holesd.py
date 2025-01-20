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

from monai.transforms import FillHolesd
from monai.utils.enums import CommonKeys
from tests.util import TEST_NDARRAYS, assert_allclose, clone

grid_1_raw = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]

grid_2_raw = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]

grid_3_raw = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]

grid_4_raw = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]

grid_1 = torch.tensor([grid_1_raw])

grid_2 = torch.tensor([grid_2_raw])

grid_3 = torch.tensor([grid_3_raw])

grid_4 = torch.tensor([grid_4_raw])

grid_5 = torch.tensor([[[1, 1, 1], [1, 0, 0], [1, 1, 1]]])

grid_6 = torch.tensor([[[1, 1, 2, 2, 2], [1, 0, 2, 0, 2], [1, 1, 2, 2, 2]]])

grid_7 = torch.tensor([[[1, 1, 2, 2, 2], [1, 0, 2, 2, 2], [1, 1, 2, 2, 2]]])

TEST_CASE_0 = ["enclosed_default_full_connectivity_default_applied_labels", {}, grid_1, grid_3]

TEST_CASE_1 = ["enclosed_full_connectivity_default_applied_labels", {"connectivity": 2}, grid_1, grid_3]

TEST_CASE_2 = [
    "enclosed_full_connectivity_applied_labels_same_single",
    {"connectivity": 2, "applied_labels": 1},
    grid_1,
    grid_3,
]

TEST_CASE_3 = [
    "enclosed_full_connectivity_applied_labels_same_list",
    {"connectivity": 2, "applied_labels": [1]},
    grid_1,
    grid_3,
]

TEST_CASE_4 = [
    "enclosed_full_connectivity_applied_labels_other_single",
    {"connectivity": 2, "applied_labels": 2},
    grid_1,
    grid_1,
]

TEST_CASE_5 = [
    "enclosed_full_connectivity_applied_labels_other_list",
    {"connectivity": 2, "applied_labels": [2]},
    grid_1,
    grid_1,
]

TEST_CASE_6 = [
    "enclosed_full_connectivity_applied_labels_same_and_other",
    {"connectivity": 2, "applied_labels": [1, 2]},
    grid_1,
    grid_3,
]

TEST_CASE_7 = ["enclosed_connectivity_1_default_applied_labels", {"connectivity": 1}, grid_1, grid_3]

TEST_CASE_8 = ["enclosed_connectivity_1_default_applied_labels", {"connectivity": 1}, grid_2, grid_4]

TEST_CASE_9 = ["open_full_connectivity_default_applied_labels", {"connectivity": 2}, grid_2, grid_2]

TEST_CASE_10 = ["open_to_edge_connectivity_1_default_applied_labels", {"connectivity": 1}, grid_5, grid_5]

TEST_CASE_11 = ["open_to_other_label_connectivity_1_default_applied_labels", {"connectivity": 1}, grid_6, grid_7]

TEST_CASE_12 = [
    "open_to_other_label_connectivity_1_applied_labels_other",
    {"connectivity": 1, "applied_labels": 1},
    grid_6,
    grid_6,
]

TEST_CASE_13 = [
    "numpy_enclosed_default_full_connectivity_default_applied_labels",
    {},
    grid_1.cpu().numpy(),
    grid_3.cpu().numpy(),
]

TEST_CASE_14 = [
    "3D_enclosed_full_connectivity_default_applied_labels",
    {"connectivity": 3},
    torch.tensor([[grid_3_raw, grid_1_raw, grid_3_raw]]),
    torch.tensor([[grid_3_raw, grid_3_raw, grid_3_raw]]),
]

TEST_CASE_15 = [
    "3D_enclosed_connectivity_1_default_applied_labels",
    {"connectivity": 1},
    torch.tensor([[grid_4_raw, grid_2_raw, grid_4_raw]]),
    torch.tensor([[grid_4_raw, grid_4_raw, grid_4_raw]]),
]

TEST_CASE_16 = [
    "3D_open_full_connectivity_default_applied_labels",
    {"connectivity": 3},
    torch.tensor([[grid_4_raw, grid_2_raw, grid_4_raw]]),
    torch.tensor([[grid_4_raw, grid_2_raw, grid_4_raw]]),
]

TEST_CASE_17 = [
    "3D_open_to_edge_connectivity_1_default_applied_labels",
    {"connectivity": 1},
    torch.tensor([[grid_1_raw, grid_1_raw, grid_3_raw]]),
    torch.tensor([[grid_1_raw, grid_1_raw, grid_3_raw]]),
]

TEST_CASE_18 = [
    "enclosed_full_connectivity_applied_labels_with_background",
    {"connectivity": 2, "applied_labels": [0, 1]},
    grid_1,
    grid_3,
]

TEST_CASE_19 = [
    "enclosed_full_connectivity_applied_labels_only_background",
    {"connectivity": 2, "applied_labels": [0]},
    grid_1,
    grid_1,
]

TEST_CASE_20 = [
    "one-hot_enclosed_connectivity_1_default_applied_labels",
    {"connectivity": 1},
    torch.tensor([grid_1_raw, grid_1_raw, grid_2_raw]),
    torch.tensor([grid_1_raw, grid_3_raw, grid_4_raw]),
]

TEST_CASE_21 = [
    "one-hot_enclosed_connectivity_1_applied_labels_2",
    {"connectivity": 1, "applied_labels": [2]},
    torch.tensor([grid_1_raw, grid_1_raw, grid_2_raw]),
    torch.tensor([grid_1_raw, grid_1_raw, grid_4_raw]),
]

TEST_CASE_22 = [
    "one-hot_full_connectivity_applied_labels_2",
    {"connectivity": 2},
    torch.tensor([grid_1_raw, grid_1_raw, grid_2_raw]),
    torch.tensor([grid_1_raw, grid_3_raw, grid_2_raw]),
]

VALID_CASES = [
    TEST_CASE_0,
    TEST_CASE_1,
    TEST_CASE_2,
    TEST_CASE_3,
    TEST_CASE_4,
    TEST_CASE_5,
    TEST_CASE_6,
    TEST_CASE_7,
    TEST_CASE_8,
    TEST_CASE_9,
    TEST_CASE_10,
    TEST_CASE_11,
    TEST_CASE_12,
    TEST_CASE_13,
    TEST_CASE_14,
    TEST_CASE_15,
    TEST_CASE_16,
    TEST_CASE_17,
    TEST_CASE_18,
    TEST_CASE_19,
    TEST_CASE_20,
    TEST_CASE_21,
    TEST_CASE_22,
]


class TestFillHoles(unittest.TestCase):
    @parameterized.expand(VALID_CASES)
    def test_correct_results(self, _, args, input_image, expected):
        key = CommonKeys.IMAGE
        converter = FillHolesd(keys=key, **args)
        for p in TEST_NDARRAYS:
            result = converter({key: p(clone(input_image))})[key]
            assert_allclose(result, p(expected), type_test="tensor")


if __name__ == "__main__":
    unittest.main()
