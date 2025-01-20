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

from monai.networks.nets.dints import Cell

TEST_CASES_3D = [
    [
        {"c_prev": 8, "c": 8, "rate": 1, "arch_code_c": None},
        torch.tensor([1, 1, 1, 1, 1]),
        torch.tensor([1, 1, 1, 1, 1]),
        (2, 8, 32, 16, 8),
        (2, 8, 64, 32, 16),
    ],
    [
        {"c_prev": 8, "c": 4, "rate": 1, "arch_code_c": [1, 1, 0, 0, 1]},
        torch.tensor([1, 1, 0, 0, 1]),
        torch.tensor([1, 0.2, 1.3, 0, 1]),
        (2, 8, 32, 16, 8),
        (2, 4, 64, 32, 16),
    ],
    [
        {"c_prev": 8, "c": 8, "rate": 0, "arch_code_c": None, "act_name": "SELU", "norm_name": "BATCH"},
        torch.tensor([1, 1, 1, 1, 1]),
        torch.tensor([0, 0, 0, 1, 0]),
        (2, 8, 32, 16, 8),
        (2, 8, 32, 16, 8),
    ],
    [
        {
            "c_prev": 8,
            "c": 8,
            "rate": -1,
            "arch_code_c": None,
            "act_name": "PRELU",
            "norm_name": ("BATCH", {"affine": False}),
        },
        torch.tensor([1, 1, 1, 1, 1]),
        torch.tensor([1, 1, 1, 1, 1]),
        (2, 8, 32, 16, 8),
        (2, 8, 16, 8, 4),
    ],
    [
        {"c_prev": 8, "c": 8, "rate": -1, "arch_code_c": [1, 0, 0, 0, 1], "act_name": "RELU", "norm_name": "INSTANCE"},
        torch.tensor([1, 0, 0, 0, 1]),
        torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2]),
        (2, 8, 32, 16, 8),
        (2, 8, 16, 8, 4),
    ],
]

TEST_CASES_2D = [
    [
        {
            "c_prev": 8,
            "c": 7,
            "rate": -1,
            "arch_code_c": [1, 0, 0, 0, 1],
            "spatial_dims": 2,
            "act_name": "PRELU",
            "norm_name": ("BATCH", {"affine": False}),
        },
        torch.tensor([1, 0]),
        torch.tensor([0.2, 0.2]),
        (2, 8, 16, 8),
        (2, 7, 8, 4),
    ],
    [
        {
            "c_prev": 8,
            "c": 8,
            "rate": -1,
            "arch_code_c": None,
            "spatial_dims": 2,
            "act_name": "SELU",
            "norm_name": "INSTANCE",
        },
        torch.tensor([1, 0]),
        torch.tensor([0.2, 0.2]),
        (2, 8, 16, 8),
        (2, 8, 8, 4),
    ],
]


class TestCell(unittest.TestCase):

    @parameterized.expand(TEST_CASES_2D + TEST_CASES_3D)
    def test_cell_3d(self, input_param, ops, weight, input_shape, expected_shape):
        net = Cell(**input_param)
        result = net(torch.randn(input_shape), weight=weight)
        self.assertEqual(result.shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
