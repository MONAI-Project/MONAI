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

from monai.networks.nets.dints import Cell, MixedOp
from tests.util import test_script_save

TEST_CASES_3D = [
    [
        {"c": 8, "arch_code_c": None},
        torch.tensor([1, 1, 1, 1, 1]),
        torch.tensor([1, 1, 1, 1, 1]),
        (2, 8, 32, 16, 8),
        (2, 8, 32, 16, 8),
    ],
    [
        {"c": 8, "arch_code_c": [1, 1, 0, 0, 1]},
        torch.tensor([1, 1, 0, 0, 1]),
        torch.tensor([1, 0.2, 1.3, 0, 1]),
        (2, 8, 64, 32, 16),
        (2, 8, 64, 32, 16),
    ],
    [
        {"c": 8, "arch_code_c": None},
        torch.tensor([1, 1, 1, 1, 1]),
        torch.tensor([0, 0, 0, 1, 0]),
        (2, 8, 32, 16, 8),
        (2, 8, 32, 16, 8),
    ],
    [
        {"c": 8, "arch_code_c": [1, 1, 1, 0, 1]},
        torch.tensor([1, 1, 1, 1, 1]),
        torch.tensor([0, 0, 0, 1, 0]),
        (2, 8, 32, 16, 8),
        (2, 8, 32, 16, 8),
    ],
]
TEST_CASES_2D = [
    [
        {"c": 32, "arch_code_c": [1, 1, 1, 0, 1]},
        torch.tensor([1, 1]),
        torch.tensor([0, 0]),
        (2, 32, 16, 8),
        (2, 32, 16, 8),
    ]
]


class TestMixOP(unittest.TestCase):
    @parameterized.expand(TEST_CASES_3D)
    def test_mixop_3d(self, input_param, ops, weight, input_shape, expected_shape):
        net = MixedOp(ops=Cell.OPS3D, **input_param)
        result = net(torch.randn(input_shape), weight=weight)
        self.assertEqual(result.shape, expected_shape)
        self.assertEqual(result.shape, expected_shape)

    @parameterized.expand(TEST_CASES_2D)
    def test_mixop_2d(self, input_param, ops, weight, input_shape, expected_shape):
        net = MixedOp(ops=Cell.OPS2D, **input_param)
        result = net(torch.randn(input_shape), weight=weight)
        self.assertEqual(result.shape, expected_shape)

    @parameterized.expand(TEST_CASES_3D)
    def test_script(self, input_param, ops, weight, input_shape, expected_shape):
        net = MixedOp(ops=Cell.OPS3D, **input_param)
        test_script_save(net, torch.randn(input_shape), weight)


if __name__ == "__main__":
    unittest.main()
