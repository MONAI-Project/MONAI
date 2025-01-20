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

from monai.networks.blocks.dints_block import P3DActiConvNormBlock

TEST_CASES_3D = [
    [
        {"in_channel": 32, "out_channel": 16, "kernel_size": 3, "padding": 0, "mode": 0},
        (7, 32, 16, 32, 8),
        (7, 16, 14, 30, 6),
    ],
    [
        {"in_channel": 32, "out_channel": 16, "kernel_size": 3, "padding": 1, "mode": 0},  # check padding
        (7, 32, 16, 32, 8),
        (7, 16, 16, 32, 8),
    ],
    [
        {"in_channel": 32, "out_channel": 16, "kernel_size": 3, "padding": 0, "mode": 1},
        (7, 32, 16, 32, 8),
        (7, 16, 14, 30, 6),
    ],
    [
        {
            "in_channel": 32,
            "out_channel": 16,
            "kernel_size": 3,
            "padding": 0,
            "mode": 2,
            "act_name": ("leakyrelu", {"inplace": True, "negative_slope": 0.2}),
        },
        (7, 32, 16, 32, 8),
        (7, 16, 14, 30, 6),
    ],
    [
        {
            "in_channel": 32,
            "out_channel": 16,
            "kernel_size": 4,
            "padding": 0,
            "mode": 0,
            "norm_name": ("INSTANCE", {"affine": True}),
        },
        (7, 32, 16, 32, 8),
        (7, 16, 13, 29, 5),
    ],
]


class TestP3D(unittest.TestCase):

    @parameterized.expand(TEST_CASES_3D)
    def test_3d(self, input_param, input_shape, expected_shape):
        net = P3DActiConvNormBlock(**input_param)
        result = net(torch.randn(input_shape))
        self.assertEqual(result.shape, expected_shape)

    def test_ill(self):
        with self.assertRaises(ValueError):
            P3DActiConvNormBlock(in_channel=32, out_channel=16, kernel_size=3, padding=0, mode=3)


if __name__ == "__main__":
    unittest.main()
