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

from monai.networks.blocks.dints_block import ActiConvNormBlock

TEST_CASES = [
    [{"in_channel": 32, "out_channel": 16, "kernel_size": 3, "padding": 1}, (7, 32, 16, 31, 7), (7, 16, 16, 31, 7)],
    [
        {"in_channel": 32, "out_channel": 16, "kernel_size": 3, "padding": 1, "spatial_dims": 2},
        (7, 32, 13, 32),
        (7, 16, 13, 32),
    ],
]


class TestACNBlock(unittest.TestCase):

    @parameterized.expand(TEST_CASES)
    def test_acn_block(self, input_param, input_shape, expected_shape):
        net = ActiConvNormBlock(**input_param)
        result = net(torch.randn(input_shape))
        self.assertEqual(result.shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
