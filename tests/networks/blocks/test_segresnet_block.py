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

from monai.networks import eval_mode
from monai.networks.blocks.segresnet_block import ResBlock

TEST_CASE_RESBLOCK = []
for spatial_dims in range(2, 4):
    for in_channels in range(1, 4):
        for kernel_size in [1, 3]:
            for norm in [("group", {"num_groups": 1}), "batch", "instance"]:
                test_case = [
                    {
                        "spatial_dims": spatial_dims,
                        "in_channels": in_channels,
                        "kernel_size": kernel_size,
                        "norm": norm,
                    },
                    (2, in_channels, *([16] * spatial_dims)),
                    (2, in_channels, *([16] * spatial_dims)),
                ]
                TEST_CASE_RESBLOCK.append(test_case)


class TestResBlock(unittest.TestCase):

    @parameterized.expand(TEST_CASE_RESBLOCK)
    def test_shape(self, input_param, input_shape, expected_shape):
        net = ResBlock(**input_param)
        with eval_mode(net):
            result = net(torch.randn(input_shape))
            self.assertEqual(result.shape, expected_shape)

    def test_ill_arg(self):
        with self.assertRaises(AssertionError):
            ResBlock(spatial_dims=3, in_channels=8, norm="group", kernel_size=2)
        with self.assertRaises(ValueError):
            ResBlock(spatial_dims=3, in_channels=8, norm="norm")


if __name__ == "__main__":
    unittest.main()
