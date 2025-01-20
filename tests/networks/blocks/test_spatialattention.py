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
from unittest import skipUnless

import torch
from parameterized import parameterized

from monai.networks import eval_mode
from monai.networks.blocks.spatialattention import SpatialAttentionBlock
from monai.utils import optional_import

einops, has_einops = optional_import("einops")

TEST_CASES = [
    [
        {"spatial_dims": 2, "num_channels": 128, "num_head_channels": 32, "norm_num_groups": 32, "norm_eps": 1e-6},
        (1, 128, 32, 32),
        (1, 128, 32, 32),
    ],
    [
        {"spatial_dims": 3, "num_channels": 16, "num_head_channels": 8, "norm_num_groups": 8, "norm_eps": 1e-6},
        (1, 16, 8, 8, 8),
        (1, 16, 8, 8, 8),
    ],
]


class TestBlock(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    @skipUnless(has_einops, "Requires einops")
    def test_shape(self, input_param, input_shape, expected_shape):
        net = SpatialAttentionBlock(**input_param)
        with eval_mode(net):
            result = net(torch.randn(input_shape))
            self.assertEqual(result.shape, expected_shape)

    def test_attention_dim_not_multiple_of_heads(self):
        with self.assertRaises(ValueError):
            SpatialAttentionBlock(spatial_dims=2, num_channels=128, num_head_channels=33)


if __name__ == "__main__":
    unittest.main()
