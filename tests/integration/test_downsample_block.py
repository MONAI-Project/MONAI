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
from monai.networks.blocks import MaxAvgPool

TEST_CASES = [
    [{"spatial_dims": 2, "kernel_size": 2}, (7, 4, 64, 48), (7, 8, 32, 24)],  # 4-channel 2D, batch 7
    [{"spatial_dims": 1, "kernel_size": 4}, (16, 4, 63), (16, 8, 15)],  # 4-channel 1D, batch 16
    [{"spatial_dims": 1, "kernel_size": 4, "padding": 1}, (16, 4, 63), (16, 8, 16)],  # 4-channel 1D, batch 16
    [  # 4-channel 3D, batch 16
        {"spatial_dims": 3, "kernel_size": 3, "ceil_mode": True},
        (16, 4, 32, 24, 48),
        (16, 8, 11, 8, 16),
    ],
    [  # 1-channel 3D, batch 16
        {"spatial_dims": 3, "kernel_size": 3, "ceil_mode": False},
        (16, 1, 32, 24, 48),
        (16, 2, 10, 8, 16),
    ],
]


class TestMaxAvgPool(unittest.TestCase):

    @parameterized.expand(TEST_CASES)
    def test_shape(self, input_param, input_shape, expected_shape):
        net = MaxAvgPool(**input_param)
        with eval_mode(net):
            result = net(torch.randn(input_shape))
            self.assertEqual(result.shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
