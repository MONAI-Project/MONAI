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
from monai.networks.nets import VNet
from tests.util import test_script_save

device = "cuda" if torch.cuda.is_available() else "cpu"

TEST_CASE_VNET_2D_1 = [
    {"spatial_dims": 2, "in_channels": 4, "out_channels": 1, "act": "elu", "dropout_dim": 1},
    (1, 4, 32, 32),
    (1, 1, 32, 32),
]
TEST_CASE_VNET_2D_2 = [
    {"spatial_dims": 2, "in_channels": 2, "out_channels": 2, "act": "prelu", "dropout_dim": 2},
    (1, 2, 32, 32),
    (1, 2, 32, 32),
]
TEST_CASE_VNET_2D_3 = [
    {"spatial_dims": 2, "in_channels": 1, "out_channels": 3, "dropout_dim": 3},
    (1, 1, 32, 32),
    (1, 3, 32, 32),
]
TEST_CASE_VNET_3D_1 = [
    {"spatial_dims": 3, "in_channels": 4, "out_channels": 1, "act": "elu", "dropout_dim": 1},
    (1, 4, 32, 32, 32),
    (1, 1, 32, 32, 32),
]
TEST_CASE_VNET_3D_2 = [
    {"spatial_dims": 3, "in_channels": 2, "out_channels": 2, "act": "prelu", "dropout_dim": 2},
    (1, 2, 32, 32, 32),
    (1, 2, 32, 32, 32),
]
TEST_CASE_VNET_3D_3 = [
    {"spatial_dims": 3, "in_channels": 1, "out_channels": 3, "dropout_dim": 3},
    (1, 1, 32, 32, 32),
    (1, 3, 32, 32, 32),
]


class TestVNet(unittest.TestCase):
    @parameterized.expand(
        [
            TEST_CASE_VNET_2D_1,
            TEST_CASE_VNET_2D_2,
            TEST_CASE_VNET_2D_3,
            TEST_CASE_VNET_3D_1,
            TEST_CASE_VNET_3D_2,
            TEST_CASE_VNET_3D_3,
        ]
    )
    def test_vnet_shape(self, input_param, input_shape, expected_shape):
        net = VNet(**input_param).to(device)
        with eval_mode(net):
            result = net.forward(torch.randn(input_shape).to(device))
            self.assertEqual(result.shape, expected_shape)

    def test_script(self):
        net = VNet(spatial_dims=3, in_channels=1, out_channels=3, dropout_dim=3)
        test_data = torch.randn(1, 1, 32, 32, 32)
        test_script_save(net, test_data)


if __name__ == "__main__":
    unittest.main()
