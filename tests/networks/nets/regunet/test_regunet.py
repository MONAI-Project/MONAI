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
from monai.networks.nets.regunet import RegUNet
from tests.util import test_script_save

device = "cuda" if torch.cuda.is_available() else "cpu"

TEST_CASE_REGUNET_2D = [
    [
        {
            "spatial_dims": 2,
            "in_channels": 2,
            "num_channel_initial": 16,
            "depth": 3,
            "out_kernel_initializer": "kaiming_uniform",
            "out_activation": None,
            "out_channels": 2,
            "pooling": False,
            "concat_skip": True,
            "encode_kernel_sizes": 3,
        },
        (1, 2, 16, 16),
        (1, 2, 16, 16),
    ]
]

TEST_CASE_REGUNET_3D = [
    [
        {
            "spatial_dims": 3,
            "in_channels": 2,
            "num_channel_initial": 16,
            "depth": 3,
            "out_kernel_initializer": "kaiming_uniform",
            "out_activation": "sigmoid",
            "out_channels": 2,
            "extract_levels": (0, 1, 2, 3),
            "pooling": True,
            "concat_skip": False,
            "encode_kernel_sizes": (3, 3, 3, 7),
        },
        (1, 2, 16, 16, 16),
        (1, 2, 16, 16, 16),
    ]
]


class TestREGUNET(unittest.TestCase):
    @parameterized.expand(TEST_CASE_REGUNET_2D + TEST_CASE_REGUNET_3D)
    def test_shape(self, input_param, input_shape, expected_shape):
        net = RegUNet(**input_param).to(device)
        with eval_mode(net):
            result = net(torch.randn(input_shape).to(device))
            self.assertEqual(result.shape, expected_shape)

    def test_ill_shape(self):
        with self.assertRaisesRegex(ValueError, ""):
            input_param, _, _ = TEST_CASE_REGUNET_2D[0]
            input_shape = (1, input_param["in_channels"], 17, 17)
            net = RegUNet(**input_param).to(device)
            net.forward(torch.randn(input_shape).to(device))

    def test_script(self):
        input_param, input_shape, _ = TEST_CASE_REGUNET_2D[0]
        net = RegUNet(**input_param)
        test_data = torch.randn(input_shape)
        test_script_save(net, test_data)


if __name__ == "__main__":
    unittest.main()
