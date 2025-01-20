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
from monai.networks.layers import Act, Norm
from monai.networks.nets import UNet
from tests.util import test_script_save

device = "cuda" if torch.cuda.is_available() else "cpu"

TEST_CASE_0 = [  # single channel 2D, batch 16, no residual
    {
        "spatial_dims": 2,
        "in_channels": 1,
        "out_channels": 3,
        "channels": (16, 32, 64),
        "strides": (2, 2),
        "num_res_units": 0,
    },
    (16, 1, 32, 32),
    (16, 3, 32, 32),
]

TEST_CASE_1 = [  # single channel 2D, batch 16
    {
        "spatial_dims": 2,
        "in_channels": 1,
        "out_channels": 3,
        "channels": (16, 32, 64),
        "strides": (2, 2),
        "num_res_units": 1,
    },
    (16, 1, 32, 32),
    (16, 3, 32, 32),
]

TEST_CASE_2 = [  # single channel 3D, batch 16
    {
        "spatial_dims": 3,
        "in_channels": 1,
        "out_channels": 3,
        "channels": (16, 32, 64),
        "strides": (2, 2),
        "num_res_units": 1,
    },
    (16, 1, 32, 24, 48),
    (16, 3, 32, 24, 48),
]

TEST_CASE_3 = [  # 4-channel 3D, batch 16
    {
        "spatial_dims": 3,
        "in_channels": 4,
        "out_channels": 3,
        "channels": (16, 32, 64),
        "strides": (2, 2),
        "num_res_units": 1,
    },
    (16, 4, 32, 64, 48),
    (16, 3, 32, 64, 48),
]

TEST_CASE_4 = [  # 4-channel 3D, batch 16, batch normalization
    {
        "spatial_dims": 3,
        "in_channels": 4,
        "out_channels": 3,
        "channels": (16, 32, 64),
        "strides": (2, 2),
        "num_res_units": 1,
        "norm": Norm.BATCH,
    },
    (16, 4, 32, 64, 48),
    (16, 3, 32, 64, 48),
]

TEST_CASE_5 = [  # 4-channel 3D, batch 16, LeakyReLU activation
    {
        "spatial_dims": 3,
        "in_channels": 4,
        "out_channels": 3,
        "channels": (16, 32, 64),
        "strides": (2, 2),
        "num_res_units": 1,
        "act": (Act.LEAKYRELU, {"negative_slope": 0.2}),
        "adn_ordering": "NA",
    },
    (16, 4, 32, 64, 48),
    (16, 3, 32, 64, 48),
]

TEST_CASE_6 = [  # 4-channel 3D, batch 16, LeakyReLU activation explicit
    {
        "spatial_dims": 3,
        "in_channels": 4,
        "out_channels": 3,
        "channels": (16, 32, 64),
        "strides": (2, 2),
        "num_res_units": 1,
        "act": (torch.nn.LeakyReLU, {"negative_slope": 0.2}),
    },
    (16, 4, 32, 64, 48),
    (16, 3, 32, 64, 48),
]

CASES = [TEST_CASE_0, TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4, TEST_CASE_5, TEST_CASE_6]

ILL_CASES = [
    [
        {  # len(channels) < 2
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 3,
            "channels": (16,),
            "strides": (2, 2),
            "num_res_units": 0,
        }
    ],
    [
        {  # len(strides) < len(channels) - 1
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 3,
            "channels": (8, 8, 8),
            "strides": (2,),
            "num_res_units": 0,
        }
    ],
    [
        {  # len(kernel_size) = 3, spatial_dims = 2
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 3,
            "channels": (8, 8, 8),
            "strides": (2, 2),
            "kernel_size": (3, 3, 3),
        }
    ],
    [
        {  # len(up_kernel_size) = 2, spatial_dims = 3
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 3,
            "channels": (8, 8, 8),
            "strides": (2, 2),
            "up_kernel_size": (3, 3),
        }
    ],
]


class TestUNET(unittest.TestCase):
    @parameterized.expand(CASES)
    def test_shape(self, input_param, input_shape, expected_shape):
        net = UNet(**input_param).to(device)
        with eval_mode(net):
            result = net.forward(torch.randn(input_shape).to(device))
            self.assertEqual(result.shape, expected_shape)

    def test_script(self):
        net = UNet(
            spatial_dims=2, in_channels=1, out_channels=3, channels=(16, 32, 64), strides=(2, 2), num_res_units=0
        )
        test_data = torch.randn(16, 1, 32, 32)
        test_script_save(net, test_data)

    def test_script_without_running_stats(self):
        net = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=3,
            channels=(16, 32, 64),
            strides=(2, 2),
            num_res_units=0,
            norm=("batch", {"track_running_stats": False}),
        )
        test_data = torch.randn(16, 1, 16, 4)
        test_script_save(net, test_data)

    def test_ill_input_shape(self):
        net = UNet(spatial_dims=2, in_channels=1, out_channels=3, channels=(16, 32, 64), strides=(2, 2))
        with eval_mode(net):
            with self.assertRaisesRegex(RuntimeError, "Sizes of tensors must match"):
                net.forward(torch.randn(2, 1, 16, 5))

    @parameterized.expand(ILL_CASES)
    def test_ill_input_hyper_params(self, input_param):
        with self.assertRaises(ValueError):
            _ = UNet(**input_param)


if __name__ == "__main__":
    unittest.main()
