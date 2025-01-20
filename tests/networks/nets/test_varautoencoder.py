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
from monai.networks.layers import Act
from monai.networks.nets import VarAutoEncoder
from tests.util import test_script_save

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TEST_CASE_0 = [  # single channel 2D, batch 4, no residual
    {
        "spatial_dims": 2,
        "in_shape": (1, 128, 128),
        "out_channels": 1,
        "latent_size": 2,
        "channels": (4, 8, 16),
        "strides": (2, 2, 2),
        "num_res_units": 0,
    },
    (1, 1, 128, 128),
    (1, 1, 128, 128),
]

TEST_CASE_1 = [  # single channel 2D, batch 4
    {
        "spatial_dims": 2,
        "in_shape": (1, 128, 128),
        "out_channels": 1,
        "latent_size": 2,
        "channels": (4, 8, 16),
        "strides": (2, 2, 2),
    },
    (1, 1, 128, 128),
    (1, 1, 128, 128),
]

TEST_CASE_2 = [  # 3-channel 2D, batch 4, LeakyReLU activation
    {
        "spatial_dims": 2,
        "in_shape": (3, 128, 128),
        "out_channels": 3,
        "latent_size": 2,
        "channels": (4, 8, 16),
        "strides": (2, 2, 2),
        "act": (Act.LEAKYRELU, {"negative_slope": 0.2}),
    },
    (1, 3, 128, 128),
    (1, 3, 128, 128),
]

TEST_CASE_3 = [  # 4-channel 3D, batch 4
    {
        "spatial_dims": 3,
        "in_shape": (4, 128, 128, 128),
        "out_channels": 3,
        "latent_size": 2,
        "channels": (4, 8, 16),
        "strides": (2, 2, 2),
    },
    (1, 4, 128, 128, 128),
    (1, 3, 128, 128, 128),
]

TEST_CASE_4 = [  # 4-channel 1D, batch 4
    {
        "spatial_dims": 1,
        "in_shape": (4, 128),
        "out_channels": 3,
        "latent_size": 2,
        "channels": (4, 8, 16),
        "strides": (2, 2, 2),
    },
    (1, 4, 128),
    (1, 3, 128),
]

TEST_CASE_5 = [  # 4-channel 1D, batch 4, use_sigmoid = False
    {
        "spatial_dims": 1,
        "in_shape": (4, 128),
        "out_channels": 3,
        "latent_size": 2,
        "channels": (4, 8, 16),
        "strides": (2, 2, 2),
        "use_sigmoid": False,
    },
    (1, 4, 128),
    (1, 3, 128),
]

CASES = [TEST_CASE_0, TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4, TEST_CASE_5]


class TestVarAutoEncoder(unittest.TestCase):
    @parameterized.expand(CASES)
    def test_shape(self, input_param, input_shape, expected_shape):
        net = VarAutoEncoder(**input_param).to(device)
        with eval_mode(net):
            result = net.forward(torch.randn(input_shape).to(device))[0]
            self.assertEqual(result.shape, expected_shape)

    def test_script(self):
        net = VarAutoEncoder(
            spatial_dims=2, in_shape=(1, 32, 32), out_channels=1, latent_size=2, channels=(4, 8), strides=(2, 2)
        )
        test_data = torch.randn(2, 1, 32, 32)
        test_script_save(net, test_data, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
