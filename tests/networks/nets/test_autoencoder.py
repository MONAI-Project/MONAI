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
from monai.networks.nets import AutoEncoder
from tests.util import test_script_save

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TEST_CASE_0 = [  # single channel 2D, batch 4, no residual
    {
        "spatial_dims": 2,
        "in_channels": 1,
        "out_channels": 1,
        "channels": (4, 8, 16),
        "strides": (2, 2, 2),
        "num_res_units": 0,
    },
    (1, 1, 128, 128),
    (1, 1, 128, 128),
]

TEST_CASE_1 = [  # single channel 2D, batch 4
    {"spatial_dims": 2, "in_channels": 1, "out_channels": 1, "channels": (4, 8, 16), "strides": (2, 2, 2)},
    (1, 1, 128, 128),
    (1, 1, 128, 128),
]

TEST_CASE_2 = [  # 3-channel 2D, batch 4, LeakyReLU activation
    {
        "spatial_dims": 2,
        "in_channels": 3,
        "out_channels": 3,
        "channels": (4, 8, 16),
        "strides": (2, 2, 2),
        "act": (Act.LEAKYRELU, {"negative_slope": 0.2}),
    },
    (1, 3, 128, 128),
    (1, 3, 128, 128),
]

TEST_CASE_3 = [  # 4-channel 3D, batch 4
    {"spatial_dims": 3, "in_channels": 4, "out_channels": 3, "channels": (4, 8, 16), "strides": (2, 2, 2)},
    (1, 4, 128, 128, 128),
    (1, 3, 128, 128, 128),
]

CASES = [TEST_CASE_0, TEST_CASE_1, TEST_CASE_2, TEST_CASE_3]


TEST_CASE_FAIL = {  # 2-channel 2D, should fail because of stride/channel mismatch.
    "spatial_dims": 2,
    "in_channels": 2,
    "out_channels": 2,
    "channels": (4, 8, 16),
    "strides": (2, 2),
}


class TestAutoEncoder(unittest.TestCase):
    @parameterized.expand(CASES)
    def test_shape(self, input_param, input_shape, expected_shape):
        net = AutoEncoder(**input_param).to(device)
        with eval_mode(net):
            result = net.forward(torch.randn(input_shape).to(device))
            self.assertEqual(result.shape, expected_shape)

    def test_script(self):
        net = AutoEncoder(spatial_dims=2, in_channels=1, out_channels=1, channels=(4, 8), strides=(2, 2))
        test_data = torch.randn(2, 1, 32, 32)
        test_script_save(net, test_data)

    def test_channel_stride_difference(self):
        with self.assertRaises(ValueError):
            AutoEncoder(**TEST_CASE_FAIL)


if __name__ == "__main__":
    unittest.main()
