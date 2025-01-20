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
from monai.networks.nets import HighResNet
from tests.util import DistTestCase, TimedCall, test_script_save

device = "cuda" if torch.cuda.is_available() else "cpu"

TEST_CASE_1 = [  # single channel 3D, batch 16
    {"spatial_dims": 3, "in_channels": 1, "out_channels": 3, "norm_type": "instance"},
    (16, 1, 32, 24, 48),
    (16, 3, 32, 24, 48),
]

TEST_CASE_2 = [  # 4-channel 3D, batch 1
    {"spatial_dims": 3, "in_channels": 4, "out_channels": 3, "acti_type": "relu6"},
    (1, 4, 17, 64, 48),
    (1, 3, 17, 64, 48),
]

TEST_CASE_3 = [  # 4-channel 2D, batch 7
    {"spatial_dims": 2, "in_channels": 4, "out_channels": 3},
    (7, 4, 64, 48),
    (7, 3, 64, 48),
]

TEST_CASE_4 = [  # 4-channel 1D, batch 16
    {"spatial_dims": 1, "in_channels": 4, "out_channels": 3, "dropout_prob": 0.1},
    (16, 4, 63),
    (16, 3, 63),
]


class TestHighResNet(DistTestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4])
    def test_shape(self, input_param, input_shape, expected_shape):
        net = HighResNet(**input_param).to(device)
        with eval_mode(net):
            result = net.forward(torch.randn(input_shape).to(device))
            self.assertEqual(result.shape, expected_shape)

    @TimedCall(seconds=800, force_quit=True)
    def test_script(self):
        input_param, input_shape, expected_shape = TEST_CASE_1
        net = HighResNet(**input_param)
        test_data = torch.randn(input_shape)
        test_script_save(net, test_data, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
