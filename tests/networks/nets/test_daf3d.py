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
from monai.networks.nets import DAF3D
from monai.utils import optional_import
from tests.util import test_script_save

_, has_tv = optional_import("torchvision")

TEST_CASES = [
    [{"in_channels": 1, "out_channels": 1}, (1, 1, 32, 32, 64), (1, 1, 32, 32, 64)],  # single channel 3D, batch 1
    [{"in_channels": 2, "out_channels": 1}, (3, 2, 32, 64, 128), (3, 1, 32, 64, 128)],  # two channel 3D, batch 3
    [
        {"in_channels": 2, "out_channels": 2},
        (3, 2, 32, 64, 128),
        (3, 2, 32, 64, 128),
    ],  # two channel 3D, same in & out channels
    [{"in_channels": 4, "out_channels": 1}, (5, 4, 35, 35, 35), (5, 1, 35, 35, 35)],  # four channel 3D, batch 5
    [
        {"in_channels": 4, "out_channels": 4},
        (5, 4, 35, 35, 35),
        (5, 4, 35, 35, 35),
    ],  # four channel 3D, same in & out channels
]


@unittest.skipUnless(has_tv, "torchvision not installed")
class TestDAF3D(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_shape(self, input_param, input_shape, expected_shape):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(input_param)
        net = DAF3D(**input_param).to(device)
        with eval_mode(net):
            result = net(torch.randn(input_shape).to(device))
        self.assertEqual(result.shape, expected_shape)

    @unittest.skip("daf3d: torchscript not currently supported")
    def test_script(self):
        net = DAF3D(in_channels=1, out_channels=1)
        test_data = torch.randn(16, 1, 32, 32)
        test_script_save(net, test_data)


if __name__ == "__main__":
    unittest.main()
