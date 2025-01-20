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
from typing import TYPE_CHECKING
from unittest import skipUnless

import torch
from parameterized import parameterized

from monai.networks import eval_mode
from monai.networks.nets import DenseNet121, Densenet169, DenseNet264, densenet201
from monai.utils import optional_import
from tests.util import skip_if_downloading_fails, skip_if_quick, test_script_save

if TYPE_CHECKING:
    import torchvision

    has_torchvision = True
else:
    torchvision, has_torchvision = optional_import("torchvision")

device = "cuda" if torch.cuda.is_available() else "cpu"

TEST_CASE_1 = [  # 4-channel 3D, batch 2
    {"pretrained": False, "spatial_dims": 3, "in_channels": 2, "out_channels": 3, "norm": ("instance", {"eps": 1e-5})},
    (2, 2, 32, 64, 48),
    (2, 3),
]

TEST_CASE_2 = [  # 4-channel 2D, batch 2
    {"pretrained": False, "spatial_dims": 2, "in_channels": 2, "out_channels": 3, "act": "PRELU"},
    (2, 2, 32, 64),
    (2, 3),
]

TEST_CASE_3 = [  # 4-channel 1D, batch 1
    {"pretrained": False, "spatial_dims": 1, "in_channels": 2, "out_channels": 3},
    (1, 2, 32),
    (1, 3),
]

TEST_CASES = []
for case in [TEST_CASE_1, TEST_CASE_2, TEST_CASE_3]:
    for model in [DenseNet121, Densenet169, densenet201, DenseNet264]:
        TEST_CASES.append([model, *case])

TEST_SCRIPT_CASES = [[model, *TEST_CASE_1] for model in [DenseNet121, Densenet169, densenet201, DenseNet264]]

TEST_PRETRAINED_2D_CASE_1 = [  # 4-channel 2D, batch 2
    DenseNet121,
    {"pretrained": True, "progress": True, "spatial_dims": 2, "in_channels": 2, "out_channels": 3},
    (1, 2, 32, 64),
    (1, 3),
]

TEST_PRETRAINED_2D_CASE_2 = [  # 4-channel 2D, batch 2
    DenseNet121,
    {"pretrained": True, "progress": False, "spatial_dims": 2, "in_channels": 2, "out_channels": 1},
    (1, 2, 32, 64),
    (1, 1),
]

TEST_PRETRAINED_2D_CASE_3 = [
    DenseNet121,
    {"pretrained": True, "progress": False, "spatial_dims": 2, "in_channels": 3, "out_channels": 1},
    (1, 3, 32, 32),
]


class TestPretrainedDENSENET(unittest.TestCase):
    @parameterized.expand([TEST_PRETRAINED_2D_CASE_1, TEST_PRETRAINED_2D_CASE_2])
    @skip_if_quick
    def test_121_2d_shape_pretrain(self, model, input_param, input_shape, expected_shape):
        with skip_if_downloading_fails():
            net = model(**input_param).to(device)
        with eval_mode(net):
            result = net.forward(torch.randn(input_shape).to(device))
            self.assertEqual(result.shape, expected_shape)

    @parameterized.expand([TEST_PRETRAINED_2D_CASE_3])
    @skipUnless(has_torchvision, "Requires `torchvision` package.")
    def test_pretrain_consistency(self, model, input_param, input_shape):
        example = torch.randn(input_shape).to(device)
        with skip_if_downloading_fails():
            net = model(**input_param).to(device)
        with eval_mode(net):
            result = net.features.forward(example)
        torchvision_net = torchvision.models.densenet121(pretrained=True).to(device)
        with eval_mode(torchvision_net):
            expected_result = torchvision_net.features.forward(example)
        self.assertTrue(torch.all(result == expected_result))


class TestDENSENET(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_densenet_shape(self, model, input_param, input_shape, expected_shape):
        net = model(**input_param).to(device)
        with eval_mode(net):
            result = net.forward(torch.randn(input_shape).to(device))
            self.assertEqual(result.shape, expected_shape)

    @parameterized.expand(TEST_SCRIPT_CASES)
    def test_script(self, model, input_param, input_shape, expected_shape):
        net = model(**input_param)
        test_data = torch.randn(input_shape)
        test_script_save(net, test_data)


if __name__ == "__main__":
    unittest.main()
