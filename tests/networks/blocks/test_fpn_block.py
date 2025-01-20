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
from collections import OrderedDict

import torch
from parameterized import parameterized

from monai.networks.blocks.backbone_fpn_utils import _resnet_fpn_extractor
from monai.networks.blocks.feature_pyramid_network import FeaturePyramidNetwork
from monai.networks.nets.resnet import resnet50
from monai.utils import optional_import
from tests.util import SkipIfBeforePyTorchVersion, test_script_save

_, has_torchvision = optional_import("torchvision")

TEST_CASES = [
    [
        {"spatial_dims": 3, "in_channels_list": [32, 64], "out_channels": 6},
        ((7, 32, 16, 32, 64), (7, 64, 8, 16, 32)),
        ((7, 6, 16, 32, 64), (7, 6, 8, 16, 32)),
    ],
    [
        {"spatial_dims": 2, "in_channels_list": [32, 64], "out_channels": 6},
        ((7, 32, 16, 32), (7, 64, 8, 16)),
        ((7, 6, 16, 32), (7, 6, 8, 16)),
    ],
]

TEST_CASES2 = [
    [{"spatial_dims": 3, "returned_layers": [1]}, (7, 3, 32, 64, 32), ((7, 256, 16, 32, 16), (7, 256, 8, 16, 8))]
]


class TestFPNBlock(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_fpn_block(self, input_param, input_shape, expected_shape):
        net = FeaturePyramidNetwork(**input_param)
        data = OrderedDict()
        data["feat0"] = torch.rand(input_shape[0])
        data["feat1"] = torch.rand(input_shape[1])
        result = net(data)
        self.assertEqual(result["feat0"].shape, expected_shape[0])
        self.assertEqual(result["feat1"].shape, expected_shape[1])

    @parameterized.expand(TEST_CASES)
    @SkipIfBeforePyTorchVersion((1, 9, 1))
    def test_script(self, input_param, input_shape, expected_shape):
        # test whether support torchscript
        net = FeaturePyramidNetwork(**input_param)
        data = OrderedDict()
        data["feat0"] = torch.rand(input_shape[0])
        data["feat1"] = torch.rand(input_shape[1])
        test_script_save(net, data)


@unittest.skipUnless(has_torchvision, "Requires torchvision")
class TestFPN(unittest.TestCase):
    @parameterized.expand(TEST_CASES2)
    def test_fpn(self, input_param, input_shape, expected_shape):
        net = _resnet_fpn_extractor(backbone=resnet50(), spatial_dims=input_param["spatial_dims"], returned_layers=[1])
        data = torch.rand(input_shape)
        result = net(data)
        self.assertEqual(result["0"].shape, expected_shape[0])
        self.assertEqual(result["pool"].shape, expected_shape[1])

    @parameterized.expand(TEST_CASES2)
    @SkipIfBeforePyTorchVersion((1, 9, 1))
    def test_script(self, input_param, input_shape, expected_shape):
        # test whether support torchscript
        net = _resnet_fpn_extractor(backbone=resnet50(), spatial_dims=input_param["spatial_dims"], returned_layers=[1])
        data = torch.rand(input_shape)
        test_script_save(net, data)


if __name__ == "__main__":
    unittest.main()
