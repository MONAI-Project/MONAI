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

import os
import unittest
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import skipUnless

import torch
from parameterized import parameterized

import monai.networks.nets.senet as se_mod
from monai.networks import eval_mode
from monai.networks.nets import SENet, SENet154, SEResNet50, SEResNet101, SEResNet152, SEResNext50, SEResNext101
from monai.utils import optional_import
from tests.util import test_is_quick, test_pretrained_networks, test_script_save, testing_data_config

if TYPE_CHECKING:
    import pretrainedmodels

    has_cadene_pretrain = True
else:
    pretrainedmodels, has_cadene_pretrain = optional_import("pretrainedmodels")

device = "cuda" if torch.cuda.is_available() else "cpu"

NET_ARGS = {"spatial_dims": 3, "in_channels": 2, "num_classes": 2}
TEST_CASE_1 = [SENet154, NET_ARGS]
TEST_CASE_2 = [SEResNet50, NET_ARGS]
TEST_CASE_3 = [SEResNet101, NET_ARGS]
TEST_CASE_4 = [SEResNet152, NET_ARGS]
TEST_CASE_5 = [SEResNext50, NET_ARGS]
TEST_CASE_6 = [SEResNext101, NET_ARGS]
TEST_CASE_7 = [
    SENet,
    {
        "spatial_dims": 3,
        "in_channels": 2,
        "num_classes": 2,
        "block": "se_bottleneck",
        "layers": (3, 8, 36, 3),
        "groups": 64,
        "reduction": 16,
    },
]

TEST_CASE_PRETRAINED_1 = [SEResNet50, {"spatial_dims": 2, "in_channels": 3, "num_classes": 2, "pretrained": True}]


class TestSENET(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4, TEST_CASE_5, TEST_CASE_6, TEST_CASE_7])
    def test_senet_shape(self, net, net_args):
        input_data = torch.randn(2, 2, 64, 64, 64).to(device)
        expected_shape = (2, 2)
        net = net(**net_args).to(device)
        with eval_mode(net):
            result = net(input_data)
            self.assertEqual(result.shape, expected_shape)

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4, TEST_CASE_5, TEST_CASE_6])
    def test_script(self, net, net_args):
        net = net(**net_args)
        input_data = torch.randn(2, 2, 64, 64, 64)
        test_script_save(net, input_data)


class TestPretrainedSENET(unittest.TestCase):
    def setUp(self):
        self.original_urls = se_mod.SE_NET_MODELS.copy()
        replace_url = test_is_quick()
        if not replace_url:
            try:
                SEResNet50(pretrained=True, spatial_dims=2, in_channels=3, num_classes=2)
            except OSError as rt_e:
                print(rt_e)
                if "certificate" in str(rt_e):  # [SSL: CERTIFICATE_VERIFY_FAILED]
                    replace_url = True
        if replace_url:
            testing_dir = Path(__file__).parents[1] / "testing_data"
            testing_data_urls = {
                "senet154": {
                    "url": testing_data_config("models", "senet154-c7b49a05", "url"),
                    "filename": "senet154-c7b49a05.pth",
                },
                "se_resnet50": {
                    "url": testing_data_config("models", "se_resnet50-ce0d4300", "url"),
                    "filename": "se_resnet50-ce0d4300.pth",
                },
                "se_resnet101": {
                    "url": testing_data_config("models", "se_resnet101-7e38fcc6", "url"),
                    "filename": "se_resnet101-7e38fcc6.pth",
                },
                "se_resnet152": {
                    "url": testing_data_config("models", "se_resnet152-d17c99b7", "url"),
                    "filename": "se_resnet152-d17c99b7.pth",
                },
                "se_resnext50_32x4d": {
                    "url": testing_data_config("models", "se_resnext50_32x4d-a260b3a4", "url"),
                    "filename": "se_resnext50_32x4d-a260b3a4.pth",
                },
                "se_resnext101_32x4d": {
                    "url": testing_data_config("models", "se_resnext101_32x4d-3b2fe3d8", "url"),
                    "filename": "se_resnext101_32x4d-3b2fe3d8.pth",
                },
            }
            for item in testing_data_urls:
                testing_data_urls[item]["filename"] = os.path.join(testing_dir, testing_data_urls[item]["filename"])
            se_mod.SE_NET_MODELS = testing_data_urls

    def tearDown(self):
        se_mod.SE_NET_MODELS = self.original_urls.copy()

    @parameterized.expand([TEST_CASE_PRETRAINED_1])
    def test_senet_shape(self, model, input_param):
        net = test_pretrained_networks(model, input_param, device)
        input_data = torch.randn(3, 3, 64, 64).to(device)
        expected_shape = (3, 2)
        net = net.to(device)
        with eval_mode(net):
            result = net(input_data)
            self.assertEqual(result.shape, expected_shape)

    @parameterized.expand([TEST_CASE_PRETRAINED_1])
    @skipUnless(has_cadene_pretrain, "Requires `pretrainedmodels` package.")
    def test_pretrain_consistency(self, model, input_param):
        input_data = torch.randn(1, 3, 64, 64).to(device)
        net = test_pretrained_networks(model, input_param, device)
        with eval_mode(net):
            result = net.features(input_data)
        cadene_net = pretrainedmodels.se_resnet50().to(device)
        with eval_mode(cadene_net):
            expected_result = cadene_net.features(input_data)
        # The difference between Cadene's senet and our version is that
        # we use nn.Linear as the FC layer, but Cadene's version uses
        # a conv layer with kernel size equals to 1. It may bring a little difference.
        self.assertTrue(torch.allclose(result, expected_result, rtol=1e-5, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
