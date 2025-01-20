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
from unittest import skipUnless

import torch
from parameterized import parameterized

from monai.networks import eval_mode
from monai.networks.nets import TorchVisionFCModel, UNet
from monai.networks.utils import look_up_named_module, set_named_module
from monai.utils import min_version, optional_import
from tests.util import skip_if_downloading_fails

Inception_V3_Weights, has_enum = optional_import("torchvision.models.inception", name="Inception_V3_Weights")

_, has_tv = optional_import("torchvision", "0.12", min_version)

device = "cuda" if torch.cuda.is_available() else "cpu"

TEST_CASE_0 = [
    {"model_name": "resnet18", "num_classes": 1, "use_conv": True, "pretrained": False},
    (2, 3, 224, 224),
    (2, 1, 1, 1),
]

TEST_CASE_1 = [
    {"model_name": "resnet18", "num_classes": 1, "use_conv": True, "pretrained": False},
    (2, 3, 256, 256),
    (2, 1, 2, 2),
]

TEST_CASE_2 = [
    {"model_name": "resnet101", "num_classes": 5, "use_conv": True, "pretrained": False},
    (2, 3, 256, 256),
    (2, 5, 2, 2),
]

TEST_CASE_3 = [
    {
        "model_name": "resnet101",
        "num_classes": 5,
        "use_conv": True,
        "pool": ("avg", {"kernel_size": 6, "stride": 1}),
        "pretrained": False,
    },
    (2, 3, 224, 224),
    (2, 5, 2, 2),
]

TEST_CASE_4 = [
    {"model_name": "resnet18", "num_classes": 1, "use_conv": False, "pool": None, "pretrained": False},
    (2, 3, 224, 224),
    (2, 1),
]

TEST_CASE_5 = [
    {"model_name": "resnet18", "num_classes": 1, "use_conv": False, "pool": None, "pretrained": False},
    (2, 3, 256, 256),
    (2, 1),
]

TEST_CASE_6 = [
    {"model_name": "resnet101", "num_classes": 5, "use_conv": False, "pool": None, "pretrained": False},
    (2, 3, 256, 256),
    (2, 5),
]

TEST_CASE_7 = [
    {
        "model_name": "inception_v3",
        "num_classes": 5,
        "use_conv": True,
        "pool": "",
        "in_channels": 2048,
        "node_name": "Mixed_7c.cat_2",
    },
    (2, 3, 299, 299),
    (2, 5, 8, 8),
]

TEST_CASE_8 = [
    {"model_name": "vit_b_16", "num_classes": 5, "in_channels": 768, "pool": None, "fc_name": "heads.head"},
    (2, 3, 224, 224),
    (2, 5),
]

TEST_CASE_PRETRAINED_0 = [
    {"model_name": "resnet18", "num_classes": 1, "use_conv": True, "pretrained": True},
    (2, 3, 224, 224),
    (2, 1, 1, 1),
    -0.010419349186122417,
]

TEST_CASE_PRETRAINED_1 = [
    {"model_name": "resnet18", "num_classes": 1, "use_conv": True, "pretrained": True},
    (2, 3, 256, 256),
    (2, 1, 2, 2),
    -0.010419349186122417,
]

TEST_CASE_PRETRAINED_2 = [
    {"model_name": "resnet18", "num_classes": 5, "use_conv": True, "pretrained": True},
    (2, 3, 256, 256),
    (2, 5, 2, 2),
    -0.010419349186122417,
]

TEST_CASE_PRETRAINED_3 = [
    {"model_name": "resnet18", "num_classes": 1, "use_conv": False, "pool": None, "pretrained": True},
    (2, 3, 224, 224),
    (2, 1),
    -0.010419349186122417,
]

TEST_CASE_PRETRAINED_4 = [
    {"model_name": "resnet18", "num_classes": 1, "use_conv": False, "pool": None, "pretrained": True},
    (2, 3, 256, 256),
    (2, 1),
    -0.010419349186122417,
]

TEST_CASE_PRETRAINED_5 = [
    {"model_name": "resnet18", "num_classes": 5, "use_conv": False, "pool": None, "pretrained": True},
    (2, 3, 256, 256),
    (2, 5),
    -0.010419349186122417,
]

TEST_CASE_PRETRAINED_6 = [
    {
        "model_name": "inception_v3",
        "num_classes": 5,
        "use_conv": False,
        "pool": None,
        "weights": Inception_V3_Weights.IMAGENET1K_V1 if has_enum else None,
    },
    (2, 3, 299, 299),
    (2, 5),
    -0.21029122173786163,
]


class TestTorchVisionFCModel(unittest.TestCase):
    @parameterized.expand(
        [TEST_CASE_0, TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4, TEST_CASE_5, TEST_CASE_6, TEST_CASE_7]
        + ([TEST_CASE_8] if has_enum else [])
    )
    @skipUnless(has_tv, "Requires TorchVision.")
    def test_without_pretrained(self, input_param, input_shape, expected_shape):
        net = TorchVisionFCModel(**input_param).to(device)
        with eval_mode(net):
            result = net.forward(torch.randn(input_shape).to(device))
            self.assertEqual(result.shape, expected_shape)

    @parameterized.expand(
        [
            TEST_CASE_PRETRAINED_0,
            TEST_CASE_PRETRAINED_1,
            TEST_CASE_PRETRAINED_2,
            TEST_CASE_PRETRAINED_3,
            TEST_CASE_PRETRAINED_4,
            TEST_CASE_PRETRAINED_5,
        ]
        + ([TEST_CASE_PRETRAINED_6] if has_enum else [])
    )
    @skipUnless(has_tv, "Requires TorchVision.")
    def test_with_pretrained(self, input_param, input_shape, expected_shape, expected_value):
        with skip_if_downloading_fails():
            net = TorchVisionFCModel(**input_param).to(device)
        with eval_mode(net):
            result = net.forward(torch.randn(input_shape).to(device))
            value = next(net.features.parameters())[0, 0, 0, 0].item()
            self.assertEqual(value, expected_value)
            self.assertEqual(result.shape, expected_shape)


class TestLookup(unittest.TestCase):
    def test_get_module(self):
        net = UNet(spatial_dims=2, in_channels=1, out_channels=1, channels=(4, 8, 16, 32, 64), strides=(2, 2, 2, 2))
        self.assertEqual(look_up_named_module("", net), net)
        mod = look_up_named_module("model.1.submodule.1.submodule.1.submodule.0.conv", net)
        self.assertTrue(str(mod).startswith("Conv2d"))
        self.assertIsInstance(set_named_module(net, "model", torch.nn.Identity()).model, torch.nn.Identity)
        self.assertIsNone(look_up_named_module("model.1.submodule.1.submodule.1.submodule.conv", net))
        self.assertIsNone(look_up_named_module("test attribute", net))


if __name__ == "__main__":
    unittest.main()
