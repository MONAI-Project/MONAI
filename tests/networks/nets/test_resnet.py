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

import copy
import os
import re
import sys
import unittest
from typing import TYPE_CHECKING

import torch
from parameterized import parameterized

from monai.networks import eval_mode
from monai.networks.nets import (
    ResNet,
    ResNetFeatures,
    get_medicalnet_pretrained_resnet_args,
    get_pretrained_resnet_medicalnet,
    resnet10,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
    resnet200,
)
from monai.networks.nets.resnet import ResNetBlock
from monai.utils import optional_import
from tests.util import (
    SkipIfNoModule,
    equal_state_dict,
    skip_if_downloading_fails,
    skip_if_no_cuda,
    skip_if_quick,
    test_script_save,
)

if TYPE_CHECKING:
    import torchvision

    has_torchvision = True
else:
    torchvision, has_torchvision = optional_import("torchvision")

has_hf_modules = "huggingface_hub" in sys.modules and "huggingface_hub.utils._errors" in sys.modules

# from torchvision.models import ResNet50_Weights, resnet50

device = "cuda" if torch.cuda.is_available() else "cpu"

TEST_CASE_1 = [  # 3D, batch 3, 2 input channel
    {
        "pretrained": False,
        "spatial_dims": 3,
        "n_input_channels": 2,
        "num_classes": 3,
        "conv1_t_size": 7,
        "conv1_t_stride": (2, 2, 2),
    },
    (3, 2, 32, 64, 48),
    (3, 3),
]

TEST_CASE_2 = [  # 2D, batch 2, 1 input channel
    {
        "pretrained": False,
        "spatial_dims": 2,
        "n_input_channels": 1,
        "num_classes": 3,
        "conv1_t_size": [7, 7],
        "conv1_t_stride": [2, 2],
    },
    (2, 1, 32, 64),
    (2, 3),
]

TEST_CASE_2_A = [  # 2D, batch 2, 1 input channel, shortcut type A
    {
        "pretrained": False,
        "spatial_dims": 2,
        "n_input_channels": 1,
        "num_classes": 3,
        "shortcut_type": "A",
        "conv1_t_size": (7, 7),
        "conv1_t_stride": 2,
    },
    (2, 1, 32, 64),
    (2, 3),
]

TEST_CASE_3 = [  # 1D, batch 1, 2 input channels
    {
        "pretrained": False,
        "spatial_dims": 1,
        "n_input_channels": 2,
        "num_classes": 3,
        "conv1_t_size": [3],
        "conv1_t_stride": 1,
        "act": ("relu", {"inplace": False}),
    },
    (1, 2, 32),
    (1, 3),
]

TEST_CASE_3_A = [  # 1D, batch 1, 2 input channels
    {"pretrained": False, "spatial_dims": 1, "n_input_channels": 2, "num_classes": 3, "shortcut_type": "A"},
    (1, 2, 32),
    (1, 3),
]

TEST_CASE_4 = [  # 2D, batch 2, 1 input channel
    {"pretrained": False, "spatial_dims": 2, "n_input_channels": 1, "num_classes": 3, "feed_forward": False},
    (2, 1, 32, 64),
    ((2, 512), (2, 2048)),
]

TEST_CASE_5 = [  # 1D, batch 1, 2 input channels
    {
        "block": "basic",
        "layers": [1, 1, 1, 1],
        "block_inplanes": [64, 128, 256, 512],
        "spatial_dims": 1,
        "n_input_channels": 2,
        "num_classes": 3,
        "conv1_t_size": [3],
        "conv1_t_stride": 1,
    },
    (1, 2, 32),
    (1, 3),
]

TEST_CASE_5_A = [  # 1D, batch 1, 2 input channels
    {
        "block": ResNetBlock,
        "layers": [1, 1, 1, 1],
        "block_inplanes": [64, 128, 256, 512],
        "spatial_dims": 1,
        "n_input_channels": 2,
        "num_classes": 3,
        "conv1_t_size": [3],
        "conv1_t_stride": 1,
    },
    (1, 2, 32),
    (1, 3),
]

TEST_CASE_6 = [  # 1D, batch 1, 2 input channels
    {
        "block": "bottleneck",
        "layers": [3, 4, 6, 3],
        "block_inplanes": [64, 128, 256, 512],
        "spatial_dims": 1,
        "n_input_channels": 2,
        "num_classes": 3,
        "conv1_t_size": [3],
        "conv1_t_stride": 1,
    },
    (1, 2, 32),
    (1, 3),
]

TEST_CASE_7 = [  # 1D, batch 1, 2 input channels, bias_downsample
    {
        "block": "bottleneck",
        "layers": [3, 4, 6, 3],
        "block_inplanes": [64, 128, 256, 512],
        "spatial_dims": 1,
        "n_input_channels": 2,
        "num_classes": 3,
        "conv1_t_size": [3],
        "conv1_t_stride": 1,
        "bias_downsample": False,  # set to False if pretrained=True (PR #5477)
    },
    (1, 2, 32),
    (1, 3),
]

TEST_CASE_8 = [
    {
        "block": "bottleneck",
        "layers": [3, 4, 6, 3],
        "block_inplanes": [64, 128, 256, 512],
        "spatial_dims": 1,
        "n_input_channels": 2,
        "num_classes": 3,
        "conv1_t_size": [3],
        "conv1_t_stride": 1,
        "act": ("relu", {"inplace": False}),
    },
    (1, 2, 32),
    (1, 3),
]

TEST_CASE_9 = [  # Layer norm
    {
        "block": ResNetBlock,
        "layers": [3, 4, 6, 3],
        "block_inplanes": [64, 128, 256, 512],
        "spatial_dims": 1,
        "n_input_channels": 2,
        "num_classes": 3,
        "conv1_t_size": [3],
        "conv1_t_stride": 1,
        "act": ("relu", {"inplace": False}),
        "norm": ("layer", {"normalized_shape": (64, 32)}),
    },
    (1, 2, 32),
    (1, 3),
]

TEST_CASES = []
PRETRAINED_TEST_CASES = []
for case in [TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_2_A, TEST_CASE_3_A]:
    for model in [resnet10, resnet18, resnet34, resnet50, resnet101, resnet152, resnet200]:
        TEST_CASES.append([model, *case])
        PRETRAINED_TEST_CASES.append([model, *case])
for case in [TEST_CASE_5, TEST_CASE_5_A, TEST_CASE_6, TEST_CASE_7, TEST_CASE_8, TEST_CASE_9]:
    TEST_CASES.append([ResNet, *case])

TEST_SCRIPT_CASES = [
    [model, *TEST_CASE_1] for model in [resnet10, resnet18, resnet34, resnet50, resnet101, resnet152, resnet200]
]

CASE_EXTRACT_FEATURES = [
    (
        {"model_name": "resnet10", "pretrained": True, "spatial_dims": 3, "in_channels": 1},
        [1, 1, 64, 64, 64],
        ([1, 64, 32, 32, 32], [1, 64, 16, 16, 16], [1, 128, 8, 8, 8], [1, 256, 4, 4, 4], [1, 512, 2, 2, 2]),
    )
]


class TestResNet(unittest.TestCase):
    def setUp(self):
        self.tmp_ckpt_filename = os.path.join("tests", "monai_unittest_tmp_ckpt.pth")

    def tearDown(self):
        if os.path.exists(self.tmp_ckpt_filename):
            try:
                os.remove(self.tmp_ckpt_filename)
            except BaseException:
                pass

    @parameterized.expand(TEST_CASES)
    def test_resnet_shape(self, model, input_param, input_shape, expected_shape):
        net = model(**input_param).to(device)
        with eval_mode(net):
            result = net.forward(torch.randn(input_shape).to(device))
            if input_param.get("feed_forward", True):
                self.assertEqual(result.shape, expected_shape)
            else:
                self.assertIn(result.shape, expected_shape)

    @parameterized.expand(PRETRAINED_TEST_CASES)
    @skip_if_quick
    @skip_if_no_cuda
    def test_resnet_pretrained(self, model, input_param, _input_shape, _expected_shape):
        net = model(**input_param).to(device)
        # Save ckpt
        torch.save(net.state_dict(), self.tmp_ckpt_filename)

        cp_input_param = copy.copy(input_param)
        # Custom pretrained weights
        cp_input_param["pretrained"] = self.tmp_ckpt_filename
        pretrained_net = model(**cp_input_param)
        equal_state_dict(net.state_dict(), pretrained_net.state_dict())

        if has_hf_modules:
            # True flag
            cp_input_param["pretrained"] = True
            resnet_depth = int(re.search(r"resnet(\d+)", model.__name__).group(1))

            bias_downsample, shortcut_type = get_medicalnet_pretrained_resnet_args(resnet_depth)

            # With orig. test cases
            if (
                input_param.get("spatial_dims", 3) == 3
                and input_param.get("n_input_channels", 3) == 1
                and input_param.get("feed_forward", True) is False
                and input_param.get("shortcut_type", "B") == shortcut_type
                and (input_param.get("bias_downsample", True) == bias_downsample)
            ):
                model(**cp_input_param)
            else:
                with self.assertRaises(NotImplementedError):
                    model(**cp_input_param)

            # forcing MedicalNet pretrained download for 3D tests cases
            cp_input_param["n_input_channels"] = 1
            cp_input_param["feed_forward"] = False
            cp_input_param["shortcut_type"] = shortcut_type
            cp_input_param["bias_downsample"] = bias_downsample
            if cp_input_param.get("spatial_dims", 3) == 3:
                with skip_if_downloading_fails():
                    pretrained_net = model(**cp_input_param).to(device)
                    medicalnet_state_dict = get_pretrained_resnet_medicalnet(resnet_depth, device=device)
                    medicalnet_state_dict = {
                        key.replace("module.", ""): value for key, value in medicalnet_state_dict.items()
                    }
                    equal_state_dict(pretrained_net.state_dict(), medicalnet_state_dict)

    @parameterized.expand(TEST_SCRIPT_CASES)
    def test_script(self, model, input_param, input_shape, expected_shape):
        net = model(**input_param)
        test_data = torch.randn(input_shape)
        test_script_save(net, test_data)


@SkipIfNoModule("hf_hub_download")
class TestExtractFeatures(unittest.TestCase):
    @parameterized.expand(CASE_EXTRACT_FEATURES)
    def test_shape(self, input_param, input_shape, expected_shapes):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        with skip_if_downloading_fails():
            net = ResNetFeatures(**input_param).to(device)

        # run inference with random tensor
        with eval_mode(net):
            features = net(torch.randn(input_shape).to(device))

        # check output shape
        self.assertEqual(len(features), len(expected_shapes))
        for feature, expected_shape in zip(features, expected_shapes):
            self.assertEqual(feature.shape, torch.Size(expected_shape))


if __name__ == "__main__":
    unittest.main()
