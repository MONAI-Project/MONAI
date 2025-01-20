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

from monai.apps.detection.networks.retinanet_network import RetinaNet, resnet_fpn_feature_extractor
from monai.networks import eval_mode
from monai.networks.nets import resnet10, resnet18, resnet34, resnet50, resnet101, resnet152, resnet200
from monai.utils import ensure_tuple, optional_import
from tests.utils import SkipIfBeforePyTorchVersion, skip_if_quick, test_onnx_save, test_script_save

_, has_onnx = optional_import("onnx")
_, has_torchvision = optional_import("torchvision")

device = "cuda" if torch.cuda.is_available() else "cpu"
num_anchors = 7

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
]

TEST_CASE_3 = [  # 1D, batch 1, 2 input channels
    {
        "pretrained": False,
        "spatial_dims": 1,
        "n_input_channels": 2,
        "num_classes": 3,
        "conv1_t_size": [3],
        "conv1_t_stride": 1,
    },
    (1, 2, 32),
]

TEST_CASE_3_A = [  # 1D, batch 1, 2 input channels
    {"pretrained": False, "spatial_dims": 1, "n_input_channels": 2, "num_classes": 3, "shortcut_type": "A"},
    (1, 2, 32),
]

TEST_CASE_4 = [  # 2D, batch 2, 1 input channel
    {"pretrained": False, "spatial_dims": 2, "n_input_channels": 1, "num_classes": 3, "feed_forward": False},
    (2, 1, 32, 64),
]

TEST_CASES = []
for case in [TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_2_A, TEST_CASE_3_A]:
    for model in [resnet10, resnet18, resnet34, resnet50, resnet101, resnet152, resnet200]:
        TEST_CASES.append([model, *case])

TEST_CASES_TS = []
for case in [TEST_CASE_1]:
    for model in [resnet10, resnet18, resnet34, resnet50, resnet101, resnet152, resnet200]:
        TEST_CASES_TS.append([model, *case])


@SkipIfBeforePyTorchVersion((1, 12))
@unittest.skipUnless(has_torchvision, "Requires torchvision")
@skip_if_quick
class TestRetinaNet(unittest.TestCase):

    @parameterized.expand(TEST_CASES)
    def test_retina_shape(self, model, input_param, input_shape):
        backbone = model(**input_param)
        feature_extractor = resnet_fpn_feature_extractor(
            backbone=backbone,
            spatial_dims=input_param["spatial_dims"],
            pretrained_backbone=input_param["pretrained"],
            trainable_backbone_layers=None,
            returned_layers=[1, 2],
        )
        net = RetinaNet(
            spatial_dims=input_param["spatial_dims"],
            num_classes=input_param["num_classes"],
            num_anchors=num_anchors,
            feature_extractor=feature_extractor,
            size_divisible=32,
        ).to(device)

        with eval_mode(net):
            result = net.forward(torch.randn(input_shape).to(device))

            base_stride = ensure_tuple(input_param["conv1_t_stride"])[0] if "conv1_t_stride" in input_param else 1
            expected_cls_channel = input_param["num_classes"] * num_anchors
            expected_cls_shape = tuple(
                (input_shape[0], expected_cls_channel)
                + tuple(input_shape[2 + a] // s // base_stride for a in range(input_param["spatial_dims"]))
                for s in [2, 4, 8]
            )
            expected_box_channel = 2 * input_param["spatial_dims"] * num_anchors
            expected_box_shape = tuple(
                (input_shape[0], expected_box_channel)
                + tuple(input_shape[2 + a] // s // base_stride for a in range(input_param["spatial_dims"]))
                for s in [2, 4, 8]
            )

            self.assertEqual(tuple(cc.shape for cc in result[net.cls_key]), expected_cls_shape)
            self.assertEqual(tuple(cc.shape for cc in result[net.box_reg_key]), expected_box_shape)

    @parameterized.expand(TEST_CASES_TS)
    def test_script(self, model, input_param, input_shape):
        try:
            idx = int(self.id().split("test_script_")[-1])
        except BaseException:
            idx = 0
        idx %= 3
        # test whether support torchscript
        data = torch.randn(input_shape)
        backbone = model(**input_param)
        if idx == 0:
            test_script_save(backbone, data)
            return
        feature_extractor = resnet_fpn_feature_extractor(
            backbone=backbone,
            spatial_dims=input_param["spatial_dims"],
            pretrained_backbone=input_param["pretrained"],
            trainable_backbone_layers=None,
            returned_layers=[1, 2],
        )
        if idx == 1:
            test_script_save(feature_extractor, data)
            return
        net = RetinaNet(
            spatial_dims=input_param["spatial_dims"],
            num_classes=input_param["num_classes"],
            num_anchors=num_anchors,
            feature_extractor=feature_extractor,
            size_divisible=32,
        )
        if idx == 2:
            test_script_save(net, data)

    @parameterized.expand(TEST_CASES_TS)
    @unittest.skipUnless(has_onnx, "Requires onnx")
    def test_onnx(self, model, input_param, input_shape):
        try:
            idx = int(self.id().split("test_onnx_")[-1])
        except BaseException:
            idx = 0
        idx %= 3
        # test whether support torchscript
        data = torch.randn(input_shape)
        backbone = model(**input_param)
        if idx == 0:
            test_onnx_save(backbone, data, rtol=2e-2, atol=1e-5)
            return
        feature_extractor = resnet_fpn_feature_extractor(
            backbone=backbone,
            spatial_dims=input_param["spatial_dims"],
            pretrained_backbone=input_param["pretrained"],
            trainable_backbone_layers=None,
            returned_layers=[1, 2],
        )
        if idx == 1:
            test_onnx_save(feature_extractor, data, rtol=2e-2, atol=1e-5)
            return
        net = RetinaNet(
            spatial_dims=input_param["spatial_dims"],
            num_classes=input_param["num_classes"],
            num_anchors=num_anchors,
            feature_extractor=feature_extractor,
            size_divisible=32,
        )
        if idx == 2:
            test_onnx_save(net, data, rtol=2e-2, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
