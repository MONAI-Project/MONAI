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

import random
import unittest

import torch
from parameterized import parameterized

from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector, retinanet_resnet50_fpn_detector
from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape
from monai.networks import eval_mode, train_mode
from monai.utils import optional_import
from tests.util import SkipIfBeforePyTorchVersion, skip_if_quick, test_script_save

_, has_torchvision = optional_import("torchvision")

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
TEST_CASES = [TEST_CASE_1, TEST_CASE_2, TEST_CASE_2_A]

TEST_CASES_TS = [TEST_CASE_1]


class NaiveNetwork(torch.nn.Module):
    def __init__(self, spatial_dims, num_classes, **kwargs):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.num_classes = num_classes
        self.num_anchors = 1
        self.cls_key = "cls"
        self.box_reg_key = "box_reg"
        self.size_divisible = 1

    def forward(self, images):
        out_cls_shape = (images.shape[0], self.num_classes * self.num_anchors) + images.shape[-self.spatial_dims :]
        out_box_reg_shape = (images.shape[0], 2 * self.spatial_dims * self.num_anchors) + images.shape[
            -self.spatial_dims :
        ]
        return {self.cls_key: [torch.randn(out_cls_shape)], self.box_reg_key: [torch.randn(out_box_reg_shape)]}


@SkipIfBeforePyTorchVersion((1, 11))
@unittest.skipUnless(has_torchvision, "Requires torchvision")
@skip_if_quick
class TestRetinaNetDetector(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_retina_detector_resnet_backbone_shape(self, input_param, input_shape):
        returned_layers = [1]
        anchor_generator = AnchorGeneratorWithAnchorShape(
            feature_map_scales=(1, 2), base_anchor_shapes=((8,) * input_param["spatial_dims"],)
        )
        detector = retinanet_resnet50_fpn_detector(
            **input_param, anchor_generator=anchor_generator, returned_layers=returned_layers
        )

        with eval_mode(detector):
            input_data = torch.randn(input_shape)
            result = detector.forward(input_data)
            assert len(result) == len(result)

            input_data = [torch.randn(input_shape[1:]) for _ in range(random.randint(1, 9))]
            result = detector.forward(input_data)
            assert len(result) == len(result)

        detector.set_atss_matcher()
        detector.set_hard_negative_sampler(10, 0.5)
        for num_gt_box in [0, 3]:  # test for both empty and non-empty boxes
            gt_box_start = torch.randint(2, (num_gt_box, input_param["spatial_dims"])).to(torch.float16)
            gt_box_end = gt_box_start + torch.randint(1, 10, (num_gt_box, input_param["spatial_dims"]))
            one_target = {
                "boxes": torch.cat((gt_box_start, gt_box_end), dim=1),
                "labels": torch.randint(input_param["num_classes"], (num_gt_box,)),
            }
            with train_mode(detector):
                input_data = torch.randn(input_shape)
                targets = [one_target] * len(input_data)
                result = detector.forward(input_data, targets)

                input_data = [torch.randn(input_shape[1:]) for _ in range(random.randint(1, 9))]
                targets = [one_target] * len(input_data)
                result = detector.forward(input_data, targets)

    @parameterized.expand(TEST_CASES)
    def test_naive_retina_detector_shape(self, input_param, input_shape):
        anchor_generator = AnchorGeneratorWithAnchorShape(
            feature_map_scales=(1,), base_anchor_shapes=((8,) * input_param["spatial_dims"],)
        )
        detector = RetinaNetDetector(network=NaiveNetwork(**input_param), anchor_generator=anchor_generator)

        with eval_mode(detector):
            input_data = torch.randn(input_shape)
            result = detector.forward(input_data)
            assert len(result) == len(result)

            input_data = [torch.randn(input_shape[1:]) for _ in range(random.randint(1, 9))]
            result = detector.forward(input_data)
            assert len(result) == len(result)

        detector.set_atss_matcher()
        detector.set_hard_negative_sampler(10, 0.5)
        gt_box_start = torch.randint(2, (3, input_param["spatial_dims"])).to(torch.float16)
        gt_box_end = gt_box_start + torch.randint(1, 10, (3, input_param["spatial_dims"]))
        one_target = {
            "boxes": torch.cat((gt_box_start, gt_box_end), dim=1),
            "labels": torch.randint(input_param["num_classes"], (3,)),
        }
        with train_mode(detector):
            input_data = torch.randn(input_shape)
            targets = [one_target] * len(input_data)
            result = detector.forward(input_data, targets)

            input_data = [torch.randn(input_shape[1:]) for _ in range(random.randint(1, 9))]
            targets = [one_target] * len(input_data)
            result = detector.forward(input_data, targets)

    @parameterized.expand(TEST_CASES_TS)
    def test_script(self, input_param, input_shape):
        # test whether support torchscript
        returned_layers = [1]
        anchor_generator = AnchorGeneratorWithAnchorShape(
            feature_map_scales=(1, 2), base_anchor_shapes=((8,) * input_param["spatial_dims"],)
        )
        detector = retinanet_resnet50_fpn_detector(
            **input_param, anchor_generator=anchor_generator, returned_layers=returned_layers
        )
        with eval_mode(detector):
            input_data = torch.randn(input_shape)
            test_script_save(detector.network, input_data)


if __name__ == "__main__":
    unittest.main()
