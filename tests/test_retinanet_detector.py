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

import random
import unittest

import torch
from parameterized import parameterized

from monai.apps.detection.networks.retinanet_detector import retinanet_resnet50_fpn_detector
from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape
from monai.networks import eval_mode
from monai.utils import optional_import
from tests.utils import SkipIfBeforePyTorchVersion, test_script_save

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
TEST_CASES = [TEST_CASE_1, TEST_CASE_2, TEST_CASE_2_A]

TEST_CASES_TS = [TEST_CASE_1]


@SkipIfBeforePyTorchVersion((1, 9))
@unittest.skipUnless(has_torchvision, "Requires torchvision")
class TestRetinaNetDetector(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_retina_detector_shape(self, input_param, input_shape):
        returned_layers = [1]
        anchor_generator = AnchorGeneratorWithAnchorShape(
            feature_map_scales=(1, 2), base_anchor_shapes=((8,) * input_param["spatial_dims"],)
        )
        detector = retinanet_resnet50_fpn_detector(
            **input_param, anchor_generator=anchor_generator, returned_layers=returned_layers
        ).to(device)

        with eval_mode(detector):
            input_data = torch.randn(input_shape).to(device)
            result = detector.forward(input_data)
            assert len(result) == len(result)

            input_data = [torch.randn(input_shape[1:]).to(device) for _ in range(random.randint(1, 9))]
            result = detector.forward(input_data)
            assert len(result) == len(result)

    @parameterized.expand(TEST_CASES_TS)
    def test_script(self, input_param, input_shape):
        # test whether support torchscript
        returned_layers = [1]
        anchor_generator = AnchorGeneratorWithAnchorShape(
            feature_map_scales=(1, 2), base_anchor_shapes=((8,) * input_param["spatial_dims"],)
        )
        detector = retinanet_resnet50_fpn_detector(
            **input_param, anchor_generator=anchor_generator, returned_layers=returned_layers
        ).to(device)
        with eval_mode(detector):
            input_data = torch.randn(input_shape).to(device)
            test_script_save(detector.network, input_data)


if __name__ == "__main__":
    unittest.main()
