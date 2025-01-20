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
from monai.networks.nets import MILModel
from monai.utils.module import optional_import
from tests.util import skip_if_downloading_fails, test_script_save

models, _ = optional_import("torchvision.models")

device = "cuda" if torch.cuda.is_available() else "cpu"

TEST_CASE_MILMODEL = []
for num_classes in [1, 5]:
    for mil_mode in ["mean", "max", "att", "att_trans", "att_trans_pyramid"]:
        test_case = [
            {"num_classes": num_classes, "mil_mode": mil_mode, "pretrained": False},
            (1, 2, 3, 512, 512),
            (1, num_classes),
        ]
        TEST_CASE_MILMODEL.append(test_case)

for trans_blocks in [1, 3]:
    test_case = [
        {"num_classes": 5, "pretrained": False, "trans_blocks": trans_blocks, "trans_dropout": 0.5},
        (1, 2, 3, 512, 512),
        (1, 5),
    ]
    TEST_CASE_MILMODEL.append(test_case)

# torchvision backbone
TEST_CASE_MILMODEL.append(
    [{"num_classes": 5, "backbone": "resnet18", "pretrained": False}, (2, 2, 3, 512, 512), (2, 5)]
)
TEST_CASE_MILMODEL.append([{"num_classes": 5, "backbone": "resnet18", "pretrained": True}, (2, 2, 3, 512, 512), (2, 5)])

# custom backbone
backbone = models.densenet121(pretrained=False)
backbone_nfeatures = backbone.classifier.in_features
backbone.classifier = torch.nn.Identity()
TEST_CASE_MILMODEL.append(
    [
        {"num_classes": 5, "backbone": backbone, "backbone_num_features": backbone_nfeatures, "pretrained": False},
        (2, 2, 3, 512, 512),
        (2, 5),
    ]
)


class TestMilModel(unittest.TestCase):
    @parameterized.expand(TEST_CASE_MILMODEL)
    def test_shape(self, input_param, input_shape, expected_shape):
        with skip_if_downloading_fails():
            net = MILModel(**input_param).to(device)
        with eval_mode(net):
            result = net(torch.randn(input_shape, dtype=torch.float).to(device))
            self.assertEqual(result.shape, expected_shape)

    def test_ill_args(self):
        with self.assertRaises(ValueError):
            MILModel(
                num_classes=5,
                pretrained=False,
                backbone="resnet50",
                backbone_num_features=2048,
                mil_mode="att_trans_pyramid",
            )

    def test_script(self):
        input_param, input_shape, expected_shape = TEST_CASE_MILMODEL[0]
        net = MILModel(**input_param)
        test_data = torch.randn(input_shape, dtype=torch.float)
        test_script_save(net, test_data)


if __name__ == "__main__":
    unittest.main()
