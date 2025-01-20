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

from monai.losses import PerceptualLoss
from monai.utils import optional_import
from tests.util import SkipIfBeforePyTorchVersion, assert_allclose, skip_if_downloading_fails, skip_if_quick

_, has_torchvision = optional_import("torchvision")
TEST_CASES = [
    [{"spatial_dims": 2, "network_type": "squeeze"}, (2, 1, 64, 64), (2, 1, 64, 64)],
    [
        {"spatial_dims": 3, "network_type": "squeeze", "is_fake_3d": True, "fake_3d_ratio": 0.1},
        (2, 1, 64, 64, 64),
        (2, 1, 64, 64, 64),
    ],
    [{"spatial_dims": 2, "network_type": "radimagenet_resnet50"}, (2, 1, 64, 64), (2, 1, 64, 64)],
    [{"spatial_dims": 2, "network_type": "radimagenet_resnet50"}, (2, 3, 64, 64), (2, 3, 64, 64)],
    [
        {"spatial_dims": 3, "network_type": "radimagenet_resnet50", "is_fake_3d": True, "fake_3d_ratio": 0.1},
        (2, 1, 64, 64, 64),
        (2, 1, 64, 64, 64),
    ],
    [
        {"spatial_dims": 3, "network_type": "medicalnet_resnet10_23datasets", "is_fake_3d": False},
        (2, 1, 64, 64, 64),
        (2, 1, 64, 64, 64),
    ],
    [
        {"spatial_dims": 3, "network_type": "medicalnet_resnet10_23datasets", "is_fake_3d": False},
        (2, 6, 64, 64, 64),
        (2, 6, 64, 64, 64),
    ],
    [
        {
            "spatial_dims": 3,
            "network_type": "medicalnet_resnet10_23datasets",
            "is_fake_3d": False,
            "channel_wise": True,
        },
        (2, 6, 64, 64, 64),
        (2, 6, 64, 64, 64),
    ],
    [
        {"spatial_dims": 3, "network_type": "medicalnet_resnet50_23datasets", "is_fake_3d": False},
        (2, 1, 64, 64, 64),
        (2, 1, 64, 64, 64),
    ],
    [
        {"spatial_dims": 3, "network_type": "medicalnet_resnet50_23datasets", "is_fake_3d": False},
        (2, 6, 64, 64, 64),
        (2, 6, 64, 64, 64),
    ],
    [
        {"spatial_dims": 3, "network_type": "resnet50", "is_fake_3d": True, "pretrained": True, "fake_3d_ratio": 0.2},
        (2, 1, 64, 64, 64),
        (2, 1, 64, 64, 64),
    ],
]


@SkipIfBeforePyTorchVersion((1, 11))
@unittest.skipUnless(has_torchvision, "Requires torchvision")
@skip_if_quick
class TestPerceptualLoss(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_shape(self, input_param, input_shape, target_shape):
        with skip_if_downloading_fails():
            loss = PerceptualLoss(**input_param)
        result = loss(torch.randn(input_shape), torch.randn(target_shape))

        if "channel_wise" in input_param.keys() and input_param["channel_wise"]:
            self.assertEqual(result.shape, torch.Size([input_shape[1]]))
        else:
            self.assertEqual(result.shape, torch.Size([]))

    @parameterized.expand(TEST_CASES)
    def test_identical_input(self, input_param, input_shape, target_shape):
        with skip_if_downloading_fails():
            loss = PerceptualLoss(**input_param)
        tensor = torch.randn(input_shape)
        result = loss(tensor, tensor)

        if "channel_wise" in input_param.keys() and input_param["channel_wise"]:
            assert_allclose(result, torch.Tensor([0.0] * input_shape[1]))
        else:
            self.assertEqual(result, torch.Tensor([0.0]))

    def test_different_shape(self):
        with skip_if_downloading_fails():
            loss = PerceptualLoss(spatial_dims=2, network_type="squeeze")
        tensor = torch.randn(2, 1, 64, 64)
        target = torch.randn(2, 1, 32, 32)
        with self.assertRaises(ValueError):
            loss(tensor, target)

    def test_1d(self):
        with self.assertRaises(NotImplementedError):
            PerceptualLoss(spatial_dims=1)

    @parameterized.expand(["medicalnet_resnet10_23datasets", "medicalnet_resnet50_23datasets"])
    def test_medicalnet_on_2d_data(self, network_type):
        with self.assertRaises(ValueError):
            PerceptualLoss(spatial_dims=2, network_type=network_type)


if __name__ == "__main__":
    unittest.main()
