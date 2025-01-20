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

import numpy as np
import torch
from parameterized import parameterized

from monai.networks import eval_mode
from monai.networks.blocks import Warp
from monai.networks.nets import GlobalNet
from monai.networks.nets.regunet import AffineHead
from tests.util import assert_allclose, test_script_save

TEST_CASES_AFFINE_TRANSFORM = [
    [
        {"spatial_dims": 3, "image_size": (2, 2, 2), "decode_size": (2, 2, 2), "in_channels": 1, "save_theta": True},
        torch.ones(2, 12),
        torch.tensor([[[1, 2], [2, 3]], [[2, 3], [3, 4]]]).unsqueeze(0).unsqueeze(0).expand(2, 3, 2, 2, 2),
    ],
    [
        {"spatial_dims": 3, "image_size": (2, 2, 2), "decode_size": (2, 2, 2), "in_channels": 1},
        torch.arange(1, 13).reshape(1, 12).to(torch.float),
        torch.tensor(
            [
                [[[4.0, 7.0], [6.0, 9.0]], [[5.0, 8.0], [7.0, 10.0]]],
                [[[8.0, 15.0], [14.0, 21.0]], [[13.0, 20.0], [19.0, 26.0]]],
                [[[12.0, 23.0], [22.0, 33.0]], [[21.0, 32.0], [31.0, 42.0]]],
            ]
        ).unsqueeze(0),
    ],
]

TEST_CASES_GLOBAL_NET = [
    [
        {
            "image_size": (16, 16),
            "spatial_dims": 2,
            "in_channels": 1,
            "num_channel_initial": 16,
            "depth": 1,
            "out_kernel_initializer": "kaiming_uniform",
            "out_activation": None,
            "pooling": True,
            "concat_skip": True,
            "encode_kernel_sizes": 3,
            "save_theta": theta,
        },
        (1, 1, 16, 16),
        (1, 2, 16, 16),
    ]
    for theta in (False, True)
]


class TestAffineHead(unittest.TestCase):
    @parameterized.expand(TEST_CASES_AFFINE_TRANSFORM)
    def test_shape(self, input_param, theta, expected_val):
        layer = AffineHead(**input_param)
        if input_param.get("save_theta"):
            assert_allclose(layer.theta, torch.Tensor())
        result = layer.affine_transform(theta)
        np.testing.assert_allclose(result.cpu().numpy(), expected_val.cpu().numpy(), rtol=1e-4, atol=1e-4)


device = "cuda" if torch.cuda.is_available() else "cpu"


class TestGlobalNet(unittest.TestCase):
    @parameterized.expand(TEST_CASES_GLOBAL_NET)
    def test_shape(self, input_param, input_shape, expected_shape):
        net = GlobalNet(**input_param).to(device)
        warp_layer = Warp()
        with eval_mode(net):
            img = torch.randn(input_shape)
            result = net(img.to(device))
            if input_param.get("save_theta"):
                assert_allclose(net.output_block.theta, torch.Tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]))
            warped = warp_layer(img.to(device), result)
            self.assertEqual(result.shape, expected_shape)
            # testing initial pred identity
            np.testing.assert_allclose(warped.detach().cpu().numpy(), img.detach().cpu().numpy(), rtol=1e-4, atol=1e-4)

    @parameterized.expand(TEST_CASES_GLOBAL_NET)
    def test_script(self, input_param, input_shape, _):
        net = GlobalNet(**input_param)
        test_data = torch.randn(input_shape)
        test_script_save(net, test_data)


if __name__ == "__main__":
    unittest.main()
