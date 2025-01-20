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
from monai.networks.nets import MultiScalePatchDiscriminator, PatchDiscriminator
from tests.util import test_script_save

TEST_PATCHGAN = [
    [
        {
            "num_layers_d": 3,
            "spatial_dims": 2,
            "channels": 8,
            "in_channels": 3,
            "out_channels": 1,
            "kernel_size": 3,
            "activation": "LEAKYRELU",
            "norm": "instance",
            "bias": False,
            "dropout": 0.1,
        },
        torch.rand([1, 3, 256, 512]),
        (1, 8, 128, 256),
        (1, 1, 32, 64),
    ],
    [
        {
            "num_layers_d": 3,
            "spatial_dims": 3,
            "channels": 8,
            "in_channels": 3,
            "out_channels": 1,
            "kernel_size": 3,
            "activation": "LEAKYRELU",
            "norm": "instance",
            "bias": False,
            "dropout": 0.1,
        },
        torch.rand([1, 3, 256, 512, 256]),
        (1, 8, 128, 256, 128),
        (1, 1, 32, 64, 32),
    ],
]

TEST_MULTISCALE_PATCHGAN = [
    [
        {
            "num_d": 2,
            "num_layers_d": 3,
            "spatial_dims": 2,
            "channels": 8,
            "in_channels": 3,
            "out_channels": 1,
            "kernel_size": 3,
            "activation": "LEAKYRELU",
            "norm": "instance",
            "bias": False,
            "dropout": 0.1,
            "minimum_size_im": 256,
        },
        torch.rand([1, 3, 256, 512]),
        [(1, 1, 32, 64), (1, 1, 4, 8)],
        [4, 7],
    ],
    [
        {
            "num_d": 2,
            "num_layers_d": 3,
            "spatial_dims": 3,
            "channels": 8,
            "in_channels": 3,
            "out_channels": 1,
            "kernel_size": 3,
            "activation": "LEAKYRELU",
            "norm": "instance",
            "bias": False,
            "dropout": 0.1,
            "minimum_size_im": 256,
        },
        torch.rand([1, 3, 256, 512, 256]),
        [(1, 1, 32, 64, 32), (1, 1, 4, 8, 4)],
        [4, 7],
    ],
]
TEST_TOO_SMALL_SIZE = [
    {
        "num_d": 2,
        "num_layers_d": 6,
        "spatial_dims": 2,
        "channels": 8,
        "in_channels": 3,
        "out_channels": 1,
        "kernel_size": 3,
        "activation": "LEAKYRELU",
        "norm": "instance",
        "bias": False,
        "dropout": 0.1,
        "minimum_size_im": 256,
    }
]


class TestPatchGAN(unittest.TestCase):
    @parameterized.expand(TEST_PATCHGAN)
    def test_shape(self, input_param, input_data, expected_shape_feature, expected_shape_output):
        net = PatchDiscriminator(**input_param)
        with eval_mode(net):
            result = net.forward(input_data)
            self.assertEqual(tuple(result[0].shape), expected_shape_feature)
            self.assertEqual(tuple(result[-1].shape), expected_shape_output)

    def test_script(self):
        net = PatchDiscriminator(
            num_layers_d=3,
            spatial_dims=2,
            channels=8,
            in_channels=3,
            out_channels=1,
            kernel_size=3,
            activation="LEAKYRELU",
            norm="instance",
            bias=False,
            dropout=0.1,
        )
        i = torch.rand([1, 3, 256, 512])
        test_script_save(net, i)


class TestMultiscalePatchGAN(unittest.TestCase):
    @parameterized.expand(TEST_MULTISCALE_PATCHGAN)
    def test_shape(self, input_param, input_data, expected_shape, features_lengths=None):
        net = MultiScalePatchDiscriminator(**input_param)
        with eval_mode(net):
            result, features = net.forward(input_data)
            for r_ind, r in enumerate(result):
                self.assertEqual(tuple(r.shape), expected_shape[r_ind])
            for o_d_ind, o_d in enumerate(features):
                self.assertEqual(len(o_d), features_lengths[o_d_ind])

    def test_too_small_shape(self):
        with self.assertRaises(AssertionError):
            MultiScalePatchDiscriminator(**TEST_TOO_SMALL_SIZE[0])

    def test_script(self):
        net = MultiScalePatchDiscriminator(
            num_d=2,
            num_layers_d=3,
            spatial_dims=2,
            channels=8,
            in_channels=3,
            out_channels=1,
            kernel_size=3,
            activation="LEAKYRELU",
            norm="instance",
            bias=False,
            dropout=0.1,
            minimum_size_im=256,
        )
        i = torch.rand([1, 3, 256, 512])
        test_script_save(net, i)


if __name__ == "__main__":
    unittest.main()
