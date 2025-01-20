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
from monai.networks.blocks.regunet_block import (
    RegistrationDownSampleBlock,
    RegistrationExtractionBlock,
    RegistrationResidualConvBlock,
)

TEST_CASE_RESIDUAL = [
    [{"spatial_dims": 2, "in_channels": 1, "out_channels": 2, "num_layers": 1}, (1, 1, 5, 5), (1, 2, 5, 5)],
    [{"spatial_dims": 3, "in_channels": 2, "out_channels": 2, "num_layers": 2}, (1, 2, 5, 5, 5), (1, 2, 5, 5, 5)],
]

TEST_CASE_DOWN_SAMPLE = [
    [{"spatial_dims": 2, "channels": 1, "pooling": False}, (1, 1, 4, 4), (1, 1, 2, 2)],
    [{"spatial_dims": 3, "channels": 2, "pooling": True}, (1, 2, 4, 4, 4), (1, 2, 2, 2, 2)],
]

TEST_CASE_EXTRACTION = [
    [
        {
            "spatial_dims": 2,
            "extract_levels": (0,),
            "num_channels": [1],
            "out_channels": 1,
            "kernel_initializer": "kaiming_uniform",
            "activation": None,
        },
        [(1, 1, 2, 2)],
        (3, 3),
        (1, 1, 3, 3),
    ],
    [
        {
            "spatial_dims": 3,
            "extract_levels": (1, 2),
            "num_channels": [1, 2, 3],
            "out_channels": 1,
            "kernel_initializer": "zeros",
            "activation": "sigmoid",
            "mode": "trilinear",
        },
        [(1, 3, 2, 2, 2), (1, 2, 4, 4, 4), (1, 1, 8, 8, 8)],
        (3, 3, 3),
        (1, 1, 3, 3, 3),
    ],
]


class TestRegistrationResidualConvBlock(unittest.TestCase):

    @parameterized.expand(TEST_CASE_RESIDUAL)
    def test_shape(self, input_param, input_shape, expected_shape):
        net = RegistrationResidualConvBlock(**input_param)
        with eval_mode(net):
            x = net(torch.randn(input_shape))
            self.assertEqual(x.shape, expected_shape)


class TestRegistrationDownSampleBlock(unittest.TestCase):

    @parameterized.expand(TEST_CASE_DOWN_SAMPLE)
    def test_shape(self, input_param, input_shape, expected_shape):
        net = RegistrationDownSampleBlock(**input_param)
        with eval_mode(net):
            x = net(torch.rand(input_shape))
            self.assertEqual(x.shape, expected_shape)

    def test_ill_shape(self):
        net = RegistrationDownSampleBlock(spatial_dims=2, channels=2, pooling=True)
        with self.assertRaises(ValueError):
            net(torch.rand((1, 2, 3, 3)))


class TestRegistrationExtractionBlock(unittest.TestCase):

    @parameterized.expand(TEST_CASE_EXTRACTION)
    def test_shape(self, input_param, input_shapes, image_size, expected_shape):
        net = RegistrationExtractionBlock(**input_param)
        with eval_mode(net):
            x = net([torch.rand(input_shape) for input_shape in input_shapes], image_size)
            self.assertEqual(x.shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
