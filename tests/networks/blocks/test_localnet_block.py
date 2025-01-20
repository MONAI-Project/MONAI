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
from monai.networks.blocks.localnet_block import (
    LocalNetDownSampleBlock,
    LocalNetFeatureExtractorBlock,
    LocalNetUpSampleBlock,
)

TEST_CASE_DOWN_SAMPLE = [
    [{"spatial_dims": spatial_dims, "in_channels": 2, "out_channels": 4, "kernel_size": 3}] for spatial_dims in [2, 3]
]

TEST_CASE_UP_SAMPLE = [
    [
        {
            "spatial_dims": spatial_dims,
            "in_channels": 4,
            "out_channels": 2,
            "mode": "bilinear" if spatial_dims == 2 else "trilinear",
        }
    ]
    for spatial_dims in [2, 3]
]

TEST_CASE_EXTRACT = [
    [{"spatial_dims": spatial_dims, "in_channels": 2, "out_channels": 3, "act": act, "initializer": initializer}]
    for spatial_dims, act, initializer in zip([2, 3], ["sigmoid", None], ["kaiming_uniform", "zeros"])
]

in_size = 4


class TestLocalNetDownSampleBlock(unittest.TestCase):

    @parameterized.expand(TEST_CASE_DOWN_SAMPLE)
    def test_shape(self, input_param):
        net = LocalNetDownSampleBlock(**input_param)
        input_shape = (1, input_param["in_channels"], *([in_size] * input_param["spatial_dims"]))
        expect_mid_shape = (1, input_param["out_channels"], *([in_size] * input_param["spatial_dims"]))
        expect_x_shape = (1, input_param["out_channels"], *([in_size / 2] * input_param["spatial_dims"]))
        with eval_mode(net):
            x, mid = net(torch.randn(input_shape))
            self.assertEqual(x.shape, expect_x_shape)
            self.assertEqual(mid.shape, expect_mid_shape)

    def test_ill_arg(self):
        # even kernel_size
        with self.assertRaises(NotImplementedError):
            LocalNetDownSampleBlock(spatial_dims=2, in_channels=2, out_channels=4, kernel_size=4)

    @parameterized.expand(TEST_CASE_DOWN_SAMPLE)
    def test_ill_shape(self, input_param):
        net = LocalNetDownSampleBlock(**input_param)
        input_shape = (1, input_param["in_channels"], *([5] * input_param["spatial_dims"]))
        with self.assertRaises(ValueError):
            with eval_mode(net):
                net(torch.randn(input_shape))


class TestLocalNetUpSampleBlock(unittest.TestCase):

    @parameterized.expand(TEST_CASE_UP_SAMPLE)
    def test_shape(self, input_param):
        net = LocalNetUpSampleBlock(**input_param)
        input_shape = (1, input_param["in_channels"], *([in_size] * input_param["spatial_dims"]))
        mid_shape = (1, input_param["out_channels"], *([in_size * 2] * input_param["spatial_dims"]))
        expected_shape = mid_shape
        with eval_mode(net):
            result = net(torch.randn(input_shape), torch.randn(mid_shape))
            self.assertEqual(result.shape, expected_shape)

    def test_ill_arg(self):
        # channel unmatch
        with self.assertRaises(ValueError):
            LocalNetUpSampleBlock(spatial_dims=2, in_channels=2, out_channels=2)

    @parameterized.expand(TEST_CASE_UP_SAMPLE)
    def test_ill_shape(self, input_param):
        net = LocalNetUpSampleBlock(**input_param)
        input_shape = (1, input_param["in_channels"], *([in_size] * input_param["spatial_dims"]))
        mid_shape = (1, input_param["out_channels"], *([in_size] * input_param["spatial_dims"]))
        with self.assertRaises(ValueError):
            with eval_mode(net):
                net(torch.randn(input_shape), torch.randn(mid_shape))


class TestExtractBlock(unittest.TestCase):

    @parameterized.expand(TEST_CASE_EXTRACT)
    def test_shape(self, input_param):
        net = LocalNetFeatureExtractorBlock(**input_param)
        input_shape = (1, input_param["in_channels"], *([in_size] * input_param["spatial_dims"]))
        expected_shape = (1, input_param["out_channels"], *([in_size] * input_param["spatial_dims"]))
        with eval_mode(net):
            result = net(torch.randn(input_shape))
            self.assertEqual(result.shape, expected_shape)

    def test_ill_arg(self):
        with self.assertRaises(ValueError):
            LocalNetFeatureExtractorBlock(spatial_dims=2, in_channels=2, out_channels=2, initializer="none")


if __name__ == "__main__":
    unittest.main()
