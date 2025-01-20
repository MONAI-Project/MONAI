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

import torch.nn as nn

from monai.networks.blocks import ConvDenseBlock, DenseBlock
from tests.util import TorchImageTestCase2D, TorchImageTestCase3D


class TestDenseBlock2D(TorchImageTestCase2D):
    def test_block_empty(self):
        block = DenseBlock([])
        out = block(self.imt)
        expected_shape = self.imt.shape
        self.assertEqual(out.shape, expected_shape)

    def test_block_conv(self):
        conv1 = nn.Conv2d(self.input_channels, self.output_channels, 3, padding=1)
        conv2 = nn.Conv2d(self.input_channels + self.output_channels, self.input_channels, 3, padding=1)
        block = DenseBlock([conv1, conv2])
        out = block(self.imt)
        expected_shape = (1, self.output_channels + self.input_channels * 2, self.im_shape[0], self.im_shape[1])
        self.assertEqual(out.shape, expected_shape)


class TestDenseBlock3D(TorchImageTestCase3D):
    def test_block_conv(self):
        conv1 = nn.Conv3d(self.input_channels, self.output_channels, 3, padding=1)
        conv2 = nn.Conv3d(self.input_channels + self.output_channels, self.input_channels, 3, padding=1)
        block = DenseBlock([conv1, conv2])
        out = block(self.imt)
        expected_shape = (
            1,
            self.output_channels + self.input_channels * 2,
            self.im_shape[0],
            self.im_shape[1],
            self.im_shape[2],
        )
        self.assertEqual(out.shape, expected_shape)


class TestConvDenseBlock2D(TorchImageTestCase2D):
    def test_block_empty(self):
        conv = ConvDenseBlock(spatial_dims=2, in_channels=self.input_channels, channels=[])
        out = conv(self.imt)
        expected_shape = self.imt.shape
        self.assertEqual(out.shape, expected_shape)

    def test_except(self):
        with self.assertRaises(ValueError):
            _ = ConvDenseBlock(spatial_dims=2, in_channels=self.input_channels, channels=[1, 2], dilations=[1, 2, 3])

    def test_block1(self):
        channels = [2, 4]
        conv = ConvDenseBlock(spatial_dims=2, in_channels=self.input_channels, channels=channels)
        out = conv(self.imt)
        expected_shape = (1, self.input_channels + sum(channels), self.im_shape[0], self.im_shape[1])
        self.assertEqual(out.shape, expected_shape)

    def test_block2(self):
        channels = [2, 4]
        dilations = [1, 2]
        conv = ConvDenseBlock(spatial_dims=2, in_channels=self.input_channels, channels=channels, dilations=dilations)
        out = conv(self.imt)
        expected_shape = (1, self.input_channels + sum(channels), self.im_shape[0], self.im_shape[1])
        self.assertEqual(out.shape, expected_shape)


class TestConvDenseBlock3D(TorchImageTestCase3D):
    def test_block_empty(self):
        conv = ConvDenseBlock(spatial_dims=3, in_channels=self.input_channels, channels=[])
        out = conv(self.imt)
        expected_shape = self.imt.shape
        self.assertEqual(out.shape, expected_shape)

    def test_block1(self):
        channels = [2, 4]
        conv = ConvDenseBlock(spatial_dims=3, in_channels=self.input_channels, channels=channels)
        out = conv(self.imt)
        expected_shape = (1, self.input_channels + sum(channels), self.im_shape[0], self.im_shape[1], self.im_shape[2])
        self.assertEqual(out.shape, expected_shape)

    def test_block2(self):
        channels = [2, 4]
        dilations = [1, 2]
        conv = ConvDenseBlock(spatial_dims=3, in_channels=self.input_channels, channels=channels, dilations=dilations)
        out = conv(self.imt)
        expected_shape = (1, self.input_channels + sum(channels), self.im_shape[0], self.im_shape[1], self.im_shape[2])
        self.assertEqual(out.shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
