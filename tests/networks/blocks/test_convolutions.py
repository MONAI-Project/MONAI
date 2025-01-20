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

from monai.networks.blocks import Convolution, ResidualUnit
from tests.util import TorchImageTestCase2D, TorchImageTestCase3D


class TestConvolution2D(TorchImageTestCase2D):
    def test_conv1(self):
        conv = Convolution(2, self.input_channels, self.output_channels)
        out = conv(self.imt)
        expected_shape = (1, self.output_channels, self.im_shape[0], self.im_shape[1])
        self.assertEqual(out.shape, expected_shape)

    def test_conv1_no_acti(self):
        conv = Convolution(2, self.input_channels, self.output_channels, act=None)
        out = conv(self.imt)
        expected_shape = (1, self.output_channels, self.im_shape[0], self.im_shape[1])
        self.assertEqual(out.shape, expected_shape)

    def test_conv_only1(self):
        conv = Convolution(2, self.input_channels, self.output_channels, conv_only=True)
        out = conv(self.imt)
        expected_shape = (1, self.output_channels, self.im_shape[0], self.im_shape[1])
        self.assertEqual(out.shape, expected_shape)

    def test_stride1(self):
        for strides in [2, [2, 2], (2, 2)]:
            conv = Convolution(2, self.input_channels, self.output_channels, strides=strides)
            out = conv(self.imt)
            expected_shape = (1, self.output_channels, self.im_shape[0] // 2, self.im_shape[1] // 2)
            self.assertEqual(out.shape, expected_shape)

    def test_dilation1(self):
        conv = Convolution(2, self.input_channels, self.output_channels, dilation=3)
        out = conv(self.imt)
        expected_shape = (1, self.output_channels, self.im_shape[0], self.im_shape[1])
        self.assertEqual(out.shape, expected_shape)

    def test_dropout1(self):
        conv = Convolution(2, self.input_channels, self.output_channels, dropout=0.15)
        out = conv(self.imt)
        expected_shape = (1, self.output_channels, self.im_shape[0], self.im_shape[1])
        self.assertEqual(out.shape, expected_shape)

    def test_transpose1(self):
        conv = Convolution(2, self.input_channels, self.output_channels, is_transposed=True)
        out = conv(self.imt)
        expected_shape = (1, self.output_channels, self.im_shape[0], self.im_shape[1])
        self.assertEqual(out.shape, expected_shape)

    def test_transpose2(self):
        conv = Convolution(2, self.input_channels, self.output_channels, strides=2, is_transposed=True)
        out = conv(self.imt)
        expected_shape = (1, self.output_channels, self.im_shape[0] * 2, self.im_shape[1] * 2)
        self.assertEqual(out.shape, expected_shape)


class TestConvolution3D(TorchImageTestCase3D):
    def test_conv1(self):
        conv = Convolution(3, self.input_channels, self.output_channels, dropout=0.1, adn_ordering="DAN")
        out = conv(self.imt)
        expected_shape = (1, self.output_channels, self.im_shape[0], self.im_shape[1], self.im_shape[2])
        self.assertEqual(out.shape, expected_shape)

    def test_conv1_no_acti(self):
        conv = Convolution(3, self.input_channels, self.output_channels, act=None, adn_ordering="AND")
        out = conv(self.imt)
        expected_shape = (1, self.output_channels, self.im_shape[0], self.im_shape[1], self.im_shape[2])
        self.assertEqual(out.shape, expected_shape)

    def test_conv_only1(self):
        conv = Convolution(3, self.input_channels, self.output_channels, conv_only=True)
        out = conv(self.imt)
        expected_shape = (1, self.output_channels, self.im_shape[0], self.im_shape[1], self.im_shape[2])
        self.assertEqual(out.shape, expected_shape)

    def test_stride1(self):
        for strides in [2, (2, 2, 2), [2, 2, 2]]:
            conv = Convolution(3, self.input_channels, self.output_channels, strides=strides)
            out = conv(self.imt)
            expected_shape = (
                1,
                self.output_channels,
                self.im_shape[0] // 2,
                self.im_shape[1] // 2,
                self.im_shape[2] // 2,
            )
            self.assertEqual(out.shape, expected_shape)

    def test_dilation1(self):
        conv = Convolution(3, self.input_channels, self.output_channels, dilation=3)
        out = conv(self.imt)
        expected_shape = (1, self.output_channels, self.im_shape[0], self.im_shape[1], self.im_shape[2])
        self.assertEqual(out.shape, expected_shape)

    def test_dropout1(self):
        conv = Convolution(3, self.input_channels, self.output_channels, dropout=0.15)
        out = conv(self.imt)
        expected_shape = (1, self.output_channels, self.im_shape[0], self.im_shape[1], self.im_shape[2])
        self.assertEqual(out.shape, expected_shape)

    def test_transpose1(self):
        conv = Convolution(3, self.input_channels, self.output_channels, is_transposed=True)
        out = conv(self.imt)
        expected_shape = (1, self.output_channels, self.im_shape[0], self.im_shape[1], self.im_shape[2])
        self.assertEqual(out.shape, expected_shape)

    def test_transpose2(self):
        conv = Convolution(3, self.input_channels, self.output_channels, strides=2, is_transposed=True)
        out = conv(self.imt)
        expected_shape = (1, self.output_channels, self.im_shape[0] * 2, self.im_shape[1] * 2, self.im_shape[2] * 2)
        self.assertEqual(out.shape, expected_shape)


class TestResidualUnit2D(TorchImageTestCase2D):
    def test_conv_only1(self):
        conv = ResidualUnit(2, 1, self.output_channels)
        out = conv(self.imt)
        expected_shape = (1, self.output_channels, self.im_shape[0], self.im_shape[1])
        self.assertEqual(out.shape, expected_shape)

    def test_stride1(self):
        for strides in [2, [2, 2], (2, 2)]:
            conv = ResidualUnit(2, 1, self.output_channels, strides=strides)
            out = conv(self.imt)
            expected_shape = (1, self.output_channels, self.im_shape[0] // 2, self.im_shape[1] // 2)
            self.assertEqual(out.shape, expected_shape)

    def test_dilation1(self):
        conv = ResidualUnit(2, 1, self.output_channels, dilation=3)
        out = conv(self.imt)
        expected_shape = (1, self.output_channels, self.im_shape[0], self.im_shape[1])
        self.assertEqual(out.shape, expected_shape)

    def test_dropout1(self):
        conv = ResidualUnit(2, 1, self.output_channels, dropout=0.15)
        out = conv(self.imt)
        expected_shape = (1, self.output_channels, self.im_shape[0], self.im_shape[1])
        self.assertEqual(out.shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
