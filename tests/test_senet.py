# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import torch
from parameterized import parameterized

from monai.networks.nets import (
    senet154,
    se_resnet50,
    se_resnet101,
    se_resnet152,
    se_resnext50_32x4d,
    se_resnext101_32x4d,
)


TEST_CASE_1 = [  # batch size: 5, channels: 2
    {"spatial_dims": 3, "in_ch": 2, "num_classes": 10},
    torch.randn(5, 2, 64, 64, 64),
    (5, 3),
]


class TestSENET(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1])
    def test_senet154_shape(self, input_param, input_data, expected_shape):
        net = senet154(**input_param)
        net.eval()
        with torch.no_grad():
            result = net.forward(input_data)
            self.assertEqual(result.shape, expected_shape)

    @parameterized.expand([TEST_CASE_1])
    def test_se_resnet50_shape(self, input_param, input_data, expected_shape):
        net = se_resnet50(**input_param)
        net.eval()
        with torch.no_grad():
            result = net.forward(input_data)
            self.assertEqual(result.shape, expected_shape)

    @parameterized.expand([TEST_CASE_1])
    def test_se_resnet101_shape(self, input_param, input_data, expected_shape):
        net = se_resnet101(**input_param)
        net.eval()
        with torch.no_grad():
            result = net.forward(input_data)
            self.assertEqual(result.shape, expected_shape)

    @parameterized.expand([TEST_CASE_1])
    def test_se_resnet152_shape(self, input_param, input_data, expected_shape):
        net = se_resnet152(**input_param)
        net.eval()
        with torch.no_grad():
            result = net.forward(input_data)
            self.assertEqual(result.shape, expected_shape)

    @parameterized.expand([TEST_CASE_1])
    def test_se_resnext50_32x4d_shape(self, input_param, input_data, expected_shape):
        net = se_resnext50_32x4d(**input_param)
        net.eval()
        with torch.no_grad():
            result = net.forward(input_data)
            self.assertEqual(result.shape, expected_shape)

    @parameterized.expand([TEST_CASE_1])
    def test_se_resnext101_32x4d_shape(self, input_param, input_data, expected_shape):
        net = se_resnext101_32x4d(**input_param)
        net.eval()
        with torch.no_grad():
            result = net.forward(input_data)
            self.assertEqual(result.shape, expected_shape)
