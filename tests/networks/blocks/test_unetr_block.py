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
from monai.networks.blocks.dynunet_block import get_padding
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from tests.util import test_script_save

TEST_CASE_UNETR_BASIC_BLOCK = []
for spatial_dims in range(1, 4):
    for kernel_size in [1, 3]:
        for stride in [2]:
            for norm_name in [("GROUP", {"num_groups": 16}), ("batch", {"track_running_stats": False}), "instance"]:
                for in_size in [15, 16]:
                    padding = get_padding(kernel_size, stride)
                    if not isinstance(padding, int):
                        padding = padding[0]
                    out_size = int((in_size + 2 * padding - kernel_size) / stride) + 1
                    test_case = [
                        {
                            "spatial_dims": spatial_dims,
                            "in_channels": 16,
                            "out_channels": 16,
                            "kernel_size": kernel_size,
                            "norm_name": norm_name,
                            "stride": stride,
                        },
                        (1, 16, *([in_size] * spatial_dims)),
                        (1, 16, *([out_size] * spatial_dims)),
                    ]
                    TEST_CASE_UNETR_BASIC_BLOCK.append(test_case)

TEST_UP_BLOCK = []
in_channels, out_channels = 4, 2
for spatial_dims in range(1, 4):
    for kernel_size in [1, 3]:
        for res_block in [False, True]:
            for norm_name in ["instance"]:
                for in_size in [15, 16]:
                    out_size = in_size * stride
                    test_case = [
                        {
                            "spatial_dims": spatial_dims,
                            "in_channels": in_channels,
                            "out_channels": out_channels,
                            "kernel_size": kernel_size,
                            "norm_name": norm_name,
                            "res_block": res_block,
                            "upsample_kernel_size": stride,
                        },
                        (1, in_channels, *([in_size] * spatial_dims)),
                        (1, out_channels, *([out_size] * spatial_dims)),
                        (1, out_channels, *([in_size * stride] * spatial_dims)),
                    ]
                    TEST_UP_BLOCK.append(test_case)

TEST_PRUP_BLOCK = []
in_channels, out_channels = 4, 2
for spatial_dims in range(1, 4):
    for kernel_size in [1, 3]:
        for upsample_kernel_size in [2, 3]:
            for stride in [1, 2]:
                for res_block in [False, True]:
                    for norm_name in ["instance"]:
                        for in_size in [15, 16]:
                            for num_layer in [0, 2]:
                                in_size_tmp = in_size
                                for _ in range(num_layer + 1):
                                    out_size = in_size_tmp * upsample_kernel_size
                                    in_size_tmp = out_size
                            test_case = [
                                {
                                    "spatial_dims": spatial_dims,
                                    "in_channels": in_channels,
                                    "out_channels": out_channels,
                                    "num_layer": num_layer,
                                    "kernel_size": kernel_size,
                                    "norm_name": norm_name,
                                    "stride": stride,
                                    "res_block": res_block,
                                    "upsample_kernel_size": upsample_kernel_size,
                                },
                                (1, in_channels, *([in_size] * spatial_dims)),
                                (1, out_channels, *([out_size] * spatial_dims)),
                            ]
                            TEST_PRUP_BLOCK.append(test_case)


class TestResBasicBlock(unittest.TestCase):
    @parameterized.expand(TEST_CASE_UNETR_BASIC_BLOCK)
    def test_shape(self, input_param, input_shape, expected_shape):
        for net in [UnetrBasicBlock(**input_param)]:
            with eval_mode(net):
                result = net(torch.randn(input_shape))
                self.assertEqual(result.shape, expected_shape)

    def test_ill_arg(self):
        with self.assertRaises(ValueError):
            UnetrBasicBlock(3, 4, 2, kernel_size=3, stride=1, norm_name="norm")
        with self.assertRaises(AssertionError):
            UnetrBasicBlock(3, 4, 2, kernel_size=1, stride=4, norm_name="batch")

    def test_script(self):
        input_param, input_shape, _ = TEST_CASE_UNETR_BASIC_BLOCK[0]
        net = UnetrBasicBlock(**input_param)
        with eval_mode(net):
            test_data = torch.randn(input_shape)
            test_script_save(net, test_data)


class TestUpBlock(unittest.TestCase):
    @parameterized.expand(TEST_UP_BLOCK)
    def test_shape(self, input_param, input_shape, expected_shape, skip_shape):
        net = UnetrUpBlock(**input_param)
        with eval_mode(net):
            result = net(torch.randn(input_shape), torch.randn(skip_shape))
            self.assertEqual(result.shape, expected_shape)

    def test_script(self):
        input_param, input_shape, _, skip_shape = TEST_UP_BLOCK[0]
        net = UnetrUpBlock(**input_param)
        test_data = torch.randn(input_shape)
        skip_data = torch.randn(skip_shape)
        test_script_save(net, test_data, skip_data)


class TestPrUpBlock(unittest.TestCase):
    @parameterized.expand(TEST_PRUP_BLOCK)
    def test_shape(self, input_param, input_shape, expected_shape):
        net = UnetrPrUpBlock(**input_param)
        with eval_mode(net):
            result = net(torch.randn(input_shape))
            self.assertEqual(result.shape, expected_shape)

    def test_script(self):
        input_param, input_shape, _ = TEST_PRUP_BLOCK[0]
        net = UnetrPrUpBlock(**input_param)
        test_data = torch.randn(input_shape)
        test_script_save(net, test_data)


if __name__ == "__main__":
    unittest.main()
