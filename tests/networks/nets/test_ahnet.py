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
from monai.networks.blocks import FCN, MCFCN
from monai.networks.nets import AHNet
from tests.util import skip_if_quick, test_pretrained_networks, test_script_save

device = "cuda" if torch.cuda.is_available() else "cpu"

TEST_CASE_FCN_1 = [  # batch 2
    {"out_channels": 3, "upsample_mode": "transpose", "pretrained": False},
    (2, 3, 32, 32),
    (2, 3, 32, 32),
]
TEST_CASE_FCN_2 = [
    {"out_channels": 2, "upsample_mode": "transpose", "pretrained": False, "progress": False},
    (1, 3, 32, 32),
    (1, 2, 32, 32),
]
TEST_CASE_FCN_3 = [
    {"out_channels": 1, "upsample_mode": "bilinear", "pretrained": False},
    (1, 3, 32, 32),
    (1, 1, 32, 32),
]
TEST_CASE_FCN_WITH_PRETRAIN_1 = [  # batch 2
    {"out_channels": 3, "upsample_mode": "transpose", "pretrained": True},
    (2, 3, 32, 32),
    (2, 3, 32, 32),
]
TEST_CASE_FCN_WITH_PRETRAIN_2 = [
    {"out_channels": 2, "upsample_mode": "transpose", "pretrained": True, "progress": False},
    (1, 3, 32, 32),
    (1, 2, 32, 32),
]

TEST_CASE_MCFCN_1 = [  # batch 5
    {"out_channels": 3, "in_channels": 8, "upsample_mode": "transpose", "pretrained": False, "progress": False},
    (5, 8, 32, 32),
    (5, 3, 32, 32),
]
TEST_CASE_MCFCN_2 = [
    {"out_channels": 2, "in_channels": 1, "upsample_mode": "transpose", "pretrained": False, "progress": True},
    (1, 1, 32, 32),
    (1, 2, 32, 32),
]
TEST_CASE_MCFCN_3 = [
    {"out_channels": 1, "in_channels": 2, "upsample_mode": "bilinear", "pretrained": False},
    (1, 2, 32, 32),
    (1, 1, 32, 32),
]
TEST_CASE_MCFCN_WITH_PRETRAIN_1 = [  # batch 5
    {"out_channels": 3, "in_channels": 8, "upsample_mode": "transpose", "pretrained": True, "progress": False},
    (5, 8, 32, 32),
    (5, 3, 32, 32),
]
TEST_CASE_MCFCN_WITH_PRETRAIN_2 = [
    {"out_channels": 2, "in_channels": 1, "upsample_mode": "transpose", "pretrained": True, "progress": True},
    (1, 1, 32, 32),
    (1, 2, 32, 32),
]

TEST_CASE_AHNET_2D_1 = [
    {"spatial_dims": 2, "upsample_mode": "bilinear", "psp_block_num": 0},
    (1, 1, 32, 64),
    (1, 1, 32, 64),
]
TEST_CASE_AHNET_2D_2 = [
    {"spatial_dims": 2, "upsample_mode": "transpose", "out_channels": 2, "psp_block_num": 1},
    (1, 1, 64, 32),
    (1, 2, 64, 32),
]
TEST_CASE_AHNET_2D_3 = [
    {"spatial_dims": 2, "upsample_mode": "bilinear", "out_channels": 2, "psp_block_num": 2},
    (1, 1, 64, 32),
    (1, 2, 64, 32),
]
TEST_CASE_AHNET_3D_1 = [
    {"spatial_dims": 3, "upsample_mode": "trilinear", "psp_block_num": 0},
    (2, 1, 32, 32, 64),
    (2, 1, 32, 32, 64),
]
TEST_CASE_AHNET_3D_2 = [
    {"spatial_dims": 3, "upsample_mode": "transpose", "out_channels": 2, "psp_block_num": 1},
    (1, 1, 32, 32, 64),
    (1, 2, 32, 32, 64),
]
TEST_CASE_AHNET_3D_3 = [
    {"spatial_dims": 3, "upsample_mode": "nearest", "out_channels": 2, "psp_block_num": 3},
    (1, 1, 96, 128, 32),
    (1, 2, 96, 128, 32),
]
TEST_CASE_AHNET_3D_WITH_PRETRAIN_1 = [
    {"spatial_dims": 3, "upsample_mode": "trilinear", "psp_block_num": 0},
    (2, 1, 32, 32, 64),
    (2, 1, 32, 32, 64),
    {"out_channels": 1, "upsample_mode": "transpose"},
]
TEST_CASE_AHNET_3D_WITH_PRETRAIN_2 = [
    {"spatial_dims": 3, "upsample_mode": "transpose", "out_channels": 2, "psp_block_num": 3},
    (1, 1, 64, 64, 64),
    (1, 2, 64, 64, 64),
    {"out_channels": 1, "upsample_mode": "bilinear"},
]
TEST_CASE_AHNET_3D_WITH_PRETRAIN_3 = [
    {"spatial_dims": 3, "upsample_mode": "transpose", "in_channels": 2, "out_channels": 3},
    (1, 2, 128, 128, 32),
    (1, 3, 128, 128, 32),
    {"out_channels": 1, "upsample_mode": "bilinear"},
]


class TestFCN(unittest.TestCase):
    @parameterized.expand([TEST_CASE_FCN_1, TEST_CASE_FCN_2, TEST_CASE_FCN_3])
    @skip_if_quick
    def test_fcn_shape(self, input_param, input_shape, expected_shape):
        net = FCN(**input_param).to(device)
        with eval_mode(net):
            result = net.forward(torch.randn(input_shape).to(device))
            self.assertEqual(result.shape, expected_shape)


class TestFCNWithPretrain(unittest.TestCase):
    @parameterized.expand([TEST_CASE_FCN_WITH_PRETRAIN_1, TEST_CASE_FCN_WITH_PRETRAIN_2])
    @skip_if_quick
    def test_fcn_shape(self, input_param, input_shape, expected_shape):
        net = test_pretrained_networks(FCN, input_param, device)
        with eval_mode(net):
            result = net.forward(torch.randn(input_shape).to(device))
            self.assertEqual(result.shape, expected_shape)


class TestMCFCN(unittest.TestCase):
    @parameterized.expand([TEST_CASE_MCFCN_1, TEST_CASE_MCFCN_2, TEST_CASE_MCFCN_3])
    def test_mcfcn_shape(self, input_param, input_shape, expected_shape):
        net = MCFCN(**input_param).to(device)
        with eval_mode(net):
            result = net.forward(torch.randn(input_shape).to(device))
            self.assertEqual(result.shape, expected_shape)


class TestMCFCNWithPretrain(unittest.TestCase):
    @parameterized.expand([TEST_CASE_MCFCN_WITH_PRETRAIN_1, TEST_CASE_MCFCN_WITH_PRETRAIN_2])
    def test_mcfcn_shape(self, input_param, input_shape, expected_shape):
        net = test_pretrained_networks(MCFCN, input_param, device)
        with eval_mode(net):
            result = net.forward(torch.randn(input_shape).to(device))
            self.assertEqual(result.shape, expected_shape)


class TestAHNET(unittest.TestCase):
    @parameterized.expand([TEST_CASE_AHNET_2D_1, TEST_CASE_AHNET_2D_2, TEST_CASE_AHNET_2D_3])
    def test_ahnet_shape_2d(self, input_param, input_shape, expected_shape):
        net = AHNet(**input_param).to(device)
        with eval_mode(net):
            result = net.forward(torch.randn(input_shape).to(device))
            self.assertEqual(result.shape, expected_shape)

    @parameterized.expand([TEST_CASE_AHNET_3D_1, TEST_CASE_AHNET_3D_2, TEST_CASE_AHNET_3D_3])
    @skip_if_quick
    def test_ahnet_shape_3d(self, input_param, input_shape, expected_shape):
        net = AHNet(**input_param).to(device)
        with eval_mode(net):
            result = net.forward(torch.randn(input_shape).to(device))
            self.assertEqual(result.shape, expected_shape)

    @skip_if_quick
    def test_script(self):
        # test 2D network
        net = AHNet(spatial_dims=2, out_channels=2)
        test_data = torch.randn(1, 1, 128, 64)
        test_script_save(net, test_data)
        # test 3D network
        net = AHNet(spatial_dims=3, out_channels=2, psp_block_num=0, upsample_mode="nearest")
        test_data = torch.randn(1, 1, 32, 32, 64)
        test_script_save(net, test_data)


class TestAHNETWithPretrain(unittest.TestCase):
    @parameterized.expand(
        [TEST_CASE_AHNET_3D_WITH_PRETRAIN_1, TEST_CASE_AHNET_3D_WITH_PRETRAIN_2, TEST_CASE_AHNET_3D_WITH_PRETRAIN_3]
    )
    def test_ahnet_shape(self, input_param, input_shape, expected_shape, fcn_input_param):
        net = AHNet(**input_param).to(device)
        net2d = FCN(**fcn_input_param).to(device)
        net.copy_from(net2d)
        with eval_mode(net):
            result = net.forward(torch.randn(input_shape).to(device))
            self.assertEqual(result.shape, expected_shape)

    @skip_if_quick
    def test_initialize_pretrained(self):
        net = AHNet(
            spatial_dims=3,
            upsample_mode="transpose",
            in_channels=2,
            out_channels=3,
            psp_block_num=2,
            pretrained=True,
            progress=True,
        ).to(device)
        input_data = torch.randn(2, 2, 32, 32, 64).to(device)
        with eval_mode(net):
            result = net.forward(input_data)
            self.assertEqual(result.shape, (2, 3, 32, 32, 64))


if __name__ == "__main__":
    unittest.main()
