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
from monai.networks.nets import SegResNet, SegResNetVAE
from monai.utils import optional_import, UpsampleMode
from tests.utils import test_onnx_save, test_script_save

_, has_onnx = optional_import("onnx")

device = "cuda" if torch.cuda.is_available() else "cpu"

TEST_CASE_SEGRESNET = []
for spatial_dims in range(2, 4):
    for init_filters in [8, 16]:
        for dropout_prob in [None, 0.2]:
            for norm in [("GROUP", {"num_groups": 8}), ("batch", {"track_running_stats": False}), "instance"]:
                for upsample_mode in UpsampleMode:
                    test_case = [
                        {
                            "spatial_dims": spatial_dims,
                            "init_filters": init_filters,
                            "dropout_prob": dropout_prob,
                            "norm": norm,
                            "upsample_mode": upsample_mode,
                            "use_conv_final": False,
                        },
                        (2, 1, *([16] * spatial_dims)),
                        (2, init_filters, *([16] * spatial_dims)),
                    ]
                    TEST_CASE_SEGRESNET.append(test_case)

TEST_CASE_SEGRESNET_2 = []
for spatial_dims in range(2, 4):
    for init_filters in [8, 16]:
        for out_channels in range(1, 3):
            for upsample_mode in UpsampleMode:
                test_case = [
                    {
                        "spatial_dims": spatial_dims,
                        "init_filters": init_filters,
                        "out_channels": out_channels,
                        "upsample_mode": upsample_mode,
                    },
                    (2, 1, *([16] * spatial_dims)),
                    (2, out_channels, *([16] * spatial_dims)),
                ]
                TEST_CASE_SEGRESNET_2.append(test_case)

TEST_CASE_SEGRESNET_VAE = []
for spatial_dims in range(2, 4):
    for init_filters in [8, 16]:
        for out_channels in range(1, 3):
            for upsample_mode in UpsampleMode:
                for vae_estimate_std in [True, False]:
                    test_case = [
                        {
                            "spatial_dims": spatial_dims,
                            "init_filters": init_filters,
                            "out_channels": out_channels,
                            "upsample_mode": upsample_mode,
                            "act": ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
                            "input_image_size": ([16] * spatial_dims),
                            "vae_estimate_std": vae_estimate_std,
                        },
                        (2, 1, *([16] * spatial_dims)),
                        (2, out_channels, *([16] * spatial_dims)),
                    ]
                TEST_CASE_SEGRESNET_VAE.append(test_case)


class TestResNet(unittest.TestCase):

    @parameterized.expand(TEST_CASE_SEGRESNET + TEST_CASE_SEGRESNET_2)
    def test_shape(self, input_param, input_shape, expected_shape):
        net = SegResNet(**input_param).to(device)
        with eval_mode(net):
            result = net(torch.randn(input_shape).to(device))
            self.assertEqual(result.shape, expected_shape)

    def test_ill_arg(self):
        with self.assertRaises(ValueError):
            SegResNet(spatial_dims=4)

    def test_script(self):
        input_param, input_shape, expected_shape = TEST_CASE_SEGRESNET[0]
        net = SegResNet(**input_param)
        test_data = torch.randn(input_shape)
        test_script_save(net, test_data)

    @unittest.skipUnless(has_onnx, "Requires onnx")
    def test_onnx(self):
        input_param, input_shape, expected_shape = TEST_CASE_SEGRESNET[0]
        net = SegResNet(**input_param)
        test_data = torch.randn(input_shape)
        test_onnx_save(net, test_data, atol=1e-4)


class TestResNetVAE(unittest.TestCase):

    @parameterized.expand(TEST_CASE_SEGRESNET_VAE)
    def test_vae_shape(self, input_param, input_shape, expected_shape):
        net = SegResNetVAE(**input_param).to(device)
        with eval_mode(net):
            result, _ = net(torch.randn(input_shape).to(device))
            self.assertEqual(result.shape, expected_shape)

    def test_script(self):
        input_param, input_shape, expected_shape = TEST_CASE_SEGRESNET_VAE[0]
        net = SegResNetVAE(**input_param)
        test_data = torch.randn(input_shape)
        test_script_save(net, test_data)

    @unittest.skipUnless(has_onnx, "Requires onnx")
    def test_onnx(self):
        input_param, input_shape, expected_shape = TEST_CASE_SEGRESNET_VAE[0]
        net = SegResNetVAE(**input_param)
        test_data = torch.randn(input_shape)
        test_onnx_save(net, test_data, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
