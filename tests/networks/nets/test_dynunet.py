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

import platform
import unittest
from collections.abc import Sequence
from typing import Any

import torch
from parameterized import parameterized

from monai.networks import eval_mode
from monai.networks.nets import DynUNet
from monai.utils import optional_import
from tests.util import assert_allclose, skip_if_no_cuda, skip_if_windows, test_script_save

InstanceNorm3dNVFuser, _ = optional_import("apex.normalization", name="InstanceNorm3dNVFuser")

ON_AARCH64 = platform.machine() == "aarch64"
if ON_AARCH64:
    rtol, atol = 1e-2, 1e-2
else:
    rtol, atol = 1e-4, 1e-4

device = "cuda" if torch.cuda.is_available() else "cpu"

strides: Sequence[Sequence[int] | int]
kernel_size: Sequence[Any]
expected_shape: Sequence[Any]

TEST_CASE_DYNUNET_2D = []
out_channels = 2
in_size = 64
spatial_dims = 2
for kernel_size in [(3, 3, 3, 1), ((3, 1), 1, (3, 3), (1, 1))]:
    for strides in [(1, 1, 1, 1), (2, 2, 2, 1)]:
        expected_shape = (1, out_channels, *[in_size // strides[0]] * spatial_dims)
        for in_channels in [2, 3]:
            for res_block in [True, False]:
                test_case = [
                    {
                        "spatial_dims": spatial_dims,
                        "in_channels": in_channels,
                        "out_channels": out_channels,
                        "kernel_size": kernel_size,
                        "strides": strides,
                        "upsample_kernel_size": strides[1:],
                        "norm_name": "batch",
                        "act_name": ("leakyrelu", {"inplace": True, "negative_slope": 0.2}),
                        "deep_supervision": False,
                        "res_block": res_block,
                        "dropout": None,
                    },
                    (1, in_channels, in_size, in_size),
                    expected_shape,
                ]
                TEST_CASE_DYNUNET_2D.append(test_case)

TEST_CASE_DYNUNET_3D = []  # in 3d cases, also test anisotropic kernel/strides
in_channels = 1
in_size = 64
for out_channels in [2, 3]:
    expected_shape = (1, out_channels, 64, 32, 64)
    for res_block in [True, False]:
        test_case = [
            {
                "spatial_dims": 3,
                "in_channels": in_channels,
                "out_channels": out_channels,
                "kernel_size": (3, (1, 1, 3), 3, 3),
                "strides": ((1, 2, 1), 2, 2, 1),
                "upsample_kernel_size": (2, 2, 1),
                "filters": (64, 96, 128, 192),
                "norm_name": ("INSTANCE", {"affine": True}),
                "deep_supervision": True,
                "res_block": res_block,
                "dropout": ("alphadropout", {"p": 0.25}),
            },
            (1, in_channels, in_size, in_size, in_size),
            expected_shape,
        ]
        TEST_CASE_DYNUNET_3D.append(test_case)

TEST_CASE_DEEP_SUPERVISION = []
for spatial_dims in [2, 3]:
    for res_block in [True, False]:
        for deep_supr_num in [1, 2]:
            for strides in [(1, 2, 1, 2, 1), (2, 2, 2, 1), (2, 1, 1, 2, 2)]:
                scale = strides[0]
                test_case = [
                    {
                        "spatial_dims": spatial_dims,
                        "in_channels": 1,
                        "out_channels": 2,
                        "kernel_size": [3] * len(strides),
                        "strides": strides,
                        "upsample_kernel_size": strides[1:],
                        "norm_name": ("group", {"num_groups": 16}),
                        "deep_supervision": True,
                        "deep_supr_num": deep_supr_num,
                        "res_block": res_block,
                    },
                    (1, 1, *[in_size] * spatial_dims),
                    (1, 1 + deep_supr_num, 2, *[in_size // scale] * spatial_dims),
                ]
                TEST_CASE_DEEP_SUPERVISION.append(test_case)


class TestDynUNet(unittest.TestCase):
    @parameterized.expand(TEST_CASE_DYNUNET_3D)
    def test_shape(self, input_param, input_shape, expected_shape):
        net = DynUNet(**input_param).to(device)
        if "alphadropout" in input_param.get("dropout"):
            self.assertTrue(any(isinstance(x, torch.nn.AlphaDropout) for x in net.modules()))
        with eval_mode(net):
            result = net(torch.randn(input_shape).to(device))
            self.assertEqual(result.shape, expected_shape)

    def test_script(self):
        input_param, input_shape, _ = TEST_CASE_DYNUNET_2D[0]
        net = DynUNet(**input_param)
        test_data = torch.randn(input_shape)
        test_script_save(net, test_data)


@skip_if_no_cuda
@skip_if_windows
class TestDynUNetWithInstanceNorm3dNVFuser(unittest.TestCase):
    def setUp(self):
        try:
            layer = InstanceNorm3dNVFuser(num_features=1, affine=False).to("cuda:0")
            inp = torch.randn([1, 1, 1, 1, 1]).to("cuda:0")
            out = layer(inp)
            del inp, out, layer
        except Exception:
            self.skipTest("NVFuser not available")

    @parameterized.expand([TEST_CASE_DYNUNET_3D[0]])
    def test_consistency(self, input_param, input_shape, _):
        for eps in [1e-4, 1e-5]:
            for momentum in [0.1, 0.01]:
                for affine in [True, False]:
                    norm_param = {"eps": eps, "momentum": momentum, "affine": affine}
                    input_param["norm_name"] = ("instance", norm_param)
                    input_param_fuser = input_param.copy()
                    input_param_fuser["norm_name"] = ("instance_nvfuser", norm_param)
                    for memory_format in [torch.contiguous_format, torch.channels_last_3d]:
                        net = DynUNet(**input_param).to("cuda:0", memory_format=memory_format)
                        net_fuser = DynUNet(**input_param_fuser).to("cuda:0", memory_format=memory_format)
                        net_fuser.load_state_dict(net.state_dict())

                        input_tensor = torch.randn(input_shape).to("cuda:0", memory_format=memory_format)
                        with eval_mode(net):
                            result = net(input_tensor)
                        with eval_mode(net_fuser):
                            result_fuser = net_fuser(input_tensor)

                        assert_allclose(result, result_fuser, rtol=rtol, atol=atol)


class TestDynUNetDeepSupervision(unittest.TestCase):
    @parameterized.expand(TEST_CASE_DEEP_SUPERVISION)
    def test_shape(self, input_param, input_shape, expected_shape):
        net = DynUNet(**input_param).to(device)
        with torch.no_grad():
            results = net(torch.randn(input_shape).to(device))
            self.assertEqual(results.shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
