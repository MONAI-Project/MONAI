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

import itertools
import platform
import unittest

import torch
from parameterized import parameterized

from monai.networks import convert_to_onnx
from monai.networks.nets import SegResNet, UNet
from monai.utils.module import pytorch_after
from tests.util import SkipIfBeforePyTorchVersion, SkipIfNoModule, optional_import, skip_if_quick

if torch.cuda.is_available():
    TORCH_DEVICE_OPTIONS = ["cpu", "cuda"]
else:
    TORCH_DEVICE_OPTIONS = ["cpu"]
TESTS = list(itertools.product(TORCH_DEVICE_OPTIONS, [True, False], [True, False]))
TESTS_ORT = list(itertools.product(TORCH_DEVICE_OPTIONS, [True]))

ON_AARCH64 = platform.machine() == "aarch64"
if ON_AARCH64:
    rtol, atol = 1e-1, 1e-2
else:
    rtol, atol = 1e-3, 1e-4

onnx, _ = optional_import("onnx")


@SkipIfNoModule("onnx")
@SkipIfBeforePyTorchVersion((1, 9))
@skip_if_quick
class TestConvertToOnnx(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_unet(self, device, use_trace, use_ort):
        if use_ort:
            _, has_onnxruntime = optional_import("onnxruntime")
            if not has_onnxruntime:
                self.skipTest("onnxruntime is not installed probably due to python version >= 3.11.")
        model = UNet(
            spatial_dims=2, in_channels=1, out_channels=3, channels=(16, 32, 64), strides=(2, 2), num_res_units=0
        )
        if pytorch_after(1, 10) or use_trace:
            onnx_model = convert_to_onnx(
                model=model,
                inputs=[torch.randn((16, 1, 32, 32), requires_grad=False)],
                input_names=["x"],
                output_names=["y"],
                verify=True,
                device=device,
                use_ort=use_ort,
                use_trace=use_trace,
                rtol=rtol,
                atol=atol,
            )
        else:
            # https://github.com/pytorch/pytorch/blob/release/1.9/torch/onnx/__init__.py#L182
            # example_outputs is required in scripting mode before PyTorch 3.10
            onnx_model = convert_to_onnx(
                model=model,
                inputs=[torch.randn((16, 1, 32, 32), requires_grad=False)],
                input_names=["x"],
                output_names=["y"],
                example_outputs=[torch.randn((16, 3, 32, 32), requires_grad=False)],
                verify=True,
                device=device,
                use_ort=use_ort,
                use_trace=use_trace,
                rtol=rtol,
                atol=atol,
            )
        self.assertTrue(isinstance(onnx_model, onnx.ModelProto))

    @parameterized.expand(TESTS_ORT)
    @SkipIfBeforePyTorchVersion((1, 12))
    def test_seg_res_net(self, device, use_ort):
        if use_ort:
            _, has_onnxruntime = optional_import("onnxruntime")
            if not has_onnxruntime:
                self.skipTest("onnxruntime is not installed probably due to python version >= 3.11.")
        model = SegResNet(
            spatial_dims=3,
            init_filters=32,
            in_channels=1,
            out_channels=105,
            dropout_prob=0.2,
            act=("RELU", {"inplace": True}),
            norm=("GROUP", {"num_groups": 8}),
            norm_name="",
            num_groups=8,
            use_conv_final=True,
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
        )
        onnx_model = convert_to_onnx(
            model=model,
            inputs=[torch.randn((1, 1, 24, 24, 24), requires_grad=False)],
            input_names=["x"],
            output_names=["y"],
            verify=True,
            device=device,
            use_ort=use_ort,
            use_trace=True,
            rtol=rtol,
            atol=atol,
        )
        self.assertTrue(isinstance(onnx_model, onnx.ModelProto))


if __name__ == "__main__":
    unittest.main()
