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

import tempfile
import unittest

import torch
from parameterized import parameterized

from monai.networks import convert_to_trt
from monai.networks.nets import UNet
from monai.utils import optional_import
from tests.util import SkipIfBeforeComputeCapabilityVersion, skip_if_no_cuda, skip_if_quick, skip_if_windows

_, has_torchtrt = optional_import(
    "torch_tensorrt",
    version="1.4.0",
    descriptor="Torch-TRT is not installed. Are you sure you have a Torch-TensorRT compilation?",
)
_, has_tensorrt = optional_import(
    "tensorrt", descriptor="TensorRT is not installed. Are you sure you have a TensorRT compilation?"
)

TEST_CASE_1 = ["fp32"]
TEST_CASE_2 = ["fp16"]


@skip_if_windows
@skip_if_no_cuda
@skip_if_quick
@SkipIfBeforeComputeCapabilityVersion((7, 5))
class TestConvertToTRT(unittest.TestCase):
    def setUp(self):
        self.gpu_device = torch.cuda.current_device()

    def tearDown(self):
        current_device = torch.cuda.current_device()
        if current_device != self.gpu_device:
            torch.cuda.set_device(self.gpu_device)

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    @unittest.skipUnless(has_torchtrt and has_tensorrt, "Torch-TensorRT is required for convert!")
    def test_value(self, precision):
        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(2, 2, 4, 8, 4),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm="batch",
        )
        with tempfile.TemporaryDirectory() as _:
            torchscript_model = convert_to_trt(
                model=model,
                precision=precision,
                input_shape=[1, 1, 96, 96, 96],
                dynamic_batchsize=[1, 4, 8],
                use_trace=False,
                verify=True,
                device=0,
                rtol=1e-2,
                atol=1e-2,
            )
            self.assertTrue(isinstance(torchscript_model, torch.nn.Module))


if __name__ == "__main__":
    unittest.main()
