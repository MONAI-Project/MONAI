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

import os
import tempfile
import unittest

import torch

from monai.networks import convert_to_torchscript
from monai.networks.nets import UNet


class TestConvertToTorchScript(unittest.TestCase):

    def test_value(self):
        model = UNet(
            spatial_dims=2, in_channels=1, out_channels=3, channels=(16, 32, 64), strides=(2, 2), num_res_units=0
        )
        with tempfile.TemporaryDirectory() as tempdir:
            torchscript_model = convert_to_torchscript(
                model=model,
                filename_or_obj=os.path.join(tempdir, "model.ts"),
                extra_files={"foo.txt": b"bar"},
                verify=True,
                inputs=[torch.randn((16, 1, 32, 32), requires_grad=False)],
                device="cuda" if torch.cuda.is_available() else "cpu",
                rtol=1e-3,
                atol=1e-4,
                optimize=None,
            )
            self.assertTrue(isinstance(torchscript_model, torch.nn.Module))


if __name__ == "__main__":
    unittest.main()
