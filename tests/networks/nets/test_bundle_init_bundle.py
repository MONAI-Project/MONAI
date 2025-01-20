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

from monai.networks.nets import UNet
from tests.util import command_line_tests, skip_if_windows


@skip_if_windows
class TestBundleInit(unittest.TestCase):
    def test_bundle(self):
        with tempfile.TemporaryDirectory() as tempdir:
            net = UNet(2, 1, 1, [4, 8], [2])
            torch.save(net.state_dict(), tempdir + "/test.pt")

            bundle_root = tempdir + "/test_bundle"

            cmd = [
                "coverage",
                "run",
                "-m",
                "monai.bundle",
                "init_bundle",
                bundle_root,
                tempdir + "/test.pt",
                "--dataset_license",
                "True",
            ]
            command_line_tests(cmd)

            self.assertTrue(os.path.exists(bundle_root + "/configs/metadata.json"))
            self.assertTrue(os.path.exists(bundle_root + "/configs/inference.json"))
            self.assertTrue(os.path.exists(bundle_root + "/models/model.pt"))
            self.assertTrue(os.path.exists(bundle_root + "/LICENSE"))
            self.assertTrue(os.path.exists(bundle_root + "/docs/README.md"))
            self.assertTrue(os.path.exists(bundle_root + "/docs/data_license.txt"))


if __name__ == "__main__":
    unittest.main()
