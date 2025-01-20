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
import shutil
import tempfile
import unittest

import torch

from monai.bundle import update_kwargs
from monai.bundle.utils import load_bundle_config
from monai.networks.nets import UNet
from monai.utils import pprint_edges
from tests.util import command_line_tests, skip_if_windows

metadata = """
{
    "test_value": 1,
    "test_list": [2,3]
}
"""

test_json = """
{
    "test_dict": {
        "a": 3,
        "b": "c"
    },
    "network_def": {
        "_target_": "UNet",
        "spatial_dims": 2,
        "in_channels": 1,
        "out_channels": 1,
        "channels": [4,8],
        "strides": [2]
    }
}
"""


@skip_if_windows
class TestLoadBundleConfig(unittest.TestCase):
    def setUp(self):
        self.bundle_dir = tempfile.TemporaryDirectory()
        self.dir_name = os.path.join(self.bundle_dir.name, "TestBundle")
        self.configs_name = os.path.join(self.dir_name, "configs")
        self.models_name = os.path.join(self.dir_name, "models")
        self.metadata_name = os.path.join(self.configs_name, "metadata.json")
        self.test_name = os.path.join(self.configs_name, "test.json")
        self.modelpt_name = os.path.join(self.models_name, "model.pt")

        self.zip_file = os.path.join(self.bundle_dir.name, "TestBundle.zip")
        self.ts_file = os.path.join(self.bundle_dir.name, "TestBundle.ts")

        # create the directories for the bundle
        os.mkdir(self.dir_name)
        os.mkdir(self.configs_name)
        os.mkdir(self.models_name)

        # fill bundle configs

        with open(self.metadata_name, "w") as o:
            o.write(metadata)

        with open(self.test_name, "w") as o:
            o.write(test_json)

        # save network
        net = UNet(2, 1, 1, [4, 8], [2])
        torch.save(net.state_dict(), self.modelpt_name)

    def tearDown(self):
        self.bundle_dir.cleanup()

    def test_load_config_dir(self):
        p = load_bundle_config(self.dir_name, "test.json")

        self.assertEqual(p["_meta_"]["test_value"], 1)

        self.assertEqual(p["test_dict"]["b"], "c")

    def test_load_config_zip(self):
        # create a zip of the bundle
        shutil.make_archive(self.zip_file[:-4], "zip", self.bundle_dir.name)

        p = load_bundle_config(self.zip_file, "test.json")

        self.assertEqual(p["_meta_"]["test_value"], 1)

        self.assertEqual(p["test_dict"]["b"], "c")

    def test_run(self):
        command_line_tests(
            [
                "python",
                "-m",
                "monai.bundle",
                "run",
                "test",
                "--test",
                "$print('hello world')",
                "--config_file",
                self.test_name,
                "--meta_file",
                self.metadata_name,
            ]
        )

    def test_load_config_ts(self):
        # create a Torchscript zip of the bundle
        cmd = ["python", "-m", "monai.bundle", "ckpt_export", "network_def", "--filepath", self.ts_file]
        cmd += ["--meta_file", self.metadata_name]
        cmd += ["--config_file", self.test_name]
        cmd += ["--ckpt_file", self.modelpt_name]

        command_line_tests(cmd)

        p = load_bundle_config(self.ts_file, "test.json")

        self.assertEqual(p["_meta_"]["test_value"], 1)

        self.assertEqual(p["test_dict"]["b"], "c")


class TestPPrintEdges(unittest.TestCase):
    def test_str(self):
        self.assertEqual(pprint_edges("", 0), "''")
        self.assertEqual(pprint_edges({"a": 1, "b": 2}, 0), "{'a': 1, 'b': 2}")
        self.assertEqual(
            pprint_edges([{"a": 1, "b": 2}] * 20, 1),
            "[{'a': 1, 'b': 2},\n\n ... omitted 18 line(s)\n\n {'a': 1, 'b': 2}]",
        )
        self.assertEqual(pprint_edges([{"a": 1, "b": 2}] * 8, 4), pprint_edges([{"a": 1, "b": 2}] * 8, 3))
        self.assertEqual(update_kwargs({"a": 1}, a=2, b=3), {"a": 2, "b": 3})


if __name__ == "__main__":
    unittest.main()
