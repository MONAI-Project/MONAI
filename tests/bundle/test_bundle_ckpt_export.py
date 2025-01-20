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

import json
import os
import tempfile
import unittest
from pathlib import Path

from parameterized import parameterized

from monai.bundle import ConfigParser
from monai.data import load_net_with_metadata
from monai.networks import save_state
from tests.util import command_line_tests, skip_if_windows

TEST_CASE_1 = ["", ""]
TEST_CASE_2 = ["model", ""]
TEST_CASE_3 = ["model", "True"]


@skip_if_windows
class TestCKPTExport(unittest.TestCase):
    def setUp(self):
        self.device = os.environ.get("CUDA_VISIBLE_DEVICES")
        if not self.device:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # default
        module_path = Path(__file__).resolve().parents[1].as_posix()
        self.meta_file = os.path.join(module_path, "testing_data", "metadata.json")
        self.config_file = os.path.join(module_path, "testing_data", "inference.json")
        self.tempdir_obj = tempfile.TemporaryDirectory()
        tempdir = self.tempdir_obj.name
        self.def_args = {"meta_file": "will be replaced by `meta_file` arg"}
        self.def_args_file = os.path.join(tempdir, "def_args.yaml")

        self.ckpt_file = os.path.join(tempdir, "model.pt")
        self.ts_file = os.path.join(tempdir, "model.ts")

        self.parser = ConfigParser()
        self.parser.export_config_file(config=self.def_args, filepath=self.def_args_file)
        self.parser.read_config(self.config_file)
        self.net = self.parser.get_parsed_content("network_def")
        self.cmd = ["coverage", "run", "-m", "monai.bundle", "ckpt_export", "network_def", "--filepath", self.ts_file]
        self.cmd += [
            "--meta_file",
            self.meta_file,
            "--config_file",
            f"['{self.config_file}','{self.def_args_file}']",
            "--ckpt_file",
        ]

    def tearDown(self):
        if self.device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.device
        else:
            del os.environ["CUDA_VISIBLE_DEVICES"]  # previously unset
        self.tempdir_obj.cleanup()

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3])
    def test_export(self, key_in_ckpt, use_trace):
        save_state(src=self.net if key_in_ckpt == "" else {key_in_ckpt: self.net}, path=self.ckpt_file)
        full_cmd = self.cmd + [self.ckpt_file, "--key_in_ckpt", key_in_ckpt, "--args_file", self.def_args_file]
        if use_trace == "True":
            full_cmd += ["--use_trace", use_trace, "--input_shape", "[1, 1, 96, 96, 96]"]
        command_line_tests(full_cmd)
        self.assertTrue(os.path.exists(self.ts_file))

        _, metadata, extra_files = load_net_with_metadata(
            self.ts_file, more_extra_files=["inference.json", "def_args.json"]
        )
        self.assertIn("schema", metadata)
        self.assertIn("meta_file", json.loads(extra_files["def_args.json"]))
        self.assertIn("network_def", json.loads(extra_files["inference.json"]))

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3])
    def test_default_value(self, key_in_ckpt, use_trace):
        ckpt_file = os.path.join(self.tempdir_obj.name, "models/model.pt")
        ts_file = os.path.join(self.tempdir_obj.name, "models/model.ts")

        save_state(src=self.net if key_in_ckpt == "" else {key_in_ckpt: self.net}, path=ckpt_file)

        # check with default value
        cmd = ["coverage", "run", "-m", "monai.bundle", "ckpt_export", "--key_in_ckpt", key_in_ckpt]
        cmd += ["--config_file", self.config_file, "--bundle_root", self.tempdir_obj.name]
        if use_trace == "True":
            cmd += ["--use_trace", use_trace, "--input_shape", "[1, 1, 96, 96, 96]"]
        command_line_tests(cmd)
        self.assertTrue(os.path.exists(ts_file))


if __name__ == "__main__":
    unittest.main()
