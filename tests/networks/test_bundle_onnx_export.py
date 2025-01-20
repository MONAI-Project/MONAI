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
from pathlib import Path

from parameterized import parameterized

from monai.bundle import ConfigParser
from monai.networks import save_state
from tests.util import SkipIfBeforePyTorchVersion, SkipIfNoModule, command_line_tests, skip_if_windows

TEST_CASE_1 = ["True"]
TEST_CASE_2 = ["False"]


@skip_if_windows
@SkipIfNoModule("onnx")
@SkipIfBeforePyTorchVersion((1, 10))
class TestONNXExport(unittest.TestCase):
    def setUp(self):
        self.device = os.environ.get("CUDA_VISIBLE_DEVICES")
        if not self.device:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # default

    def tearDown(self):
        if self.device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.device
        else:
            del os.environ["CUDA_VISIBLE_DEVICES"]  # previously unset

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    def test_onnx_export(self, use_trace):
        tests_path = Path(__file__).parents[1]
        meta_file = os.path.join(tests_path, "testing_data", "metadata.json")
        config_file = os.path.join(tests_path, "testing_data", "inference.json")
        with tempfile.TemporaryDirectory() as tempdir:
            def_args = {"meta_file": "will be replaced by `meta_file` arg"}
            def_args_file = os.path.join(tempdir, "def_args.yaml")

            ckpt_file = os.path.join(tempdir, "model.pt")
            onnx_file = os.path.join(tempdir, "model.onnx")

            parser = ConfigParser()
            parser.export_config_file(config=def_args, filepath=def_args_file)
            parser.read_config(config_file)
            net = parser.get_parsed_content("network_def")
            save_state(src=net, path=ckpt_file)

            cmd = ["python", "-m", "monai.bundle", "onnx_export", "network_def", "--filepath", onnx_file]
            cmd += ["--meta_file", meta_file, "--config_file", f"['{config_file}','{def_args_file}']"]
            cmd += ["--ckpt_file", ckpt_file, "--args_file", def_args_file, "--input_shape", "[1, 1, 96, 96, 96]"]
            cmd += ["--use_trace", use_trace]
            command_line_tests(cmd)
            self.assertTrue(os.path.exists(onnx_file))


if __name__ == "__main__":
    unittest.main()
