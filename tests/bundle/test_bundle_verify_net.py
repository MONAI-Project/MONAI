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

from monai.bundle import ConfigParser, verify_net_in_out
from tests.util import command_line_tests, skip_if_no_cuda, skip_if_windows

TESTS_PATH = Path(__file__).parents[1].as_posix()

TEST_CASE_1 = [
    os.path.join(TESTS_PATH, "testing_data", "metadata.json"),
    os.path.join(TESTS_PATH, "testing_data", "inference.json"),
]


@skip_if_windows
class TestVerifyNetwork(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1])
    def test_verify(self, meta_file, config_file):
        with tempfile.TemporaryDirectory() as tempdir:
            def_args = {"meta_file": "will be replaced by `meta_file` arg", "p": 2}
            def_args_file = os.path.join(tempdir, "def_args.json")
            ConfigParser.export_config_file(config=def_args, filepath=def_args_file)

            cmd = ["coverage", "run", "-m", "monai.bundle", "verify_net_in_out", "network_def", "--meta_file"]
            cmd += [meta_file, "--config_file", config_file, "-n", "4", "--any", "16", "--args_file", def_args_file]
            cmd += ["--device", "cpu", "--_meta_::network_data_format::inputs#image#spatial_shape", "[16,'*','2**p*n']"]
            command_line_tests(cmd)

    @parameterized.expand([TEST_CASE_1])
    @skip_if_no_cuda
    def test_verify_fp16(self, meta_file, config_file):
        with tempfile.TemporaryDirectory() as tempdir:
            def_args = {"meta_file": "will be replaced by `meta_file` arg", "p": 2}
            def_args_file = os.path.join(tempdir, "def_args.json")
            ConfigParser.export_config_file(config=def_args, filepath=def_args_file)

            cmd = ["coverage", "run", "-m", "monai.bundle", "verify_net_in_out", "network_def", "--meta_file"]
            cmd += [meta_file, "--config_file", config_file, "-n", "4", "--any", "16", "--args_file", def_args_file]
            cmd += ["--device", "cuda", "--_meta_#network_data_format#inputs#image#spatial_shape", "[16,'*','2**p*n']"]
            cmd += ["--_meta_#network_data_format#inputs#image#dtype", "float16"]
            cmd += ["--_meta_::network_data_format::outputs::pred::dtype", "float16"]
            command_line_tests(cmd)

    @parameterized.expand([TEST_CASE_1])
    @skip_if_no_cuda
    def test_verify_fp16_extra_forward_args(self, meta_file, config_file):
        verify_net_in_out(
            net_id="network_def",
            meta_file=meta_file,
            config_file=config_file,
            n=4,
            any=16,
            extra_forward_args={"extra_arg1": 1, "extra_arg2": 2},
            **{"network_def#_target_": "tests.testing_data.bundle_test_network.TestMultiInputUNet"},
        )


if __name__ == "__main__":
    unittest.main()
