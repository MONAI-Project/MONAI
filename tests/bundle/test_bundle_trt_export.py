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
from monai.utils import optional_import
from tests.util import (
    SkipIfBeforeComputeCapabilityVersion,
    command_line_tests,
    skip_if_no_cuda,
    skip_if_quick,
    skip_if_windows,
)

_, has_torchtrt = optional_import(
    "torch_tensorrt",
    version="1.4.0",
    descriptor="Torch-TRT is not installed. Are you sure you have a Torch-TensorRT compilation?",
)
_, has_tensorrt = optional_import(
    "tensorrt", descriptor="TensorRT is not installed. Are you sure you have a TensorRT compilation?"
)

_, has_onnx = optional_import("onnx", descriptor="Onnx is not installed. Will not test the onnx option.")

TEST_CASE_1 = ["fp32", [], []]

TEST_CASE_2 = ["fp16", [], []]

TEST_CASE_3 = ["fp32", [1, 1, 96, 96, 96], [1, 4, 8]]

TEST_CASE_4 = ["fp16", [1, 1, 96, 96, 96], [1, 4, 8]]


@skip_if_windows
@skip_if_no_cuda
@skip_if_quick
@SkipIfBeforeComputeCapabilityVersion((7, 5))
class TestTRTExport(unittest.TestCase):
    def setUp(self):
        self.device = os.environ.get("CUDA_VISIBLE_DEVICES")
        if not self.device:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # default

    def tearDown(self):
        if self.device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.device
        else:
            del os.environ["CUDA_VISIBLE_DEVICES"]  # previously unset

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4])
    @unittest.skipUnless(has_torchtrt and has_tensorrt, "Torch-TensorRT is required for conversion!")
    def test_trt_export(self, convert_precision, input_shape, dynamic_batch):
        tests_dir = Path(__file__).resolve().parent
        meta_file = os.path.join(tests_dir, "testing_data", "metadata.json")
        config_file = os.path.join(tests_dir, "testing_data", "inference.json")
        with tempfile.TemporaryDirectory() as tempdir:
            def_args = {"meta_file": "will be replaced by `meta_file` arg"}
            def_args_file = os.path.join(tempdir, "def_args.yaml")

            ckpt_file = os.path.join(tempdir, "model.pt")
            ts_file = os.path.join(tempdir, f"model_trt_{convert_precision}.ts")

            parser = ConfigParser()
            parser.export_config_file(config=def_args, filepath=def_args_file)
            parser.read_config(config_file)
            net = parser.get_parsed_content("network_def")
            save_state(src=net, path=ckpt_file)

            cmd = ["python", "-m", "monai.bundle", "trt_export", "network_def", "--filepath", ts_file]
            cmd += ["--meta_file", meta_file, "--config_file", f"['{config_file}','{def_args_file}']", "--ckpt_file"]
            cmd += [ckpt_file, "--args_file", def_args_file, "--precision", convert_precision]
            if input_shape:
                cmd += ["--input_shape", str(input_shape)]
            if dynamic_batch:
                cmd += ["--dynamic_batch", str(dynamic_batch)]
            command_line_tests(cmd)
            self.assertTrue(os.path.exists(ts_file))

            _, metadata, extra_files = load_net_with_metadata(
                ts_file, more_extra_files=["inference.json", "def_args.json"]
            )
            self.assertIn("schema", metadata)
            self.assertIn("meta_file", json.loads(extra_files["def_args.json"]))
            self.assertIn("network_def", json.loads(extra_files["inference.json"]))

    @parameterized.expand([TEST_CASE_3, TEST_CASE_4])
    @unittest.skipUnless(
        has_onnx and has_torchtrt and has_tensorrt, "Onnx and TensorRT are required for onnx-trt conversion!"
    )
    def test_onnx_trt_export(self, convert_precision, input_shape, dynamic_batch):
        tests_dir = Path(__file__).resolve().parent
        meta_file = os.path.join(tests_dir, "testing_data", "metadata.json")
        config_file = os.path.join(tests_dir, "testing_data", "inference.json")
        with tempfile.TemporaryDirectory() as tempdir:
            def_args = {"meta_file": "will be replaced by `meta_file` arg"}
            def_args_file = os.path.join(tempdir, "def_args.yaml")

            ckpt_file = os.path.join(tempdir, "model.pt")
            ts_file = os.path.join(tempdir, f"model_trt_{convert_precision}.ts")

            parser = ConfigParser()
            parser.export_config_file(config=def_args, filepath=def_args_file)
            parser.read_config(config_file)
            net = parser.get_parsed_content("network_def")
            save_state(src=net, path=ckpt_file)

            cmd = ["python", "-m", "monai.bundle", "trt_export", "network_def", "--filepath", ts_file]
            cmd += ["--meta_file", meta_file, "--config_file", f"['{config_file}','{def_args_file}']", "--ckpt_file"]
            cmd += [ckpt_file, "--args_file", def_args_file, "--precision", convert_precision]
            cmd += ["--use_onnx", "True"]
            if input_shape:
                cmd += ["--input_shape", str(input_shape)]
            if dynamic_batch:
                cmd += ["--dynamic_batch", str(dynamic_batch)]
            command_line_tests(cmd)
            self.assertTrue(os.path.exists(ts_file))

            _, metadata, extra_files = load_net_with_metadata(
                ts_file, more_extra_files=["inference.json", "def_args.json"]
            )
            self.assertIn("schema", metadata)
            self.assertIn("meta_file", json.loads(extra_files["def_args.json"]))
            self.assertIn("network_def", json.loads(extra_files["inference.json"]))


if __name__ == "__main__":
    unittest.main()
