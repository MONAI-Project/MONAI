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
import shutil
import subprocess
import sys
import tempfile
import unittest
from glob import glob
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from parameterized import parameterized

from monai.bundle import ConfigParser
from monai.bundle.utils import DEFAULT_HANDLERS_ID
from monai.transforms import LoadImage
from monai.utils import path_to_uri
from tests.util import command_line_tests

TESTS_PATH = Path(__file__).parents[1]
TEST_CASE_1 = [os.path.join(TESTS_PATH, "testing_data", "inference.json"), (128, 128, 128)]

TEST_CASE_2 = [os.path.join(TESTS_PATH, "testing_data", "inference.yaml"), (128, 128, 128)]


class _Runnable42:
    def __init__(self, val=1):
        self.val = val

    def run(self):
        assert self.val == 42  # defined in `TestBundleRun.test_tiny``
        return self.val


class _Runnable43:
    def __init__(self, func):
        self.func = func

    def run(self):
        self.func()


class TestBundleRun(unittest.TestCase):
    def setUp(self):
        self.data_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.data_dir)

    def test_tiny(self):
        config_file = os.path.join(self.data_dir, "tiny_config.json")
        meta_file = os.path.join(self.data_dir, "tiny_meta.json")
        with open(config_file, "w") as f:
            json.dump(
                {
                    "trainer": {"_target_": "tests.test_integration_bundle_run._Runnable42", "val": 42},
                    # keep this test case to cover the "run_id" arg
                    "training": "$@trainer.run()",
                },
                f,
            )
        with open(meta_file, "w") as f:
            json.dump(
                {"version": "0.1.0", "monai_version": "1.1.0", "pytorch_version": "1.13.1", "numpy_version": "1.22.2"},
                f,
            )
        cmd = ["coverage", "run", "-m", "monai.bundle"]
        # test both CLI entry "run" and "run_workflow"
        command_line_tests(cmd + ["run", "training", "--config_file", config_file, "--meta_file", meta_file])
        command_line_tests(
            cmd + ["run_workflow", "--run_id", "training", "--config_file", config_file, "--meta_file", meta_file]
        )
        with self.assertRaises(RuntimeError):
            # test wrong run_id="run"
            command_line_tests(cmd + ["run", "run", "--config_file", config_file])
        # test missing meta file
        self.assertIn("ERROR", command_line_tests(cmd + ["run", "training", "--config_file", config_file]))

    def test_scripts_fold(self):
        # test scripts directory has been added to Python search directories automatically
        config_file = os.path.join(self.data_dir, "tiny_config.json")
        meta_file = os.path.join(self.data_dir, "tiny_meta.json")
        scripts_dir = os.path.join(self.data_dir, "scripts")
        script_file = os.path.join(scripts_dir, "test_scripts_fold.py")
        init_file = os.path.join(scripts_dir, "__init__.py")

        with open(config_file, "w") as f:
            json.dump(
                {
                    "imports": ["$import scripts"],
                    "trainer": {
                        "_target_": "tests.test_integration_bundle_run._Runnable43",
                        "func": "$scripts.tiny_test",
                    },
                    # keep this test case to cover the "run_id" arg
                    "training": "$@trainer.run()",
                },
                f,
            )
        with open(meta_file, "w") as f:
            json.dump(
                {"version": "0.1.0", "monai_version": "1.1.0", "pytorch_version": "1.13.1", "numpy_version": "1.22.2"},
                f,
            )

        os.mkdir(scripts_dir)
        script_file_lines = ["def tiny_test():\n", "    print('successfully added scripts fold!') \n"]
        init_file_line = "from .test_scripts_fold import tiny_test\n"
        with open(script_file, "w") as f:
            f.writelines(script_file_lines)
            f.close()
        with open(init_file, "w") as f:
            f.write(init_file_line)
            f.close()

        cmd = ["coverage", "run", "-m", "monai.bundle"]
        # test both CLI entry "run" and "run_workflow"
        expected_condition = "successfully added scripts fold!"
        command_run = cmd + ["run", "training", "--config_file", config_file, "--meta_file", meta_file]
        completed_process = subprocess.run(command_run, check=True, capture_output=True, text=True)
        output = repr(completed_process.stdout).replace("\\n", "\n").replace("\\t", "\t")  # Get the captured output

        self.assertIn(expected_condition, output)
        command_run_workflow = cmd + [
            "run_workflow",
            "--run_id",
            "training",
            "--config_file",
            config_file,
            "--meta_file",
            meta_file,
        ]
        completed_process = subprocess.run(command_run_workflow, check=True, capture_output=True, text=True)
        output = repr(completed_process.stdout).replace("\\n", "\n").replace("\\t", "\t")  # Get the captured output
        self.assertIn(expected_condition, output)

        # test missing meta file
        self.assertIn("ERROR", command_line_tests(cmd + ["run", "training", "--config_file", config_file]))

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    def test_shape(self, config_file, expected_shape):
        test_image = np.random.rand(*expected_shape)
        tempdir = self.data_dir
        filename = os.path.join(tempdir, "image.nii")
        nib.save(nib.Nifti1Image(test_image, np.eye(4)), filename)

        # generate default args in a JSON file
        logging_conf = os.path.join(TESTS_PATH, "testing_data", "logging.conf")
        def_args = {"config_file": "will be replaced by `config_file` arg", "logging_file": logging_conf}
        def_args_file = os.path.join(tempdir, "def_args.json")
        ConfigParser.export_config_file(config=def_args, filepath=def_args_file)

        meta = {"datalist": [{"image": filename}], "window": (96, 96, 96)}
        # test YAML file
        meta_file = os.path.join(tempdir, "meta.yaml")
        ConfigParser.export_config_file(config=meta, filepath=meta_file, fmt="yaml")

        # test MLFlow settings
        settings = {
            "handlers_id": DEFAULT_HANDLERS_ID,
            "configs": {
                "no_epoch": True,  # test override config in the settings file
                "evaluator": {
                    "_target_": "MLFlowHandler",
                    "tracking_uri": "$monai.utils.path_to_uri(@output_dir) + '/mlflow_override1'",
                    "iteration_log": "@no_epoch",
                },
            },
        }
        settings_file = os.path.join(tempdir, "mlflow.json")
        ConfigParser.export_config_file(config=settings, filepath=settings_file, fmt="json")

        # test override with file, up case postfix
        overridefile1 = os.path.join(tempdir, "override1.JSON")
        with open(overridefile1, "w") as f:
            # test override with part of the overriding file
            json.dump({"move_net": "$@network_def.to(@device)"}, f)
        os.makedirs(os.path.join(tempdir, "jsons"), exist_ok=True)
        overridefile2 = os.path.join(tempdir, "jsons/override2.JSON")
        with open(overridefile2, "w") as f:
            # test override with the whole overriding file
            json.dump("Dataset", f)

        if sys.platform == "win32":
            override = "--network $@network_def.to(@device) --dataset#_target_ Dataset"
        else:
            override = f"--network %{overridefile1}#move_net --dataset#_target_ %{overridefile2}"
        device = "$torch.device('cuda:0')" if torch.cuda.is_available() else "$torch.device('cpu')"
        # test with `monai.bundle` as CLI entry directly
        cmd = "-m monai.bundle run --postprocessing#transforms#2#output_postfix seg"
        cmd += f" {override} --no_epoch False --output_dir {tempdir} --device {device}"
        la = ["coverage", "run"] + cmd.split(" ") + ["--meta_file", meta_file] + ["--config_file", config_file]
        test_env = os.environ.copy()
        print(f"CUDA_VISIBLE_DEVICES in {__file__}", test_env.get("CUDA_VISIBLE_DEVICES"))
        command_line_tests(la + ["--args_file", def_args_file] + ["--tracking", settings_file])
        loader = LoadImage(image_only=True)
        self.assertTupleEqual(loader(os.path.join(tempdir, "image", "image_seg.nii.gz")).shape, expected_shape)
        self.assertTrue(os.path.exists(f"{tempdir}/mlflow_override1"))

        tracking_uri = path_to_uri(tempdir) + "/mlflow_override2"  # test override experiment management configs
        # here test the script with `google fire` tool as CLI
        cmd = "-m fire monai.bundle.scripts run --tracking mlflow --evaluator#amp False"
        cmd += f" --tracking_uri {tracking_uri} {override} --output_dir {tempdir} --device {device}"
        la = ["coverage", "run"] + cmd.split(" ") + ["--meta_file", meta_file] + ["--config_file", config_file]
        command_line_tests(la)
        self.assertTupleEqual(loader(os.path.join(tempdir, "image", "image_trans.nii.gz")).shape, expected_shape)
        self.assertTrue(os.path.exists(f"{tempdir}/mlflow_override2"))
        # test the saved execution configs
        self.assertTrue(len(glob(f"{tempdir}/config_*.json")), 2)

    def test_customized_workflow(self):
        expected_shape = (64, 64, 64)
        test_image = np.random.rand(*expected_shape)
        filename = os.path.join(self.data_dir, "image.nii")
        nib.save(nib.Nifti1Image(test_image, np.eye(4)), filename)

        cmd = "-m fire monai.bundle.scripts run_workflow --workflow_name tests.nonconfig_workflow.NonConfigWorkflow"
        cmd += f" --filename {filename} --output_dir {self.data_dir}"
        command_line_tests(["coverage", "run"] + cmd.split(" "))
        loader = LoadImage(image_only=True)
        self.assertTupleEqual(loader(os.path.join(self.data_dir, "image", "image_seg.nii.gz")).shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
