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
from unittest.case import skipUnless
from unittest.mock import patch

import numpy as np
import torch
from parameterized import parameterized

import monai.networks.nets as nets
from monai.apps import check_hash
from monai.bundle import ConfigParser, create_workflow, load
from monai.bundle.scripts import _examine_monai_version, _list_latest_versions, download
from monai.utils import optional_import
from tests.util import (
    SkipIfBeforePyTorchVersion,
    assert_allclose,
    command_line_tests,
    skip_if_downloading_fails,
    skip_if_no_cuda,
    skip_if_quick,
)

_, has_huggingface_hub = optional_import("huggingface_hub")

TEST_CASE_1 = ["test_bundle", None]

TEST_CASE_2 = ["test_bundle", "0.1.1"]

TEST_CASE_3 = [
    ["model.pt", "model.ts", "network.json", "test_output.pt", "test_input.pt"],
    "test_bundle",
    "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/test_bundle.zip",
    "a131d39a0af717af32d19e565b434928",
]

TEST_CASE_4 = [
    ["model.pt", "model.ts", "network.json", "test_output.pt", "test_input.pt"],
    "test_bundle",
    "monai-test/test_bundle",
]

TEST_CASE_5 = [
    ["models/model.pt", "models/model.ts", "configs/train.json"],
    "brats_mri_segmentation",
    "https://api.ngc.nvidia.com/v2/models/nvidia/monaihosting/brats_mri_segmentation/versions/0.4.0/files/brats_mri_segmentation_v0.4.0.zip",
]

TEST_CASE_6 = [["models/model.pt", "configs/train.json"], "renalStructures_CECT_segmentation", "0.1.0"]

TEST_CASE_7 = [
    ["model.pt", "model.ts", "network.json", "test_output.pt", "test_input.pt"],
    "test_bundle",
    "Project-MONAI/MONAI-extra-test-data/0.8.1",
    "cuda" if torch.cuda.is_available() else "cpu",
    "model.pt",
]

TEST_CASE_8 = [
    "spleen_ct_segmentation",
    "cuda" if torch.cuda.is_available() else "cpu",
    {"spatial_dims": 3, "out_channels": 5},
]

TEST_CASE_9 = [
    ["test_output.pt", "test_input.pt"],
    "test_bundle",
    "0.1.1",
    "Project-MONAI/MONAI-extra-test-data/0.8.1",
    "cuda" if torch.cuda.is_available() else "cpu",
    "model.ts",
]

TEST_CASE_10 = [
    ["network.json", "test_output.pt", "test_input.pt", "large_files.yaml"],
    "test_bundle",
    "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/test_bundle_v0.1.3.zip",
    {"model.pt": "27952767e2e154e3b0ee65defc5aed38", "model.ts": "97746870fe591f69ac09827175b00675"},
]


class TestDownload(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    @skip_if_quick
    def test_github_download_bundle(self, bundle_name, version):
        bundle_files = ["model.pt", "model.ts", "network.json", "test_output.pt", "test_input.pt"]
        repo = "Project-MONAI/MONAI-extra-test-data/0.8.1"
        hash_val = "a131d39a0af717af32d19e565b434928"
        with skip_if_downloading_fails():
            # download a whole bundle from github releases
            with tempfile.TemporaryDirectory() as tempdir:
                cmd = ["coverage", "run", "-m", "monai.bundle", "download", "--name", bundle_name, "--source", "github"]
                cmd += ["--bundle_dir", tempdir, "--repo", repo]
                if version is not None:
                    cmd += ["--version", version]
                command_line_tests(cmd)
                for file in bundle_files:
                    file_path = os.path.join(tempdir, "test_bundle", file)
                    self.assertTrue(os.path.exists(file_path))
                    if file == "network.json":
                        self.assertTrue(check_hash(filepath=file_path, val=hash_val))

    @parameterized.expand([TEST_CASE_3])
    @skip_if_quick
    def test_url_download_bundle(self, bundle_files, bundle_name, url, hash_val):
        with skip_if_downloading_fails():
            # download a single file from url, also use `args_file`
            with tempfile.TemporaryDirectory() as tempdir:
                def_args = {"name": bundle_name, "bundle_dir": tempdir, "url": ""}
                def_args_file = os.path.join(tempdir, "def_args.json")
                parser = ConfigParser()
                parser.export_config_file(config=def_args, filepath=def_args_file)
                cmd = ["coverage", "run", "-m", "monai.bundle", "download", "--args_file", def_args_file]
                cmd += ["--url", url, "--source", "github"]
                command_line_tests(cmd)
                for file in bundle_files:
                    file_path = os.path.join(tempdir, bundle_name, file)
                    self.assertTrue(os.path.exists(file_path))
                if file == "network.json":
                    self.assertTrue(check_hash(filepath=file_path, val=hash_val))

    @parameterized.expand([TEST_CASE_4])
    @skip_if_quick
    @skipUnless(has_huggingface_hub, "Requires `huggingface_hub`.")
    def test_hf_hub_download_bundle(self, bundle_files, bundle_name, repo):
        with skip_if_downloading_fails():
            with tempfile.TemporaryDirectory() as tempdir:
                cmd = [
                    "coverage",
                    "run",
                    "-m",
                    "monai.bundle",
                    "download",
                    "--name",
                    bundle_name,
                    "--source",
                    "huggingface_hub",
                ]
                cmd += ["--bundle_dir", tempdir, "--repo", repo, "--progress", "False"]
                command_line_tests(cmd)
                for file in bundle_files:
                    file_path = os.path.join(tempdir, bundle_name, file)
                    self.assertTrue(os.path.exists(file_path))

    @parameterized.expand([TEST_CASE_5])
    @skip_if_quick
    def test_monaihosting_url_download_bundle(self, bundle_files, bundle_name, url):
        with skip_if_downloading_fails():
            # download a single file from url, also use `args_file`
            with tempfile.TemporaryDirectory() as tempdir:
                def_args = {"name": bundle_name, "bundle_dir": tempdir, "url": ""}
                def_args_file = os.path.join(tempdir, "def_args.json")
                parser = ConfigParser()
                parser.export_config_file(config=def_args, filepath=def_args_file)
                cmd = ["coverage", "run", "-m", "monai.bundle", "download", "--args_file", def_args_file]
                cmd += ["--url", url, "--progress", "False"]
                command_line_tests(cmd)
                for file in bundle_files:
                    file_path = os.path.join(tempdir, bundle_name, file)
                    self.assertTrue(os.path.exists(file_path))

    @parameterized.expand([TEST_CASE_5])
    @skip_if_quick
    def test_ngc_private_source_download_bundle(self, bundle_files, bundle_name, _url):
        with skip_if_downloading_fails():
            # download a single file from url, also use `args_file`
            with tempfile.TemporaryDirectory() as tempdir:
                def_args = {"name": bundle_name, "bundle_dir": tempdir}
                def_args_file = os.path.join(tempdir, "def_args.json")
                parser = ConfigParser()
                parser.export_config_file(config=def_args, filepath=def_args_file)
                cmd = ["coverage", "run", "-m", "monai.bundle", "download", "--args_file", def_args_file]
                cmd += ["--progress", "False", "--source", "ngc_private"]
                command_line_tests(cmd)
                for file in bundle_files:
                    file_path = os.path.join(tempdir, bundle_name, file)
                    self.assertTrue(os.path.exists(file_path))

    @parameterized.expand([TEST_CASE_6])
    @skip_if_quick
    def test_monaihosting_source_download_bundle(self, bundle_files, bundle_name, version):
        with skip_if_downloading_fails():
            # download a single file from url, also use `args_file`
            with tempfile.TemporaryDirectory() as tempdir:
                def_args = {"name": bundle_name, "bundle_dir": tempdir, "version": version}
                def_args_file = os.path.join(tempdir, "def_args.json")
                parser = ConfigParser()
                parser.export_config_file(config=def_args, filepath=def_args_file)
                cmd = ["coverage", "run", "-m", "monai.bundle", "download", "--args_file", def_args_file]
                cmd += ["--progress", "False", "--source", "monaihosting"]
                command_line_tests(cmd)
                for file in bundle_files:
                    file_path = os.path.join(tempdir, bundle_name, file)
                    self.assertTrue(os.path.exists(file_path))

    @patch("monai.bundle.scripts.get_versions", return_value={"version": "1.2"})
    def test_examine_monai_version(self, mock_get_versions):
        self.assertTrue(_examine_monai_version("1.1")[0])  # Should return True, compatible
        self.assertTrue(_examine_monai_version("1.2rc1")[0])  # Should return True, compatible
        self.assertFalse(_examine_monai_version("1.3")[0])  # Should return False, not compatible

    @patch("monai.bundle.scripts.get_versions", return_value={"version": "1.2rc1"})
    def test_examine_monai_version_rc(self, mock_get_versions):
        self.assertTrue(_examine_monai_version("1.2")[0])  # Should return True, compatible
        self.assertFalse(_examine_monai_version("1.3")[0])  # Should return False, not compatible

    def test_list_latest_versions(self):
        """Test listing of the latest versions."""
        data = {
            "modelVersions": [
                {"createdDate": "2021-01-01", "versionId": "1.0"},
                {"createdDate": "2021-01-02", "versionId": "1.1"},
                {"createdDate": "2021-01-03", "versionId": "1.2"},
            ]
        }
        self.assertEqual(_list_latest_versions(data), ["1.2", "1.1", "1.0"])
        self.assertEqual(_list_latest_versions(data, max_versions=2), ["1.2", "1.1"])
        data = {
            "modelVersions": [
                {"createdDate": "2021-01-01", "versionId": "1.0"},
                {"createdDate": "2021-01-02", "versionId": "1.1"},
            ]
        }
        self.assertEqual(_list_latest_versions(data), ["1.1", "1.0"])

    @skip_if_quick
    @patch("monai.bundle.scripts.get_versions", return_value={"version": "1.2"})
    def test_download_monaihosting(self, mock_get_versions):
        """Test checking MONAI version from a metadata file."""
        with patch("monai.bundle.scripts.logger") as mock_logger:
            with tempfile.TemporaryDirectory() as tempdir:
                download(name="spleen_ct_segmentation", bundle_dir=tempdir, source="monaihosting")
                # Should have a warning message because the latest version is using monai > 1.2
                mock_logger.warning.assert_called_once()

    @skip_if_quick
    @patch("monai.bundle.scripts.get_versions", return_value={"version": "1.3"})
    def test_download_ngc(self, mock_get_versions):
        """Test checking MONAI version from a metadata file."""
        with patch("monai.bundle.scripts.logger") as mock_logger:
            with tempfile.TemporaryDirectory() as tempdir:
                download(name="spleen_ct_segmentation", bundle_dir=tempdir, source="ngc")
                mock_logger.warning.assert_not_called()


@skip_if_no_cuda
class TestLoad(unittest.TestCase):
    @parameterized.expand([TEST_CASE_7])
    @skip_if_quick
    def test_load_weights(self, bundle_files, bundle_name, repo, device, model_file):
        with skip_if_downloading_fails():
            # download bundle, and load weights from the downloaded path
            with tempfile.TemporaryDirectory() as tempdir:
                # load weights
                weights = load(
                    name=bundle_name,
                    model_file=model_file,
                    bundle_dir=tempdir,
                    repo=repo,
                    source="github",
                    progress=False,
                    device=device,
                    return_state_dict=True,
                )
                # prepare network
                with open(os.path.join(tempdir, bundle_name, bundle_files[2])) as f:
                    net_args = json.load(f)["network_def"]
                model_name = net_args["_target_"]
                del net_args["_target_"]
                model = getattr(nets, model_name)(**net_args)
                model.to(device)
                model.load_state_dict(weights)
                model.eval()

                # prepare data and test
                input_tensor = torch.load(os.path.join(tempdir, bundle_name, bundle_files[4]), map_location=device)
                output = model.forward(input_tensor)
                expected_output = torch.load(os.path.join(tempdir, bundle_name, bundle_files[3]), map_location=device)
                assert_allclose(output, expected_output, atol=1e-4, rtol=1e-4, type_test=False)

                # load instantiated model directly and test, since the bundle has been downloaded,
                # there is no need to input `repo`
                _model_2 = getattr(nets, model_name)(**net_args)
                model_2 = load(
                    name=bundle_name,
                    model=_model_2,
                    model_file=model_file,
                    bundle_dir=tempdir,
                    progress=False,
                    device=device,
                    source="github",
                    return_state_dict=False,
                )
                model_2.eval()
                output_2 = model_2.forward(input_tensor)
                assert_allclose(output_2, expected_output, atol=1e-4, rtol=1e-4, type_test=False)

                # test compatibility with return_state_dict=True.
                model_3 = load(
                    name=bundle_name,
                    model_file=model_file,
                    bundle_dir=tempdir,
                    progress=False,
                    device=device,
                    net_name=model_name,
                    source="github",
                    return_state_dict=False,
                    **net_args,
                )
                model_3.eval()
                output_3 = model_3.forward(input_tensor)
                assert_allclose(output_3, expected_output, atol=1e-4, rtol=1e-4, type_test=False)

    @parameterized.expand([TEST_CASE_8])
    @skip_if_quick
    def test_load_weights_with_net_override(self, bundle_name, device, net_override):
        with skip_if_downloading_fails():
            # download bundle, and load weights from the downloaded path
            with tempfile.TemporaryDirectory() as tempdir:
                # load weights
                model = load(
                    name=bundle_name,
                    bundle_dir=tempdir,
                    source="monaihosting",
                    progress=False,
                    device=device,
                    return_state_dict=False,
                )

                # prepare data and test
                input_tensor = torch.rand(1, 1, 96, 96, 96).to(device)
                output = model(input_tensor)
                model_path = f"{tempdir}/spleen_ct_segmentation/models/model.pt"
                workflow = create_workflow(
                    config_file=f"{tempdir}/spleen_ct_segmentation/configs/train.json", workflow_type="train"
                )
                expected_model = workflow.network_def.to(device)
                expected_model.load_state_dict(torch.load(model_path))
                expected_output = expected_model(input_tensor)
                assert_allclose(output, expected_output, atol=1e-4, rtol=1e-4, type_test=False)

                # using net_override to override kwargs in network directly
                model_2 = load(
                    name=bundle_name,
                    bundle_dir=tempdir,
                    source="monaihosting",
                    progress=False,
                    device=device,
                    return_state_dict=False,
                    net_override=net_override,
                )

                # prepare data and test
                input_tensor = torch.rand(1, 1, 96, 96, 96).to(device)
                output = model_2(input_tensor)
                expected_shape = (1, 5, 96, 96, 96)
                np.testing.assert_equal(output.shape, expected_shape)

    @parameterized.expand([TEST_CASE_9])
    @skip_if_quick
    @SkipIfBeforePyTorchVersion((1, 7, 1))
    def test_load_ts_module(self, bundle_files, bundle_name, version, repo, device, model_file):
        with skip_if_downloading_fails():
            # load ts module
            with tempfile.TemporaryDirectory() as tempdir:
                # load ts module
                model_ts, metadata, extra_file_dict = load(
                    name=bundle_name,
                    version=version,
                    model_file=model_file,
                    load_ts_module=True,
                    bundle_dir=tempdir,
                    repo=repo,
                    progress=False,
                    device=device,
                    source="github",
                    config_files=("network.json",),
                )

                # prepare and test ts
                input_tensor = torch.load(os.path.join(tempdir, bundle_name, bundle_files[1]), map_location=device)
                output = model_ts.forward(input_tensor)
                expected_output = torch.load(os.path.join(tempdir, bundle_name, bundle_files[0]), map_location=device)
                assert_allclose(output, expected_output, atol=1e-4, rtol=1e-4, type_test=False)
                # test metadata
                self.assertTrue(metadata["pytorch_version"] == "1.7.1")
                # test extra_file_dict
                self.assertTrue("network.json" in extra_file_dict.keys())


class TestDownloadLargefiles(unittest.TestCase):
    @parameterized.expand([TEST_CASE_10])
    @skip_if_quick
    def test_url_download_large_files(self, bundle_files, bundle_name, url, hash_val):
        with skip_if_downloading_fails():
            # download a single file from url, also use `args_file`
            with tempfile.TemporaryDirectory() as tempdir:
                def_args = {"name": bundle_name, "bundle_dir": tempdir, "url": ""}
                def_args_file = os.path.join(tempdir, "def_args.json")
                parser = ConfigParser()
                parser.export_config_file(config=def_args, filepath=def_args_file)
                cmd = ["coverage", "run", "-m", "monai.bundle", "download", "--args_file", def_args_file]
                cmd += ["--url", url, "--source", "github"]
                command_line_tests(cmd)
                for file in bundle_files:
                    file_path = os.path.join(tempdir, bundle_name, file)
                    print(file_path)
                    self.assertTrue(os.path.exists(file_path))

                # download large files
                bundle_path = os.path.join(tempdir, bundle_name)
                cmd = ["coverage", "run", "-m", "monai.bundle", "download_large_files", "--bundle_path", bundle_path]
                command_line_tests(cmd)
                for file in ["model.pt", "model.ts"]:
                    file_path = os.path.join(tempdir, bundle_name, f"models/{file}")
                    self.assertTrue(check_hash(filepath=file_path, val=hash_val[file]))


if __name__ == "__main__":
    unittest.main()
