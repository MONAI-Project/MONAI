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
import sys
import tempfile
import unittest

import torch
from parameterized import parameterized

from monai.apps import check_hash
from monai.apps.mmars import MODEL_DESC, load_from_mmar
from monai.bundle import download, load
from monai.config import print_debug_info
from monai.networks.utils import copy_model_state
from tests.util import assert_allclose, skip_if_downloading_fails, skip_if_quick, skip_if_windows

TEST_CASE_NGC_1 = [
    "spleen_ct_segmentation",
    "0.3.7",
    None,
    "monai_spleen_ct_segmentation",
    "models/model.pt",
    "b418a2dc8672ce2fd98dc255036e7a3d",
]
TEST_CASE_NGC_2 = [
    "monai_spleen_ct_segmentation",
    "0.3.7",
    "monai_",
    "spleen_ct_segmentation",
    "models/model.pt",
    "b418a2dc8672ce2fd98dc255036e7a3d",
]

TESTCASE_WEIGHTS = {
    "key": "model.0.conv.unit0.adn.N.bias",
    "value": torch.tensor(
        [
            -0.0705,
            -0.0937,
            -0.0422,
            -0.2068,
            0.1023,
            -0.2007,
            -0.0883,
            0.0018,
            -0.1719,
            0.0116,
            0.0285,
            -0.0044,
            0.1223,
            -0.1287,
            -0.1858,
            0.0460,
        ]
    ),
}


@skip_if_windows
class TestNgcBundleDownload(unittest.TestCase):
    @parameterized.expand([TEST_CASE_NGC_1, TEST_CASE_NGC_2])
    @skip_if_quick
    def test_ngc_download_bundle(self, bundle_name, version, remove_prefix, download_name, file_path, hash_val):
        with skip_if_downloading_fails():
            with tempfile.TemporaryDirectory() as tempdir:
                download(
                    name=bundle_name, source="ngc", version=version, bundle_dir=tempdir, remove_prefix=remove_prefix
                )
                full_file_path = os.path.join(tempdir, download_name, file_path)
                self.assertTrue(os.path.exists(full_file_path))
                self.assertTrue(check_hash(filepath=full_file_path, val=hash_val))

                model = load(
                    name=bundle_name,
                    source="ngc",
                    version=version,
                    bundle_dir=tempdir,
                    remove_prefix=remove_prefix,
                    return_state_dict=False,
                )
                assert_allclose(
                    model.state_dict()[TESTCASE_WEIGHTS["key"]],
                    TESTCASE_WEIGHTS["value"],
                    atol=1e-4,
                    rtol=1e-4,
                    type_test=False,
                )


@unittest.skip("deprecating mmar tests")
class TestAllDownloadingMMAR(unittest.TestCase):
    def setUp(self):
        print_debug_info()
        self.test_dir = "./"

    @parameterized.expand((item,) for item in MODEL_DESC)
    def test_loading_mmar(self, item):
        if item["name"] == "clara_pt_self_supervised_learning_segmentation":  # test the byow model
            default_model_file = os.path.join("ssl_models_2gpu", "best_metric_model.pt")
            pretrained_weights = load_from_mmar(
                item=item["name"],
                mmar_dir="./",
                map_location="cpu",
                api=True,
                model_file=default_model_file,
                weights_only=True,
            )
            pretrained_weights = {k.split(".", 1)[1]: v for k, v in pretrained_weights["state_dict"].items()}
            sys.path.append(os.path.join(f"{item['name']}", "custom"))  # custom model folder
            from vit_network import ViTAutoEnc  # pylint: disable=E0401

            model = ViTAutoEnc(
                in_channels=1,
                img_size=(96, 96, 96),
                patch_size=(16, 16, 16),
                proj_type="conv",
                hidden_size=768,
                mlp_dim=3072,
            )
            _, loaded, not_loaded = copy_model_state(model, pretrained_weights)
            self.assertTrue(len(loaded) > 0 and len(not_loaded) == 0)
            return
        if item["name"] == "clara_pt_fed_learning_brain_tumor_mri_segmentation":
            default_model_file = os.path.join("models", "server", "best_FL_global_model.pt")
        else:
            default_model_file = None
        pretrained_model = load_from_mmar(
            item=item["name"], mmar_dir="./", map_location="cpu", api=True, model_file=default_model_file
        )
        self.assertTrue(isinstance(pretrained_model, torch.nn.Module))

    def tearDown(self):
        print(os.listdir(self.test_dir))


if __name__ == "__main__":
    unittest.main()
