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
from copy import deepcopy
from pathlib import Path

import numpy as np
from parameterized import parameterized

from monai import config as monai_config
from monai.bundle import ConfigParser
from monai.data import CacheDataset, DataLoader, MetaTensor, decollate_batch
from monai.data.utils import TraceKeys
from monai.transforms import InvertD, SaveImageD, reset_ops_id
from monai.utils import optional_import, set_determinism
from tests.util import assert_allclose, download_url_or_skip_test, testing_data_config

nib, has_nib = optional_import("nibabel")
TINY_DIFF = 0.1

keys = ("img", "seg")
key, key_1 = "MNI152_T1_2mm", "MNI152_T1_2mm_strucseg"
TESTS_PATH = Path(__file__).parents[1]
FILE_PATH = os.path.join(TESTS_PATH, "testing_data", f"{key}.nii.gz")
FILE_PATH_1 = os.path.join(TESTS_PATH, "testing_data", f"{key_1}.nii.gz")
TEST_CASES = os.path.join(TESTS_PATH, "testing_data", "transform_metatensor_cases.yaml")


@unittest.skipUnless(has_nib, "Requires nibabel package.")
class TestMetaTensorIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        for k, n in ((key, FILE_PATH), (key_1, FILE_PATH_1)):
            config = testing_data_config("images", f"{k}")
            download_url_or_skip_test(filepath=n, **config)
        cls.files = [{keys[0]: x, keys[1]: y} for (x, y) in [[FILE_PATH, FILE_PATH_1]] * 4]

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        set_determinism(None)

    @parameterized.expand(["TEST_CASE_1", "TEST_CASE_2", "TEST_CASE_3"])
    def test_transforms(self, case_id):
        set_determinism(2022)
        config = ConfigParser()
        config.read_config(TEST_CASES)
        config["input_keys"] = keys
        test_case = config.get_parsed_content(id=case_id, instantiate=True, lazy=False)  # transform instance

        dataset = CacheDataset(self.files, transform=test_case)
        loader = DataLoader(dataset, batch_size=3, shuffle=True)
        for x in loader:
            self.assertIsInstance(x[keys[0]], MetaTensor)
            self.assertIsInstance(x[keys[1]], MetaTensor)
            out = decollate_batch(x)  # decollate every batch should work

        # test forward patches
        loaded = out[0]
        if not monai_config.USE_META_DICT:
            self.assertEqual(len(loaded), len(keys))
        else:
            self.assertNotEqual(len(loaded), len(keys))
        img, seg = loaded[keys[0]], loaded[keys[1]]
        expected = config.get_parsed_content(id=f"{case_id}_answer", instantiate=True)  # expected results
        self.assertEqual(expected["load_shape"], list(x[keys[0]].shape))
        assert_allclose(expected["affine"], img.affine, type_test=False, atol=TINY_DIFF, rtol=TINY_DIFF)
        assert_allclose(expected["affine"], seg.affine, type_test=False, atol=TINY_DIFF, rtol=TINY_DIFF)
        test_cls = [type(x).__name__ for x in test_case.transforms]
        tracked_cls = [x[TraceKeys.CLASS_NAME] for x in img.applied_operations]
        self.assertTrue(len(tracked_cls) <= len(test_cls))  # tracked items should  be no more than the compose items.
        with tempfile.TemporaryDirectory() as tempdir:  # test writer
            SaveImageD(keys, resample=False, output_dir=tempdir, output_postfix=case_id)(loaded)
        test_data = reset_ops_id(deepcopy(loaded))
        for val in test_data.values():
            if isinstance(val, MetaTensor) and val.applied_operations:
                self.assertEqual(val.applied_operations[-1][TraceKeys.ID], TraceKeys.NONE)

        # test inverse
        inv = InvertD(keys, orig_keys=keys, transform=test_case, nearest_interp=True)
        out = inv(loaded)
        img, seg = out[keys[0]], out[keys[1]]
        assert_allclose(expected["inv_affine"], img.affine, type_test=False, atol=TINY_DIFF, rtol=TINY_DIFF)
        assert_allclose(expected["inv_affine"], seg.affine, type_test=False, atol=TINY_DIFF, rtol=TINY_DIFF)
        self.assertFalse(img.applied_operations)
        self.assertFalse(seg.applied_operations)
        assert_allclose(expected["inv_shape"], img.shape, type_test=False, atol=TINY_DIFF, rtol=TINY_DIFF)
        assert_allclose(expected["inv_shape"], seg.shape, type_test=False, atol=TINY_DIFF, rtol=TINY_DIFF)
        with tempfile.TemporaryDirectory() as tempdir:  # test writer
            SaveImageD(keys, resample=False, output_dir=tempdir, output_postfix=case_id)(out)
            seg_file = os.path.join(tempdir, key_1, f"{key_1}_{case_id}.nii.gz")
            segout = nib.load(seg_file).get_fdata()
            segin = nib.load(FILE_PATH_1).get_fdata()
            ndiff = np.sum(np.abs(segout - segin) > 0)
            total = np.prod(segout.shape)
        self.assertTrue(ndiff / total < 0.4, f"{ndiff / total}")


if __name__ == "__main__":
    unittest.main()
