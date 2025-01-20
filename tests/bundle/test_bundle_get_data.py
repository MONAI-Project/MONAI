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

import unittest

from parameterized import parameterized

from monai.bundle import get_all_bundles_list, get_bundle_info, get_bundle_versions
from monai.utils import optional_import
from tests.util import SkipIfNoModule, skip_if_downloading_fails, skip_if_quick, skip_if_windows

requests, _ = optional_import("requests")

TEST_CASE_1 = [{"bundle_name": "brats_mri_segmentation", "tag": "hosting_storage_v1"}]

TEST_CASE_2 = [
    {"bundle_name": "spleen_ct_segmentation", "version": "0.1.0", "auth_token": None, "tag": "hosting_storage_v1"}
]

TEST_CASE_3 = [{"tag": "hosting_storage_v1"}]

TEST_CASE_4 = [{"tag": "dev"}]

TEST_CASE_5 = [{"bundle_name": "brats_mri_segmentation", "tag": "dev"}]

TEST_CASE_6 = [{"bundle_name": "spleen_ct_segmentation", "version": "0.1.0", "auth_token": None, "tag": "dev"}]

TEST_CASE_FAKE_TOKEN_1 = [{"bundle_name": "spleen_ct_segmentation", "version": "0.1.0", "auth_token": "ghp_errortoken"}]

TEST_CASE_FAKE_TOKEN_2 = [
    {"bundle_name": "spleen_ct_segmentation", "version": "0.1.0", "auth_token": "ghp_errortoken", "tag": "dev"}
]


@skip_if_windows
@SkipIfNoModule("requests")
class TestGetBundleData(unittest.TestCase):
    @parameterized.expand([TEST_CASE_3, TEST_CASE_4])
    @skip_if_quick
    def test_get_all_bundles_list(self, params):
        with skip_if_downloading_fails():
            output = get_all_bundles_list(**params)
            self.assertIsInstance(output, list)
            self.assertIsInstance(output[0], tuple)
            self.assertTrue(len(output[0]) == 2)

    @parameterized.expand([TEST_CASE_1, TEST_CASE_5])
    @skip_if_quick
    def test_get_bundle_versions(self, params):
        with skip_if_downloading_fails():
            output = get_bundle_versions(**params)
            self.assertIsInstance(output, dict)
            self.assertIn("latest_version", output)
            self.assertIn("all_versions", output)
            self.assertIn("0.1.0", output["all_versions"])

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    @skip_if_quick
    def test_get_bundle_info(self, params):
        with skip_if_downloading_fails():
            output = get_bundle_info(**params)
            self.assertIsInstance(output, dict)
            for key in ["id", "name", "size", "download_count", "browser_download_url"]:
                self.assertTrue(key in output)

    @parameterized.expand([TEST_CASE_5, TEST_CASE_6])
    @skip_if_quick
    def test_get_bundle_info_monaihosting(self, params):
        with skip_if_downloading_fails():
            output = get_bundle_info(**params)
            self.assertIsInstance(output, dict)
            for key in ["name", "browser_download_url"]:
                self.assertTrue(key in output)

    @parameterized.expand([TEST_CASE_FAKE_TOKEN_1, TEST_CASE_FAKE_TOKEN_2])
    @skip_if_quick
    def test_fake_token(self, params):
        with skip_if_downloading_fails():
            with self.assertRaises(requests.exceptions.HTTPError):
                get_bundle_info(**params)


if __name__ == "__main__":
    unittest.main()
