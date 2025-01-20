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

import tempfile
import unittest
from unittest.case import skipUnless
from unittest.mock import patch

from parameterized import parameterized

from monai.bundle import push_to_hf_hub
from monai.utils import optional_import
from tests.util import skip_if_quick

huggingface_hub, has_huggingface_hub = optional_import("huggingface_hub")

TEST_CASE_1 = ["monai-test/test_bundle_push", "test_bundle"]


class TestPushToHuggingFaceHub(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1])
    @skip_if_quick
    @skipUnless(has_huggingface_hub, "Requires `huggingface_hub` package.")
    @patch.object(huggingface_hub.HfApi, "create_repo")
    @patch.object(huggingface_hub.HfApi, "upload_folder")
    @patch.object(huggingface_hub.HfApi, "create_tag")
    def test_push_to_huggingface_hub(self, repo, bundle_name, test_createrepo, test_uploadfolder, test_createtag):
        test_uploadfolder.return_value = "https://hf.co/repo/test"
        with tempfile.TemporaryDirectory() as tempdir:
            repo_url = push_to_hf_hub(repo, bundle_name, tempdir)
            self.assertEqual("https://hf.co/repo/test", repo_url)
