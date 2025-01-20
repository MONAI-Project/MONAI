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
from pathlib import Path
from urllib.error import ContentTooShortError, HTTPError

from parameterized import parameterized

from monai.apps import download_and_extract, download_url, extractall
from tests.utils import SkipIfNoModule, skip_if_downloading_fails, skip_if_quick, testing_data_config


@SkipIfNoModule("requests")
class TestDownloadAndExtract(unittest.TestCase):
    @skip_if_quick
    def test_actions(self):
        testing_dir = Path(__file__).parents[1] / "testing_data"
        config_dict = testing_data_config("images", "mednist")
        url = config_dict["url"]
        filepath = Path(testing_dir) / "MedNIST.tar.gz"
        output_dir = Path(testing_dir)
        hash_val, hash_type = config_dict["hash_val"], config_dict["hash_type"]
        with skip_if_downloading_fails():
            download_and_extract(url, filepath, output_dir, hash_val=hash_val, hash_type=hash_type)
            download_and_extract(url, filepath, output_dir, hash_val=hash_val, hash_type=hash_type)

        wrong_md5 = "0"
        with self.assertLogs(logger="monai.apps", level="ERROR"):
            try:
                download_url(url, filepath, wrong_md5)
            except (ContentTooShortError, HTTPError, RuntimeError) as e:
                if isinstance(e, RuntimeError):
                    # FIXME: skip MD5 check as current downloading method may fail
                    self.assertTrue(str(e).startswith("md5 check"))
                return  # skipping this test due the network connection errors

        try:
            extractall(filepath, output_dir, wrong_md5)
        except RuntimeError as e:
            self.assertTrue(str(e).startswith("md5 check"))

    @skip_if_quick
    @parameterized.expand((("icon", "tar"), ("favicon", "zip")))
    def test_default(self, key, file_type):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with skip_if_downloading_fails():
                img_spec = testing_data_config("images", key)
                download_and_extract(
                    img_spec["url"],
                    output_dir=tmp_dir,
                    hash_val=img_spec["hash_val"],
                    hash_type=img_spec["hash_type"],
                    file_type=file_type,
                )


if __name__ == "__main__":
    unittest.main()
