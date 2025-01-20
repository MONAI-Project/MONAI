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
from urllib.error import HTTPError

from monai.apps.utils import download_url

YANDEX_MODEL_URL = (
    "https://cloud-api.yandex.net/v1/disk/public/resources/download?"
    "public_key=https%3A%2F%2Fdisk.yandex.ru%2Fd%2Fxs0gzlj2_irgWA"
)
YANDEX_MODEL_FLAWED_URL = (
    "https://cloud-api.yandex.net/v1/disk/public/resources/download?"
    "public_key=https%3A%2F%2Fdisk.yandex.ru%2Fd%2Fxs0gzlj2_irgWA-url-with-error"
)


class TestDownloadUrlYandex(unittest.TestCase):

    @unittest.skip("data source unstable")
    def test_verify(self):
        with tempfile.TemporaryDirectory() as tempdir:
            download_url(url=YANDEX_MODEL_URL, filepath=os.path.join(tempdir, "model.pt"))

    def test_verify_error(self):
        with tempfile.TemporaryDirectory() as tempdir:
            with self.assertRaises(HTTPError):
                download_url(url=YANDEX_MODEL_FLAWED_URL, filepath=os.path.join(tempdir, "model.pt"))


if __name__ == "__main__":
    unittest.main()
