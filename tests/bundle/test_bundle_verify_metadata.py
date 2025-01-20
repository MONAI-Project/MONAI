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

from monai.bundle import ConfigParser, verify_metadata
from tests.util import command_line_tests, download_url_or_skip_test, skip_if_windows, testing_data_config

TESTS_DIR = Path(__file__).parents[1]
SCHEMA_FILE = os.path.join(TESTS_DIR, "testing_data", "schema.json")

TEST_CASE_1 = [os.path.join(TESTS_DIR, "testing_data", "metadata.json"), SCHEMA_FILE]


@skip_if_windows
class TestVerifyMetaData(unittest.TestCase):
    def setUp(self):
        self.config = testing_data_config("configs", "test_meta_file")
        download_url_or_skip_test(
            url=self.config["url"],
            filepath=SCHEMA_FILE,
            hash_val=self.config.get("hash_val"),
            hash_type=self.config.get("hash_type", "sha256"),
        )

    @parameterized.expand([TEST_CASE_1])
    def test_verify(self, meta_file, schema_file):
        with tempfile.TemporaryDirectory() as tempdir:
            def_args = {"meta_file": "will be replaced by `meta_file` arg"}
            def_args_file = os.path.join(tempdir, "def_args.json")
            ConfigParser.export_config_file(config=def_args, filepath=def_args_file)

            cmd = ["coverage", "run", "-m", "monai.bundle", "verify_metadata", "--meta_file", meta_file]
            cmd += ["--filepath", schema_file, "--hash_val", self.config["hash_val"], "--args_file", def_args_file]
            command_line_tests(cmd)

    def test_verify_error(self):
        with tempfile.TemporaryDirectory() as tempdir:
            filepath = os.path.join(tempdir, "schema.json")
            metafile = os.path.join(tempdir, "metadata.json")
            meta_dict = {"schema": self.config["url"], "wrong_meta": "wrong content"}
            with open(metafile, "w") as f:
                json.dump(meta_dict, f)

            with self.assertRaises(ValueError):
                verify_metadata(meta_file=metafile, filepath=filepath)


if __name__ == "__main__":
    unittest.main()
