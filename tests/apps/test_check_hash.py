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

import numpy as np
from parameterized import parameterized

from monai.apps import check_hash

TEST_CASE_1 = ["b94716452086a054208395e8c9d1ae2a", "md5", True]

TEST_CASE_2 = ["abcdefg", "md5", False]

TEST_CASE_3 = [None, "md5", True]

TEST_CASE_4 = [None, "sha1", True]

TEST_CASE_5 = ["b4dc3c246b298eae37cefdfdd2a50b091ffd5e69", "sha1", True]


class TestCheckMD5(unittest.TestCase):

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4, TEST_CASE_5])
    def test_result(self, md5_value, t, expected_result):
        test_image = np.ones((5, 5, 3))
        with tempfile.TemporaryDirectory() as tempdir:
            filename = os.path.join(tempdir, "test_file.png")
            test_image.tofile(filename)

            result = check_hash(filename, md5_value, hash_type=t)
            self.assertTrue(result == expected_result)

    def test_hash_type_error(self):
        with self.assertRaises(ValueError):
            with tempfile.TemporaryDirectory() as tempdir:
                check_hash(tempdir, "test_hash", "test_type")


if __name__ == "__main__":
    unittest.main()
