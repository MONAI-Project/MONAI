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

from monai.data import is_supported_format

TEST_CASE_1 = [{"filename": "testfile.nii.gz", "suffixes": ["nii", "nii.gz"]}, True]

TEST_CASE_2 = [{"filename": "./testfile.nii.gz", "suffixes": ["nii", "nii.gz"]}, True]

TEST_CASE_3 = [{"filename": "./test.data/file.nii.gz", "suffixes": ["nii", "nii.gz"]}, True]

TEST_CASE_4 = [{"filename": "./test.data/file.nii", "suffixes": ["nii", "nii.gz"]}, True]

TEST_CASE_5 = [{"filename": "C:\\documents\\testfile.nii.gz", "suffixes": ["nii", "nii.gz"]}, True]

TEST_CASE_6 = [{"filename": "1.3.12.2.1107.5.4.4.145.nii.gz", "suffixes": ["nii.gz"]}, True]

TEST_CASE_7 = [{"filename": "test.PNG", "suffixes": ["bmp", "png"]}, True]


class TestIsSupportedFormat(unittest.TestCase):

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4, TEST_CASE_5, TEST_CASE_6, TEST_CASE_7])
    def test_value(self, input_param, result):
        self.assertEqual(is_supported_format(**input_param), result)


if __name__ == "__main__":
    unittest.main()
