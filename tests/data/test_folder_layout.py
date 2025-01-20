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
from pathlib import Path

from parameterized import parameterized

from monai.data.folder_layout import FolderLayout

TEST_CASES = [
    ({"output_dir": ""}, {}, "subject"),
    ({"output_dir": Path(".")}, {}, "subject"),
    ({"output_dir": Path(".")}, {"idx": 1}, "subject_1"),
    (dict(output_dir=Path("/test_run_1"), extension=".seg", makedirs=False), {}, "/test_run_1/subject.seg"),
    (dict(output_dir=Path("/test_run_1"), extension=None, makedirs=False), {}, "/test_run_1/subject"),
    (
        dict(output_dir=Path("/test_run_1"), postfix="seg", extension=".test", makedirs=False),
        {},  # using the default subject name
        "/test_run_1/subject_seg.test",
    ),
    (
        dict(output_dir=Path("/test_run_1"), postfix="seg", extension=".test", makedirs=False),
        {"subject": "test.abc"},
        "/test_run_1/test_seg.test",  # subject's extension is ignored
    ),
    (
        dict(output_dir=Path("/test_run_1/dest/test1/"), data_root_dir="/test_run", makedirs=False),
        {"subject": "/test_run/source/test.abc"},
        "/test_run_1/dest/test1/source/test",  # preserves the structure from `subject`
    ),
    (
        dict(output_dir=Path("/test_run_1/dest/test1/"), makedirs=False),
        {"subject": "/test_run/source/test.abc"},
        "/test_run_1/dest/test1/test",  # data_root_dir used
    ),
    (
        dict(output_dir=Path("/test_run_1/dest/test1/"), makedirs=False),
        {"subject": "/test_run/source/test.abc", "key": "value"},
        "/test_run_1/dest/test1/test_key-value",  # data_root_dir used
    ),
    (
        dict(output_dir=Path("/test_run_1/"), postfix="seg", extension=".nii", makedirs=False),
        dict(subject=Path("Sub-A"), idx="00", modality="T1"),
        "/test_run_1/Sub-A_seg_00_modality-T1.nii",  # test the code example
    ),
]


class TestFolderLayout(unittest.TestCase):

    @parameterized.expand(TEST_CASES)
    def test_value(self, con_params, f_params, expected):
        fname = FolderLayout(**con_params).filename(**f_params)
        self.assertEqual(Path(fname), Path(expected))

    def test_mkdir(self):
        """mkdir=True should create the directory if it does not exist."""
        with tempfile.TemporaryDirectory() as tempdir:
            output_tmp = os.path.join(tempdir, "output")
            FolderLayout(output_tmp, makedirs=True).filename("subject_test", "001")
            self.assertTrue(os.path.exists(os.path.join(output_tmp)))


if __name__ == "__main__":
    unittest.main()
