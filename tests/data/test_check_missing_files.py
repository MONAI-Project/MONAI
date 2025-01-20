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

import nibabel as nib
import numpy as np

from monai.data import check_missing_files


class TestCheckMissingFiles(unittest.TestCase):

    def test_content(self):
        test_image = nib.Nifti1Image(np.random.randint(0, 2, size=[128, 128, 128]).astype(float), np.eye(4))
        with tempfile.TemporaryDirectory() as tempdir:
            nib.save(test_image, os.path.join(tempdir, "test_image1.nii.gz"))
            nib.save(test_image, os.path.join(tempdir, "test_label1.nii.gz"))
            nib.save(test_image, os.path.join(tempdir, "test_extra1.nii.gz"))
            nib.save(test_image, os.path.join(tempdir, "test_image2.nii.gz"))

            datalist = [
                {
                    "image": os.path.join(tempdir, "test_image1.nii.gz"),
                    "label": [os.path.join(tempdir, "test_label1.nii.gz"), os.path.join(tempdir, "test_extra1.nii.gz")],
                },
                {
                    "image": Path(os.path.join(tempdir, "test_image2.nii.gz")),
                    "label": Path(os.path.join(tempdir, "test_label_missing.nii.gz")),
                },
            ]

            missings = check_missing_files(datalist=datalist, keys=["image", "label"])
            self.assertEqual(len(missings), 1)
            self.assertEqual(str(missings[0]), os.path.join(tempdir, "test_label_missing.nii.gz"))

            # test with missing key and relative path
            datalist = [{"image": "test_image1.nii.gz", "label": "test_label_missing.nii.gz"}]
            missings = check_missing_files(
                datalist=datalist, keys=["image", "label", "test"], root_dir=tempdir, allow_missing_keys=True
            )
            self.assertEqual(f"{missings[0]}", os.path.join(tempdir, "test_label_missing.nii.gz"))


if __name__ == "__main__":
    unittest.main()
