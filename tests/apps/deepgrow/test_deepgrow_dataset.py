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
import shutil
import tempfile
import unittest

import nibabel as nib
import numpy as np
from parameterized import parameterized

from monai.apps.deepgrow.dataset import create_dataset
from monai.utils import set_determinism

TEST_CASE_1 = [{"dimension": 2, "pixdim": (1, 1)}, {"length": 3}, 9, 1]

TEST_CASE_2 = [{"dimension": 2, "pixdim": (1, 1), "limit": 1}, {"length": 3}, 3, 1]

TEST_CASE_3 = [{"dimension": 2, "pixdim": (1, 1)}, {"length": 1}, 3, 1]

TEST_CASE_4 = [{"dimension": 3, "pixdim": (1, 1, 1)}, {"length": 1}, 1, 1]

TEST_CASE_5 = [{"dimension": 3, "pixdim": (1, 1, 1)}, {"length": 1, "image_channel": 1}, 1, 1]

TEST_CASE_6 = [{"dimension": 2, "pixdim": (1, 1)}, {"length": 1, "image_channel": 1}, 3, 1]

TEST_CASE_7 = [
    {"dimension": 2, "pixdim": (1, 1), "label_key": None},
    {"length": 1, "image_channel": 1, "with_label": False},
    40,
    None,
]

TEST_CASE_8 = [
    {"dimension": 3, "pixdim": (1, 1, 1), "label_key": None},
    {"length": 1, "image_channel": 1, "with_label": False},
    1,
    None,
]


class TestCreateDataset(unittest.TestCase):

    def setUp(self):
        set_determinism(1)
        self.tempdir = tempfile.mkdtemp()

    def _create_data(self, length=1, image_channel=1, with_label=True):
        affine = np.eye(4)
        datalist = []
        for i in range(length):
            if image_channel == 1:
                image = np.random.randint(0, 2, size=(128, 128, 40))
            else:
                image = np.random.randint(0, 2, size=(128, 128, 40, image_channel))
            image_file = os.path.join(self.tempdir, f"image{i}.nii.gz")
            nib.save(nib.Nifti1Image(image.astype(float), affine), image_file)

            if with_label:
                # 3 slices has label
                label = np.zeros((128, 128, 40))
                label[0][1][0] = 1
                label[0][1][1] = 1
                label[0][0][2] = 1
                label[0][1][2] = 1
                label_file = os.path.join(self.tempdir, f"label{i}.nii.gz")
                nib.save(nib.Nifti1Image(label.astype(float), affine), label_file)
                datalist.append({"image": image_file, "label": label_file})
            else:
                datalist.append({"image": image_file})

        return datalist

    @parameterized.expand(
        [TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4, TEST_CASE_5, TEST_CASE_6, TEST_CASE_7, TEST_CASE_8]
    )
    def test_create_dataset(self, args, data_args, expected_length, expected_region):
        datalist = self._create_data(**data_args)
        deepgrow_datalist = create_dataset(datalist=datalist, output_dir=self.tempdir, **args)
        self.assertEqual(len(deepgrow_datalist), expected_length)
        if expected_region is not None:
            self.assertEqual(deepgrow_datalist[0]["region"], expected_region)

    def test_invalid_dim(self):
        with self.assertRaises(ValueError):
            create_dataset(datalist=self._create_data(), output_dir=self.tempdir, dimension=4, pixdim=(1, 1, 1, 1))

    def test_empty_datalist(self):
        with self.assertRaises(ValueError):
            create_dataset(datalist=[], output_dir=self.tempdir, dimension=3, pixdim=(1, 1, 1))

    def tearDown(self):
        shutil.rmtree(self.tempdir)
        set_determinism(None)


if __name__ == "__main__":
    unittest.main()
