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

import nibabel as nib
import numpy as np
from parameterized import parameterized

from monai.data import CacheDataset
from monai.transforms import Compose, LoadImaged

TEST_CASE_1 = [0, 5, Compose([LoadImaged(keys=["image", "label", "extra"])])]

TEST_CASE_2 = [4, 5, Compose([LoadImaged(keys=["image", "label", "extra"])])]

TEST_CASE_3 = [4, 5, None]


class TestCacheDatasetParallel(unittest.TestCase):

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3])
    def test_shape(self, num_workers, dataset_size, transform):
        test_image = nib.Nifti1Image(np.random.randint(0, 2, size=[8, 8, 8]).astype(float), np.eye(4))
        with tempfile.TemporaryDirectory() as tempdir:
            nib.save(test_image, os.path.join(tempdir, "test_image1.nii.gz"))
            nib.save(test_image, os.path.join(tempdir, "test_label1.nii.gz"))
            nib.save(test_image, os.path.join(tempdir, "test_extra1.nii.gz"))
            test_data = [
                {
                    "image": os.path.join(tempdir, "test_image1.nii.gz"),
                    "label": os.path.join(tempdir, "test_label1.nii.gz"),
                    "extra": os.path.join(tempdir, "test_extra1.nii.gz"),
                }
            ] * dataset_size
            dataset = CacheDataset(data=test_data, transform=transform, cache_rate=1, num_workers=num_workers)

        self.assertEqual(len(dataset._cache), dataset.cache_num)
        for i in range(dataset.cache_num):
            self.assertIsNotNone(dataset._cache[i])


if __name__ == "__main__":
    unittest.main()
