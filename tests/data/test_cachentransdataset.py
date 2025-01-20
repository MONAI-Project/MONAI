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

from monai.data import CacheNTransDataset
from monai.transforms import LoadImaged, ShiftIntensityd

TEST_CASE_1 = [
    [
        LoadImaged(keys="image"),
        ShiftIntensityd(keys="image", offset=1.0),
        ShiftIntensityd(keys="image", offset=2.0),
        ShiftIntensityd(keys="image", offset=3.0),
    ],
    (128, 128, 128),
]


class TestCacheNTransDataset(unittest.TestCase):

    @parameterized.expand([TEST_CASE_1])
    def test_n_trans(self, transform, expected_shape):
        data_array = np.random.randint(0, 2, size=[128, 128, 128]).astype(float)
        test_image = nib.Nifti1Image(data_array, np.eye(4))
        with tempfile.TemporaryDirectory() as tempdir:
            nib.save(test_image, os.path.join(tempdir, "test_image.nii.gz"))
            test_data = [{"image": os.path.join(tempdir, "test_image.nii.gz")}]

            cache_dir = os.path.join(os.path.join(tempdir, "cache"), "data")
            dataset_precached = CacheNTransDataset(
                data=test_data, transform=transform, cache_dir=cache_dir, cache_n_trans=2
            )
            data_precached = dataset_precached[0]
            self.assertTupleEqual(data_precached["image"].shape, expected_shape)

            dataset_postcached = CacheNTransDataset(
                data=test_data, transform=transform, cache_dir=cache_dir, cache_n_trans=2
            )
            data_postcached = dataset_postcached[0]
            self.assertTupleEqual(data_postcached["image"].shape, expected_shape)
            np.testing.assert_allclose(data_array + 6.0, data_postcached["image"])
            np.testing.assert_allclose(data_precached["image"], data_postcached["image"])


if __name__ == "__main__":
    unittest.main()
