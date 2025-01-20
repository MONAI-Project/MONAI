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

import copy
import os
import sys
import tempfile
import unittest

import nibabel as nib
import numpy as np
import torch
from parameterized import parameterized

from monai.data import DataLoader, SmartCacheDataset
from monai.transforms import Compose, Lambda, LoadImaged
from tests.util import assert_allclose

TEST_CASE_1 = [0.1, 0, Compose([LoadImaged(keys=["image", "label", "extra"])])]

TEST_CASE_2 = [0.1, 4, Compose([LoadImaged(keys=["image", "label", "extra"])])]

TEST_CASE_3 = [0.1, None, Compose([LoadImaged(keys=["image", "label", "extra"])])]

TEST_CASE_4 = [0.1, 4, None]

TEST_CASE_5 = [0.5, 2, Compose([LoadImaged(keys=["image", "label", "extra"])])]


class TestSmartCacheDataset(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4, TEST_CASE_5])
    def test_shape(self, replace_rate, num_replace_workers, transform):
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
            ] * 20
            dataset = SmartCacheDataset(
                data=test_data,
                transform=transform,
                replace_rate=replace_rate,
                cache_num=16,
                num_init_workers=4,
                num_replace_workers=num_replace_workers,
            )
            if transform is None:
                # Check without providing transfrom
                dataset2 = SmartCacheDataset(
                    data=test_data,
                    replace_rate=replace_rate,
                    cache_num=16,
                    num_init_workers=4,
                    num_replace_workers=num_replace_workers,
                )
                for k in ["image", "label", "extra"]:
                    self.assertEqual(dataset[0][k], dataset2[0][k])

            self.assertEqual(len(dataset._cache), dataset.cache_num)
            for i in range(dataset.cache_num):
                self.assertIsNotNone(dataset._cache[i])

            for _ in range(2):
                dataset.start()
                for _ in range(3):
                    dataset.update_cache()
                    self.assertIsNotNone(dataset[15])
                    if isinstance(dataset[15]["image"], (np.ndarray, torch.Tensor)):
                        assert_allclose(dataset[15]["image"], dataset[15]["label"])
                    else:
                        self.assertIsInstance(dataset[15]["image"], str)
                dataset.shutdown()

    def test_update_cache(self):
        # Given
        test_data = [{"image": f"test_image{i}.nii.gz", "label": f"test_image{i}.nii.gz"} for i in range(40)]
        dataset = SmartCacheDataset(
            data=test_data,
            transform=None,
            replace_rate=0.2,
            cache_num=10,
            num_init_workers=4,
            num_replace_workers=4,
            shuffle=False,
        )
        dataset.start()
        start_num = int(0.2 * 10)
        remain_num = int((1 - 0.2) * 10)

        old_cache = copy.deepcopy(dataset._cache)
        # When
        with dataset._update_lock:
            replacements = copy.deepcopy(dataset._replacements)
        dataset.update_cache()
        new_cache = dataset._cache
        kept_cache = old_cache[start_num:]
        # Then
        for string1, string2 in zip(kept_cache, new_cache[0:remain_num]):
            assert string1 == string2
        for string_new, string_replacement in zip(replacements, new_cache[remain_num:]):
            assert string_new == string_replacement

    def test_shuffle(self):
        test_data = [{"image": f"test_image{i}.nii.gz"} for i in range(20)]
        dataset = SmartCacheDataset(
            data=test_data,
            transform=None,
            replace_rate=0.1,
            cache_num=16,
            num_init_workers=4,
            num_replace_workers=4,
            shuffle=True,
            seed=123,
        )

        dataset.start()
        for i in range(3):
            dataset.update_cache()

            if i == 0:
                self.assertEqual(dataset[15]["image"], "test_image18.nii.gz")
            elif i == 1:
                self.assertEqual(dataset[15]["image"], "test_image13.nii.gz")
            else:
                self.assertEqual(dataset[15]["image"], "test_image5.nii.gz")

        dataset.shutdown()

    @unittest.skip("https://github.com/Project-MONAI/MONAI/issues/5660 blocks the ci")
    def test_set_data(self):
        data_list1 = list(range(10))

        transform = Lambda(func=lambda x: np.array([x * 10]))

        dataset = SmartCacheDataset(
            data=data_list1,
            transform=transform,
            cache_rate=0.5,
            replace_rate=0.4,
            num_init_workers=4,
            num_replace_workers=2,
            shuffle=False,
            progress=True,
        )

        num_workers = 2 if sys.platform == "linux" else 0
        dataloader = DataLoader(dataset=dataset, num_workers=num_workers, batch_size=1)

        dataset.start()
        for i, d in enumerate(dataloader):
            np.testing.assert_allclose([[data_list1[i] * 10]], d)
        # replace cache content, move forward 2(5 * 0.4) items
        dataset.update_cache()
        for i, d in enumerate(dataloader):
            np.testing.assert_allclose([[data_list1[i + 2] * 10]], d)
        # shutdown to update data
        dataset.shutdown()
        # update the datalist and fill the cache content
        data_list2 = list(range(-10, 0))
        dataset.set_data(data=data_list2)
        # restart the dataset
        dataset.start()
        # rerun with updated cache content
        for i, d in enumerate(dataloader):
            np.testing.assert_allclose([[data_list2[i] * 10]], d)
        # replace cache content, move forward 2(5 * 0.4) items
        dataset.update_cache()
        for i, d in enumerate(dataloader):
            np.testing.assert_allclose([[data_list2[i + 2] * 10]], d)
        # finally shutdown the dataset
        dataset.shutdown()

    def test_datalist(self):
        data_list = [np.array([i]) for i in range(5)]
        data_list_backup = copy.copy(data_list)

        SmartCacheDataset(data=data_list, transform=None, cache_rate=0.5, replace_rate=0.4, shuffle=True)
        np.testing.assert_allclose(data_list, data_list_backup)


if __name__ == "__main__":
    unittest.main()
