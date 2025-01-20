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
import pickle
import tempfile
import unittest

import nibabel as nib
import numpy as np
from parameterized import parameterized

from monai.data import PersistentDataset, json_hashing
from monai.transforms import Compose, Flip, Identity, LoadImaged, SimulateDelayd, Transform

TEST_CASE_1 = [
    Compose(
        [
            LoadImaged(keys=["image", "label", "extra"]),
            SimulateDelayd(keys=["image", "label", "extra"], delay_time=[1e-7, 1e-6, 1e-5]),
        ]
    ),
    (128, 128, 128),
]

TEST_CASE_2 = [
    [
        LoadImaged(keys=["image", "label", "extra"]),
        SimulateDelayd(keys=["image", "label", "extra"], delay_time=[1e-7, 1e-6, 1e-5]),
    ],
    (128, 128, 128),
]

TEST_CASE_3 = [None, (128, 128, 128)]


class _InplaceXform(Transform):

    def __call__(self, data):
        if data:
            data[0] = data[0] + np.pi
        else:
            data.append(1)
        return data


class TestDataset(unittest.TestCase):

    def test_cache(self):
        """testing no inplace change to the hashed item"""
        items = [[list(range(i))] for i in range(5)]

        with tempfile.TemporaryDirectory() as tempdir:
            ds = PersistentDataset(
                data=items,
                transform=_InplaceXform(),
                cache_dir=tempdir,
                pickle_module="pickle",
                pickle_protocol=pickle.HIGHEST_PROTOCOL,
            )
            self.assertEqual(items, [[[]], [[0]], [[0, 1]], [[0, 1, 2]], [[0, 1, 2, 3]]])
            ds1 = PersistentDataset(items, transform=_InplaceXform(), cache_dir=tempdir)
            self.assertEqual(list(ds1), list(ds))
            self.assertEqual(items, [[[]], [[0]], [[0, 1]], [[0, 1, 2]], [[0, 1, 2, 3]]])

            ds = PersistentDataset(items, transform=_InplaceXform(), cache_dir=tempdir, hash_func=json_hashing)
            self.assertEqual(items, [[[]], [[0]], [[0, 1]], [[0, 1, 2]], [[0, 1, 2, 3]]])
            ds1 = PersistentDataset(items, transform=_InplaceXform(), cache_dir=tempdir, hash_func=json_hashing)
            self.assertEqual(list(ds1), list(ds))
            self.assertEqual(items, [[[]], [[0]], [[0, 1]], [[0, 1, 2]], [[0, 1, 2, 3]]])

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3])
    def test_shape(self, transform, expected_shape):
        test_image = nib.Nifti1Image(np.random.randint(0, 2, size=[128, 128, 128]).astype(float), np.eye(4))
        with tempfile.TemporaryDirectory() as tempdir:
            nib.save(test_image, os.path.join(tempdir, "test_image1.nii.gz"))
            nib.save(test_image, os.path.join(tempdir, "test_label1.nii.gz"))
            nib.save(test_image, os.path.join(tempdir, "test_extra1.nii.gz"))
            nib.save(test_image, os.path.join(tempdir, "test_image2.nii.gz"))
            nib.save(test_image, os.path.join(tempdir, "test_label2.nii.gz"))
            nib.save(test_image, os.path.join(tempdir, "test_extra2.nii.gz"))
            test_data = [
                {
                    "image": os.path.join(tempdir, "test_image1.nii.gz"),
                    "label": os.path.join(tempdir, "test_label1.nii.gz"),
                    "extra": os.path.join(tempdir, "test_extra1.nii.gz"),
                },
                {
                    "image": os.path.join(tempdir, "test_image2.nii.gz"),
                    "label": os.path.join(tempdir, "test_label2.nii.gz"),
                    "extra": os.path.join(tempdir, "test_extra2.nii.gz"),
                },
            ]

            cache_dir = os.path.join(os.path.join(tempdir, "cache"), "data")
            dataset_precached = PersistentDataset(data=test_data, transform=transform, cache_dir=cache_dir)
            data1_precached = dataset_precached[0]
            data2_precached = dataset_precached[1]

            dataset_postcached = PersistentDataset(data=test_data, transform=transform, cache_dir=cache_dir)
            data1_postcached = dataset_postcached[0]
            data2_postcached = dataset_postcached[1]
            data3_postcached = dataset_postcached[0:2]

            if transform is None:
                self.assertEqual(data1_precached["image"], os.path.join(tempdir, "test_image1.nii.gz"))
                self.assertEqual(data2_precached["label"], os.path.join(tempdir, "test_label2.nii.gz"))
                self.assertEqual(data1_postcached["image"], os.path.join(tempdir, "test_image1.nii.gz"))
                self.assertEqual(data2_postcached["extra"], os.path.join(tempdir, "test_extra2.nii.gz"))
            else:
                self.assertTupleEqual(data1_precached["image"].shape, expected_shape)
                self.assertTupleEqual(data1_precached["label"].shape, expected_shape)
                self.assertTupleEqual(data1_precached["extra"].shape, expected_shape)
                self.assertTupleEqual(data2_precached["image"].shape, expected_shape)
                self.assertTupleEqual(data2_precached["label"].shape, expected_shape)
                self.assertTupleEqual(data2_precached["extra"].shape, expected_shape)

                self.assertTupleEqual(data1_postcached["image"].shape, expected_shape)
                self.assertTupleEqual(data1_postcached["label"].shape, expected_shape)
                self.assertTupleEqual(data1_postcached["extra"].shape, expected_shape)
                self.assertTupleEqual(data2_postcached["image"].shape, expected_shape)
                self.assertTupleEqual(data2_postcached["label"].shape, expected_shape)
                self.assertTupleEqual(data2_postcached["extra"].shape, expected_shape)
                for d in data3_postcached:
                    self.assertTupleEqual(d["image"].shape, expected_shape)

            # update the data to cache
            test_data_new = [
                {
                    "image": os.path.join(tempdir, "test_image1_new.nii.gz"),
                    "label": os.path.join(tempdir, "test_label1_new.nii.gz"),
                    "extra": os.path.join(tempdir, "test_extra1_new.nii.gz"),
                },
                {
                    "image": os.path.join(tempdir, "test_image2_new.nii.gz"),
                    "label": os.path.join(tempdir, "test_label2_new.nii.gz"),
                    "extra": os.path.join(tempdir, "test_extra2_new.nii.gz"),
                },
            ]
            dataset_postcached.set_data(data=test_data_new)
            # test new exchanged cache content
            if transform is None:
                self.assertEqual(dataset_postcached[0]["image"], os.path.join(tempdir, "test_image1_new.nii.gz"))
                self.assertEqual(dataset_postcached[0]["label"], os.path.join(tempdir, "test_label1_new.nii.gz"))
                self.assertEqual(dataset_postcached[1]["extra"], os.path.join(tempdir, "test_extra2_new.nii.gz"))

    def test_different_transforms(self):
        """
        Different instances of `PersistentDataset` with the same cache_dir,
        same input data, but different transforms should give different results.
        """
        shape = (1, 10, 9, 8)
        im = np.arange(0, np.prod(shape)).reshape(shape)
        with tempfile.TemporaryDirectory() as path:
            im1 = PersistentDataset([im], Identity(), cache_dir=path, hash_transform=json_hashing)[0]
            im2 = PersistentDataset([im], Flip(1), cache_dir=path, hash_transform=json_hashing)[0]
            l2 = ((im1 - im2) ** 2).sum() ** 0.5
            self.assertGreater(l2, 1)


if __name__ == "__main__":
    unittest.main()
