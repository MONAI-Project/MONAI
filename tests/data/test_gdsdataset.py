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

import numpy as np
import torch
from parameterized import parameterized

from monai.data import GDSDataset, json_hashing
from monai.transforms import Compose, Flip, Identity, LoadImaged, SimulateDelayd, Transform
from monai.utils import optional_import
from tests.util import TEST_NDARRAYS, assert_allclose, skip_if_no_cuda

_, has_cp = optional_import("cupy")
nib, has_nib = optional_import("nibabel")
_, has_kvikio_numpy = optional_import("kvikio.numpy")

TEST_CASE_1 = [
    Compose(
        [
            LoadImaged(keys=["image", "label", "extra"], image_only=True),
            SimulateDelayd(keys=["image", "label", "extra"], delay_time=[1e-7, 1e-6, 1e-5]),
        ]
    ),
    (128, 128, 128),
]

TEST_CASE_2 = [
    [
        LoadImaged(keys=["image", "label", "extra"], image_only=True),
        SimulateDelayd(keys=["image", "label", "extra"], delay_time=[1e-7, 1e-6, 1e-5]),
    ],
    (128, 128, 128),
]

TEST_CASE_3 = [None, (128, 128, 128)]

DTYPES = {
    np.dtype(np.uint8): torch.uint8,
    np.dtype(np.int8): torch.int8,
    np.dtype(np.int16): torch.int16,
    np.dtype(np.int32): torch.int32,
    np.dtype(np.int64): torch.int64,
    np.dtype(np.float16): torch.float16,
    np.dtype(np.float32): torch.float32,
    np.dtype(np.float64): torch.float64,
    np.dtype(np.complex64): torch.complex64,
    np.dtype(np.complex128): torch.complex128,
}


class _InplaceXform(Transform):
    def __call__(self, data):
        data[0] = data[0] + 1
        return data


@skip_if_no_cuda
@unittest.skipUnless(has_cp, "Requires CuPy library.")
@unittest.skipUnless(has_cp and has_kvikio_numpy, "Requires CuPy and kvikio library.")
class TestDataset(unittest.TestCase):
    def test_cache(self):
        """testing no inplace change to the hashed item"""
        for p in TEST_NDARRAYS[:2]:
            shape = (1, 10, 9, 8)
            items = [p(np.arange(0, np.prod(shape)).reshape(shape))]

            with tempfile.TemporaryDirectory() as tempdir:
                ds = GDSDataset(
                    data=items,
                    transform=_InplaceXform(),
                    cache_dir=tempdir,
                    device=0,
                    pickle_module="pickle",
                    pickle_protocol=pickle.HIGHEST_PROTOCOL,
                )
                assert_allclose(items[0], p(np.arange(0, np.prod(shape)).reshape(shape)))
                ds1 = GDSDataset(items, transform=_InplaceXform(), cache_dir=tempdir, device=0)
                assert_allclose(ds[0], ds1[0], type_test=False)
                assert_allclose(items[0], p(np.arange(0, np.prod(shape)).reshape(shape)))

                ds = GDSDataset(
                    items, transform=_InplaceXform(), cache_dir=tempdir, hash_transform=json_hashing, device=0
                )
                assert_allclose(items[0], p(np.arange(0, np.prod(shape)).reshape(shape)))
                ds1 = GDSDataset(
                    items, transform=_InplaceXform(), cache_dir=tempdir, hash_transform=json_hashing, device=0
                )
                assert_allclose(ds[0], ds1[0], type_test=False)
                assert_allclose(items[0], p(np.arange(0, np.prod(shape)).reshape(shape)))

    def test_metatensor(self):
        shape = (1, 10, 9, 8)
        items = [TEST_NDARRAYS[-1](np.arange(0, np.prod(shape)).reshape(shape))]
        with tempfile.TemporaryDirectory() as tempdir:
            ds = GDSDataset(data=items, transform=_InplaceXform(), cache_dir=tempdir, device=0)
            assert_allclose(ds[0], ds[0][0], type_test=False)

    def test_dtype(self):
        shape = (1, 10, 9, 8)
        data = np.arange(0, np.prod(shape)).reshape(shape)
        for _dtype in DTYPES.keys():
            items = [np.array(data).astype(_dtype)]
            with tempfile.TemporaryDirectory() as tempdir:
                ds = GDSDataset(data=items, transform=_InplaceXform(), cache_dir=tempdir, device=0)
                ds1 = GDSDataset(data=items, transform=_InplaceXform(), cache_dir=tempdir, device=0)
                self.assertEqual(ds[0].dtype, _dtype)
                self.assertEqual(ds1[0].dtype, DTYPES[_dtype])

        for _dtype in DTYPES.keys():
            items = [torch.tensor(data, dtype=DTYPES[_dtype])]
            with tempfile.TemporaryDirectory() as tempdir:
                ds = GDSDataset(data=items, transform=_InplaceXform(), cache_dir=tempdir, device=0)
                ds1 = GDSDataset(data=items, transform=_InplaceXform(), cache_dir=tempdir, device=0)
                self.assertEqual(ds[0].dtype, DTYPES[_dtype])
                self.assertEqual(ds1[0].dtype, DTYPES[_dtype])

    @unittest.skipUnless(has_nib, "Requires nibabel package.")
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
            dataset_precached = GDSDataset(data=test_data, transform=transform, cache_dir=cache_dir, device=0)
            data1_precached = dataset_precached[0]
            data2_precached = dataset_precached[1]

            dataset_postcached = GDSDataset(data=test_data, transform=transform, cache_dir=cache_dir, device=0)
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
        Different instances of `GDSDataset` with the same cache_dir,
        same input data, but different transforms should give different results.
        """
        shape = (1, 10, 9, 8)
        im = np.arange(0, np.prod(shape)).reshape(shape)
        with tempfile.TemporaryDirectory() as path:
            im1 = GDSDataset([im], Identity(), cache_dir=path, hash_transform=json_hashing, device=0)[0]
            im2 = GDSDataset([im], Flip(1), cache_dir=path, hash_transform=json_hashing, device=0)[0]
            l2 = ((im1 - im2) ** 2).sum() ** 0.5
            self.assertTrue(l2 > 1)


if __name__ == "__main__":
    unittest.main()
