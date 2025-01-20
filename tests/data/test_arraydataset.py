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
import sys
import tempfile
import unittest

import nibabel as nib
import numpy as np
from parameterized import parameterized
from torch.utils.data import DataLoader

from monai.data import ArrayDataset
from monai.transforms import Compose, EnsureChannelFirst, LoadImage, RandAdjustContrast, RandGaussianNoise, Spacing

TEST_CASE_1 = [
    Compose([LoadImage(image_only=True), EnsureChannelFirst(channel_dim="no_channel"), RandGaussianNoise(prob=1.0)]),
    Compose([LoadImage(image_only=True), EnsureChannelFirst(channel_dim="no_channel"), RandGaussianNoise(prob=1.0)]),
    (0, 1),
    (1, 128, 128, 128),
]

TEST_CASE_2 = [
    Compose([LoadImage(image_only=True), EnsureChannelFirst(channel_dim="no_channel"), RandAdjustContrast(prob=1.0)]),
    Compose([LoadImage(image_only=True), EnsureChannelFirst(channel_dim="no_channel"), RandAdjustContrast(prob=1.0)]),
    (0, 1),
    (1, 128, 128, 128),
]


class TestCompose(Compose):
    __test__ = False  # indicate to pytest that this class is not intended for collection

    def __call__(self, input_, lazy=False):
        img = self.transforms[0](input_)
        metadata = img.meta
        img = self.transforms[1](img)
        img = self.transforms[2](img, lazy=lazy)
        metadata = img.meta
        return self.transforms[3](img), metadata


TEST_CASE_3 = [
    TestCompose(
        [
            LoadImage(image_only=True),
            EnsureChannelFirst(channel_dim="no_channel"),
            Spacing(pixdim=(2, 2, 4)),
            RandAdjustContrast(prob=1.0),
        ]
    ),
    TestCompose(
        [
            LoadImage(image_only=True),
            EnsureChannelFirst(channel_dim="no_channel"),
            Spacing(pixdim=(2, 2, 4)),
            RandAdjustContrast(prob=1.0),
        ]
    ),
    (0, 2),
    (1, 64, 64, 33),
]

TEST_CASE_4 = [
    Compose([LoadImage(image_only=True), EnsureChannelFirst(channel_dim="no_channel"), RandGaussianNoise(prob=1.0)]),
    (1, 128, 128, 128),
]


class TestArrayDataset(unittest.TestCase):

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3])
    def test_shape(self, img_transform, label_transform, indices, expected_shape):
        test_image = nib.Nifti1Image(np.random.randint(0, 2, size=(128, 128, 128)).astype(float), np.eye(4))
        with tempfile.TemporaryDirectory() as tempdir:
            test_image1 = os.path.join(tempdir, "test_image1.nii.gz")
            test_seg1 = os.path.join(tempdir, "test_seg1.nii.gz")
            test_image2 = os.path.join(tempdir, "test_image2.nii.gz")
            test_seg2 = os.path.join(tempdir, "test_seg2.nii.gz")
            nib.save(test_image, test_image1)
            nib.save(test_image, test_seg1)
            nib.save(test_image, test_image2)
            nib.save(test_image, test_seg2)
            test_images = [test_image1, test_image2]
            test_segs = [test_seg1, test_seg2]
            test_labels = [1, 1]
            dataset = ArrayDataset(test_images, img_transform, test_segs, label_transform, test_labels, None)
            self.assertEqual(len(dataset), 2)
            dataset.set_random_state(1234)
            data1 = dataset[0]
            data2 = dataset[1]

            self.assertTupleEqual(data1[indices[0]].shape, expected_shape)
            self.assertTupleEqual(data1[indices[1]].shape, expected_shape)
            np.testing.assert_allclose(data1[indices[0]], data1[indices[1]])
            self.assertTupleEqual(data2[indices[0]].shape, expected_shape)
            self.assertTupleEqual(data2[indices[1]].shape, expected_shape)
            np.testing.assert_allclose(data2[indices[0]], data2[indices[0]])

            dataset = ArrayDataset(test_images, img_transform, test_segs, label_transform, test_labels, None)
            dataset.set_random_state(1234)
            _ = dataset[0]
            data2_new = dataset[1]
            np.testing.assert_allclose(data2[indices[0]], data2_new[indices[0]], atol=1e-3)

    @parameterized.expand([TEST_CASE_4])
    def test_default_none(self, img_transform, expected_shape):
        test_image = nib.Nifti1Image(np.random.randint(0, 2, size=(128, 128, 128)).astype(float), np.eye(4))
        with tempfile.TemporaryDirectory() as tempdir:
            test_image1 = os.path.join(tempdir, "test_image1.nii.gz")
            test_image2 = os.path.join(tempdir, "test_image2.nii.gz")
            nib.save(test_image, test_image1)
            nib.save(test_image, test_image2)
            test_images = [test_image1, test_image2]
            dataset = ArrayDataset(test_images, img_transform)
            self.assertEqual(len(dataset), 2)
            dataset.set_random_state(1234)
            data1 = dataset[0]
            data2 = dataset[1]
            self.assertTupleEqual(data1.shape, expected_shape)
            self.assertTupleEqual(data2.shape, expected_shape)

            dataset = ArrayDataset(test_images, img_transform)
            dataset.set_random_state(1234)
            _ = dataset[0]
            data2_new = dataset[1]
            np.testing.assert_allclose(data2, data2_new, atol=1e-3)

    @parameterized.expand([TEST_CASE_4])
    def test_dataloading_img(self, img_transform, expected_shape):
        test_image = nib.Nifti1Image(np.random.randint(0, 2, size=(128, 128, 128)).astype(float), np.eye(4))
        with tempfile.TemporaryDirectory() as tempdir:
            test_image1 = os.path.join(tempdir, "test_image1.nii.gz")
            test_image2 = os.path.join(tempdir, "test_image2.nii.gz")
            nib.save(test_image, test_image1)
            nib.save(test_image, test_image2)
            test_images = [test_image1, test_image2]
            dataset = ArrayDataset(test_images, img_transform)
            self.assertEqual(len(dataset), 2)
            dataset.set_random_state(1234)
            n_workers = 0 if sys.platform == "win32" else 2
            loader = DataLoader(dataset, batch_size=10, num_workers=n_workers)
            imgs = next(iter(loader))  # test batching
            np.testing.assert_allclose(imgs.shape, [2] + list(expected_shape))

            dataset.set_random_state(1234)
            new_imgs = next(iter(loader))  # test batching
            np.testing.assert_allclose(imgs, new_imgs, atol=1e-3)

    @parameterized.expand([TEST_CASE_4])
    def test_dataloading_img_label(self, img_transform, expected_shape):
        test_image = nib.Nifti1Image(np.random.randint(0, 2, size=(128, 128, 128)).astype(float), np.eye(4))
        with tempfile.TemporaryDirectory() as tempdir:
            test_image1 = os.path.join(tempdir, "test_image1.nii.gz")
            test_image2 = os.path.join(tempdir, "test_image2.nii.gz")
            test_label1 = os.path.join(tempdir, "test_label1.nii.gz")
            test_label2 = os.path.join(tempdir, "test_label2.nii.gz")
            nib.save(test_image, test_image1)
            nib.save(test_image, test_image2)
            nib.save(test_image, test_label1)
            nib.save(test_image, test_label2)
            test_images = [test_image1, test_image2]
            test_labels = [test_label1, test_label2]
            dataset = ArrayDataset(test_images, img_transform, test_labels, img_transform)
            self.assertEqual(len(dataset), 2)
            dataset.set_random_state(1234)
            n_workers = 0 if sys.platform == "win32" else 2
            loader = DataLoader(dataset, batch_size=10, num_workers=n_workers)
            data = next(iter(loader))  # test batching
            np.testing.assert_allclose(data[0].shape, [2] + list(expected_shape))

            dataset.set_random_state(1234)
            new_data = next(iter(loader))  # test batching
            np.testing.assert_allclose(data[0], new_data[0], atol=1e-3)


if __name__ == "__main__":
    unittest.main()
