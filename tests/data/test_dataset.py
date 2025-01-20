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

import logging
import os
import tempfile
import unittest
from copy import deepcopy
from io import StringIO

import nibabel as nib
import numpy as np
from parameterized import parameterized

from monai.data import Dataset
from monai.transforms import Compose, Lambda, LoadImage, LoadImaged, SimulateDelay, SimulateDelayd
from tests.transforms.compose.test_compose import TEST_COMPOSE_LAZY_ON_CALL_LOGGING_TEST_CASES, data_from_keys

TEST_CASE_1 = [(128, 128, 128)]


class TestDataset(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1])
    def test_shape(self, expected_shape):
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
            test_transform = Compose(
                [
                    LoadImaged(keys=["image", "label", "extra"]),
                    SimulateDelayd(keys=["image", "label", "extra"], delay_time=[1e-7, 1e-6, 1e-5]),
                ]
            )
            dataset = Dataset(data=test_data, transform=test_transform)
            data1 = dataset[0]
            data2 = dataset[1]

            self.assertTupleEqual(data1["image"].shape, expected_shape)
            self.assertTupleEqual(data1["label"].shape, expected_shape)
            self.assertTupleEqual(data1["extra"].shape, expected_shape)
            self.assertTupleEqual(data2["image"].shape, expected_shape)
            self.assertTupleEqual(data2["label"].shape, expected_shape)
            self.assertTupleEqual(data2["extra"].shape, expected_shape)

            dataset = Dataset(data=test_data, transform=LoadImaged(keys=["image", "label", "extra"]))
            data1_simple = dataset[0]
            data2_simple = dataset[1]
            data3_simple = dataset[-1]
            data4_simple = dataset[[0, 1]]

            self.assertTupleEqual(data1_simple["image"].shape, expected_shape)
            self.assertTupleEqual(data1_simple["label"].shape, expected_shape)
            self.assertTupleEqual(data1_simple["extra"].shape, expected_shape)
            self.assertTupleEqual(data2_simple["image"].shape, expected_shape)
            self.assertTupleEqual(data2_simple["label"].shape, expected_shape)
            self.assertTupleEqual(data2_simple["extra"].shape, expected_shape)
            self.assertTupleEqual(data3_simple["image"].shape, expected_shape)
            self.assertTupleEqual(data3_simple["label"].shape, expected_shape)
            self.assertTupleEqual(data3_simple["extra"].shape, expected_shape)
            self.assertTupleEqual(data4_simple[0]["image"].shape, expected_shape)
            self.assertTupleEqual(data4_simple[1]["label"].shape, expected_shape)
            self.assertTupleEqual(data4_simple[-1]["extra"].shape, expected_shape)

            data4_list = dataset[0:1]
            self.assertEqual(len(data4_list), 1)
            for d in data4_list:
                self.assertTupleEqual(d["image"].shape, expected_shape)

    def test_dataset_lazy_on_call(self):
        data = np.zeros((1, 5, 5))
        data[0, 0:2, 0:2] = 1


class TestTupleDataset(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1])
    def test_shape(self, expected_shape):
        test_image = nib.Nifti1Image(np.random.randint(0, 2, size=[128, 128, 128]).astype(float), np.eye(4))
        with tempfile.TemporaryDirectory() as tempdir:
            nib.save(test_image, os.path.join(tempdir, "test_image1.nii.gz"))
            nib.save(test_image, os.path.join(tempdir, "test_label1.nii.gz"))
            nib.save(test_image, os.path.join(tempdir, "test_image2.nii.gz"))
            nib.save(test_image, os.path.join(tempdir, "test_label2.nii.gz"))
            test_data = [
                (os.path.join(tempdir, "test_image1.nii.gz"), os.path.join(tempdir, "test_label1.nii.gz")),
                (os.path.join(tempdir, "test_image2.nii.gz"), os.path.join(tempdir, "test_label2.nii.gz")),
            ]

            test_transform = Compose([LoadImage(), SimulateDelay(delay_time=1e-5)])

            # Here test_transform is applied element by element for the tuple.
            dataset = Dataset(data=test_data, transform=test_transform)
            data1 = dataset[0]
            data2 = dataset[1]

            # Output is a list/tuple
            self.assertTrue(isinstance(data1, (list, tuple)))
            self.assertTrue(isinstance(data2, (list, tuple)))

            # Number of elements are 2
            self.assertEqual(len(data1), 2)
            self.assertEqual(len(data2), 2)

            # Output shapes are as expected
            self.assertTupleEqual(data1[0].shape, expected_shape)
            self.assertTupleEqual(data1[1].shape, expected_shape)
            self.assertTupleEqual(data2[0].shape, expected_shape)
            self.assertTupleEqual(data2[1].shape, expected_shape)

            # Here test_transform is applied to the tuple as a whole.
            test_transform = Compose(
                [
                    # LoadImage creates a channel-stacked image when applied to a tuple
                    LoadImage(),
                    # Get the channel-stacked image and the label
                    Lambda(func=lambda x: (x[0].permute(2, 1, 0), x[1])),
                ],
                map_items=False,
            )

            dataset = Dataset(data=test_data, transform=test_transform)
            data1 = dataset[0]
            data2 = dataset[1]

            # Output is a list/tuple
            self.assertTrue(isinstance(data1, (list, tuple)))
            self.assertTrue(isinstance(data2, (list, tuple)))

            # Number of elements are 2
            self.assertEqual(len(data1), 2)
            self.assertEqual(len(data2), 2)

            # Output shapes are as expected
            self.assertTupleEqual(data1[0].shape, expected_shape)
            self.assertTupleEqual(data1[1].shape, expected_shape)
            self.assertTupleEqual(data2[0].shape, expected_shape)
            self.assertTupleEqual(data2[1].shape, expected_shape)


class TestDatsesetWithLazy(unittest.TestCase):
    LOGGER_NAME = "a_logger_name"

    def init_logger(self, name=LOGGER_NAME):
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        formatter = logging.Formatter("%(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        while len(logger.handlers) > 0:
            logger.removeHandler(logger.handlers[-1])
        logger.addHandler(handler)
        return handler, stream

    @parameterized.expand(TEST_COMPOSE_LAZY_ON_CALL_LOGGING_TEST_CASES)
    def test_dataset_lazy_with_logging(self, compose_type, pipeline, lazy, expected):
        handler, stream = self.init_logger(name=self.LOGGER_NAME)

        data = data_from_keys(None, 12, 16)
        c = compose_type(deepcopy(pipeline), log_stats=self.LOGGER_NAME, lazy=lazy)
        ds = Dataset([data], transform=c)
        ds[0]

        handler.flush()
        actual = stream.getvalue()
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
