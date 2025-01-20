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
import unittest
from pathlib import Path
from unittest import skipUnless

import numpy as np
from numpy.testing import assert_array_equal
from parameterized import parameterized

from monai.data import PatchWSIDataset
from monai.data.wsi_reader import CuCIMWSIReader, OpenSlideWSIReader
from monai.utils import optional_import
from monai.utils.enums import WSIPatchKeys
from tests.util import download_url_or_skip_test, testing_data_config

cucim, has_cim = optional_import("cucim")
has_cim = has_cim and hasattr(cucim, "CuImage")
openslide, has_osl = optional_import("openslide")
imwrite, has_tiff = optional_import("tifffile", name="imwrite")
_, has_codec = optional_import("imagecodecs")
has_tiff = has_tiff and has_codec

FILE_KEY = "wsi_generic_tiff"
FILE_URL = testing_data_config("images", FILE_KEY, "url")
TESTS_PATH = Path(__file__).parents[1].as_posix()
FILE_PATH = os.path.join(TESTS_PATH, "testing_data", f"temp_{FILE_KEY}.tiff")

TEST_CASE_0 = [
    {
        "data": [{"image": FILE_PATH, WSIPatchKeys.LOCATION.value: [0, 0], "label": [1], "patch_level": 0}],
        "patch_size": (1, 1),
    },
    {"image": np.array([[[239]], [[239]], [[239]]], dtype=np.uint8), "label": np.array([1])},
]

TEST_CASE_0_L1 = [
    {
        "data": [{"image": FILE_PATH, WSIPatchKeys.LOCATION.value: [0, 0], "label": [1]}],
        "patch_size": (1, 1),
        "patch_level": 1,
    },
    {"image": np.array([[[239]], [[239]], [[239]]], dtype=np.uint8), "label": np.array([1])},
]

TEST_CASE_0_L2 = [
    {
        "data": [{"image": FILE_PATH, WSIPatchKeys.LOCATION.value: [0, 0], "label": [1]}],
        "patch_size": (1, 1),
        "patch_level": 1,
    },
    {"image": np.array([[[239]], [[239]], [[239]]], dtype=np.uint8), "label": np.array([1])},
]
TEST_CASE_1 = [
    {"data": [{"image": FILE_PATH, WSIPatchKeys.LOCATION.value: [0, 0], WSIPatchKeys.SIZE.value: 1, "label": [1]}]},
    {"image": np.array([[[239]], [[239]], [[239]]], dtype=np.uint8), "label": np.array([1])},
]

TEST_CASE_2 = [
    {
        "data": [{"image": FILE_PATH, WSIPatchKeys.LOCATION.value: [0, 0], "label": [1]}],
        "patch_size": 1,
        "patch_level": 0,
    },
    {"image": np.array([[[239]], [[239]], [[239]]], dtype=np.uint8), "label": np.array([1])},
]

TEST_CASE_3 = [
    {"data": [{"image": FILE_PATH, WSIPatchKeys.LOCATION.value: [0, 0], "label": [[[0, 1], [1, 0]]]}], "patch_size": 1},
    {"image": np.array([[[239]], [[239]], [[239]]], dtype=np.uint8), "label": np.array([[[0, 1], [1, 0]]])},
]

TEST_CASE_4 = [
    {
        "data": [
            {"image": FILE_PATH, WSIPatchKeys.LOCATION.value: [0, 0], "label": [[[0, 1], [1, 0]]]},
            {"image": FILE_PATH, WSIPatchKeys.LOCATION.value: [0, 0], "label": [[[1, 0], [0, 0]]]},
        ],
        "patch_size": 1,
    },
    [
        {"image": np.array([[[239]], [[239]], [[239]]], dtype=np.uint8), "label": np.array([[[0, 1], [1, 0]]])},
        {"image": np.array([[[239]], [[239]], [[239]]], dtype=np.uint8), "label": np.array([[[1, 0], [0, 0]]])},
    ],
]

TEST_CASE_5 = [
    {
        "data": [
            {
                "image": FILE_PATH,
                WSIPatchKeys.LOCATION.value: [0, 0],
                "label": [[[0, 1], [1, 0]]],
                WSIPatchKeys.SIZE.value: 1,
                WSIPatchKeys.LEVEL.value: 1,
            },
            {
                "image": FILE_PATH,
                WSIPatchKeys.LOCATION.value: [100, 100],
                "label": [[[1, 0], [0, 0]]],
                WSIPatchKeys.SIZE.value: 1,
                WSIPatchKeys.LEVEL.value: 1,
            },
        ]
    },
    [
        {"image": np.array([[[239]], [[239]], [[239]]], dtype=np.uint8), "label": np.array([[[0, 1], [1, 0]]])},
        {"image": np.array([[[243]], [[243]], [[243]]], dtype=np.uint8), "label": np.array([[[1, 0], [0, 0]]])},
    ],
]


@skipUnless(has_cim or has_osl or has_tiff, "Requires cucim, openslide, or tifffile!")
def setUpModule():
    hash_type = testing_data_config("images", FILE_KEY, "hash_type")
    hash_val = testing_data_config("images", FILE_KEY, "hash_val")
    download_url_or_skip_test(FILE_URL, FILE_PATH, hash_type=hash_type, hash_val=hash_val)


class PatchWSIDatasetTests:
    class Tests(unittest.TestCase):
        backend = None

        @parameterized.expand([TEST_CASE_0, TEST_CASE_0_L1, TEST_CASE_0_L2, TEST_CASE_1, TEST_CASE_2, TEST_CASE_3])
        def test_read_patches_str(self, input_parameters, expected):
            dataset = PatchWSIDataset(reader=self.backend, **input_parameters)
            sample = dataset[0]
            self.assertTupleEqual(sample["label"].shape, expected["label"].shape)
            self.assertTupleEqual(sample["image"].shape, expected["image"].shape)
            self.assertIsNone(assert_array_equal(sample["label"], expected["label"]))
            self.assertIsNone(assert_array_equal(sample["image"], expected["image"]))

        @parameterized.expand([TEST_CASE_0, TEST_CASE_0_L1, TEST_CASE_0_L2, TEST_CASE_1, TEST_CASE_2, TEST_CASE_3])
        def test_read_patches_class(self, input_parameters, expected):
            if self.backend == "openslide":
                reader = OpenSlideWSIReader
            elif self.backend == "cucim":
                reader = CuCIMWSIReader
            else:
                raise ValueError("Unsupported backend: {self.backend}")
            dataset = PatchWSIDataset(reader=reader, **input_parameters)
            sample = dataset[0]
            self.assertTupleEqual(sample["label"].shape, expected["label"].shape)
            self.assertTupleEqual(sample["image"].shape, expected["image"].shape)
            self.assertIsNone(assert_array_equal(sample["label"], expected["label"]))
            self.assertIsNone(assert_array_equal(sample["image"], expected["image"]))

        @parameterized.expand([TEST_CASE_0, TEST_CASE_0_L1, TEST_CASE_0_L2, TEST_CASE_1, TEST_CASE_2, TEST_CASE_3])
        def test_read_patches_object(self, input_parameters, expected):
            if self.backend == "openslide":
                reader = OpenSlideWSIReader(level=input_parameters.get("patch_level", 0))
            elif self.backend == "cucim":
                reader = CuCIMWSIReader(level=input_parameters.get("patch_level", 0))
            else:
                raise ValueError("Unsupported backend: {self.backend}")
            dataset = PatchWSIDataset(reader=reader, **input_parameters)
            sample = dataset[0]
            self.assertTupleEqual(sample["label"].shape, expected["label"].shape)
            self.assertTupleEqual(sample["image"].shape, expected["image"].shape)
            self.assertIsNone(assert_array_equal(sample["label"], expected["label"]))
            self.assertIsNone(assert_array_equal(sample["image"], expected["image"]))

        @parameterized.expand([TEST_CASE_4, TEST_CASE_5])
        def test_read_patches_str_multi(self, input_parameters, expected):
            dataset = PatchWSIDataset(reader=self.backend, **input_parameters)
            for i, item in enumerate(dataset):
                self.assertTupleEqual(item["label"].shape, expected[i]["label"].shape)
                self.assertTupleEqual(item["image"].shape, expected[i]["image"].shape)
                self.assertIsNone(assert_array_equal(item["label"], expected[i]["label"]))
                self.assertIsNone(assert_array_equal(item["image"], expected[i]["image"]))


@skipUnless(has_cim, "Requires cucim")
class TestPatchWSIDatasetCuCIM(PatchWSIDatasetTests.Tests):
    @classmethod
    def setUpClass(cls):
        cls.backend = "cucim"


@skipUnless(has_osl, "Requires openslide")
class TestPatchWSIDatasetOpenSlide(PatchWSIDatasetTests.Tests):
    @classmethod
    def setUpClass(cls):
        cls.backend = "openslide"


if __name__ == "__main__":
    unittest.main()
