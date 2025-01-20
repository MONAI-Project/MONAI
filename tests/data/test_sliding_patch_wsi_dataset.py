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

from monai.data import SlidingPatchWSIDataset
from monai.utils import WSIPatchKeys, optional_import, set_determinism
from tests.util import download_url_or_skip_test, testing_data_config

set_determinism(0)

cucim, has_cucim = optional_import("cucim")
has_cucim = has_cucim and hasattr(cucim, "CuImage")
openslide, has_osl = optional_import("openslide")
imwrite, has_tiff = optional_import("tifffile", name="imwrite")
_, has_codec = optional_import("imagecodecs")
has_tiff = has_tiff and has_codec

FILE_KEY = "wsi_generic_tiff"
FILE_URL = testing_data_config("images", FILE_KEY, "url")
TESTS_PATH = Path(__file__).parents[1].as_posix()
FILE_PATH = os.path.join(TESTS_PATH, "testing_data", f"temp_{FILE_KEY}.tiff")

FILE_PATH_SMALL_0 = os.path.join(TESTS_PATH, "testing_data", "temp_wsi_inference_0.tiff")
FILE_PATH_SMALL_1 = os.path.join(TESTS_PATH, "testing_data", "temp_wsi_inference_1.tiff")
ARRAY_SMALL_0 = np.random.randint(low=0, high=255, size=(3, 4, 4), dtype=np.uint8)
ARRAY_SMALL_1 = np.random.randint(low=0, high=255, size=(3, 5, 5), dtype=np.uint8)

TEST_CASE_SMALL_0 = [
    {"data": [{"image": FILE_PATH_SMALL_0, WSIPatchKeys.LEVEL: 0}], "patch_size": (2, 2)},
    [
        {"image": ARRAY_SMALL_0[:, :2, :2]},
        {"image": ARRAY_SMALL_0[:, :2, 2:]},
        {"image": ARRAY_SMALL_0[:, 2:, :2]},
        {"image": ARRAY_SMALL_0[:, 2:, 2:]},
    ],
]

TEST_CASE_SMALL_1 = [
    {"data": [{"image": FILE_PATH_SMALL_0, WSIPatchKeys.LEVEL: 0, WSIPatchKeys.SIZE: (2, 2)}]},
    [
        {"image": ARRAY_SMALL_0[:, :2, :2]},
        {"image": ARRAY_SMALL_0[:, :2, 2:]},
        {"image": ARRAY_SMALL_0[:, 2:, :2]},
        {"image": ARRAY_SMALL_0[:, 2:, 2:]},
    ],
]

TEST_CASE_SMALL_2 = [
    {"data": [{"image": FILE_PATH_SMALL_0, WSIPatchKeys.LEVEL: 0}], "patch_size": (2, 2), "overlap": 0.5},
    [
        {"image": ARRAY_SMALL_0[:, 0:2, 0:2]},
        {"image": ARRAY_SMALL_0[:, 0:2, 1:3]},
        {"image": ARRAY_SMALL_0[:, 0:2, 2:4]},
        {"image": ARRAY_SMALL_0[:, 1:3, 0:2]},
        {"image": ARRAY_SMALL_0[:, 1:3, 1:3]},
        {"image": ARRAY_SMALL_0[:, 1:3, 2:4]},
        {"image": ARRAY_SMALL_0[:, 2:4, 0:2]},
        {"image": ARRAY_SMALL_0[:, 2:4, 1:3]},
        {"image": ARRAY_SMALL_0[:, 2:4, 2:4]},
    ],
]

TEST_CASE_SMALL_3 = [
    {"data": [{"image": FILE_PATH_SMALL_0, WSIPatchKeys.LEVEL: 0}], "patch_size": (3, 3), "overlap": 2.0 / 3.0},
    [
        {"image": ARRAY_SMALL_0[:, :3, :3]},
        {"image": ARRAY_SMALL_0[:, :3, 1:]},
        {"image": ARRAY_SMALL_0[:, 1:, :3]},
        {"image": ARRAY_SMALL_0[:, 1:, 1:]},
    ],
]

TEST_CASE_SMALL_4 = [
    {
        "data": [
            {"image": FILE_PATH_SMALL_0, WSIPatchKeys.LEVEL: 0},
            {"image": FILE_PATH_SMALL_1, WSIPatchKeys.LEVEL: 0},
        ],
        "patch_size": (2, 2),
    },
    [
        {"image": ARRAY_SMALL_0[:, 0:2, 0:2]},
        {"image": ARRAY_SMALL_0[:, 0:2, 2:4]},
        {"image": ARRAY_SMALL_0[:, 2:4, 0:2]},
        {"image": ARRAY_SMALL_0[:, 2:4, 2:4]},
        {"image": ARRAY_SMALL_1[:, 0:2, 0:2]},
        {"image": ARRAY_SMALL_1[:, 0:2, 2:4]},
        {"image": ARRAY_SMALL_1[:, 2:4, 0:2]},
        {"image": ARRAY_SMALL_1[:, 2:4, 2:4]},
    ],
]

TEST_CASE_SMALL_5 = [
    {
        "data": [
            {"image": FILE_PATH_SMALL_0, WSIPatchKeys.LEVEL: 0, WSIPatchKeys.SIZE: (2, 2)},
            {"image": FILE_PATH_SMALL_1, WSIPatchKeys.LEVEL: 0, WSIPatchKeys.SIZE: (3, 3)},
        ]
    },
    [
        {"image": ARRAY_SMALL_0[:, 0:2, 0:2]},
        {"image": ARRAY_SMALL_0[:, 0:2, 2:4]},
        {"image": ARRAY_SMALL_0[:, 2:4, 0:2]},
        {"image": ARRAY_SMALL_0[:, 2:4, 2:4]},
        {"image": ARRAY_SMALL_1[:, 0:3, 0:3]},
    ],
]

TEST_CASE_SMALL_6 = [
    {
        "data": [
            {"image": FILE_PATH_SMALL_0, WSIPatchKeys.LEVEL: 1, WSIPatchKeys.SIZE: (1, 1)},
            {"image": FILE_PATH_SMALL_1, WSIPatchKeys.LEVEL: 2, WSIPatchKeys.SIZE: (4, 4)},
        ],
        "patch_size": (2, 2),
        "patch_level": 0,
    },
    [
        {"image": ARRAY_SMALL_0[:, 0:2, 0:2]},
        {"image": ARRAY_SMALL_0[:, 0:2, 2:4]},
        {"image": ARRAY_SMALL_0[:, 2:4, 0:2]},
        {"image": ARRAY_SMALL_0[:, 2:4, 2:4]},
        {"image": ARRAY_SMALL_1[:, 0:2, 0:2]},
        {"image": ARRAY_SMALL_1[:, 0:2, 2:4]},
        {"image": ARRAY_SMALL_1[:, 2:4, 0:2]},
        {"image": ARRAY_SMALL_1[:, 2:4, 2:4]},
    ],
]

TEST_CASE_SMALL_7 = [
    {"data": [{"image": FILE_PATH_SMALL_0, WSIPatchKeys.LEVEL: 0, WSIPatchKeys.SIZE: (2, 2)}], "offset": (1, 0)},
    [{"image": ARRAY_SMALL_0[:, 1:3, :2]}, {"image": ARRAY_SMALL_0[:, 1:3, 2:]}],
]

TEST_CASE_SMALL_8 = [
    {
        "data": [{"image": FILE_PATH_SMALL_0, WSIPatchKeys.LEVEL: 0, WSIPatchKeys.SIZE: (2, 2)}],
        "offset": "random",
        "offset_limits": (0, 2),
    },
    [{"image": ARRAY_SMALL_0[:, 1:3, :2]}, {"image": ARRAY_SMALL_0[:, 1:3, 2:]}],
]

TEST_CASE_SMALL_9 = [
    {
        "data": [{"image": FILE_PATH_SMALL_0, WSIPatchKeys.LEVEL: 0, WSIPatchKeys.SIZE: (2, 2)}],
        "offset": "random",
        "offset_limits": ((0, 3), (0, 2)),
    },
    [{"image": ARRAY_SMALL_0[:, :2, 1:3]}, {"image": ARRAY_SMALL_0[:, 2:, 1:3]}],
]

TEST_CASE_LARGE_0 = [
    {"data": [{"image": FILE_PATH, WSIPatchKeys.LEVEL: 8, WSIPatchKeys.SIZE: (64, 50)}]},
    [
        {"step_loc": (0, 0), "patch_size": (64, 50), "patch_level": 8, "ratio": 257.06195068359375},
        {"step_loc": (0, 1), "patch_size": (64, 50), "patch_level": 8, "ratio": 257.06195068359375},
        {"step_loc": (0, 2), "patch_size": (64, 50), "patch_level": 8, "ratio": 257.06195068359375},
        {"step_loc": (1, 0), "patch_size": (64, 50), "patch_level": 8, "ratio": 257.06195068359375},
        {"step_loc": (1, 1), "patch_size": (64, 50), "patch_level": 8, "ratio": 257.06195068359375},
        {"step_loc": (1, 2), "patch_size": (64, 50), "patch_level": 8, "ratio": 257.06195068359375},
    ],
]

TEST_CASE_LARGE_1 = [
    {
        "data": [
            {"image": FILE_PATH, WSIPatchKeys.LEVEL: 8, WSIPatchKeys.SIZE: (64, 50)},
            {"image": FILE_PATH, WSIPatchKeys.LEVEL: 7, WSIPatchKeys.SIZE: (125, 110)},
        ]
    },
    [
        {"step_loc": (0, 0), "patch_size": (64, 50), "patch_level": 8, "ratio": 257.06195068359375},
        {"step_loc": (0, 1), "patch_size": (64, 50), "patch_level": 8, "ratio": 257.06195068359375},
        {"step_loc": (0, 2), "patch_size": (64, 50), "patch_level": 8, "ratio": 257.06195068359375},
        {"step_loc": (1, 0), "patch_size": (64, 50), "patch_level": 8, "ratio": 257.06195068359375},
        {"step_loc": (1, 1), "patch_size": (64, 50), "patch_level": 8, "ratio": 257.06195068359375},
        {"step_loc": (1, 2), "patch_size": (64, 50), "patch_level": 8, "ratio": 257.06195068359375},
        {"step_loc": (0, 0), "patch_size": (125, 110), "patch_level": 7, "ratio": 128.10186767578125},
        {"step_loc": (0, 1), "patch_size": (125, 110), "patch_level": 7, "ratio": 128.10186767578125},
        {"step_loc": (0, 2), "patch_size": (125, 110), "patch_level": 7, "ratio": 128.10186767578125},
        {"step_loc": (1, 0), "patch_size": (125, 110), "patch_level": 7, "ratio": 128.10186767578125},
        {"step_loc": (1, 1), "patch_size": (125, 110), "patch_level": 7, "ratio": 128.10186767578125},
        {"step_loc": (1, 2), "patch_size": (125, 110), "patch_level": 7, "ratio": 128.10186767578125},
    ],
]


@skipUnless(has_cucim or has_tiff, "Requires cucim, openslide, or tifffile!")
def setUpModule():
    for info in [(ARRAY_SMALL_0, FILE_PATH_SMALL_0), (ARRAY_SMALL_1, FILE_PATH_SMALL_1)]:
        array = info[0].transpose([1, 2, 0])
        imwrite(info[1], array, shape=array.shape, photometric="rgb")
    hash_type = testing_data_config("images", FILE_KEY, "hash_type")
    hash_val = testing_data_config("images", FILE_KEY, "hash_val")
    download_url_or_skip_test(FILE_URL, FILE_PATH, hash_type=hash_type, hash_val=hash_val)


class SlidingPatchWSIDatasetTests:
    class Tests(unittest.TestCase):
        backend = None

        @parameterized.expand(
            [
                TEST_CASE_SMALL_0,
                TEST_CASE_SMALL_1,
                TEST_CASE_SMALL_2,
                TEST_CASE_SMALL_3,
                TEST_CASE_SMALL_4,
                TEST_CASE_SMALL_5,
                TEST_CASE_SMALL_6,
                TEST_CASE_SMALL_7,
                TEST_CASE_SMALL_8,
                TEST_CASE_SMALL_9,
            ]
        )
        def test_read_patches(self, input_parameters, expected):
            if self.backend == "openslide":
                return
            dataset = SlidingPatchWSIDataset(reader=self.backend, **input_parameters)
            self.assertEqual(len(dataset), len(expected))
            for i, sample in enumerate(dataset):
                self.assertTupleEqual(sample["image"].shape, expected[i]["image"].shape)

        @parameterized.expand([TEST_CASE_LARGE_0, TEST_CASE_LARGE_1])
        def test_read_patches_large(self, input_parameters, expected):
            dataset = SlidingPatchWSIDataset(reader=self.backend, **input_parameters)
            self.assertEqual(len(dataset), len(expected))
            for i, sample in enumerate(dataset):
                self.assertEqual(sample["image"].meta[WSIPatchKeys.LEVEL], expected[i]["patch_level"])
                assert_array_equal(sample["image"].meta[WSIPatchKeys.SIZE], expected[i]["patch_size"])
                steps = [round(expected[i]["ratio"] * s) for s in expected[i]["patch_size"]]
                expected_location = tuple(expected[i]["step_loc"][j] * steps[j] for j in range(len(steps)))
                assert_array_equal(sample["image"].meta[WSIPatchKeys.LOCATION], expected_location)


@skipUnless(has_cucim, "Requires cucim")
class TestSlidingPatchWSIDatasetCuCIM(SlidingPatchWSIDatasetTests.Tests):
    @classmethod
    def setUpClass(cls):
        cls.backend = "cucim"


@skipUnless(has_osl, "Requires openslide")
class TestSlidingPatchWSIDatasetOpenSlide(SlidingPatchWSIDatasetTests.Tests):
    @classmethod
    def setUpClass(cls):
        cls.backend = "openslide"


if __name__ == "__main__":
    unittest.main()
