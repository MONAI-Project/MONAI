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

import torch
from parameterized import parameterized

from monai.data import CuCIMWSIReader, ImageReader, OpenSlideWSIReader, WSIReader
from monai.inferers import WSISlidingWindowSplitter
from tests.util import download_url_or_skip_test, optional_import, testing_data_config

cucim, has_cucim = optional_import("cucim")
has_cucim = has_cucim and hasattr(cucim, "CuImage")
_, has_osl = optional_import("openslide")

WSI_READER_STR = None
WSI_READER_CLASS: type[CuCIMWSIReader] | type[OpenSlideWSIReader] | None = None
if has_cucim:
    WSI_READER_STR = "cuCIM"
    WSI_READER_CLASS = CuCIMWSIReader
elif has_osl:
    WSI_READER_STR = "OpenSlide"
    WSI_READER_CLASS = OpenSlideWSIReader

WSI_GENERIC_TIFF_KEY = "wsi_generic_tiff"
TESTS_PATH = Path(__file__).parents[1]
WSI_GENERIC_TIFF_PATH = os.path.join(TESTS_PATH, "testing_data", f"temp_{WSI_GENERIC_TIFF_KEY}.tiff")

HEIGHT = 32914
WIDTH = 46000

# ----------------------------------------------------------------------------
# WSI test cases
# ----------------------------------------------------------------------------

TEST_CASE_WSI_0_BASE = [
    WSI_GENERIC_TIFF_PATH,
    {"patch_size": (1000, 1000), "reader": WSI_READER_STR},
    {"locations": [(0, 0), (0, 1000), (0, 2000), (0, 3000)]},
]

TEST_CASE_WSI_1_BASE = [
    WSI_GENERIC_TIFF_PATH,
    {"patch_size": (1000, 1000), "level": 1, "reader": WSI_READER_STR},
    {"locations": [(0, 0), (0, 1000), (0, 2000), (0, 3000)]},
]

# Check readers
if WSI_READER_STR is not None:
    TEST_CASE_WSI_2_READER = [
        WSI_GENERIC_TIFF_PATH,
        {"patch_size": (1000, 1000), "reader": WSIReader(backend=WSI_READER_STR)},
        {"locations": [(0, 0), (0, 1000), (0, 2000), (0, 3000)]},
    ]
else:
    TEST_CASE_WSI_2_READER = []
TEST_CASE_WSI_3_READER = [
    WSI_GENERIC_TIFF_PATH,
    {"patch_size": (1000, 1000), "overlap": 0.0, "reader": WSIReader, "backend": WSI_READER_STR},
    {"locations": [(0, 0), (0, 1000), (0, 2000), (0, 3000)]},
]
TEST_CASE_WSI_4_READER = [
    WSI_GENERIC_TIFF_PATH,
    {"patch_size": (1000, 1000), "overlap": 0.0, "reader": WSI_READER_CLASS},
    {"locations": [(0, 0), (0, 1000), (0, 2000), (0, 3000)]},
]
TEST_CASE_WSI_5_READER = [
    WSI_GENERIC_TIFF_PATH,
    {"patch_size": (1000, 1000), "overlap": 0.0, "level": 1, "reader": WSI_READER_STR},
    {"locations": [(0, 0), (0, 1000), (0, 2000), (0, 3000)]},
]

# Check overlaps
TEST_CASE_WSI_6_OVERLAP = [
    WSI_GENERIC_TIFF_PATH,
    {"patch_size": (1000, 1000), "overlap": 0.0, "reader": WSI_READER_STR},
    {"locations": [(0, 0), (0, 1000), (0, 2000), (0, 3000)]},
]
TEST_CASE_WSI_7_OVERLAP = [
    WSI_GENERIC_TIFF_PATH,
    {"patch_size": (1000, 1000), "overlap": 0.5, "reader": WSI_READER_STR},
    {"locations": [(0, 0), (0, 500), (0, 1000), (0, 1500)]},
]
TEST_CASE_WSI_8_OVERLAP = [
    WSI_GENERIC_TIFF_PATH,
    {"patch_size": (1000, 1000), "overlap": 0.999, "reader": WSI_READER_STR},
    {"locations": [(0, 0), (0, 1), (0, 2), (0, 3)]},
]


# Filtering functions test cases
def gen_location_filter(locations):
    def my_filter(patch, loc):
        if loc in locations:
            return False
        return True

    return my_filter


TEST_CASE_WSI_9_FILTER = [
    WSI_GENERIC_TIFF_PATH,
    {"patch_size": (1000, 1000), "reader": WSI_READER_STR, "filter_fn": gen_location_filter([(0, 0), (0, 2000)])},
    {"locations": [(0, 1000), (0, 3000)]},
]


# ----------------------------------------------------------------------------
# Error test cases
# ----------------------------------------------------------------------------
def extra_parameter_filter(patch, location, extra):
    return


def missing_parameter_filter(patch):
    return


# invalid overlap: float 1.0
TEST_CASE_ERROR_0 = [
    WSI_GENERIC_TIFF_PATH,
    {"patch_size": (2, 2), "overlap": 1.0, "reader": WSI_READER_STR},
    ValueError,
]
# invalid overlap: negative float
TEST_CASE_ERROR_1 = [
    WSI_GENERIC_TIFF_PATH,
    {"patch_size": (2, 2), "overlap": -0.1, "reader": WSI_READER_STR},
    ValueError,
]
# invalid overlap: negative integer
TEST_CASE_ERROR_2 = [WSI_GENERIC_TIFF_PATH, {"patch_size": (2, 2), "overlap": -1, "reader": WSI_READER_STR}, ValueError]
# invalid overlap: integer larger than patch size
TEST_CASE_ERROR_3 = [WSI_GENERIC_TIFF_PATH, {"patch_size": (2, 2), "overlap": 3, "reader": WSI_READER_STR}, ValueError]

# invalid offset: positive and larger than image size
TEST_CASE_ERROR_4 = [
    WSI_GENERIC_TIFF_PATH,
    {"patch_size": (2, 2), "offset": WIDTH, "reader": WSI_READER_STR},
    ValueError,
]
# invalid offset: negative and larger than patch size (in magnitude)
TEST_CASE_ERROR_5 = [
    WSI_GENERIC_TIFF_PATH,
    {"patch_size": (2, 2), "offset": -3, "pad_mode": "constant", "reader": WSI_READER_STR},
    ValueError,
]
# invalid offset: negative and no padding
TEST_CASE_ERROR_6 = [
    WSI_GENERIC_TIFF_PATH,
    {"patch_size": (2, 2), "pad_mode": None, "offset": -1, "reader": WSI_READER_STR},
    ValueError,
]

# invalid filter function: with more than two positional parameters
TEST_CASE_ERROR_7 = [
    WSI_GENERIC_TIFF_PATH,
    {"patch_size": (2, 2), "filter_fn": extra_parameter_filter, "reader": WSI_READER_STR},
    ValueError,
]
# invalid filter function: with less than two positional parameters
TEST_CASE_ERROR_8 = [
    WSI_GENERIC_TIFF_PATH,
    {"patch_size": (2, 2), "filter_fn": missing_parameter_filter, "reader": WSI_READER_STR},
    ValueError,
]
# invalid filter function: non-callable
TEST_CASE_ERROR_9 = [
    WSI_GENERIC_TIFF_PATH,
    {"patch_size": (2, 2), "filter_fn": 1, "reader": WSI_READER_STR},
    ValueError,
]

# invalid reader
TEST_CASE_ERROR_10 = [WSI_GENERIC_TIFF_PATH, {"patch_size": (2, 2), "reader": ImageReader}, ValueError]


@skipUnless(WSI_READER_STR, "Requires cucim or openslide!")
def setUpModule():
    download_url_or_skip_test(
        testing_data_config("images", WSI_GENERIC_TIFF_KEY, "url"),
        WSI_GENERIC_TIFF_PATH,
        hash_type=testing_data_config("images", WSI_GENERIC_TIFF_KEY, "hash_type"),
        hash_val=testing_data_config("images", WSI_GENERIC_TIFF_KEY, "hash_val"),
    )


class WSISlidingWindowSplitterTests(unittest.TestCase):
    @parameterized.expand(
        [
            TEST_CASE_WSI_0_BASE,
            TEST_CASE_WSI_1_BASE,
            TEST_CASE_WSI_2_READER,
            TEST_CASE_WSI_3_READER,
            TEST_CASE_WSI_4_READER,
            TEST_CASE_WSI_5_READER,
            TEST_CASE_WSI_6_OVERLAP,
            TEST_CASE_WSI_7_OVERLAP,
            TEST_CASE_WSI_8_OVERLAP,
            TEST_CASE_WSI_9_FILTER,
        ]
    )
    def test_split_patches_wsi(self, filepath, arguments, expected):
        patches = WSISlidingWindowSplitter(**arguments)(filepath)
        for sample, expected_loc in zip(patches, expected["locations"]):
            patch = sample[0]
            loc = sample[1]
            self.assertTrue(isinstance(patch, torch.Tensor))
            self.assertTupleEqual(patch.shape[2:], arguments["patch_size"])
            self.assertTrue(isinstance(loc, tuple))
            self.assertTupleEqual(loc, expected_loc)

    @parameterized.expand(
        [
            TEST_CASE_ERROR_0,
            TEST_CASE_ERROR_1,
            TEST_CASE_ERROR_2,
            TEST_CASE_ERROR_3,
            TEST_CASE_ERROR_4,
            TEST_CASE_ERROR_5,
            TEST_CASE_ERROR_6,
            TEST_CASE_ERROR_7,
            TEST_CASE_ERROR_8,
            TEST_CASE_ERROR_9,
            TEST_CASE_ERROR_10,
        ]
    )
    def test_split_patches_errors(self, image, arguments, expected_error):
        with self.assertRaises(expected_error):
            patches = WSISlidingWindowSplitter(**arguments)(image)
            patches = list(patches)


if __name__ == "__main__":
    unittest.main()
