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

import unittest

import numpy as np
from parameterized import parameterized

from monai.transforms.intensity.array import ForegroundMask
from monai.utils import min_version, optional_import, set_determinism
from tests.util import TEST_NDARRAYS, assert_allclose

skimage, has_skimage = optional_import("skimage", "0.19.0", min_version)
set_determinism(1234)

A = np.random.randint(64, 128, (3, 3, 2)).astype(np.uint8)
A3D = np.random.randint(64, 128, (3, 3, 2, 2)).astype(np.uint8)
B = np.ones_like(A[:1])
B3D = np.ones_like(A3D[:1])
MASK = np.pad(B, ((0, 0), (2, 2), (2, 2)), constant_values=0)
MASK3D = np.pad(B3D, ((0, 0), (2, 2), (2, 2), (2, 2)), constant_values=0)
IMAGE1 = np.pad(A, ((0, 0), (2, 2), (2, 2)), constant_values=255)
IMAGE3D = np.pad(A3D, ((0, 0), (2, 2), (2, 2), (2, 2)), constant_values=255)
IMAGE2 = np.copy(IMAGE1)
IMAGE2[0] = 0
IMAGE3 = np.pad(A, ((0, 0), (2, 2), (2, 2)), constant_values=0)
TEST_CASE_0 = [{}, IMAGE1, MASK]
TEST_CASE_1 = [{"threshold": "otsu"}, IMAGE1, MASK]
TEST_CASE_2 = [{"threshold": "otsu"}, IMAGE2, MASK]
TEST_CASE_3 = [{"threshold": 140}, IMAGE1, MASK]
TEST_CASE_4 = [{"threshold": "otsu", "invert": True}, IMAGE3, MASK]
TEST_CASE_5 = [{"threshold": 0.5}, MASK, np.logical_not(MASK)]
TEST_CASE_6 = [{"threshold": 140}, IMAGE2, np.ones_like(MASK)]
TEST_CASE_7 = [{"threshold": {"R": "otsu", "G": "otsu", "B": "otsu"}}, IMAGE2, MASK]
TEST_CASE_8 = [{"threshold": {"R": 140, "G": "otsu", "B": "otsu"}}, IMAGE2, np.ones_like(MASK)]
TEST_CASE_9 = [{"threshold": {"R": 140, "G": skimage.filters.threshold_otsu, "B": "otsu"}}, IMAGE2, np.ones_like(MASK)]
TEST_CASE_10 = [{"threshold": skimage.filters.threshold_mean}, IMAGE1, MASK]
TEST_CASE_11 = [{"threshold": None, "hsv_threshold": "otsu"}, IMAGE1, np.ones_like(MASK)]
TEST_CASE_12 = [{"threshold": None, "hsv_threshold": {"S": "otsu"}}, IMAGE1, MASK]
TEST_CASE_13 = [{"threshold": 100, "invert": True}, IMAGE1, np.logical_not(MASK)]
TEST_CASE_14 = [{}, IMAGE3D, MASK3D]
TEST_CASE_15 = [{"hsv_threshold": {"S": 0.1}}, IMAGE3D, MASK3D]

TEST_CASE_ERROR_1 = [{"threshold": None}, IMAGE1]
TEST_CASE_ERROR_2 = [{"threshold": {"K": 1}}, IMAGE1]

TESTS = []
for p in TEST_NDARRAYS:
    TESTS.append([p, *TEST_CASE_0])
    TESTS.append([p, *TEST_CASE_1])
    TESTS.append([p, *TEST_CASE_2])
    TESTS.append([p, *TEST_CASE_3])
    TESTS.append([p, *TEST_CASE_4])
    TESTS.append([p, *TEST_CASE_5])
    TESTS.append([p, *TEST_CASE_6])
    TESTS.append([p, *TEST_CASE_7])
    TESTS.append([p, *TEST_CASE_8])
    TESTS.append([p, *TEST_CASE_9])
    TESTS.append([p, *TEST_CASE_10])
    TESTS.append([p, *TEST_CASE_11])
    TESTS.append([p, *TEST_CASE_12])
    TESTS.append([p, *TEST_CASE_13])
    TESTS.append([p, *TEST_CASE_14])
    TESTS.append([p, *TEST_CASE_15])

TESTS_ERROR = []
for p in TEST_NDARRAYS:
    TESTS_ERROR.append([p, *TEST_CASE_ERROR_1])
    TESTS_ERROR.append([p, *TEST_CASE_ERROR_2])


@unittest.skipUnless(has_skimage, "Requires sci-kit image")
class TestForegroundMask(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_foreground_mask(self, in_type, arguments, image, mask):
        input_image = in_type(image)
        result = ForegroundMask(**arguments)(input_image)
        assert_allclose(result, mask, type_test="tensor")

    @parameterized.expand(TESTS_ERROR)
    def test_foreground_mask_error(self, in_type, arguments, image):
        input_image = in_type(image)
        with self.assertRaises(ValueError):
            ForegroundMask(**arguments)(input_image)


if __name__ == "__main__":
    unittest.main()
