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
from unittest import skipUnless

import numpy as np
from parameterized import parameterized

from monai.apps.pathology.metrics import LesionFROC
from monai.utils import optional_import

_cucim, has_cucim = optional_import("cucim")
has_cucim = has_cucim and hasattr(_cucim, "CuImage")
_, has_skimage = optional_import("skimage.measure")
_, has_sp = optional_import("scipy.ndimage")
imwrite, has_tif = optional_import("tifffile", name="imwrite")


def save_as_tif(filename, array):
    array = array[::-1, ...]  # Upside-down
    if not filename.endswith(".tif"):
        filename += ".tif"
    file_path = os.path.join("tests", "testing_data", filename)
    imwrite(file_path, array, compression="jpeg", tile=(16, 16))


def around(val, interval=3):
    return slice(val - interval, val + interval)


# mask and prediction image size
HEIGHT = 101
WIDTH = 800


def prepare_test_data():
    # -------------------------------------
    # Ground Truth - Binary Masks
    # -------------------------------------
    # ground truth with no tumor
    ground_truth = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    save_as_tif("temp_ground_truth_0", ground_truth)

    # ground truth with one tumor
    ground_truth[around(HEIGHT // 2), around(1 * WIDTH // 7)] = 1
    save_as_tif("temp_ground_truth_1", ground_truth)

    # ground truth with two tumors
    ground_truth[around(HEIGHT // 2), around(2 * WIDTH // 7)] = 1
    save_as_tif("temp_ground_truth_2", ground_truth)

    # ground truth with three tumors
    ground_truth[around(HEIGHT // 2), around(3 * WIDTH // 7)] = 1
    save_as_tif("temp_ground_truth_3", ground_truth)

    # ground truth with four tumors
    ground_truth[around(HEIGHT // 2), around(4 * WIDTH // 7)] = 1
    save_as_tif("temp_ground_truth_4", ground_truth)

    # -------------------------------------
    # predictions - Probability Maps
    # -------------------------------------

    # prediction with no tumor
    prob_map = np.zeros((HEIGHT, WIDTH))
    np.save("./tests/testing_data/temp_prob_map_0_0.npy", prob_map)

    # prediction with one incorrect tumor
    prob_map[HEIGHT // 2, 5 * WIDTH // 7] = 0.6
    np.save("./tests/testing_data/temp_prob_map_0_1.npy", prob_map)

    # prediction with correct first tumors and an incorrect tumor
    prob_map[HEIGHT // 2, 1 * WIDTH // 7] = 0.8
    np.save("./tests/testing_data/temp_prob_map_1_1.npy", prob_map)

    # prediction with correct firt two tumors and an incorrect tumor
    prob_map[HEIGHT // 2, 2 * WIDTH // 7] = 0.8
    np.save("./tests/testing_data/temp_prob_map_2_1.npy", prob_map)

    # prediction with two incorrect tumors
    prob_map = np.zeros((HEIGHT, WIDTH))
    prob_map[HEIGHT // 2, 5 * WIDTH // 7] = 0.6
    prob_map[HEIGHT // 2, 6 * WIDTH // 7] = 0.4
    np.save("./tests/testing_data/temp_prob_map_0_2.npy", prob_map)

    # prediction with correct first tumors and two incorrect tumors
    prob_map[HEIGHT // 2, 1 * WIDTH // 7] = 0.8
    np.save("./tests/testing_data/temp_prob_map_1_2.npy", prob_map)


TEST_CASE_0 = [
    {
        "data": [
            {
                "prob_map": "./tests/testing_data/temp_prob_map_0_0.npy",
                "tumor_mask": "./tests/testing_data/temp_ground_truth_0.tif",
                "level": 0,
                "pixel_spacing": 1,
            }
        ],
        "grow_distance": 2,
        "itc_diameter": 0,
    },
    np.nan,
]

TEST_CASE_1 = [
    {
        "data": [
            {
                "prob_map": "./tests/testing_data/temp_prob_map_0_0.npy",
                "tumor_mask": "./tests/testing_data/temp_ground_truth_1.tif",
                "level": 0,
                "pixel_spacing": 1,
            }
        ],
        "grow_distance": 2,
        "itc_diameter": 0,
    },
    0.0,
]

TEST_CASE_2 = [
    {
        "data": [
            {
                "prob_map": "./tests/testing_data/temp_prob_map_1_1.npy",
                "tumor_mask": "./tests/testing_data/temp_ground_truth_1.tif",
                "level": 0,
                "pixel_spacing": 1,
            }
        ],
        "grow_distance": 2,
        "itc_diameter": 0,
    },
    1.0,
]

TEST_CASE_3 = [
    {
        "data": [
            {
                "prob_map": "./tests/testing_data/temp_prob_map_2_1.npy",
                "tumor_mask": "./tests/testing_data/temp_ground_truth_1.tif",
                "level": 0,
                "pixel_spacing": 1,
            }
        ],
        "grow_distance": 2,
        "itc_diameter": 0,
    },
    1.0,
]

TEST_CASE_4 = [
    {
        "data": [
            {
                "prob_map": "./tests/testing_data/temp_prob_map_2_1.npy",
                "tumor_mask": "./tests/testing_data/temp_ground_truth_2.tif",
                "level": 0,
                "pixel_spacing": 1,
            }
        ],
        "grow_distance": 2,
        "itc_diameter": 0,
    },
    1.0,
]

TEST_CASE_5 = [
    {
        "data": [
            {
                "prob_map": "./tests/testing_data/temp_prob_map_1_2.npy",
                "tumor_mask": "./tests/testing_data/temp_ground_truth_2.tif",
                "level": 0,
                "pixel_spacing": 1,
            }
        ],
        "grow_distance": 2,
        "itc_diameter": 0,
    },
    0.5,
]

TEST_CASE_6 = [
    {
        "data": [
            {
                "prob_map": "./tests/testing_data/temp_prob_map_1_1.npy",
                "tumor_mask": "./tests/testing_data/temp_ground_truth_1.tif",
                "level": 0,
                "pixel_spacing": 1,
            },
            {
                "prob_map": "./tests/testing_data/temp_prob_map_1_2.npy",
                "tumor_mask": "./tests/testing_data/temp_ground_truth_2.tif",
                "level": 0,
                "pixel_spacing": 1,
            },
        ],
        "grow_distance": 2,
        "itc_diameter": 0,
    },
    2.0 / 3.0,
]

TEST_CASE_7 = [
    {
        "data": [
            {
                "prob_map": "./tests/testing_data/temp_prob_map_1_1.npy",
                "tumor_mask": "./tests/testing_data/temp_ground_truth_3.tif",
                "level": 0,
                "pixel_spacing": 1,
            },
            {
                "prob_map": "./tests/testing_data/temp_prob_map_1_2.npy",
                "tumor_mask": "./tests/testing_data/temp_ground_truth_2.tif",
                "level": 0,
                "pixel_spacing": 1,
            },
        ],
        "grow_distance": 2,
        "itc_diameter": 0,
    },
    0.4,
]

TEST_CASE_8 = [
    {
        "data": [
            {
                "prob_map": "./tests/testing_data/temp_prob_map_0_1.npy",
                "tumor_mask": "./tests/testing_data/temp_ground_truth_1.tif",
                "level": 0,
                "pixel_spacing": 1,
            },
            {
                "prob_map": "./tests/testing_data/temp_prob_map_1_1.npy",
                "tumor_mask": "./tests/testing_data/temp_ground_truth_3.tif",
                "level": 0,
                "pixel_spacing": 1,
            },
            {
                "prob_map": "./tests/testing_data/temp_prob_map_1_2.npy",
                "tumor_mask": "./tests/testing_data/temp_ground_truth_2.tif",
                "level": 0,
                "pixel_spacing": 1,
            },
        ],
        "grow_distance": 2,
        "itc_diameter": 0,
    },
    1.0 / 3.0,
]

TEST_CASE_9 = [
    {
        "data": [
            {
                "prob_map": "./tests/testing_data/temp_prob_map_0_2.npy",
                "tumor_mask": "./tests/testing_data/temp_ground_truth_4.tif",
                "level": 0,
                "pixel_spacing": 1,
            },
            {
                "prob_map": "./tests/testing_data/temp_prob_map_1_1.npy",
                "tumor_mask": "./tests/testing_data/temp_ground_truth_3.tif",
                "level": 0,
                "pixel_spacing": 1,
            },
            {
                "prob_map": "./tests/testing_data/temp_prob_map_1_2.npy",
                "tumor_mask": "./tests/testing_data/temp_ground_truth_2.tif",
                "level": 0,
                "pixel_spacing": 1,
            },
        ],
        "grow_distance": 2,
        "itc_diameter": 0,
    },
    2.0 / 9.0,
]


class TestEvaluateTumorFROC(unittest.TestCase):

    @skipUnless(has_cucim, "Requires cucim")
    @skipUnless(has_skimage, "Requires skimage")
    @skipUnless(has_sp, "Requires scipy")
    @skipUnless(has_tif, "Requires tifffile")
    def setUp(self):
        prepare_test_data()

    @parameterized.expand(
        [
            TEST_CASE_0,
            TEST_CASE_1,
            TEST_CASE_2,
            TEST_CASE_3,
            TEST_CASE_4,
            TEST_CASE_5,
            TEST_CASE_6,
            TEST_CASE_7,
            TEST_CASE_8,
            TEST_CASE_9,
        ]
    )
    def test_read_patches_cucim(self, input_parameters, expected):
        froc = LesionFROC(**input_parameters)
        froc_score = froc.evaluate()
        if np.isnan(expected):
            self.assertTrue(np.isnan(froc_score))
        else:
            self.assertAlmostEqual(froc_score, expected)


if __name__ == "__main__":
    unittest.main()
