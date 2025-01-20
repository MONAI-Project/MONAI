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

from monai.apps.deepgrow.transforms import (
    AddGuidanceFromPointsd,
    AddGuidanceSignald,
    AddInitialSeedPointd,
    AddRandomGuidanced,
    Fetch2DSliced,
    FindAllValidSlicesd,
    FindDiscrepancyRegionsd,
    ResizeGuidanced,
    RestoreLabeld,
    SpatialCropForegroundd,
    SpatialCropGuidanced,
)
from monai.utils.enums import PostFix

IMAGE = np.array([[[[1, 0, 2, 0, 1], [0, 1, 2, 1, 0], [2, 2, 3, 2, 2], [0, 1, 2, 1, 0], [1, 0, 2, 0, 1]]]])
LABEL = np.array([[[[0, 0, 0, 0, 0], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0], [0, 1, 0, 1, 0], [0, 0, 0, 0, 0]]]])

DATA_1 = {"image": IMAGE, "label": LABEL, PostFix.meta("image"): {}, PostFix.meta("label"): {}}

DATA_2 = {
    "image": np.array(
        [
            [
                [[1, 2, 3, 2, 1], [1, 1, 3, 2, 1], [0, 0, 0, 0, 0], [1, 1, 1, 2, 1], [0, 2, 2, 2, 1]],
                [[1, 0, 2, 0, 1], [0, 1, 2, 1, 0], [2, 2, 3, 2, 2], [0, 1, 2, 1, 0], [1, 0, 2, 0, 1]],
            ]
        ]
    ),
    "label": np.array(
        [
            [
                [[0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0], [0, 0, 1, 0, 0]],
                [[0, 0, 0, 0, 0], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0], [0, 1, 0, 1, 0], [0, 0, 0, 0, 0]],
            ]
        ]
    ),
    "guidance": np.array([[[1, 0, 2, 2], [1, 1, 2, 2]], [[-1, -1, -1, -1], [-1, -1, -1, -1]]]),
}

DATA_3 = {
    "image": IMAGE,
    "label": LABEL,
    "pred": np.array([[[[0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 1, 1, 0, 0], [0, 1, 0, 1, 0], [0, 0, 0, 0, 0]]]]),
}

DATA_4 = {
    "image": IMAGE,
    "label": LABEL,
    "guidance": np.array([[[1, 0, 2, 2]], [[-1, -1, -1, -1]]]),
    "discrepancy": np.array(
        [
            [[[[0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]],
            [[[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]],
        ]
    ),
    "probability": 1.0,
}

DATA_5 = {
    "image": np.arange(25).reshape((1, 5, 5)),
    PostFix.meta("image"): {"spatial_shape": [5, 5, 1]},
    "foreground": [[2, 2, 0]],
    "background": [],
}

DATA_6 = {
    "image": np.arange(25).reshape((1, 5, 5)),
    PostFix.meta("image"): {"spatial_shape": [5, 2, 1]},
    "foreground": [[2, 1, 0]],
    "background": [[1, 0, 0]],
}

DATA_7 = {
    "image": np.arange(500).reshape((5, 10, 10)),
    PostFix.meta("image"): {"spatial_shape": [20, 20, 10]},
    "foreground": [[10, 14, 6], [10, 14, 8]],
    "background": [[10, 16, 8]],
    "slice": 6,
}

DATA_8 = {
    "image": np.arange(500).reshape((1, 5, 10, 10)),
    PostFix.meta("image"): {"spatial_shape": [20, 20, 10]},
    "guidance": [[[3, 5, 7], [4, 5, 7]], [[4, 5, 8]]],
}

DATA_9 = {
    "image": np.arange(1000).reshape((1, 5, 10, 20)),
    PostFix.meta("image"): {"foreground_cropped_shape": (1, 10, 20, 40)},
    "guidance": [[[6, 10, 14], [8, 10, 14]], [[8, 10, 16]]],
}

DATA_10 = {
    "image": np.arange(9).reshape((1, 1, 3, 3)),
    PostFix.meta("image"): {
        "spatial_shape": [3, 3, 1],
        "foreground_start_coord": np.array([0, 0, 0]),
        "foreground_end_coord": np.array([1, 3, 3]),
        "foreground_original_shape": (1, 1, 3, 3),
        "foreground_cropped_shape": (1, 1, 3, 3),
        "original_affine": np.array(
            [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]
        ),
    },
    "pred": np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]),
}

DATA_11 = {
    "image": np.arange(500).reshape((1, 5, 10, 10)),
    PostFix.meta("image"): {
        "spatial_shape": [20, 20, 10],
        "foreground_start_coord": np.array([2, 2, 2]),
        "foreground_end_coord": np.array([4, 4, 4]),
        "foreground_original_shape": (1, 5, 10, 10),
        "foreground_cropped_shape": (1, 2, 2, 2),
        "original_affine": np.array(
            [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]
        ),
    },
    "pred": np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]),
}

DATA_12 = {"image": np.arange(27).reshape(3, 3, 3), PostFix.meta("image"): {}, "guidance": [[0, 0, 0], [0, 1, 1], 1]}

DATA_13 = {
    "image": np.arange(64).reshape((1, 4, 4, 4)),
    PostFix.meta("image"): {
        "spatial_shape": [8, 8, 4],
        "foreground_start_coord": np.array([1, 1, 1]),
        "foreground_end_coord": np.array([3, 3, 3]),
        "foreground_original_shape": (1, 4, 4, 4),
        "foreground_cropped_shape": (1, 2, 2, 2),
        "original_affine": np.array(
            [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]]
        ),
    },
    "pred": np.array([[[[10, 20], [30, 40]], [[50, 60], [70, 80]]]]),
}

FIND_SLICE_TEST_CASE_1 = [{"label": "label", "sids": "sids"}, DATA_1, [0]]

FIND_SLICE_TEST_CASE_2 = [{"label": "label", "sids": "sids"}, DATA_2, [0, 1]]

CROP_TEST_CASE_1 = [
    {
        "keys": ["image", "label"],
        "source_key": "label",
        "select_fn": lambda x: x > 0,
        "channel_indices": None,
        "margin": 0,
        "spatial_size": [1, 4, 4],
    },
    DATA_1,
    np.array([[[[1, 2, 1], [2, 3, 2], [1, 2, 1]]]]),
]

CROP_TEST_CASE_2 = [
    {
        "keys": ["image", "label"],
        "source_key": "label",
        "select_fn": lambda x: x > 0,
        "channel_indices": None,
        "margin": 0,
        "spatial_size": [2, 4, 4],
    },
    DATA_1,
    np.array([1, 1, 4, 4]),
]

ADD_INITIAL_POINT_TEST_CASE_1 = [
    {"label": "label", "guidance": "guidance", "sids": "sids"},
    DATA_1,
    "[[[1, 0, 2, 2]], [[-1, -1, -1, -1]]]",
]

ADD_GUIDANCE_TEST_CASE_1 = [
    {"image": "image", "guidance": "guidance"},
    DATA_2,
    np.array(
        [
            [
                [[1, 2, 3, 2, 1], [1, 1, 3, 2, 1], [0, 0, 0, 0, 0], [1, 1, 1, 2, 1], [0, 2, 2, 2, 1]],
                [[1, 0, 2, 0, 1], [0, 1, 2, 1, 0], [2, 2, 3, 2, 2], [0, 1, 2, 1, 0], [1, 0, 2, 0, 1]],
            ],
            [
                [
                    [0.0, 0.26689214, 0.37996644, 0.26689214, 0.0],
                    [0.26689214, 0.65222847, 0.81548417, 0.65222847, 0.26689214],
                    [0.37996635, 0.81548399, 1.0, 0.81548399, 0.37996635],
                    [0.26689214, 0.65222847, 0.81548417, 0.65222847, 0.26689214],
                    [0.0, 0.26689214, 0.37996644, 0.26689214, 0.0],
                ],
                [
                    [0.0, 0.26689214, 0.37996644, 0.26689214, 0.0],
                    [0.26689214, 0.65222847, 0.81548417, 0.65222847, 0.26689214],
                    [0.37996635, 0.81548399, 1.0, 0.81548399, 0.37996635],
                    [0.26689214, 0.65222847, 0.81548417, 0.65222847, 0.26689214],
                    [0.0, 0.26689214, 0.37996644, 0.26689214, 0.0],
                ],
            ],
            [
                [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            ],
        ]
    ),
]

FIND_DISCREPANCY_TEST_CASE_1 = [
    {"label": "label", "pred": "pred", "discrepancy": "discrepancy"},
    DATA_3,
    np.array(
        [
            [[[[0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]],
            [[[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]],
        ]
    ),
]

ADD_RANDOM_GUIDANCE_TEST_CASE_1 = [
    {"guidance": "guidance", "discrepancy": "discrepancy", "probability": "probability"},
    DATA_4,
    "[[[1, 0, 2, 2], [1, 0, 1, 3]], [[-1, -1, -1, -1], [-1, -1, -1, -1]]]",
]

ADD_GUIDANCE_FROM_POINTS_TEST_CASE_1 = [
    {"ref_image": "image", "spatial_dims": 3, "guidance": "guidance", "depth_first": True},
    DATA_5,
    [[0, 2, 2]],
    [],
]

ADD_GUIDANCE_FROM_POINTS_TEST_CASE_2 = [
    {"ref_image": "image", "spatial_dims": 3, "guidance": "guidance", "depth_first": True},
    DATA_6,
    [[0, 2, 2]],
    [[0, 1, 0]],
]

ADD_GUIDANCE_FROM_POINTS_TEST_CASE_3 = [
    {"ref_image": "image", "spatial_dims": 3, "guidance": "guidance", "depth_first": True},
    DATA_7,
    [[3, 5, 7], [4, 5, 7]],
    [[4, 5, 8]],
]

ADD_GUIDANCE_FROM_POINTS_TEST_CASE_4 = [
    {"ref_image": "image", "spatial_dims": 2, "guidance": "guidance", "depth_first": True},
    DATA_6,
    [[2, 2]],
    [[1, 0]],
]

ADD_GUIDANCE_FROM_POINTS_TEST_CASE_5 = [
    {"ref_image": "image", "spatial_dims": 2, "guidance": "guidance", "depth_first": True, "slice_key": "slice"},
    DATA_7,
    [[5, 7]],
    [],
]

ADD_GUIDANCE_FROM_POINTS_TEST_CASE_6 = [
    {"ref_image": "image", "spatial_dims": 2, "guidance": "guidance", "depth_first": True},
    DATA_5,
    [[2, 2]],
    [],
]

SPATIAL_CROP_GUIDANCE_TEST_CASE_1 = [
    {"keys": ["image"], "guidance": "guidance", "spatial_size": [1, 4, 4], "margin": 0},
    DATA_8,
    np.array([[[[357, 358]], [[457, 458]]]]),
]

SPATIAL_CROP_GUIDANCE_TEST_CASE_2 = [
    {"keys": ["image"], "guidance": "guidance", "spatial_size": [2, 2], "margin": 1},
    DATA_8,
    np.array(
        [
            [
                [[246, 247, 248, 249], [256, 257, 258, 259], [266, 267, 268, 269]],
                [[346, 347, 348, 349], [356, 357, 358, 359], [366, 367, 368, 369]],
                [[446, 447, 448, 449], [456, 457, 458, 459], [466, 467, 468, 469]],
            ]
        ]
    ),
]

SPATIAL_CROP_GUIDANCE_TEST_CASE_3 = [
    {"keys": ["image"], "guidance": "guidance", "spatial_size": [3, 3], "margin": 0},
    DATA_8,
    np.array(
        [
            [
                [[47, 48, 49], [57, 58, 59], [67, 68, 69]],
                [[147, 148, 149], [157, 158, 159], [167, 168, 169]],
                [[247, 248, 249], [257, 258, 259], [267, 268, 269]],
                [[347, 348, 349], [357, 358, 359], [367, 368, 369]],
                [[447, 448, 449], [457, 458, 459], [467, 468, 469]],
            ]
        ]
    ),
]

RESIZE_GUIDANCE_TEST_CASE_1 = [
    {"ref_image": "image", "guidance": "guidance"},
    DATA_9,
    [[[3, 5, 7], [4, 5, 7]], [[4, 5, 8]]],
]

RESTORE_LABEL_TEST_CASE_1 = [
    {"keys": ["pred"], "ref_image": "image", "mode": "nearest"},
    DATA_10,
    np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]),
]

RESULT = np.zeros((10, 20, 20))
RESULT[4:8, 4:8, 4:8] = np.array(
    [
        [[1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0], [3.0, 3.0, 4.0, 4.0], [3.0, 3.0, 4.0, 4.0]],
        [[1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0], [3.0, 3.0, 4.0, 4.0], [3.0, 3.0, 4.0, 4.0]],
        [[5.0, 5.0, 6.0, 6.0], [5.0, 5.0, 6.0, 6.0], [7.0, 7.0, 8.0, 8.0], [7.0, 7.0, 8.0, 8.0]],
        [[5.0, 5.0, 6.0, 6.0], [5.0, 5.0, 6.0, 6.0], [7.0, 7.0, 8.0, 8.0], [7.0, 7.0, 8.0, 8.0]],
    ]
)

RESTORE_LABEL_TEST_CASE_2 = [{"keys": ["pred"], "ref_image": "image", "mode": "nearest"}, DATA_11, RESULT]

RESTORE_LABEL_TEST_CASE_3_RESULT = np.zeros((10, 20, 20))
RESTORE_LABEL_TEST_CASE_3_RESULT[:5, 0:10, 0:10] = 1
RESTORE_LABEL_TEST_CASE_3_RESULT[:5, 0:10, 10:20] = 2
RESTORE_LABEL_TEST_CASE_3_RESULT[:5, 10:20, 0:10] = 3
RESTORE_LABEL_TEST_CASE_3_RESULT[:5, 10:20, 10:20] = 4
RESTORE_LABEL_TEST_CASE_3_RESULT[5:10, 0:10, 0:10] = 5
RESTORE_LABEL_TEST_CASE_3_RESULT[5:10, 0:10, 10:20] = 6
RESTORE_LABEL_TEST_CASE_3_RESULT[5:10, 10:20, 0:10] = 7
RESTORE_LABEL_TEST_CASE_3_RESULT[5:10, 10:20, 10:20] = 8

RESTORE_LABEL_TEST_CASE_3 = [
    {"keys": ["pred"], "ref_image": "image", "mode": "nearest", "restore_cropping": False},
    DATA_11,
    RESTORE_LABEL_TEST_CASE_3_RESULT,
]

RESTORE_LABEL_TEST_CASE_4_RESULT = np.zeros((4, 8, 8))
RESTORE_LABEL_TEST_CASE_4_RESULT[1, 2:6, 2:6] = np.array(
    [[10.0, 10.0, 20.0, 20.0], [10.0, 10.0, 20.0, 20.0], [30.0, 30.0, 40.0, 40.0], [30.0, 30.0, 40.0, 40.0]]
)
RESTORE_LABEL_TEST_CASE_4_RESULT[2, 2:6, 2:6] = np.array(
    [[50.0, 50.0, 60.0, 60.0], [50.0, 50.0, 60.0, 60.0], [70.0, 70.0, 80.0, 80.0], [70.0, 70.0, 80.0, 80.0]]
)

RESTORE_LABEL_TEST_CASE_4 = [
    {"keys": ["pred"], "ref_image": "image", "mode": "nearest", "restore_resizing": False},
    DATA_13,
    RESTORE_LABEL_TEST_CASE_4_RESULT,
]

RESTORE_LABEL_TEST_CASE_5_RESULT = np.zeros((4, 4, 4))
RESTORE_LABEL_TEST_CASE_5_RESULT[1, 1:3, 1:3] = np.array([[10.0, 20.0], [30.0, 40.0]])
RESTORE_LABEL_TEST_CASE_5_RESULT[2, 1:3, 1:3] = np.array([[50.0, 60.0], [70.0, 80.0]])

RESTORE_LABEL_TEST_CASE_5 = [
    {"keys": ["pred"], "ref_image": "image", "mode": "nearest", "restore_spacing": False},
    DATA_13,
    RESTORE_LABEL_TEST_CASE_5_RESULT,
]

RESTORE_LABEL_TEST_CASE_6_RESULT = np.zeros((1, 4, 8, 8))
RESTORE_LABEL_TEST_CASE_6_RESULT[-1, 1, 2:6, 2:6] = np.array(
    [[10.0, 10.0, 20.0, 20.0], [10.0, 10.0, 20.0, 20.0], [30.0, 30.0, 40.0, 40.0], [30.0, 30.0, 40.0, 40.0]]
)
RESTORE_LABEL_TEST_CASE_6_RESULT[-1, 2, 2:6, 2:6] = np.array(
    [[50.0, 50.0, 60.0, 60.0], [50.0, 50.0, 60.0, 60.0], [70.0, 70.0, 80.0, 80.0], [70.0, 70.0, 80.0, 80.0]]
)

RESTORE_LABEL_TEST_CASE_6 = [
    {"keys": ["pred"], "ref_image": "image", "mode": "nearest", "restore_slicing": False},
    DATA_13,
    RESTORE_LABEL_TEST_CASE_6_RESULT,
]

RESTORE_LABEL_TEST_CASE_7 = [
    {
        "keys": ["pred"],
        "ref_image": "image",
        "mode": "nearest",
        "restore_resizing": False,
        "restore_cropping": False,
        "restore_spacing": False,
        "restore_slicing": False,
    },
    DATA_11,
    np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]),
]

FETCH_2D_SLICE_TEST_CASE_1 = [
    {"keys": ["image"], "guidance": "guidance"},
    DATA_12,
    np.array([[9, 10, 11], [12, 13, 14], [15, 16, 17]]),
]


class TestFindAllValidSlicesd(unittest.TestCase):

    @parameterized.expand([FIND_SLICE_TEST_CASE_1, FIND_SLICE_TEST_CASE_2])
    def test_correct_results(self, arguments, input_data, expected_result):
        result = FindAllValidSlicesd(**arguments)(input_data)
        np.testing.assert_allclose(result[arguments["sids"]], expected_result)


class TestSpatialCropForegroundd(unittest.TestCase):

    @parameterized.expand([CROP_TEST_CASE_1])
    def test_correct_results(self, arguments, input_data, expected_result):
        result = SpatialCropForegroundd(**arguments)(input_data)
        np.testing.assert_allclose(result["image"], expected_result)

    @parameterized.expand([CROP_TEST_CASE_2])
    def test_correct_shape(self, arguments, input_data, expected_shape):
        result = SpatialCropForegroundd(**arguments)(input_data)
        np.testing.assert_equal(result["image"].shape, expected_shape)

    @parameterized.expand([CROP_TEST_CASE_1])
    def test_foreground_position(self, arguments, input_data, _):
        result = SpatialCropForegroundd(**arguments)(input_data)
        np.testing.assert_allclose(result[PostFix.meta("image")]["foreground_start_coord"], np.array([0, 1, 1]))
        np.testing.assert_allclose(result[PostFix.meta("image")]["foreground_end_coord"], np.array([1, 4, 4]))

        arguments["start_coord_key"] = "test_start_coord"
        arguments["end_coord_key"] = "test_end_coord"
        result = SpatialCropForegroundd(**arguments)(input_data)
        np.testing.assert_allclose(result[PostFix.meta("image")]["test_start_coord"], np.array([0, 1, 1]))
        np.testing.assert_allclose(result[PostFix.meta("image")]["test_end_coord"], np.array([1, 4, 4]))


class TestAddInitialSeedPointd(unittest.TestCase):

    @parameterized.expand([ADD_INITIAL_POINT_TEST_CASE_1])
    def test_correct_results(self, arguments, input_data, expected_result):
        seed = 0
        add_fn = AddInitialSeedPointd(**arguments)
        add_fn.set_random_state(seed)
        result = add_fn(input_data)
        self.assertEqual(result[arguments["guidance"]], expected_result)


class TestAddGuidanceSignald(unittest.TestCase):

    @parameterized.expand([ADD_GUIDANCE_TEST_CASE_1])
    def test_correct_results(self, arguments, input_data, expected_result):
        result = AddGuidanceSignald(**arguments)(input_data)
        np.testing.assert_allclose(result["image"], expected_result, rtol=1e-5)


class TestFindDiscrepancyRegionsd(unittest.TestCase):

    @parameterized.expand([FIND_DISCREPANCY_TEST_CASE_1])
    def test_correct_results(self, arguments, input_data, expected_result):
        result = FindDiscrepancyRegionsd(**arguments)(input_data)
        np.testing.assert_allclose(result[arguments["discrepancy"]], expected_result)


class TestAddRandomGuidanced(unittest.TestCase):

    @parameterized.expand([ADD_RANDOM_GUIDANCE_TEST_CASE_1])
    def test_correct_results(self, arguments, input_data, expected_result):
        seed = 0
        add_fn = AddRandomGuidanced(**arguments)
        add_fn.set_random_state(seed)
        result = add_fn(input_data)
        self.assertEqual(result[arguments["guidance"]], expected_result)


class TestAddGuidanceFromPointsd(unittest.TestCase):

    @parameterized.expand(
        [
            ADD_GUIDANCE_FROM_POINTS_TEST_CASE_1,
            ADD_GUIDANCE_FROM_POINTS_TEST_CASE_2,
            ADD_GUIDANCE_FROM_POINTS_TEST_CASE_3,
            ADD_GUIDANCE_FROM_POINTS_TEST_CASE_4,
            ADD_GUIDANCE_FROM_POINTS_TEST_CASE_5,
            ADD_GUIDANCE_FROM_POINTS_TEST_CASE_6,
        ]
    )
    def test_correct_results(self, arguments, input_data, expected_pos, expected_neg):
        result = AddGuidanceFromPointsd(**arguments)(input_data)
        self.assertEqual(result[arguments["guidance"]][0], expected_pos)
        self.assertEqual(result[arguments["guidance"]][1], expected_neg)


class TestSpatialCropGuidanced(unittest.TestCase):

    @parameterized.expand(
        [SPATIAL_CROP_GUIDANCE_TEST_CASE_1, SPATIAL_CROP_GUIDANCE_TEST_CASE_2, SPATIAL_CROP_GUIDANCE_TEST_CASE_3]
    )
    def test_correct_results(self, arguments, input_data, expected_result):
        result = SpatialCropGuidanced(**arguments)(input_data)
        np.testing.assert_allclose(result["image"], expected_result)


class TestResizeGuidanced(unittest.TestCase):

    @parameterized.expand([RESIZE_GUIDANCE_TEST_CASE_1])
    def test_correct_results(self, arguments, input_data, expected_result):
        result = ResizeGuidanced(**arguments)(input_data)
        self.assertEqual(result[arguments["guidance"]], expected_result)


class TestRestoreLabeld(unittest.TestCase):

    @parameterized.expand(
        [
            RESTORE_LABEL_TEST_CASE_1,
            RESTORE_LABEL_TEST_CASE_2,
            RESTORE_LABEL_TEST_CASE_3,
            RESTORE_LABEL_TEST_CASE_4,
            RESTORE_LABEL_TEST_CASE_5,
            RESTORE_LABEL_TEST_CASE_6,
            RESTORE_LABEL_TEST_CASE_7,
        ]
    )
    def test_correct_results(self, arguments, input_data, expected_result):
        result = RestoreLabeld(**arguments)(input_data)
        np.testing.assert_allclose(result["pred"], expected_result)


class TestFetch2DSliced(unittest.TestCase):

    @parameterized.expand([FETCH_2D_SLICE_TEST_CASE_1])
    def test_correct_results(self, arguments, input_data, expected_result):
        result = Fetch2DSliced(**arguments)(input_data)
        np.testing.assert_allclose(result["image"], expected_result)


if __name__ == "__main__":
    unittest.main()
