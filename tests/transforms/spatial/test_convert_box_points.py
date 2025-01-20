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

import torch
from parameterized import parameterized

from monai.data.box_utils import convert_box_to_standard_mode
from monai.transforms.spatial.array import ConvertBoxToPoints, ConvertPointsToBoxes
from tests.util import assert_allclose

TEST_CASE_POINTS_2D = [
    [
        torch.tensor([[10, 20, 30, 40], [50, 60, 70, 80]]),
        "xyxy",
        torch.tensor([[[10, 20], [30, 20], [30, 40], [10, 40]], [[50, 60], [70, 60], [70, 80], [50, 80]]]),
    ],
    [torch.tensor([[10, 20, 20, 20]]), "ccwh", torch.tensor([[[0, 10], [20, 10], [20, 30], [0, 30]]])],
]
TEST_CASE_POINTS_3D = [
    [
        torch.tensor([[10, 20, 30, 40, 50, 60], [70, 80, 90, 100, 110, 120]]),
        "xyzxyz",
        torch.tensor(
            [
                [
                    [10, 20, 30],
                    [40, 20, 30],
                    [40, 50, 30],
                    [10, 50, 30],
                    [10, 20, 60],
                    [40, 20, 60],
                    [40, 50, 60],
                    [10, 50, 60],
                ],
                [
                    [70, 80, 90],
                    [100, 80, 90],
                    [100, 110, 90],
                    [70, 110, 90],
                    [70, 80, 120],
                    [100, 80, 120],
                    [100, 110, 120],
                    [70, 110, 120],
                ],
            ]
        ),
    ],
    [
        torch.tensor([[10, 20, 30, 10, 10, 10]]),
        "cccwhd",
        torch.tensor(
            [
                [
                    [5, 15, 25],
                    [15, 15, 25],
                    [15, 25, 25],
                    [5, 25, 25],
                    [5, 15, 35],
                    [15, 15, 35],
                    [15, 25, 35],
                    [5, 25, 35],
                ]
            ]
        ),
    ],
    [
        torch.tensor([[10, 20, 30, 40, 50, 60]]),
        "xxyyzz",
        torch.tensor(
            [
                [
                    [10, 30, 50],
                    [20, 30, 50],
                    [20, 40, 50],
                    [10, 40, 50],
                    [10, 30, 60],
                    [20, 30, 60],
                    [20, 40, 60],
                    [10, 40, 60],
                ]
            ]
        ),
    ],
]

TEST_CASES = TEST_CASE_POINTS_2D + TEST_CASE_POINTS_3D


class TestConvertBoxToPoints(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_convert_box_to_points(self, boxes, mode, expected_points):
        transform = ConvertBoxToPoints(mode=mode)
        converted_points = transform(boxes)
        assert_allclose(converted_points, expected_points, type_test=False)


class TestConvertPointsToBoxes(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_convert_box_to_points(self, boxes, mode, points):
        transform = ConvertPointsToBoxes()
        converted_boxes = transform(points)
        expected_boxes = convert_box_to_standard_mode(boxes, mode)
        assert_allclose(converted_boxes, expected_boxes, type_test=False)


if __name__ == "__main__":
    unittest.main()
