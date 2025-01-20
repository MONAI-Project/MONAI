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

from monai.transforms import RandCoarseDropoutd
from monai.utils import fall_back_tuple

TEST_CASE_0 = [
    {"keys": "img", "holes": 2, "spatial_size": [2, 2, 2], "fill_value": 5, "prob": 1.0},
    {"img": np.random.randint(0, 2, size=[3, 3, 3, 4])},
]

TEST_CASE_1 = [
    {"keys": "img", "holes": 1, "spatial_size": [1, 2, 3], "fill_value": 5, "max_holes": 5, "prob": 1.0},
    {"img": np.random.randint(0, 2, size=[3, 3, 3, 4])},
]

TEST_CASE_2 = [
    {"keys": "img", "holes": 2, "spatial_size": [2, 2, 2], "fill_value": 5, "max_spatial_size": [4, 4, 3], "prob": 1.0},
    {"img": np.random.randint(0, 2, size=[3, 3, 3, 4])},
]

TEST_CASE_3 = [
    {
        "keys": "img",
        "holes": 2,
        "spatial_size": [2, -1, 2],
        "fill_value": 5,
        "max_spatial_size": [4, 4, -1],
        "prob": 1.0,
    },
    {"img": np.random.randint(0, 2, size=[3, 3, 3, 4])},
]

TEST_CASE_4 = [
    {"keys": "img", "holes": 2, "spatial_size": [2, 2, 2], "fill_value": (0.2, 0.6), "prob": 1.0},
    {"img": np.random.rand(3, 3, 3, 4)},
]

TEST_CASE_5 = [
    {"keys": "img", "holes": 2, "spatial_size": [2, 2, 2], "fill_value": None, "prob": 1.0},
    {"img": np.random.rand(3, 3, 3, 4)},
]

TEST_CASE_6 = [
    {"keys": "img", "holes": 2, "spatial_size": [2, 2, 2], "dropout_holes": False, "fill_value": 0.5, "prob": 1.0},
    {"img": np.random.rand(3, 3, 3, 4)},
]


class TestRandCoarseDropoutd(unittest.TestCase):

    @parameterized.expand([TEST_CASE_0, TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4, TEST_CASE_5, TEST_CASE_6])
    def test_value(self, input_param, input_data):
        dropout = RandCoarseDropoutd(**input_param)
        result = dropout(input_data)["img"]
        holes = input_param.get("holes")
        max_holes = input_param.get("max_holes")
        spatial_size = fall_back_tuple(input_param.get("spatial_size"), input_data["img"].shape[1:])
        max_spatial_size = fall_back_tuple(input_param.get("max_spatial_size"), input_data["img"].shape[1:])

        if max_holes is None:
            self.assertEqual(len(dropout.dropper.hole_coords), holes)
        else:
            self.assertGreaterEqual(len(dropout.dropper.hole_coords), holes)
            self.assertLessEqual(len(dropout.dropper.hole_coords), max_holes)

        for h in dropout.dropper.hole_coords:
            data = result[h]
            # test hole value
            if input_param.get("dropout_holes", True):
                fill_value = input_param.get("fill_value", 0)
                if isinstance(fill_value, (int, float)):
                    np.testing.assert_allclose(data, fill_value)
                elif fill_value is not None:
                    min_value = data.min()
                    max_value = data.max()
                    self.assertGreaterEqual(max_value, min_value)
                    self.assertGreaterEqual(min_value, fill_value[0])
                    self.assertLess(max_value, fill_value[1])
            else:
                np.testing.assert_allclose(data, input_data["img"][h])

            if max_spatial_size is None:
                self.assertTupleEqual(data.shape[1:], tuple(spatial_size))
            else:
                for d, s, m in zip(data.shape[1:], spatial_size, max_spatial_size):
                    self.assertGreaterEqual(d, s)
                    self.assertLessEqual(d, m)


if __name__ == "__main__":
    unittest.main()
