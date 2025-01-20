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
import torch
from parameterized import parameterized

from monai.transforms import RandCoarseDropout
from monai.utils import fall_back_tuple
from tests.util import TEST_NDARRAYS, assert_allclose

TEST_CASE_0 = [
    {"holes": 2, "spatial_size": [2, 2, 2], "fill_value": 5, "prob": 1.0},
    np.random.randint(0, 2, size=[3, 3, 3, 4]),
]

TEST_CASE_1 = [
    {"holes": 1, "spatial_size": [1, 2, 3], "fill_value": 5, "max_holes": 5, "prob": 1.0},
    np.random.randint(0, 2, size=[3, 3, 3, 4]),
]

TEST_CASE_2 = [
    {"holes": 2, "spatial_size": [2, 2, 2], "fill_value": 5, "max_spatial_size": [4, 4, 3], "prob": 1.0},
    np.random.randint(0, 2, size=[3, 3, 3, 4]),
]

TEST_CASE_3 = [
    {"holes": 2, "spatial_size": [2, -1, 2], "fill_value": 5, "max_spatial_size": [4, 4, -1], "prob": 1.0},
    np.random.randint(0, 2, size=[3, 3, 3, 4]),
]

TEST_CASE_4 = [
    {"holes": 2, "spatial_size": [2, 2, 2], "fill_value": (3, 6), "prob": 1.0},
    np.random.randint(0, 2, size=[3, 3, 3, 4]),
]

TEST_CASE_5 = [
    {"holes": 2, "spatial_size": [2, 2, 2], "fill_value": None, "prob": 1.0},
    np.random.randint(0, 2, size=[3, 3, 3, 4]),
]

TEST_CASE_6 = [
    {"holes": 2, "spatial_size": [2, 2, 2], "dropout_holes": False, "fill_value": (3, 6), "prob": 1.0},
    np.random.randint(0, 2, size=[3, 3, 3, 4]),
]

TEST_CASE_7 = [
    {"holes": 2, "spatial_size": [2, 2, 2], "dropout_holes": False, "fill_value": (3, 6), "prob": 1.0},
    torch.randint(0, 2, size=[3, 3, 3, 4]),
]


class TestRandCoarseDropout(unittest.TestCase):
    @parameterized.expand(
        [TEST_CASE_0, TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4, TEST_CASE_5, TEST_CASE_6, TEST_CASE_7]
    )
    def test_value(self, input_param, input_data):
        for p in TEST_NDARRAYS:
            dropout = RandCoarseDropout(**input_param)
            im = p(input_data)
            result = dropout(im)
        holes = input_param.get("holes")
        max_holes = input_param.get("max_holes")
        spatial_size = fall_back_tuple(input_param.get("spatial_size"), input_data.shape[1:])
        max_spatial_size = fall_back_tuple(input_param.get("max_spatial_size"), input_data.shape[1:])

        if max_holes is None:
            self.assertEqual(len(dropout.hole_coords), holes)
        else:
            self.assertGreaterEqual(len(dropout.hole_coords), holes)
            self.assertLessEqual(len(dropout.hole_coords), max_holes)

        for h in dropout.hole_coords:
            data = result[h]
            # test hole value
            if input_param.get("dropout_holes", True):
                fill_value = input_param.get("fill_value", None)
                if isinstance(fill_value, (int, float)):
                    assert_allclose(data, fill_value, type_test=False)
                elif fill_value is not None:
                    min_value = data.min()
                    max_value = data.max()
                    self.assertGreaterEqual(max_value, min_value)
                    self.assertGreaterEqual(min_value, fill_value[0])
                    self.assertLess(max_value, fill_value[1])
            else:
                assert_allclose(data, input_data[h], type_test=False)

            if max_spatial_size is None:
                self.assertTupleEqual(data.shape[1:], tuple(spatial_size))
            else:
                for d, s, m in zip(data.shape[1:], spatial_size, max_spatial_size):
                    self.assertGreaterEqual(d, s)
                    self.assertLessEqual(d, m)


if __name__ == "__main__":
    unittest.main()
