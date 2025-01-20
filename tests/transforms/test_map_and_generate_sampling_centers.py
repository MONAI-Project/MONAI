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
from copy import deepcopy

import numpy as np
from parameterized import parameterized

from monai.transforms import map_and_generate_sampling_centers
from monai.utils.misc import set_determinism
from tests.util import TEST_NDARRAYS, assert_allclose

TEST_CASE_1 = [
    # test Argmax data
    {
        "label": (np.array([[[0, 1, 2], [2, 0, 1], [1, 2, 0]]])),
        "spatial_size": [2, 2, 2],
        "num_samples": 2,
        "label_spatial_shape": [3, 3, 3],
        "num_classes": 3,
        "image": None,
        "ratios": [0, 1, 2],
        "image_threshold": 0.0,
    },
    tuple,
    2,
    3,
]

TEST_CASE_2 = [
    {
        "label": (
            np.array(
                [
                    [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
                    [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
                ]
            )
        ),
        "spatial_size": [2, 2, 2],
        "num_samples": 1,
        "ratios": None,
        "label_spatial_shape": [3, 3, 3],
        "image": None,
        "image_threshold": 0.0,
    },
    tuple,
    1,
    3,
]


class TestMapAndGenerateSamplingCenters(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    def test_map_and_generate_sampling_centers(self, input_data, expected_type, expected_count, expected_shape):
        results = []
        for p in TEST_NDARRAYS + (None,):
            input_data = deepcopy(input_data)
            if p is not None:
                input_data["label"] = p(input_data["label"])
            set_determinism(0)
            result = map_and_generate_sampling_centers(**input_data)
            self.assertIsInstance(result, expected_type)
            self.assertEqual(len(result), expected_count)
            self.assertEqual(len(result[0]), expected_shape)
            # check for consistency between numpy, torch and torch.cuda
            results.append(result)
            if len(results) > 1:
                for x, y in zip(result[0], result[-1]):
                    assert_allclose(x, y, type_test=False)


if __name__ == "__main__":
    unittest.main()
