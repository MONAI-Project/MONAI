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

from monai.transforms import RandCoarseShuffle

TEST_CASES = [
    [
        {"holes": 5, "spatial_size": 1, "max_spatial_size": -1, "prob": 0.0},
        {"img": np.arange(8).reshape((1, 2, 2, 2))},
        np.arange(8).reshape((1, 2, 2, 2)),
    ],
    [
        {"holes": 10, "spatial_size": 1, "max_spatial_size": -1, "prob": 1.0},
        {"img": np.arange(27).reshape((1, 3, 3, 3))},
        np.asarray(
            [
                [
                    [[8, 19, 26], [24, 6, 15], [0, 13, 25]],
                    [[17, 3, 5], [10, 1, 12], [22, 4, 11]],
                    [[21, 20, 23], [14, 2, 16], [18, 9, 7]],
                ]
            ]
        ),
    ],
    [
        {"holes": 2, "spatial_size": 1, "max_spatial_size": -1, "prob": 1.0},
        {"img": np.arange(16).reshape((2, 2, 2, 2))},
        np.asarray([[[[6, 1], [4, 3]], [[0, 2], [7, 5]]], [[[14, 10], [9, 8]], [[12, 15], [13, 11]]]]),
    ],
    [
        {"holes": 2, "spatial_size": 1, "max_spatial_size": -1, "prob": 1.0},
        {"img": torch.arange(16).reshape((2, 2, 2, 2))},
        torch.as_tensor([[[[6, 1], [4, 3]], [[0, 2], [7, 5]]], [[[14, 10], [9, 8]], [[12, 15], [13, 11]]]]),
    ],
]


class TestRandCoarseShuffle(unittest.TestCase):

    @parameterized.expand(TEST_CASES)
    def test_shuffle(self, input_param, input_data, expected_val):
        g = RandCoarseShuffle(**input_param)
        g.set_random_state(seed=12)
        result = g(**input_data)
        np.testing.assert_allclose(result, expected_val, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
