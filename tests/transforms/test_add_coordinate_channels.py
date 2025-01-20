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

from monai.transforms import AddCoordinateChannels
from tests.util import TEST_NDARRAYS, assert_allclose

TESTS, TEST_CASES_ERROR_1, TEST_CASES_ERROR_2 = [], [], []
for p in TEST_NDARRAYS:
    TESTS.append([{"spatial_dims": (0, 1, 2)}, p(np.random.randint(0, 2, size=(1, 3, 3, 3))), (4, 3, 3, 3)])
    TESTS.append([{"spatial_dims": (0,)}, p(np.random.randint(0, 2, size=(1, 3, 3, 3))), (2, 3, 3, 3)])
    TEST_CASES_ERROR_1.append([{"spatial_dims": (2,)}, p(np.random.randint(0, 2, size=(1, 3, 3)))])
    TEST_CASES_ERROR_2.append([{"spatial_dims": (-1, 0, 1)}, p(np.random.randint(0, 2, size=(1, 3, 3)))])


class TestAddCoordinateChannels(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_shape(self, input_param, input, expected_shape):
        result = AddCoordinateChannels(**input_param)(input)
        self.assertEqual(type(result), type(input))
        if isinstance(result, torch.Tensor):
            self.assertEqual(result.device, input.device)
        self.assertEqual(list(result.shape), list(expected_shape))
        assert_allclose(input[0, ...], result[0, ...])

    @parameterized.expand(TEST_CASES_ERROR_1)
    def test_max_channel(self, input_param, input):
        with self.assertRaises(ValueError):
            AddCoordinateChannels(**input_param)(input)

    @parameterized.expand(TEST_CASES_ERROR_2)
    def test_channel_dim(self, input_param, input):
        with self.assertRaises(ValueError):
            AddCoordinateChannels(**input_param)(input)


if __name__ == "__main__":
    unittest.main()
