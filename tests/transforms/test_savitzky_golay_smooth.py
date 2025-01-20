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

from monai.transforms import SavitzkyGolaySmooth
from tests.util import TEST_NDARRAYS, assert_allclose

# Zero-padding trivial tests

TEST_CASE_SINGLE_VALUE = [
    {"window_length": 3, "order": 1},
    np.expand_dims(np.array([1.0]), 0),  # Input data: Single value
    np.expand_dims(np.array([1 / 3]), 0),  # Expected output: With a window length of 3 and polyorder 1
    # output should be equal to mean of 0, 1 and 0 = 1/3 (because input will be zero-padded and a linear fit performed)
    1e-5,  # absolute tolerance
]

TEST_CASE_2D_AXIS_2 = [
    {"window_length": 3, "order": 1, "axis": 2},  # along axis 2 (second spatial dim)
    np.expand_dims(np.ones((2, 3)), 0),
    np.expand_dims(np.array([[2 / 3, 1.0, 2 / 3], [2 / 3, 1.0, 2 / 3]]), 0),
    1e-5,  # absolute tolerance
]

# Replicated-padding trivial tests

TEST_CASE_SINGLE_VALUE_REP = [
    {"window_length": 3, "order": 1, "mode": "replicate"},
    np.expand_dims(np.array([1.0]), 0),  # Input data: Single value
    np.expand_dims(np.array([1.0]), 0),  # Expected output: With a window length of 3 and polyorder 1
    # output will be equal to mean of [1, 1, 1] = 1 (input will be nearest-neighbour-padded and a linear fit performed)
    1e-5,  # absolute tolerance
]

# Sine smoothing

TEST_CASE_SINE_SMOOTH = [
    {"window_length": 3, "order": 1},
    # Sine wave with period equal to savgol window length (windowed to reduce edge effects).
    np.expand_dims(np.sin(2 * np.pi * 1 / 3 * np.arange(100)) * np.hanning(100), 0),
    # Should be smoothed out to zeros
    np.expand_dims(np.zeros(100), 0),
    # tolerance chosen by examining output of SciPy.signal.savgol_filter() when provided the above input
    2e-2,  # absolute tolerance
]


class TestSavitzkyGolaySmooth(unittest.TestCase):
    @parameterized.expand(
        [TEST_CASE_SINGLE_VALUE, TEST_CASE_2D_AXIS_2, TEST_CASE_SINE_SMOOTH, TEST_CASE_SINGLE_VALUE_REP]
    )
    def test_value(self, arguments, image, expected_data, atol):
        for p in TEST_NDARRAYS:
            result = SavitzkyGolaySmooth(**arguments)(p(image.astype(np.float32)))
            assert_allclose(result, p(expected_data.astype(np.float32)), rtol=1e-4, atol=atol, type_test="tensor")


if __name__ == "__main__":
    unittest.main()
