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

from monai.networks.layers import SavitzkyGolayFilter
from tests.util import skip_if_no_cuda

# Zero-padding trivial tests

TEST_CASE_SINGLE_VALUE = [
    {"window_length": 3, "order": 1},
    torch.Tensor([1.0]).unsqueeze(0).unsqueeze(0),  # Input data: Single value
    torch.Tensor([1 / 3]).unsqueeze(0).unsqueeze(0),  # Expected output: With a window length of 3 and polyorder 1
    # output should be equal to mean of 0, 1 and 0 = 1/3 (because input will be zero-padded and a linear fit performed)
    1e-6,  # absolute tolerance
]

TEST_CASE_1D = [
    {"window_length": 3, "order": 1},
    torch.Tensor([1.0, 1.0, 1.0]).unsqueeze(0).unsqueeze(0),  # Input data
    torch.Tensor([2 / 3, 1.0, 2 / 3])
    .unsqueeze(0)
    .unsqueeze(0),  # Expected output: zero padded, so linear interpolation
    # over length-3 windows will result in output of [2/3, 1, 2/3].
    1e-6,  # absolute tolerance
]

TEST_CASE_2D_AXIS_2 = [
    {"window_length": 3, "order": 1},  # along default axis (2, first spatial dim)
    torch.ones((3, 2)).unsqueeze(0).unsqueeze(0),
    torch.Tensor([[2 / 3, 2 / 3], [1.0, 1.0], [2 / 3, 2 / 3]]).unsqueeze(0).unsqueeze(0),
    1e-6,  # absolute tolerance
]

TEST_CASE_2D_AXIS_3 = [
    {"window_length": 3, "order": 1, "axis": 3},  # along axis 3 (second spatial dim)
    torch.ones((2, 3)).unsqueeze(0).unsqueeze(0),
    torch.Tensor([[2 / 3, 1.0, 2 / 3], [2 / 3, 1.0, 2 / 3]]).unsqueeze(0).unsqueeze(0),
    1e-6,  # absolute tolerance
]

# Replicated-padding trivial tests

TEST_CASE_SINGLE_VALUE_REP = [
    {"window_length": 3, "order": 1, "mode": "replicate"},
    torch.Tensor([1.0]).unsqueeze(0).unsqueeze(0),  # Input data: Single value
    torch.Tensor([1.0]).unsqueeze(0).unsqueeze(0),  # Expected output: With a window length of 3 and polyorder 1
    # output will be equal to mean of [1, 1, 1] = 1 (input will be nearest-neighbour-padded and a linear fit performed)
    1e-6,  # absolute tolerance
]

TEST_CASE_1D_REP = [
    {"window_length": 3, "order": 1, "mode": "replicate"},
    torch.Tensor([1.0, 1.0, 1.0]).unsqueeze(0).unsqueeze(0),  # Input data
    torch.Tensor([1.0, 1.0, 1.0]).unsqueeze(0).unsqueeze(0),  # Expected output: zero padded, so linear interpolation
    # over length-3 windows will result in output of [2/3, 1, 2/3].
    1e-6,  # absolute tolerance
]

TEST_CASE_2D_AXIS_2_REP = [
    {"window_length": 3, "order": 1, "mode": "replicate"},  # along default axis (2, first spatial dim)
    torch.ones((3, 2)).unsqueeze(0).unsqueeze(0),
    torch.Tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]).unsqueeze(0).unsqueeze(0),
    1e-6,  # absolute tolerance
]

TEST_CASE_2D_AXIS_3_REP = [
    {"window_length": 3, "order": 1, "axis": 3, "mode": "replicate"},  # along axis 3 (second spatial dim)
    torch.ones((2, 3)).unsqueeze(0).unsqueeze(0),
    torch.Tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]).unsqueeze(0).unsqueeze(0),
    1e-6,  # absolute tolerance
]

# Sine smoothing

TEST_CASE_SINE_SMOOTH = [
    {"window_length": 3, "order": 1},
    # Sine wave with period equal to savgol window length (windowed to reduce edge effects).
    torch.as_tensor(np.sin(2 * np.pi * 1 / 3 * np.arange(100)) * np.hanning(100)).unsqueeze(0).unsqueeze(0),
    # Should be smoothed out to zeros
    torch.zeros(100).unsqueeze(0).unsqueeze(0),
    # tolerance chosen by examining output of SciPy.signal.savgol_filter when provided the above input
    2e-2,  # absolute tolerance
]


class TestSavitzkyGolayCPU(unittest.TestCase):
    @parameterized.expand(
        [TEST_CASE_SINGLE_VALUE, TEST_CASE_1D, TEST_CASE_2D_AXIS_2, TEST_CASE_2D_AXIS_3, TEST_CASE_SINE_SMOOTH]
    )
    def test_value(self, arguments, image, expected_data, atol, rtol=1e-5):
        result = SavitzkyGolayFilter(**arguments)(image)
        np.testing.assert_allclose(result, expected_data, atol=atol, rtol=rtol)


class TestSavitzkyGolayCPUREP(unittest.TestCase):
    @parameterized.expand(
        [TEST_CASE_SINGLE_VALUE_REP, TEST_CASE_1D_REP, TEST_CASE_2D_AXIS_2_REP, TEST_CASE_2D_AXIS_3_REP]
    )
    def test_value(self, arguments, image, expected_data, atol, rtol=1e-5):
        result = SavitzkyGolayFilter(**arguments)(image)
        np.testing.assert_allclose(result, expected_data, atol=atol, rtol=rtol)


@skip_if_no_cuda
class TestSavitzkyGolayGPU(unittest.TestCase):
    @parameterized.expand(
        [TEST_CASE_SINGLE_VALUE, TEST_CASE_1D, TEST_CASE_2D_AXIS_2, TEST_CASE_2D_AXIS_3, TEST_CASE_SINE_SMOOTH]
    )
    def test_value(self, arguments, image, expected_data, atol, rtol=1e-5):
        result = SavitzkyGolayFilter(**arguments)(image.to(device="cuda"))
        np.testing.assert_allclose(result.cpu(), expected_data, atol=atol, rtol=rtol)


@skip_if_no_cuda
class TestSavitzkyGolayGPUREP(unittest.TestCase):
    @parameterized.expand(
        [TEST_CASE_SINGLE_VALUE_REP, TEST_CASE_1D_REP, TEST_CASE_2D_AXIS_2_REP, TEST_CASE_2D_AXIS_3_REP]
    )
    def test_value(self, arguments, image, expected_data, atol, rtol=1e-5):
        result = SavitzkyGolayFilter(**arguments)(image.to(device="cuda"))
        np.testing.assert_allclose(result.cpu(), expected_data, atol=atol, rtol=rtol)


if __name__ == "__main__":
    unittest.main()
