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

from monai.transforms import RandHistogramShift
from tests.util import TEST_NDARRAYS, assert_allclose

TESTS = []
for p in TEST_NDARRAYS:
    TESTS.append(
        [
            {"num_control_points": 5, "prob": 0.0},
            {"img": p(np.arange(8).reshape((1, 2, 2, 2)))},
            np.arange(8).reshape((1, 2, 2, 2)),
        ]
    )
    TESTS.append(
        [
            {"num_control_points": 5, "prob": 0.9},
            {"img": p(np.arange(8).reshape((1, 2, 2, 2)).astype(np.float32))},
            np.array([[[[0.0, 0.57227867], [1.1391707, 1.68990281]], [[2.75833219, 4.34445884], [5.70913743, 7.0]]]]),
        ]
    )
    TESTS.append(
        [
            {"num_control_points": (5, 20), "prob": 0.9},
            {"img": p(np.arange(8).reshape((1, 2, 2, 2)).astype(np.float32))},
            np.array([[[[0.0, 1.17472492], [2.21553091, 2.88292011]], [[3.98407301, 5.01302123], [6.09275004, 7.0]]]]),
        ]
    )

WARN_TESTS = []
for p in TEST_NDARRAYS:
    WARN_TESTS.append(
        [
            {"num_control_points": 5, "prob": 1.0},
            {"img": p(np.zeros(8).reshape((1, 2, 2, 2)))},
            np.zeros(8).reshape((1, 2, 2, 2)),
        ]
    )


class TestRandHistogramShift(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_rand_histogram_shift(self, input_param, input_data, expected_val):
        g = RandHistogramShift(**input_param)
        g.set_random_state(123)
        result = g(**input_data)
        assert_allclose(result, expected_val, rtol=1e-4, atol=1e-4, type_test="tensor")

    def test_interp(self):
        tr = RandHistogramShift()
        for array_type in (torch.tensor, np.array):
            x = array_type([0.0, 4.0, 6.0, 10.0])
            y = array_type([1.0, -1.0, 3.0, 5.0])

            yi = tr.interp(array_type([0, 2, 4, 8, 10]), x, y)
            self.assertEqual(yi.shape, (5,))
            assert_allclose(yi, array_type([1.0, 0.0, -1.0, 4.0, 5.0]))

            yi = tr.interp(array_type([-1, 11, 10.001, -0.001]), x, y)
            self.assertEqual(yi.shape, (4,))
            assert_allclose(yi, array_type([1.0, 5.0, 5.0, 1.0]))

            yi = tr.interp(array_type([[-2, 11], [1, 3], [8, 10]]), x, y)
            self.assertEqual(yi.shape, (3, 2))
            assert_allclose(yi, array_type([[1.0, 5.0], [0.5, -0.5], [4.0, 5.0]]))

    @parameterized.expand(WARN_TESTS)
    def test_warn(self, input_param, input_data, expected_val):
        with self.assertWarns(Warning):
            result = RandHistogramShift(**input_param)(**input_data)
            assert_allclose(result, expected_val, type_test="tensor")


if __name__ == "__main__":
    unittest.main()
