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

from monai.metrics import R2Metric, compute_r2_score

_device = "cuda:0" if torch.cuda.is_available() else "cpu"
TEST_CASE_1 = [
    torch.tensor([0.1, -0.25, 3.0, 0.99], device=_device),
    torch.tensor([0.1, -0.2, -2.7, 1.58], device=_device),
    "uniform_average",
    0,
    -2.469944,
]

TEST_CASE_2 = [
    torch.tensor([0.1, -0.25, 3.0, 0.99]),
    torch.tensor([0.1, -0.2, 2.7, 1.58]),
    "uniform_average",
    2,
    0.75828,
]

TEST_CASE_3 = [
    torch.tensor([[0.1], [-0.25], [3.0], [0.99]]),
    torch.tensor([[0.1], [-0.2], [2.7], [1.58]]),
    "raw_values",
    2,
    0.75828,
]

TEST_CASE_4 = [
    torch.tensor([[0.1, 1.0], [-0.25, 0.5], [3.0, -0.2], [0.99, 2.1]]),
    torch.tensor([[0.1, 0.82], [-0.2, 0.01], [2.7, -0.1], [1.58, 2.0]]),
    "raw_values",
    1,
    [0.87914, 0.844375],
]

TEST_CASE_5 = [
    torch.tensor([[0.1, 1.0], [-0.25, 0.5], [3.0, -0.2], [0.99, 2.1]]),
    torch.tensor([[0.1, 0.82], [-0.2, 0.01], [2.7, -0.1], [1.58, 2.0]]),
    "variance_weighted",
    1,
    0.867314,
]

TEST_CASE_6 = [
    torch.tensor([[0.1, 1.0], [-0.25, 0.5], [3.0, -0.2], [0.99, 2.1]]),
    torch.tensor([[0.1, 0.82], [-0.2, 0.01], [2.7, -0.1], [1.58, 2.0]]),
    "uniform_average",
    0,
    0.907838,
]

TEST_CASE_ERROR_1 = [
    torch.tensor([[0.1, 1.0], [-0.25, 0.5], [3.0, -0.2], [0.99, 2.1]]),
    torch.tensor([[0.1, 0.82], [-0.2, 0.01], [2.7, -0.1], [1.58, 2.0]]),
    "abc",
    0,
]

TEST_CASE_ERROR_2 = [
    torch.tensor([[0.1, 1.0], [-0.25, 0.5], [3.0, -0.2], [0.99, 2.1]]),
    torch.tensor([[0.1, 0.82], [-0.2, 0.01], [2.7, -0.1], [1.58, 2.0]]),
    "uniform_average",
    -1,
]

TEST_CASE_ERROR_3 = [
    torch.tensor([[0.1, 1.0], [-0.25, 0.5], [3.0, -0.2], [0.99, 2.1]]),
    np.array([[0.1, 0.82], [-0.2, 0.01], [2.7, -0.1], [1.58, 2.0]]),
    "uniform_average",
    0,
]

TEST_CASE_ERROR_4 = [
    torch.tensor([[0.1, 1.0], [-0.25, 0.5], [3.0, -0.2], [0.99, 2.1]]),
    torch.tensor([[0.1, 0.82], [-0.2, 0.01], [2.7, -0.1]]),
    "uniform_average",
    0,
]

TEST_CASE_ERROR_5 = [
    torch.tensor([[[0.1, 1.0], [-0.25, 0.5], [3.0, -0.2], [0.99, 2.1]]]),
    torch.tensor([[[0.1, 0.82], [-0.2, 0.01], [2.7, -0.1], [1.58, 2.0]]]),
    "uniform_average",
    0,
]

TEST_CASE_ERROR_6 = [
    torch.tensor([[0.1, 1.0], [-0.25, 0.5], [3.0, -0.2], [0.99, 2.1]]),
    torch.tensor([[0.1, 0.82], [-0.2, 0.01], [2.7, -0.1], [1.58, 2.0]]),
    "uniform_average",
    3,
]

TEST_CASE_ERROR_7 = [torch.tensor([[0.1, 1.0]]), torch.tensor([[0.1, 0.82]]), "uniform_average", 0]


class TestComputeR2Score(unittest.TestCase):

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4, TEST_CASE_5, TEST_CASE_6])
    def test_value(self, y_pred, y, multi_output, p, expected_value):
        result = compute_r2_score(y_pred=y_pred, y=y, multi_output=multi_output, p=p)
        np.testing.assert_allclose(expected_value, result, rtol=1e-5)

    @parameterized.expand(
        [
            TEST_CASE_ERROR_1,
            TEST_CASE_ERROR_2,
            TEST_CASE_ERROR_3,
            TEST_CASE_ERROR_4,
            TEST_CASE_ERROR_5,
            TEST_CASE_ERROR_6,
            TEST_CASE_ERROR_7,
        ]
    )
    def test_error(self, y_pred, y, multi_output, p):
        with self.assertRaises(ValueError):
            _ = compute_r2_score(y_pred=y_pred, y=y, multi_output=multi_output, p=p)

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4, TEST_CASE_5, TEST_CASE_6])
    def test_class_value(self, y_pred, y, multi_output, p, expected_value):
        metric = R2Metric(multi_output=multi_output, p=p)
        metric(y_pred=y_pred, y=y)
        result = metric.aggregate()
        np.testing.assert_allclose(expected_value, result, rtol=1e-5)
        result = metric.aggregate(multi_output=multi_output)  # test optional argument
        metric.reset()
        np.testing.assert_allclose(expected_value, result, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
