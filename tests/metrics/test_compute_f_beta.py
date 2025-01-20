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

from monai.metrics import FBetaScore
from tests.util import assert_allclose

_device = "cuda:0" if torch.cuda.is_available() else "cpu"


class TestFBetaScore(unittest.TestCase):
    def test_expecting_success_and_device(self):
        metric = FBetaScore()
        y_pred = torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]], device=_device)
        y = torch.tensor([[1, 0, 1], [0, 1, 0], [1, 0, 1]], device=_device)
        metric(y_pred=y_pred, y=y)
        result = metric.aggregate()[0]
        assert_allclose(result, torch.Tensor([0.714286]), atol=1e-6, rtol=1e-6)
        np.testing.assert_equal(result.device, y_pred.device)

    @parameterized.expand(
        [
            (0.5, torch.Tensor([[1, 0, 1], [0, 1, 0], [1, 0, 1]]), torch.Tensor([0.609756])),  # success_beta_0_5
            (2, torch.Tensor([[1, 0, 1], [0, 1, 0], [1, 0, 1]]), torch.Tensor([0.862069])),  # success_beta_2
            (
                2,  # success_beta_2, denominator_zero
                torch.Tensor([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
                torch.Tensor([0.0]),
            ),
        ]
    )
    def test_success_and_zero(self, beta, y, expected_score):
        metric = FBetaScore(beta=beta)
        metric(y_pred=torch.Tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]]), y=y)
        assert_allclose(metric.aggregate()[0], expected_score, atol=1e-6, rtol=1e-6)

    def test_number_of_dimensions_less_than_2_should_raise_error(self):
        metric = FBetaScore()
        with self.assertRaises(ValueError):
            metric(y_pred=torch.Tensor([1, 1, 1]), y=torch.Tensor([0, 0, 0]))

    def test_with_nan_values(self):
        metric = FBetaScore(get_not_nans=True)
        metric(
            y_pred=torch.Tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
            y=torch.Tensor([[1, 0, 1], [np.nan, np.nan, np.nan], [1, 0, 1]]),
        )
        assert_allclose(metric.aggregate()[0][0], torch.Tensor([0.727273]), atol=1e-6, rtol=1e-6)

    def test_do_not_include_background(self):
        metric = FBetaScore(include_background=False)
        metric(
            y_pred=torch.Tensor([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]),
            y=torch.Tensor([[0, 1, 1, 1], [0, 1, 1, 1], [0, 1, 1, 1]]),
        )
        assert_allclose(metric.aggregate()[0], torch.Tensor([1.0]), atol=1e-7, rtol=1e-7)

    def test_prediction_and_result_have_different_shape(self):
        metric = FBetaScore()
        with self.assertRaises(ValueError):
            metric(y_pred=torch.Tensor([[1, 1, 1], [1, 1, 1]]), y=torch.Tensor([1, 1, 1]))


if __name__ == "__main__":
    unittest.main()
