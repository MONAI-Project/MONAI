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

from monai.metrics import GeneralizedDiceScore, compute_generalized_dice

_device = "cuda:0" if torch.cuda.is_available() else "cpu"

# keep background
TEST_CASE_1 = [  # y (1, 1, 2, 2), y_pred (1, 1, 2, 2), expected out (1, 1) with compute_generalized_dice
    {
        "y_pred": torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], device=_device),
        "y": torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]], device=_device),
        "include_background": True,
    },
    [[0.8]],
]

# remove background
TEST_CASE_2 = [  # y (2, 3, 2, 2), y_pred (2, 3, 2, 2), expected out (2) (no background) with GeneralizedDiceScore
    {
        "y_pred": torch.tensor(
            [
                [[[0.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 1.0]], [[1.0, 0.0], [0.0, 0.0]]],
                [[[0.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 0.0]], [[0.0, 1.0], [1.0, 0.0]]],
            ]
        ),
        "y": torch.tensor(
            [
                [[[0.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [0.0, 0.0]]],
                [[[0.0, 0.0], [0.0, 1.0]], [[1.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 0.0]]],
            ]
        ),
        "include_background": False,
        "reduction": "mean_batch",
    },
    [0.583333, 0.333333],
]

TEST_CASE_3 = [  # y (2, 3, 2, 2), y_pred (2, 3, 2, 2), expected out (1) with GeneralizedDiceScore
    {
        "y_pred": torch.tensor(
            [
                [[[1.0, 1.0], [1.0, 0.0]], [[0.0, 1.0], [0.0, 0.0]], [[0.0, 1.0], [1.0, 1.0]]],
                [[[1.0, 0.0], [1.0, 1.0]], [[0.0, 1.0], [1.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]],
            ]
        ),
        "y": torch.tensor(
            [
                [[[1.0, 1.0], [1.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
                [[[0.0, 0.0], [0.0, 1.0]], [[1.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 0.0]]],
            ]
        ),
        "include_background": True,
        "reduction": "mean",
    },
    [0.5454],
]

TEST_CASE_4 = [  # y (2, 3, 2, 2), y_pred (2, 3, 2, 2), expected out (1) with GeneralizedDiceScore
    {
        "y_pred": torch.tensor(
            [
                [[[1.0, 1.0], [1.0, 0.0]], [[0.0, 1.0], [0.0, 0.0]], [[0.0, 1.0], [1.0, 1.0]]],
                [[[1.0, 0.0], [1.0, 1.0]], [[0.0, 1.0], [1.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]],
            ]
        ),
        "y": torch.tensor(
            [
                [[[1.0, 1.0], [1.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
                [[[1.0, 1.0], [1.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
            ]
        ),
        "include_background": True,
        "reduction": "sum",
    },
    [1.045455],
]

TEST_CASE_5 = [  # y (2, 2, 3, 3) y_pred (2, 2, 3, 3) expected out (2, 2) with compute_generalized_dice
    {"y": torch.ones((2, 2, 3, 3)), "y_pred": torch.ones((2, 2, 3, 3))},
    [[1.0000, 1.0000], [1.0000, 1.0000]],
]

TEST_CASE_6 = [  # y (2, 2, 3, 3) y_pred (2, 2, 3, 3) expected out (2, 2) with compute_generalized_dice
    {"y": torch.zeros((2, 2, 3, 3)), "y_pred": torch.ones((2, 2, 3, 3))},
    [[0.0000, 0.0000], [0.0000, 0.0000]],
]

TEST_CASE_7 = [  # y (2, 2, 3, 3) y_pred (2, 2, 3, 3) expected out (2, 2) with compute_generalized_dice
    {"y": torch.ones((2, 2, 3, 3)), "y_pred": torch.zeros((2, 2, 3, 3))},
    [[0.0000, 0.0000], [0.0000, 0.0000]],
]

TEST_CASE_8 = [  # y (2, 2, 3, 3) y_pred (2, 2, 3, 3) expected out (2, 2) with compute_generalized_dice
    {"y": torch.zeros((2, 2, 3, 3)), "y_pred": torch.zeros((2, 2, 3, 3))},
    [[1.0000, 1.0000], [1.0000, 1.0000]],
]

TEST_CASE_9 = [  # y (2, 3, 2, 2) y_pred (2, 3, 2, 2) expected out (2) with GeneralizedDiceScore
    {
        "y_pred": torch.tensor(
            [
                [[[1.0, 1.0], [1.0, 0.0]], [[0.0, 1.0], [0.0, 0.0]], [[0.0, 1.0], [1.0, 1.0]]],
                [[[1.0, 0.0], [1.0, 1.0]], [[0.0, 1.0], [1.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]],
            ]
        ),
        "y": torch.tensor(
            [
                [[[1.0, 1.0], [1.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
                [[[0.0, 0.0], [0.0, 1.0]], [[1.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 0.0]]],
            ]
        ),
        "include_background": True,
        "reduction": "mean_channel",
    },
    [0.545455, 0.545455],
]


TEST_CASE_10 = [  # y (2, 3, 2, 2) y_pred (2, 3, 2, 2) expected out (2, 3) with compute_generalized_dice
    # and (3) with GeneralizedDiceScore "mean_batch"
    {
        "y_pred": torch.tensor(
            [
                [[[1.0, 1.0], [1.0, 0.0]], [[0.0, 1.0], [0.0, 0.0]], [[0.0, 1.0], [1.0, 1.0]]],
                [[[1.0, 0.0], [1.0, 1.0]], [[0.0, 1.0], [1.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]],
            ]
        ),
        "y": torch.tensor(
            [
                [[[1.0, 1.0], [1.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
                [[[0.0, 0.0], [0.0, 1.0]], [[1.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 0.0]]],
            ]
        ),
        "include_background": True,
    },
    [[0.857143, 0.0, 0.0], [0.5, 0.4, 0.666667]],
]

TEST_CASE_11 = [  # y (2, 3, 2, 2) y_pred (2, 3, 2, 2) expected out (2, 1) with compute_generalized_dice (summed over classes)
    # and (2) with GeneralizedDiceScore "mean_channel"
    {
        "y_pred": torch.tensor(
            [
                [[[1.0, 1.0], [1.0, 0.0]], [[0.0, 1.0], [0.0, 0.0]], [[0.0, 1.0], [1.0, 1.0]]],
                [[[1.0, 0.0], [1.0, 1.0]], [[0.0, 1.0], [1.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]],
            ]
        ),
        "y": torch.tensor(
            [
                [[[1.0, 1.0], [1.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
                [[[0.0, 0.0], [0.0, 1.0]], [[1.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 0.0]]],
            ]
        ),
        "include_background": True,
        "sum_over_classes": True,
    },
    [[0.545455], [0.545455]],
]


class TestComputeGeneralizedDiceScore(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1])
    def test_device(self, input_data, _expected_value):
        """
        Test if the result tensor is on the same device as the input tensor.
        """
        result = compute_generalized_dice(**input_data)
        np.testing.assert_equal(result.device, input_data["y_pred"].device)

    @parameterized.expand([TEST_CASE_1, TEST_CASE_5, TEST_CASE_6, TEST_CASE_7, TEST_CASE_8])
    def test_value(self, input_data, expected_value):
        """
        Test if the computed generalized dice score matches the expected value.
        """
        result = compute_generalized_dice(**input_data)
        np.testing.assert_allclose(result.cpu().numpy(), expected_value, atol=1e-4)

    @parameterized.expand([TEST_CASE_2, TEST_CASE_3, TEST_CASE_4, TEST_CASE_9])
    def test_value_class(self, input_data, expected_value):
        """
        Test if the GeneralizedDiceScore class computes the correct values.
        """
        y_pred = input_data.pop("y_pred")
        y = input_data.pop("y")
        generalized_dice_score = GeneralizedDiceScore(**input_data)
        generalized_dice_score(y_pred=y_pred, y=y)
        result = generalized_dice_score.aggregate()
        np.testing.assert_allclose(result.cpu().numpy(), expected_value, atol=1e-4)

    @parameterized.expand([TEST_CASE_10])
    def test_values_compare(self, input_data, expected_value):
        """
        Compare the results of compute_generalized_dice function and GeneralizedDiceScore class.
        """
        result = compute_generalized_dice(**input_data)
        np.testing.assert_allclose(result.cpu().numpy(), expected_value, atol=1e-4)

        y_pred = input_data.pop("y_pred")
        y = input_data.pop("y")
        generalized_dice_score = GeneralizedDiceScore(**input_data, reduction="mean_batch")
        generalized_dice_score(y_pred=y_pred, y=y)
        result_class_mean = generalized_dice_score.aggregate()
        np.testing.assert_allclose(result_class_mean.cpu().numpy(), np.mean(expected_value, axis=0), atol=1e-4)

    @parameterized.expand([TEST_CASE_11])
    def test_values_compare_sum_over_classes(self, input_data, expected_value):
        """
        Compare the results when summing over classes between compute_generalized_dice function and GeneralizedDiceScore class.
        """
        result = compute_generalized_dice(**input_data)
        np.testing.assert_allclose(result.cpu().numpy(), expected_value, atol=1e-4)

        y_pred = input_data.pop("y_pred")
        y = input_data.pop("y")
        input_data.pop("sum_over_classes")
        generalized_dice_score = GeneralizedDiceScore(**input_data, reduction="mean_channel")
        generalized_dice_score(y_pred=y_pred, y=y)
        result_class_mean = generalized_dice_score.aggregate()
        np.testing.assert_allclose(result_class_mean.cpu().numpy(), np.mean(expected_value, axis=1), atol=1e-4)


if __name__ == "__main__":
    unittest.main()
