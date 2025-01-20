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

from monai.metrics import MeanIoU, compute_iou

_device = "cuda:0" if torch.cuda.is_available() else "cpu"
# keep background
TEST_CASE_1 = [  # y (1, 1, 2, 2), y_pred (1, 1, 2, 2), expected out (1, 1)
    {
        "y_pred": torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], device=_device),
        "y": torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]], device=_device),
        "include_background": True,
    },
    [[0.6667]],
]

# remove background and not One-Hot target
TEST_CASE_2 = [  # y (2, 3, 2, 2), y_pred (2, 3, 2, 2), expected out (2, 2) (no background)
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
    },
    [[0.3333, 0.0000], [0.5000, 0.5000]],
]

# should return Nan for all labels=0 case and skip for MeanIoU
TEST_CASE_3 = [
    {
        "y_pred": torch.tensor(
            [
                [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]], [[1.0, 1.0], [1.0, 1.0]]],
                [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]], [[1.0, 1.0], [1.0, 1.0]]],
            ]
        ),
        "y": torch.tensor(
            [
                [[[1.0, 1.0], [1.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
                [[[0.0, 1.0], [1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]]],
            ]
        ),
        "include_background": True,
    },
    [[False, True, True], [False, False, True]],
]

TEST_CASE_4 = [
    {"include_background": True, "reduction": "mean_batch", "get_not_nans": True},
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
    },
    [0.5416, 0.2500, 0.5000],
]

TEST_CASE_5 = [
    {"include_background": True, "reduction": "mean", "get_not_nans": True},
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
    },
    0.5555,
]

TEST_CASE_6 = [
    {"include_background": True, "reduction": "sum_batch", "get_not_nans": True},
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
    },
    [1.5000, 0.0000, 0.0000],
]

TEST_CASE_7 = [
    {"include_background": True, "reduction": "mean", "get_not_nans": True},
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
    },
    0.7500,
]

TEST_CASE_8 = [
    {"include_background": False, "reduction": "sum_batch", "get_not_nans": True},
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
    },
    [0.0000, 0.0000],
]

TEST_CASE_9 = [
    {"y": torch.ones((2, 2, 3, 3)), "y_pred": torch.ones((2, 2, 3, 3))},
    [[1.0000, 1.0000], [1.0000, 1.0000]],
]

TEST_CASE_10 = [
    {"y": [torch.ones((2, 3, 3)), torch.ones((2, 3, 3))], "y_pred": [torch.ones((2, 3, 3)), torch.ones((2, 3, 3))]},
    [[1.0000, 1.0000], [1.0000, 1.0000]],
]

TEST_CASE_11 = [
    {"y": torch.zeros((2, 2, 3, 3)), "y_pred": torch.zeros((2, 2, 3, 3)), "ignore_empty": False},
    [[1.0000, 1.0000], [1.0000, 1.0000]],
]

TEST_CASE_12 = [
    {"y": torch.zeros((2, 2, 3, 3)), "y_pred": torch.ones((2, 2, 3, 3)), "ignore_empty": False},
    [[0.0000, 0.0000], [0.0000, 0.0000]],
]


class TestComputeMeanIoU(unittest.TestCase):

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_9, TEST_CASE_11, TEST_CASE_12])
    def test_value(self, input_data, expected_value):
        result = compute_iou(**input_data)
        np.testing.assert_allclose(result.cpu().numpy(), expected_value, atol=1e-4)
        np.testing.assert_equal(result.device, input_data["y_pred"].device)

    @parameterized.expand([TEST_CASE_3])
    def test_nans(self, input_data, expected_value):
        result = compute_iou(**input_data)
        self.assertTrue(np.allclose(np.isnan(result.cpu().numpy()), expected_value))

    # MeanIoU class tests
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_10])
    def test_value_class(self, input_data, expected_value):
        # same test as for compute_iou
        vals = {}
        vals["y_pred"] = input_data.pop("y_pred")
        vals["y"] = input_data.pop("y")
        iou_metric = MeanIoU(**input_data)
        iou_metric(**vals)
        result = iou_metric.aggregate(reduction="none")
        np.testing.assert_allclose(result.cpu().numpy(), expected_value, atol=1e-4)

    @parameterized.expand([TEST_CASE_4, TEST_CASE_5, TEST_CASE_6, TEST_CASE_7, TEST_CASE_8])
    def test_nans_class(self, params, input_data, expected_value):
        iou_metric = MeanIoU(**params)
        iou_metric(**input_data)
        result, _ = iou_metric.aggregate()
        np.testing.assert_allclose(result.cpu().numpy(), expected_value, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
