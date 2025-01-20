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

from monai.data import decollate_batch
from monai.metrics import ROCAUCMetric, compute_roc_auc
from monai.transforms import Activations, AsDiscrete, Compose, ToTensor

_device = "cuda:0" if torch.cuda.is_available() else "cpu"
TEST_CASE_1 = [
    torch.tensor([[0.1, 0.9], [0.3, 1.4], [0.2, 0.1], [0.1, 0.5]], device=_device),
    torch.tensor([[0], [1], [0], [1]], device=_device),
    True,
    2,
    "macro",
    0.75,
]

TEST_CASE_2 = [
    torch.tensor([[0.5], [0.5], [0.2], [8.3]]),
    torch.tensor([[0], [1], [0], [1]]),
    False,
    None,
    "macro",
    0.875,
]

TEST_CASE_3 = [torch.tensor([[0.5], [0.5], [0.2], [8.3]]), torch.tensor([0, 1, 0, 1]), False, None, "macro", 0.875]

TEST_CASE_4 = [torch.tensor([0.5, 0.5, 0.2, 8.3]), torch.tensor([0, 1, 0, 1]), False, None, "macro", 0.875]

TEST_CASE_5 = [
    torch.tensor([[0.1, 0.9], [0.3, 1.4], [0.2, 0.1], [0.1, 0.5]]),
    torch.tensor([[0], [1], [0], [1]]),
    True,
    2,
    "none",
    [0.75, 0.75],
]

TEST_CASE_6 = [
    torch.tensor([[0.1, 0.9], [0.3, 1.4], [0.2, 0.1], [0.1, 0.5], [0.1, 0.5]]),
    torch.tensor([[1, 0], [0, 1], [0, 0], [1, 1], [0, 1]]),
    True,
    None,
    "weighted",
    0.56667,
]

TEST_CASE_7 = [
    torch.tensor([[0.1, 0.9], [0.3, 1.4], [0.2, 0.1], [0.1, 0.5], [0.1, 0.5]]),
    torch.tensor([[1, 0], [0, 1], [0, 0], [1, 1], [0, 1]]),
    True,
    None,
    "micro",
    0.62,
]

TEST_CASE_8 = [
    torch.tensor([[0.1, 0.9], [0.3, 1.4], [0.2, 0.1], [0.1, 0.5]]),
    torch.tensor([[0], [0], [0], [0]]),
    True,
    2,
    "macro",
    float("nan"),
]

TEST_CASE_9 = [
    torch.tensor([[0.1, 0.9], [0.3, 1.4], [0.2, 0.1], [0.1, 0.5]]),
    torch.tensor([[1], [1], [1], [1]]),
    True,
    2,
    "macro",
    float("nan"),
]

TEST_CASE_10 = [
    torch.tensor([[0.1, 0.9], [0.3, 1.4], [0.2, 0.1], [0.1, 0.5]]),
    torch.tensor([[0, 0], [1, 1], [2, 2], [3, 3]]),
    True,
    None,
    "macro",
    float("nan"),
]


class TestComputeROCAUC(unittest.TestCase):

    @parameterized.expand(
        [
            TEST_CASE_1,
            TEST_CASE_2,
            TEST_CASE_3,
            TEST_CASE_4,
            TEST_CASE_5,
            TEST_CASE_6,
            TEST_CASE_7,
            TEST_CASE_8,
            TEST_CASE_9,
            TEST_CASE_10,
        ]
    )
    def test_value(self, y_pred, y, softmax, to_onehot, average, expected_value):
        y_pred_trans = Compose([ToTensor(), Activations(softmax=softmax)])
        y_trans = Compose([ToTensor(), AsDiscrete(to_onehot=to_onehot)])
        y_pred = torch.stack([y_pred_trans(i) for i in decollate_batch(y_pred)], dim=0)
        y = torch.stack([y_trans(i) for i in decollate_batch(y)], dim=0)
        result = compute_roc_auc(y_pred=y_pred, y=y, average=average)
        np.testing.assert_allclose(expected_value, result, rtol=1e-5)

    @parameterized.expand(
        [
            TEST_CASE_1,
            TEST_CASE_2,
            TEST_CASE_3,
            TEST_CASE_4,
            TEST_CASE_5,
            TEST_CASE_6,
            TEST_CASE_7,
            TEST_CASE_8,
            TEST_CASE_9,
            TEST_CASE_10,
        ]
    )
    def test_class_value(self, y_pred, y, softmax, to_onehot, average, expected_value):
        y_pred_trans = Compose([ToTensor(), Activations(softmax=softmax)])
        y_trans = Compose([ToTensor(), AsDiscrete(to_onehot=to_onehot)])
        y_pred = [y_pred_trans(i) for i in decollate_batch(y_pred)]
        y = [y_trans(i) for i in decollate_batch(y)]
        metric = ROCAUCMetric(average=average)
        metric(y_pred=y_pred, y=y)
        result = metric.aggregate()
        np.testing.assert_allclose(expected_value, result, rtol=1e-5)
        result = metric.aggregate(average=average)  # test optional argument
        metric.reset()
        np.testing.assert_allclose(expected_value, result, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
