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
from typing import Any

import torch
from ignite.engine import Engine
from parameterized import parameterized

from monai.handlers import ConfusionMatrix
from tests.util import assert_allclose

TEST_CASE_1 = [{"include_background": True, "save_details": False, "metric_name": "f1"}, 0.75]
TEST_CASE_2 = [{"include_background": False, "save_details": False, "metric_name": "ppv"}, 1.0]
TEST_CASE_3 = [{"save_details": False, "metric_name": "f1", "reduction": "mean_batch"}, torch.tensor([0.6667, 0.8000])]
TEST_CASE_SEG_1 = [{"include_background": True, "metric_name": "tpr"}, 0.7]

data_1: dict[Any, Any] = {
    "y_pred": torch.tensor(
        [
            [[[0.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 1.0]], [[1.0, 0.0], [0.0, 0.0]]],
            [[[0.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 0.0]], [[0.0, 1.0], [1.0, 0.0]]],
        ]
    ),
    "y": torch.tensor(
        [
            [[[0.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 1.0]], [[1.0, 0.0], [0.0, 0.0]]],
            [[[0.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 0.0]], [[0.0, 1.0], [1.0, 0.0]]],
        ]
    ),
}

data_2: dict[Any, Any] = {
    "y_pred": torch.tensor([[[[0.0, 1.0], [1.0, 0.0]], [[1.0, 0.0], [1.0, 1.0]], [[0.0, 1.0], [0.0, 0.0]]]]),
    "y": torch.tensor([[[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]]]),
}


class TestHandlerConfusionMatrix(unittest.TestCase):
    # TODO test multi node averaged confusion matrix

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3])
    def test_compute(self, input_params, expected_avg):
        metric = ConfusionMatrix(**input_params)
        # test input a list of channel-first tensor
        y_pred = [torch.Tensor([[0], [1]]), torch.Tensor([[1], [0]])]
        y = [torch.Tensor([[0], [1]]), torch.Tensor([[0], [1]])]
        metric.update([y_pred, y])

        y_pred = torch.Tensor([[[0], [1]], [[1], [0]]])
        y = torch.Tensor([[[0], [1]], [[1], [0]]])
        metric.update([y_pred, y])

        avg_metric = metric.compute()
        assert_allclose(avg_metric, expected_avg, atol=1e-4, rtol=1e-4, type_test=False)

    @parameterized.expand([TEST_CASE_SEG_1])
    def test_compute_seg(self, input_params, expected_avg):
        metric = ConfusionMatrix(**input_params)

        def _val_func(engine, batch):
            pass

        engine = Engine(_val_func)
        metric.attach(engine, "confusion_matrix")

        y_pred = data_1["y_pred"]
        y = data_1["y"]
        metric.update([y_pred, y])

        y_pred = data_2["y_pred"]
        y = data_2["y"]
        metric.update([y_pred, y])

        avg_metric = metric.compute()
        self.assertAlmostEqual(avg_metric, expected_avg, places=4)

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    def test_shape_mismatch(self, input_params, _expected):
        metric = ConfusionMatrix(**input_params)
        with self.assertRaises((AssertionError, ValueError)):
            y_pred = torch.Tensor([[0, 1], [1, 0]])
            y = torch.ones((2, 3))
            metric.update([y_pred, y])

        with self.assertRaises((AssertionError, ValueError)):
            y_pred = torch.Tensor([[0, 1], [1, 0]])
            y = torch.ones((3, 2))
            metric.update([y_pred, y])


if __name__ == "__main__":
    unittest.main()
