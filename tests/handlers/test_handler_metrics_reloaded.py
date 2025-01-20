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

import torch
from ignite.engine import Engine, Events
from parameterized import parameterized

from monai.handlers import MetricsReloadedBinaryHandler, MetricsReloadedCategoricalHandler, from_engine
from monai.utils import optional_import
from tests.util import assert_allclose

_, has_metrics = optional_import("MetricsReloaded")

TEST_CASE_BIN_1 = [
    {"metric_name": "Volume Difference"},
    [torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]), torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])],
    [torch.tensor([[[1.0, 0.0], [1.0, 1.0]]]), torch.tensor([[[1.0, 0.0], [1.0, 1.0]]])],
    0.3333,
]

TEST_CASE_BIN_2 = [
    {"metric_name": "Boundary IoU"},
    [torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]), torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])],
    [torch.tensor([[[1.0, 0.0], [1.0, 1.0]]]), torch.tensor([[[1.0, 0.0], [1.0, 1.0]]])],
    0.6667,
]

TEST_CASE_BIN_3 = [
    {"metric_name": "xTh Percentile Hausdorff Distance"},
    [torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]), torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])],
    [torch.tensor([[[1.0, 0.0], [1.0, 1.0]]]), torch.tensor([[[1.0, 0.0], [1.0, 1.0]]])],
    0.9,
]

TEST_CASE_CAT_1 = [
    {"metric_name": "Weighted Cohens Kappa"},
    [
        torch.tensor([[[0, 0], [0, 1]], [[0, 0], [0, 0]], [[1, 1], [1, 0]]]),
        torch.tensor([[[0, 0], [0, 1]], [[0, 0], [0, 0]], [[1, 1], [1, 0]]]),
    ],
    [
        torch.tensor([[[1, 0], [0, 1]], [[0, 1], [0, 0]], [[0, 0], [1, 0]]]),
        torch.tensor([[[1, 0], [0, 1]], [[0, 1], [0, 0]], [[0, 0], [1, 0]]]),
    ],
    0.272727,
]

TEST_CASE_CAT_2 = [
    {"metric_name": "Matthews Correlation Coefficient"},
    [
        torch.tensor([[[0, 0], [0, 1]], [[0, 0], [0, 0]], [[1, 1], [1, 0]]]),
        torch.tensor([[[0, 0], [0, 1]], [[0, 0], [0, 0]], [[1, 1], [1, 0]]]),
    ],
    [
        torch.tensor([[[1, 0], [0, 1]], [[0, 1], [0, 0]], [[0, 0], [1, 0]]]),
        torch.tensor([[[1, 0], [0, 1]], [[0, 1], [0, 0]], [[0, 0], [1, 0]]]),
    ],
    0.387298,
]


@unittest.skipIf(not has_metrics, "MetricsReloaded not available.")
class TestHandlerMetricsReloadedBinary(unittest.TestCase):
    @parameterized.expand([TEST_CASE_BIN_1, TEST_CASE_BIN_2, TEST_CASE_BIN_3])
    def test_compute(self, input_params, y_pred, y, expected_value):
        input_params["output_transform"] = from_engine(["pred", "label"])
        metric = MetricsReloadedBinaryHandler(**input_params)

        # set up engine

        def _val_func(engine, batch):
            pass

        engine = Engine(_val_func)
        metric.attach(engine=engine, name=input_params["metric_name"])
        engine.state.output = {"pred": y_pred, "label": y}
        engine.fire_event(Events.ITERATION_COMPLETED)

        engine.state.output = {"pred": y_pred, "label": y}
        engine.fire_event(Events.ITERATION_COMPLETED)

        engine.fire_event(Events.EPOCH_COMPLETED)
        assert_allclose(
            engine.state.metrics[input_params["metric_name"]], expected_value, atol=1e-4, rtol=1e-4, type_test=False
        )

    @parameterized.expand([TEST_CASE_BIN_1, TEST_CASE_BIN_2, TEST_CASE_BIN_3])
    def test_shape_mismatch(self, input_params, _y_pred, _y, _expected_value):
        input_params["output_transform"] = from_engine(["pred", "label"])
        metric = MetricsReloadedBinaryHandler(**input_params)
        with self.assertRaises((AssertionError, ValueError)):
            y_pred = torch.Tensor([[0, 1], [1, 0]])
            y = torch.ones((2, 3))
            metric.update([y_pred, y])

        with self.assertRaises((AssertionError, ValueError)):
            y_pred = [torch.ones((2, 1, 1)), torch.ones((1, 1, 1))]
            y = [torch.ones((2, 1, 1)), torch.ones((1, 1, 1))]
            metric.update([y_pred, y])


@unittest.skipIf(not has_metrics, "MetricsReloaded not available.")
class TestMetricsReloadedCategorical(unittest.TestCase):
    @parameterized.expand([TEST_CASE_CAT_1, TEST_CASE_CAT_2])
    def test_compute(self, input_params, y_pred, y, expected_value):
        input_params["output_transform"] = from_engine(["pred", "label"])
        metric = MetricsReloadedCategoricalHandler(**input_params)

        # set up engine

        def _val_func(engine, batch):
            pass

        engine = Engine(_val_func)
        metric.attach(engine=engine, name=input_params["metric_name"])
        engine.state.output = {"pred": y_pred, "label": y}
        engine.fire_event(Events.ITERATION_COMPLETED)

        engine.state.output = {"pred": y_pred, "label": y}
        engine.fire_event(Events.ITERATION_COMPLETED)

        engine.fire_event(Events.EPOCH_COMPLETED)
        assert_allclose(
            engine.state.metrics[input_params["metric_name"]], expected_value, atol=1e-4, rtol=1e-4, type_test=False
        )

    @parameterized.expand([TEST_CASE_CAT_1, TEST_CASE_CAT_2])
    def test_shape_mismatch(self, input_params, y_pred, y, _expected_value):
        input_params["output_transform"] = from_engine(["pred", "label"])
        metric = MetricsReloadedCategoricalHandler(**input_params)
        with self.assertRaises((AssertionError, ValueError)):
            y_pred[0] = torch.zeros([3, 2, 1])
            metric.update([y_pred, y])


if __name__ == "__main__":
    unittest.main()
