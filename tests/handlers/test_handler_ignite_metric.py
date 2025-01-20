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
from parameterized import parameterized

from monai.handlers import IgniteMetricHandler, from_engine
from monai.losses import DiceLoss
from monai.metrics import LossMetric
from tests.util import SkipIfNoModule, assert_allclose, optional_import

try:
    _, has_ignite = optional_import("ignite")
    from ignite.engine import Engine, Events
except ImportError:
    has_ignite = False

TEST_CASE_1 = [
    {"reduction": "none", "include_background": True},
    {},
    {"output_transform": from_engine(["pred", "label"])},
    0.25,
]
TEST_CASE_2 = [
    {"reduction": "mean", "include_background": False},
    {},
    {"output_transform": from_engine(["pred", "label"])},
    0.5,
]
TEST_CASE_3 = [
    {"reduction": "none"},
    {"reduction": "mean_channel"},
    {"output_transform": from_engine(["pred", "label"])},
    torch.Tensor([0.5, 0]),
]

TEST_CASES = [
    [
        {"include_background": True, "smooth_nr": 1e-6, "smooth_dr": 1e-6},
        {
            "input": torch.tensor([[[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]]]),
            "target": torch.tensor([[[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]]]),
        },
        0,
    ],
    [
        {"include_background": False, "smooth_nr": 1e-6, "smooth_dr": 1e-6},
        {
            "input": torch.tensor([[[[0.0, 0.0], [0.0, 0.0]], [[1.0, 1.0], [1.0, 1.0]]]]),
            "target": torch.tensor([[[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]]]),
        },
        0,
    ],
    [
        {"include_background": True, "smooth_nr": 1e-6, "smooth_dr": 1e-6},
        {
            "input": torch.tensor([[[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]]]),
            "target": torch.tensor([[[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]]),
        },
        1,
    ],
    [
        {"include_background": False, "smooth_nr": 1e-6, "smooth_dr": 1e-6},
        {
            "input": torch.tensor([[[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]]),
            "target": torch.tensor([[[[0.0, 0.0], [0.0, 0.0]], [[1.0, 1.0], [1.0, 1.0]]]]),
        },
        1,
    ],
    [
        {"include_background": True, "smooth_nr": 1e-6, "smooth_dr": 1e-6},
        {
            "input": torch.tensor([[[[0.0, 1.0], [0.0, 1.0]], [[0.0, 1.0], [0.0, 1.0]]]]),
            "target": torch.tensor([[[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]]]),
        },
        0.333333,
    ],
    [
        {"include_background": False, "smooth_nr": 1e-6, "smooth_dr": 1e-6},
        {
            "input": torch.tensor([[[[0.0, 1.0], [0.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]]]),
            "target": torch.tensor([[[[1.0, 0.0], [1.0, 0.0]], [[1.0, 1.0], [1.0, 1.0]]]]),
        },
        0,
    ],
]


class TestHandlerIgniteMetricHandler(unittest.TestCase):
    @SkipIfNoModule("ignite")
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3])
    def test_metric_fn(self, loss_params, metric_params, handler_params, expected_avg):
        loss_fn = DiceLoss(**loss_params)
        metric_fn = LossMetric(loss_fn=loss_fn, **metric_params)
        ignite_metric = IgniteMetricHandler(metric_fn=metric_fn, **handler_params)

        def _val_func(engine, batch):
            pass

        engine = Engine(_val_func)
        ignite_metric.attach(engine=engine, name="ignite_dice_loss")
        y_pred = torch.tensor([[[[0.0, 1.0]], [[1.0, 0.0]]]])
        y = torch.tensor([[[[0.0, 1.0]], [[0.0, 1.0]]]])
        engine.state.output = {"pred": y_pred, "label": y}
        engine.fire_event(Events.ITERATION_COMPLETED)

        y_pred = torch.tensor([[[[0.0, 1.0]], [[1.0, 0.0]]]])
        y = torch.tensor([[[[0.0, 1.0]], [[1.0, 0.0]]]])
        engine.state.output = {"pred": y_pred, "label": y}
        engine.fire_event(Events.ITERATION_COMPLETED)

        engine.fire_event(Events.EPOCH_COMPLETED)
        assert_allclose(engine.state.metrics["ignite_dice_loss"], expected_avg, atol=1e-4, rtol=1e-4, type_test=False)

    @SkipIfNoModule("ignite")
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3])
    def test_loss_fn(self, loss_params, metric_params, handler_params, expected_avg):
        loss_fn = DiceLoss(**loss_params)
        ignite_metric = IgniteMetricHandler(loss_fn=loss_fn, **handler_params, **metric_params)

        def _val_func(engine, batch):
            pass

        engine = Engine(_val_func)
        ignite_metric.attach(engine=engine, name="ignite_dice_loss")
        y_pred = torch.tensor([[[[0.0, 1.0]], [[1.0, 0.0]]]])
        y = torch.tensor([[[[0.0, 1.0]], [[0.0, 1.0]]]])
        engine.state.output = {"pred": y_pred, "label": y}
        engine.fire_event(Events.ITERATION_COMPLETED)

        y_pred = torch.tensor([[[[0.0, 1.0]], [[1.0, 0.0]]]])
        y = torch.tensor([[[[0.0, 1.0]], [[1.0, 0.0]]]])
        engine.state.output = {"pred": y_pred, "label": y}
        engine.fire_event(Events.ITERATION_COMPLETED)

        engine.fire_event(Events.EPOCH_COMPLETED)
        assert_allclose(engine.state.metrics["ignite_dice_loss"], expected_avg, atol=1e-4, rtol=1e-4, type_test=False)

    @SkipIfNoModule("ignite")
    @parameterized.expand(TEST_CASES)
    def test_dice_loss(self, input_param, input_data, expected_val):
        loss_fn = DiceLoss(**input_param)
        ignite_metric = IgniteMetricHandler(loss_fn=loss_fn, output_transform=from_engine(["pred", "label"]))

        def _val_func(engine, batch):
            pass

        engine = Engine(_val_func)
        ignite_metric.attach(engine=engine, name="ignite_dice_loss")
        y_pred = input_data["input"]
        y = input_data["target"]
        engine.state.output = {"pred": y_pred, "label": y}
        engine.fire_event(Events.ITERATION_COMPLETED)

        engine.fire_event(Events.EPOCH_COMPLETED)
        assert_allclose(engine.state.metrics["ignite_dice_loss"], expected_val, atol=1e-4, rtol=1e-4, type_test=False)

    @SkipIfNoModule("ignite")
    @parameterized.expand(TEST_CASES[0:2])
    def test_old_ignite_metric(self, input_param, input_data, expected_val):
        loss_fn = DiceLoss(**input_param)
        ignite_metric = IgniteMetricHandler(loss_fn=loss_fn, output_transform=from_engine(["pred", "label"]))

        def _val_func(engine, batch):
            pass

        engine = Engine(_val_func)
        ignite_metric.attach(engine=engine, name="ignite_dice_loss")
        y_pred = input_data["input"]
        y = input_data["target"]
        engine.state.output = {"pred": y_pred, "label": y}
        engine.fire_event(Events.ITERATION_COMPLETED)

        engine.fire_event(Events.EPOCH_COMPLETED)
        assert_allclose(engine.state.metrics["ignite_dice_loss"], expected_val, atol=1e-4, rtol=1e-4, type_test=False)


if __name__ == "__main__":
    unittest.main()
