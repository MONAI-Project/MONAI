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

from monai.handlers import MeanIoUHandler, from_engine
from tests.util import assert_allclose

TEST_CASE_1 = [{"include_background": True, "output_transform": from_engine(["pred", "label"])}, 0.75, (4, 2)]
TEST_CASE_2 = [{"include_background": False, "output_transform": from_engine(["pred", "label"])}, 2 / 3, (4, 1)]
TEST_CASE_3 = [
    {"reduction": "mean_channel", "output_transform": from_engine(["pred", "label"])},
    torch.Tensor([1.0, 0.0, 1.0, 1.0]),
    (4, 2),
]


class TestHandlerMeanIoU(unittest.TestCase):
    # TODO test multi node averaged iou

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3])
    def test_compute(self, input_params, expected_avg, details_shape):
        iou_metric = MeanIoUHandler(**input_params)

        # set up engine

        def _val_func(engine, batch):
            pass

        engine = Engine(_val_func)
        iou_metric.attach(engine=engine, name="mean_iou")
        # test input a list of channel-first tensor
        y_pred = [torch.Tensor([[0], [1]]), torch.Tensor([[1], [0]])]
        y = torch.Tensor([[[0], [1]], [[0], [1]]])
        engine.state.output = {"pred": y_pred, "label": y}
        engine.fire_event(Events.ITERATION_COMPLETED)

        y_pred = [torch.Tensor([[0], [1]]), torch.Tensor([[1], [0]])]
        y = torch.Tensor([[[0], [1]], [[1], [0]]])
        engine.state.output = {"pred": y_pred, "label": y}
        engine.fire_event(Events.ITERATION_COMPLETED)

        engine.fire_event(Events.EPOCH_COMPLETED)
        assert_allclose(engine.state.metrics["mean_iou"], expected_avg)
        self.assertTupleEqual(tuple(engine.state.metric_details["mean_iou"].shape), details_shape)

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    def test_shape_mismatch(self, input_params, _expected_avg, _details_shape):
        iou_metric = MeanIoUHandler(**input_params)
        with self.assertRaises((AssertionError, ValueError)):
            y_pred = torch.Tensor([[0, 1], [1, 0]])
            y = torch.ones((3, 30))
            iou_metric.update([y_pred, y])

        with self.assertRaises((AssertionError, ValueError)):
            y_pred = torch.Tensor([[0, 1], [1, 0]])
            y = torch.ones((8, 30))
            iou_metric.update([y_pred, y])


if __name__ == "__main__":
    unittest.main()
