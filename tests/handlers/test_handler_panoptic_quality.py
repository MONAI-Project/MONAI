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

from monai.handlers import PanopticQuality, from_engine
from tests.util import SkipIfNoModule, assert_allclose

sample_1_pred = torch.as_tensor(
    [[[0, 1, 1, 1], [0, 0, 5, 5], [2, 0, 3, 3], [2, 2, 2, 0]], [[0, 1, 1, 1], [0, 0, 0, 0], [2, 0, 3, 3], [4, 2, 2, 0]]]
)

sample_1_gt = torch.as_tensor(
    [[[0, 6, 6, 6], [1, 0, 5, 5], [1, 0, 3, 3], [1, 3, 2, 0]], [[0, 1, 1, 1], [0, 0, 1, 1], [2, 0, 3, 3], [4, 4, 4, 3]]]
)

sample_2_pred = torch.as_tensor(
    [[[3, 1, 1, 1], [3, 1, 1, 4], [3, 1, 4, 4], [3, 2, 2, 4]], [[0, 1, 1, 1], [2, 2, 2, 2], [2, 0, 0, 3], [4, 2, 2, 3]]]
)

sample_2_gt = torch.as_tensor(
    [[[0, 6, 6, 6], [1, 0, 5, 5], [1, 0, 3, 3], [1, 3, 2, 0]], [[0, 1, 1, 1], [2, 1, 1, 3], [2, 0, 0, 3], [4, 2, 2, 3]]]
)

TEST_CASE_1 = [{"num_classes": 4, "output_transform": from_engine(["pred", "label"])}, [0.6667, 0.1538, 0.6667, 0.5714]]
TEST_CASE_2 = [
    {
        "num_classes": 5,
        "output_transform": from_engine(["pred", "label"]),
        "metric_name": "rq",
        "match_iou_threshold": 0.3,
    },
    [0.6667, 0.7692, 0.8889, 0.5714, 0.0000],
]
TEST_CASE_3 = [
    {
        "num_classes": 5,
        "reduction": "mean",
        "output_transform": from_engine(["pred", "label"]),
        "metric_name": "SQ",
        "match_iou_threshold": 0.2,
    },
    0.8235,
]


@SkipIfNoModule("scipy.optimize")
class TestHandlerPanopticQuality(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3])
    def test_compute(self, input_params, expected_avg):
        metric = PanopticQuality(**input_params)

        # set up engine

        def _val_func(engine, batch):
            pass

        engine = Engine(_val_func)
        metric.attach(engine=engine, name="panoptic_quality")
        # test input a list of channel-first tensor
        y_pred = [sample_1_pred, sample_2_pred]
        y = [sample_1_gt, sample_2_gt]
        engine.state.output = {"pred": y_pred, "label": y}
        engine.fire_event(Events.ITERATION_COMPLETED)
        y_pred = [sample_1_pred, sample_1_pred]
        y = [sample_1_gt, sample_1_gt]
        engine.state.output = {"pred": y_pred, "label": y}
        engine.fire_event(Events.ITERATION_COMPLETED)

        engine.fire_event(Events.EPOCH_COMPLETED)
        assert_allclose(engine.state.metrics["panoptic_quality"], expected_avg, atol=1e-4, rtol=1e-4, type_test=False)


if __name__ == "__main__":
    unittest.main()
