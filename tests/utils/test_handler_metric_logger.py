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

from monai.utils import optional_import
from tests.util import SkipIfNoModule

try:
    _, has_ignite = optional_import("ignite")
    from ignite.engine import Engine, Events

    from monai.handlers import MetricLogger
except ImportError:
    has_ignite = False


class TestHandlerMetricLogger(unittest.TestCase):
    @SkipIfNoModule("ignite")
    def test_metric_logging(self):
        dummy_name = "dummy"

        # set up engine
        def _train_func(engine, batch):
            return torch.tensor(0.0)

        engine = Engine(_train_func)

        # set up dummy metric
        @engine.on(Events.EPOCH_COMPLETED)
        def _update_metric(engine):
            engine.state.metrics[dummy_name] = 1

        # set up testing handler
        handler = MetricLogger(loss_transform=lambda output: output.item())
        handler.attach(engine)

        engine.run(range(3), max_epochs=2)

        expected_loss = [(1, 0.0), (2, 0.0), (3, 0.0), (4, 0.0), (5, 0.0), (6, 0.0)]
        expected_metric = [(4, 1), (5, 1), (6, 1)]

        self.assertSetEqual({dummy_name}, set(handler.metrics))

        self.assertListEqual(expected_loss, handler.loss)
        self.assertListEqual(expected_metric, handler.metrics[dummy_name])


if __name__ == "__main__":
    unittest.main()
