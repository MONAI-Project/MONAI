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
import torch.distributed as dist
from ignite.engine import Engine

from monai.handlers import ConfusionMatrix
from tests.util import DistCall, DistTestCase


class DistributedConfusionMatrix(DistTestCase):
    @DistCall(nnodes=1, nproc_per_node=2)
    def test_compute(self):
        self._compute()

    def _compute(self):
        device = f"cuda:{dist.get_rank()}" if torch.cuda.is_available() else "cpu"
        metric = ConfusionMatrix(include_background=True, metric_name="tpr")

        def _val_func(engine, batch):
            pass

        engine = Engine(_val_func)
        metric.attach(engine, "confusion_matrix")
        if dist.get_rank() == 0:
            y_pred = torch.tensor(
                [
                    [[[0.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 1.0]], [[1.0, 0.0], [0.0, 0.0]]],
                    [[[0.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 0.0]], [[0.0, 1.0], [1.0, 0.0]]],
                ],
                device=device,
            )
            y = torch.tensor(
                [
                    [[[0.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 1.0]], [[1.0, 0.0], [0.0, 0.0]]],
                    [[[0.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 0.0]], [[0.0, 1.0], [1.0, 0.0]]],
                ],
                device=device,
            )
            metric.update([y_pred, y])

        if dist.get_rank() == 1:
            y_pred = torch.tensor(
                [[[[0.0, 1.0], [1.0, 0.0]], [[1.0, 0.0], [1.0, 1.0]], [[0.0, 1.0], [0.0, 0.0]]]], device=device
            )
            y = torch.tensor(
                [[[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]]], device=device
            )
            metric.update([y_pred, y])

        avg_metric = metric.compute()
        np.testing.assert_allclose(avg_metric, 0.7, rtol=1e-04, atol=1e-04)


if __name__ == "__main__":
    unittest.main()
