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

from monai.losses import DiceLoss
from monai.metrics import LossMetric

_device = "cuda:0" if torch.cuda.is_available() else "cpu"
TEST_CASE_1 = [  # y (1, 1, 2, 2), y_pred (1, 1, 2, 2), expected out (1, 1)
    {
        "loss_class": DiceLoss,
        "loss_kwargs": {"include_background": True},
        "reduction": "mean",
        "get_not_nans": False,
        "y_pred": torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], device=_device),
        "y": torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]], device=_device),
        "include_background": True,
    },
    [0.2],
]


class TestComputeLossMetric(unittest.TestCase):

    @parameterized.expand([TEST_CASE_1])
    def test_value_class(self, input_data, expected_value):
        loss_fn = input_data["loss_class"](**input_data["loss_kwargs"])
        loss_metric = LossMetric(
            loss_fn=loss_fn, reduction=input_data["reduction"], get_not_nans=input_data["get_not_nans"]
        )

        loss_metric(y_pred=input_data.get("y_pred"), y=input_data.get("y"))
        loss_metric(y_pred=input_data.get("y_pred"), y=input_data.get("y"))
        result = loss_metric.aggregate()
        np.testing.assert_allclose(result.cpu().numpy(), expected_value, atol=1e-4)
        loss_metric.reset()
        result = loss_metric.aggregate()
        np.testing.assert_allclose(result.cpu().numpy(), 0.0, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
