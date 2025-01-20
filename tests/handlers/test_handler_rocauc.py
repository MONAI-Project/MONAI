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

from monai.handlers import ROCAUC
from monai.transforms import Activations, AsDiscrete


class TestHandlerROCAUC(unittest.TestCase):

    def test_compute(self):
        auc_metric = ROCAUC()
        act = Activations(softmax=True)
        to_onehot = AsDiscrete(to_onehot=2)

        y_pred = [torch.Tensor([0.1, 0.9]), torch.Tensor([0.3, 1.4])]
        y = [torch.Tensor([0]), torch.Tensor([1])]
        y_pred = [act(p) for p in y_pred]
        y = [to_onehot(y_) for y_ in y]
        auc_metric.update([y_pred, y])

        y_pred = [torch.Tensor([0.2, 0.1]), torch.Tensor([0.1, 0.5])]
        y = [torch.Tensor([0]), torch.Tensor([1])]
        y_pred = [act(p) for p in y_pred]
        y = [to_onehot(y_) for y_ in y]

        auc_metric.update([y_pred, y])

        auc = auc_metric.compute()
        np.testing.assert_allclose(0.75, auc)


if __name__ == "__main__":
    unittest.main()
