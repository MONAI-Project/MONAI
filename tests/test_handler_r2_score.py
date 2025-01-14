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

from monai.handlers import R2Score


class TestHandlerR2Score(unittest.TestCase):

    def test_compute(self):
        r2_score = R2Score(multi_output="variance_weighted", p=1)

        y_pred = [torch.Tensor([0.1, 1.0]), torch.Tensor([-0.25, 0.5])]
        y = [torch.Tensor([0.1, 0.82]), torch.Tensor([-0.2, 0.01])]
        r2_score.update([y_pred, y])

        y_pred = [torch.Tensor([3.0, -0.2]), torch.Tensor([0.99, 2.1])]
        y = [torch.Tensor([2.7, -0.1]), torch.Tensor([1.58, 2.0])]

        r2_score.update([y_pred, y])

        r2 = r2_score.compute()
        np.testing.assert_allclose(0.867314, r2, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
