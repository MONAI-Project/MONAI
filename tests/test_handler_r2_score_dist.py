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

from monai.handlers import R2Score
from tests.utils import DistCall, DistTestCase


class DistributedR2Score(DistTestCase):

    @DistCall(nnodes=1, nproc_per_node=2, node_rank=0)
    def test_compute(self):
        r2_score = R2Score(multi_output="variance_weighted", p=1)

        device = f"cuda:{dist.get_rank()}" if torch.cuda.is_available() else "cpu"
        if dist.get_rank() == 0:
            y_pred = [torch.tensor([0.1, 1.0], device=device), torch.tensor([-0.25, 0.5], device=device)]
            y = [torch.tensor([0.1, 0.82], device=device), torch.tensor([-0.2, 0.01], device=device)]

        if dist.get_rank() == 1:
            y_pred = [
                torch.tensor([3.0, -0.2], device=device),
                torch.tensor([0.99, 2.1], device=device),
                torch.tensor([-0.1, 0.0], device=device),
            ]
            y = [
                torch.tensor([2.7, -0.1], device=device),
                torch.tensor([1.58, 2.0], device=device),
                torch.tensor([-1.0, -0.1], device=device),
            ]

        r2_score.update([y_pred, y])

        result = r2_score.compute()
        np.testing.assert_allclose(0.829185, result, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
