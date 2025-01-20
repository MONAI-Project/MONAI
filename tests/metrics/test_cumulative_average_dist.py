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

from monai.metrics import CumulativeAverage
from tests.util import DistCall, DistTestCase, SkipIfBeforePyTorchVersion


@SkipIfBeforePyTorchVersion((1, 8))
class DistributedCumulativeAverage(DistTestCase):
    @DistCall(nnodes=1, nproc_per_node=2)
    def test_value(self):
        rank = dist.get_rank()
        nprocs = dist.get_world_size()
        is_cuda = dist.get_backend() == dist.Backend.NCCL
        if is_cuda:
            torch.cuda.set_device(rank)

        device = torch.device(rank) if is_cuda else torch.device("cpu")

        avg_meter = CumulativeAverage()  # each process rank has it's own AverageMeter
        n_iter = 10
        for i in range(n_iter):
            val = torch.as_tensor(rank + i, device=device)
            avg_meter.append(val=val)

        avg_val = avg_meter.aggregate()  # average across all processes
        expected_val = sum(sum(list(range(rank_i, rank_i + n_iter))) for rank_i in range(nprocs)) / (n_iter * nprocs)
        np.testing.assert_equal(avg_val, expected_val)


if __name__ == "__main__":
    unittest.main()
