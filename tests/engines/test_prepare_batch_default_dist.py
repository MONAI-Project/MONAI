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
import torch.distributed as dist
from parameterized import parameterized

from monai.engines import PrepareBatchDefault, SupervisedEvaluator
from tests.util import DistCall, DistTestCase, assert_allclose

TEST_CASE_1 = [
    [
        # data for rank 0, has 1 iteration
        [{"image": torch.tensor([1, 1]), "label": torch.tensor([1, 0])}],
        # data for rank 1, has 2 iterations
        [
            {"image": torch.tensor([1, 0]), "label": torch.tensor([1, 0])},
            {"image": torch.tensor([1]), "label": torch.tensor([0])},
        ],
    ]
]

TEST_CASE_2 = [
    [
        # data for rank 0
        [{"image": torch.tensor([0, 1, 1, 0, 1]), "label": torch.tensor([1, 1, 0, 0, 1])}],
        # data for rank 1
        [],
    ]
]


class TestNet(torch.nn.Module):
    __test__ = False  # indicate to pytest that this class is not intended for collection

    def forward(self, x: torch.Tensor):
        return x


class DistributedPrepareBatchDefault(DistTestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    @DistCall(nnodes=1, nproc_per_node=2, node_rank=0)
    def test_compute(self, dataloaders):
        device = torch.device(f"cuda:{dist.get_rank()}" if torch.cuda.is_available() else "cpu")
        dataloader = dataloaders[dist.get_rank()]
        # set up engine
        evaluator = SupervisedEvaluator(
            device=device,
            val_data_loader=dataloader,
            epoch_length=len(dataloader),
            network=TestNet(),
            non_blocking=False,
            prepare_batch=PrepareBatchDefault(),
            decollate=False,
        )
        evaluator.run()
        output = evaluator.state.output
        if len(dataloader) > 0:
            assert_allclose(output["image"], dataloader[-1]["image"].to(device=device))
            assert_allclose(output["label"], dataloader[-1]["label"].to(device=device))


if __name__ == "__main__":
    unittest.main()
