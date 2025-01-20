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
from parameterized import parameterized

from monai.engines import PrepareBatchDefault, SupervisedEvaluator
from tests.util import assert_allclose


class TestNet(torch.nn.Module):
    __test__ = False  # indicate to pytest that this class is not intended for collection

    def forward(self, x: torch.Tensor):
        return x


class TestPrepareBatchDefault(unittest.TestCase):
    @parameterized.expand(
        [
            (
                [
                    {
                        "image": torch.tensor([1, 2]),
                        "label": torch.tensor([3, 4]),
                        "extra1": torch.tensor([5, 6]),
                        "extra2": 16,
                        "extra3": "test",
                    }
                ],
                TestNet(),
                True,
            ),  # dict_content
            ([torch.tensor([1, 2])], torch.nn.Identity(), True),  # tensor_content
            ([(torch.tensor([1, 2]), torch.tensor([3, 4]))], torch.nn.Identity(), True),  # pair_content
            ([], TestNet(), False),  # empty_data
        ]
    )
    def test_prepare_batch(self, dataloader, network, should_run):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        evaluator = SupervisedEvaluator(
            device=device,
            val_data_loader=dataloader,
            epoch_length=len(dataloader) if should_run else 0,
            network=network,
            non_blocking=False,
            prepare_batch=PrepareBatchDefault(),
            decollate=False,
            mode="eval" if should_run else "train",
        )
        evaluator.run()

        if should_run:
            output = evaluator.state.output
            if isinstance(dataloader[0], dict) or isinstance(dataloader[0], tuple):
                assert_allclose(output["image"], torch.tensor([1, 2], device=device))
                assert_allclose(output["label"], torch.tensor([3, 4], device=device))
            else:
                assert_allclose(output["image"], torch.tensor([1, 2], device=device))
                self.assertTrue(output["label"] is None)


if __name__ == "__main__":
    unittest.main()
