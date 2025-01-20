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

from monai.engines import PrepareBatchExtraInput, SupervisedEvaluator
from tests.util import assert_allclose

TEST_CASE_0 = [
    {"extra_keys": "extra1"},
    {"x": torch.tensor([1, 2]), "t1": torch.tensor([5, 6]), "t2": None, "t3": None},
]

TEST_CASE_1 = [
    {"extra_keys": ["extra1", "extra3"]},
    {"x": torch.tensor([1, 2]), "t1": torch.tensor([5, 6]), "t2": "test", "t3": None},
]

TEST_CASE_2 = [
    {"extra_keys": {"t1": "extra2", "t2": "extra3", "t3": "extra1"}},
    {"x": torch.tensor([1, 2]), "t1": 16, "t2": "test", "t3": torch.tensor([5, 6])},
]


class TestNet(torch.nn.Module):
    __test__ = False  # indicate to pytest that this class is not intended for collection

    def forward(self, x: torch.Tensor, t1=None, t2=None, t3=None):
        return {"x": x, "t1": t1, "t2": t2, "t3": t3}


class TestPrepareBatchExtraInput(unittest.TestCase):
    @parameterized.expand([TEST_CASE_0, TEST_CASE_1, TEST_CASE_2])
    def test_content(self, input_args, expected_value):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataloader = [
            {
                "image": torch.tensor([1, 2]),
                "label": torch.tensor([3, 4]),
                "extra1": torch.tensor([5, 6]),
                "extra2": 16,
                "extra3": "test",
            }
        ]
        # set up engine
        evaluator = SupervisedEvaluator(
            device=device,
            val_data_loader=dataloader,
            epoch_length=1,
            network=TestNet(),
            non_blocking=True,
            prepare_batch=PrepareBatchExtraInput(**input_args),
            decollate=False,
        )
        evaluator.run()
        output = evaluator.state.output
        assert_allclose(output["image"], torch.tensor([1, 2], device=device))
        assert_allclose(output["label"], torch.tensor([3, 4], device=device))
        for k, v in output["pred"].items():
            if isinstance(v, torch.Tensor):
                assert_allclose(v, expected_value[k].to(device))
            else:
                self.assertEqual(v, expected_value[k])


if __name__ == "__main__":
    unittest.main()
