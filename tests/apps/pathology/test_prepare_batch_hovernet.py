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

from monai.apps.pathology.engines import PrepareBatchHoVerNet
from monai.engines import SupervisedEvaluator
from monai.utils.enums import HoVerNetBranch
from tests.util import assert_allclose

TEST_CASE_0 = [
    {"extra_keys": ["extra_label1", "extra_label2"]},
    {HoVerNetBranch.NP: torch.tensor([1, 2]), HoVerNetBranch.NC: torch.tensor([4, 4]), HoVerNetBranch.HV: 16},
]


class TestNet(torch.nn.Module):
    __test__ = False  # indicate to pytest that this class is not intended for collection

    def forward(self, x: torch.Tensor):
        return {HoVerNetBranch.NP: torch.tensor([1, 2]), HoVerNetBranch.NC: torch.tensor([4, 4]), HoVerNetBranch.HV: 16}


class TestPrepareBatchHoVerNet(unittest.TestCase):
    @parameterized.expand([TEST_CASE_0])
    def test_content(self, input_args, expected_value):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataloader = [
            {
                "image": torch.tensor([1, 2]),
                "label": torch.tensor([1, 2]),
                "extra_label1": torch.tensor([3, 4]),
                "extra_label2": 16,
            }
        ]
        # set up engine
        evaluator = SupervisedEvaluator(
            device=device,
            val_data_loader=dataloader,
            epoch_length=1,
            network=TestNet(),
            non_blocking=True,
            prepare_batch=PrepareBatchHoVerNet(**input_args),
            decollate=False,
        )
        evaluator.run()
        output = evaluator.state.output
        assert_allclose(output["image"], torch.tensor([1, 2], device=device))
        for k, v in output["pred"].items():
            if isinstance(v, torch.Tensor):
                assert_allclose(v, expected_value[k].to(device))
            else:
                self.assertEqual(v, expected_value[k])


if __name__ == "__main__":
    unittest.main()
