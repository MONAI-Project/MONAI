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

from monai.losses import ContrastiveLoss

TEST_CASES = [
    [  # shape: (1, 4), (1, 4)
        {"temperature": 0.5},
        {"input": torch.tensor([[1.0, 1.0, 0.0, 0.0]]), "target": torch.tensor([[1.0, 1.0, 0.0, 0.0]])},
        0.0,
    ],
    [  # shape: (2, 4), (2, 4)
        {"temperature": 0.5},
        {
            "input": torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]]),
            "target": torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]]),
        },
        1.0986,
    ],
    [  # shape: (1, 4), (1, 4)
        {"temperature": 0.5},
        {
            "input": torch.tensor([[1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 0.0, 0.0]]),
            "target": torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]]),
        },
        0.8719,
    ],
    [  # shape: (1, 4), (1, 4)
        {"temperature": 0.5},
        {"input": torch.tensor([[0.0, 0.0, 1.0, 1.0]]), "target": torch.tensor([[1.0, 1.0, 0.0, 0.0]])},
        0.0,
    ],
    [  # shape: (1, 4), (1, 4)
        {"temperature": 0.05},
        {"input": torch.tensor([[0.0, 0.0, 1.0, 1.0]]), "target": torch.tensor([[1.0, 1.0, 0.0, 0.0]])},
        0.0,
    ],
]


class TestContrastiveLoss(unittest.TestCase):

    @parameterized.expand(TEST_CASES)
    def test_result(self, input_param, input_data, expected_val):
        contrastiveloss = ContrastiveLoss(**input_param)
        result = contrastiveloss(**input_data)
        np.testing.assert_allclose(result.detach().cpu().numpy(), expected_val, atol=1e-4, rtol=1e-4)

    def test_ill_shape(self):
        loss = ContrastiveLoss(temperature=0.5)
        with self.assertRaisesRegex(ValueError, ""):
            loss(torch.ones((1, 2, 3)), torch.ones((1, 1, 2, 3)))

    def test_with_cuda(self):
        loss = ContrastiveLoss(temperature=0.5)
        i = torch.ones((1, 10))
        j = torch.ones((1, 10))
        if torch.cuda.is_available():
            i = i.cuda()
            j = j.cuda()
        output = loss(i, j)
        np.testing.assert_allclose(output.detach().cpu().numpy(), 0.0, atol=1e-4, rtol=1e-4)

    def check_warning_rasied(self):
        with self.assertWarns(Warning):
            ContrastiveLoss(temperature=0.5, batch_size=1)


if __name__ == "__main__":
    unittest.main()
