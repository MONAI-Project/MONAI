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

from monai.losses import BarlowTwinsLoss

TEST_CASES = [
    [  # shape: (2, 4), (2, 4)
        {"lambd": 5e-3},
        {
            "input": torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]]),
            "target": torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]]),
        },
        4.0,
    ],
    [  # shape: (2, 4), (2, 4)
        {"lambd": 5e-3},
        {
            "input": torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]]),
            "target": torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]]),
        },
        4.0,
    ],
    [  # shape: (2, 4), (2, 4)
        {"lambd": 5e-3},
        {
            "input": torch.tensor([[1.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 0.0]]),
            "target": torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 1.0]]),
        },
        5.2562,
    ],
    [  # shape: (2, 4), (2, 4)
        {"lambd": 5e-4},
        {
            "input": torch.tensor([[2.0, 3.0, 1.0, 2.0], [0.0, 1.0, 2.0, 5.0]]),
            "target": torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]),
        },
        5.0015,
    ],
    [  # shape: (4, 4), (4, 4)
        {"lambd": 5e-3},
        {
            "input": torch.tensor(
                [[1.0, 2.0, 1.0, 1.0], [3.0, 1.0, 1.0, 2.0], [1.0, 1.0, 1.0, 1.0], [2.0, 1.0, 1.0, 0.0]]
            ),
            "target": torch.tensor(
                [
                    [0.0, 1.0, -1.0, 0.0],
                    [1 / 3, 0.0, -2 / 3, 1 / 3],
                    [-2 / 3, -1.0, 7 / 3, 1 / 3],
                    [1 / 3, 0.0, 1 / 3, -2 / 3],
                ]
            ),
        },
        1.4736,
    ],
]


class TestBarlowTwinsLoss(unittest.TestCase):

    @parameterized.expand(TEST_CASES)
    def test_result(self, input_param, input_data, expected_val):
        barlowtwinsloss = BarlowTwinsLoss(**input_param)
        result = barlowtwinsloss(**input_data)
        np.testing.assert_allclose(result.detach().cpu().numpy(), expected_val, atol=1e-4, rtol=1e-4)

    def test_ill_shape(self):
        loss = BarlowTwinsLoss(lambd=5e-3)
        with self.assertRaises(ValueError):
            loss(torch.ones((1, 2, 3)), torch.ones((1, 1, 2, 3)))

    def test_ill_batch_size(self):
        loss = BarlowTwinsLoss(lambd=5e-3)
        with self.assertRaises(ValueError):
            loss(torch.ones((1, 2)), torch.ones((1, 2)))

    def test_with_cuda(self):
        loss = BarlowTwinsLoss(lambd=5e-3)
        i = torch.ones((2, 10))
        j = torch.ones((2, 10))
        if torch.cuda.is_available():
            i = i.cuda()
            j = j.cuda()
        output = loss(i, j)
        np.testing.assert_allclose(output.detach().cpu().numpy(), 10.0, atol=1e-4, rtol=1e-4)

    def check_warning_raised(self):
        with self.assertWarns(Warning):
            BarlowTwinsLoss(lambd=5e-3, batch_size=1)


if __name__ == "__main__":
    unittest.main()
