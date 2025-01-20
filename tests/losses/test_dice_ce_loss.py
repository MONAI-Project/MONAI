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

from monai.losses import DiceCELoss

TEST_CASES = [
    [  # shape: (2, 2, 3), (2, 1, 3)
        {"to_onehot_y": True},
        {
            "input": torch.tensor([[[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]]),
            "target": torch.tensor([[[0.0, 0.0, 1.0]], [[0.0, 1.0, 0.0]]]),
        },
        0.3133,  # the result equals to -1 + np.log(1 + np.exp(1))
    ],
    [  # shape: (2, 2, 3), (2, 2, 3), one-hot target
        {"to_onehot_y": False},
        {
            "input": torch.tensor([[[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]]),
            "target": torch.tensor([[[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]]),
        },
        0.3133,
    ],
    [  # shape: (2, 2, 3), (2, 2, 3), one-hot target
        {"to_onehot_y": False},
        {
            "input": torch.tensor([[[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]]),
            "target": torch.tensor([[[1, 1, 0], [0, 0, 1]], [[1, 0, 1], [0, 1, 0]]], dtype=torch.uint8),
        },
        0.3133,
    ],
    [  # shape: (2, 2, 3), (2, 1, 3)
        {"include_background": False, "to_onehot_y": True, "weight": torch.tensor([1.0, 1.0])},
        {
            "input": torch.tensor([[[100.0, 100.0, 0.0], [0.0, 0.0, 1.0]], [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]]),
            "target": torch.tensor([[[0.0, 0.0, 1.0]], [[0.0, 1.0, 0.0]]]),
        },
        0.2088,
    ],
    [  # shape: (2, 2, 3), (2, 1, 3) lambda_dice: 1.0, lambda_ce: 2.0
        {
            "include_background": False,
            "to_onehot_y": True,
            "weight": torch.tensor([1.0, 1.0]),
            "lambda_dice": 1.0,
            "lambda_ce": 2.0,
        },
        {
            "input": torch.tensor([[[100.0, 100.0, 0.0], [0.0, 0.0, 1.0]], [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]]),
            "target": torch.tensor([[[0.0, 0.0, 1.0]], [[0.0, 1.0, 0.0]]]),
        },
        0.4176,
    ],
    [  # shape: (2, 2, 3), (2, 1, 3), do not include class 0
        {"include_background": False, "to_onehot_y": True, "weight": torch.tensor([0.0, 1.0])},
        {
            "input": torch.tensor([[[100.0, 100.0, 0.0], [0.0, 0.0, 1.0]], [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]]),
            "target": torch.tensor([[[0.0, 0.0, 1.0]], [[0.0, 1.0, 0.0]]]),
        },
        0.3133,
    ],
    [  # shape: (2, 1, 3), (2, 1, 3), bceloss
        {"weight": torch.tensor([0.5]), "sigmoid": True},
        {
            "input": torch.tensor([[[0.8, 0.6, 0.0]], [[0.0, 0.0, 0.9]]]),
            "target": torch.tensor([[[0.0, 0.0, 1.0]], [[0.0, 1.0, 0.0]]]),
        },
        1.445239,
    ],
]


class TestDiceCELoss(unittest.TestCase):

    @parameterized.expand(TEST_CASES)
    def test_result(self, input_param, input_data, expected_val):
        diceceloss = DiceCELoss(**input_param)
        result = diceceloss(**input_data)
        np.testing.assert_allclose(result.detach().cpu().numpy(), expected_val, atol=1e-4, rtol=1e-4)

    def test_ill_shape(self):
        loss = DiceCELoss()
        with self.assertRaises(AssertionError):
            loss.forward(torch.ones((1, 2, 3)), torch.ones((1, 2, 5)))

    def test_ill_shape2(self):
        loss = DiceCELoss()
        with self.assertRaises(ValueError):
            loss.forward(torch.ones((1, 2, 3)), torch.ones((1, 1, 2, 3)))

    def test_ill_shape3(self):
        loss = DiceCELoss()
        with self.assertRaises(ValueError):
            loss.forward(torch.ones((1, 3, 4, 4)), torch.ones((1, 2, 4, 4)))

    # def test_ill_reduction(self):
    #     with self.assertRaisesRegex(ValueError, ""):
    #         loss = DiceCELoss(reduction="none")
    #         loss(torch.ones((1, 2, 3)), torch.ones((1, 1, 2, 3)))

    # def test_script(self):
    #     loss = DiceCELoss()
    #     test_input = torch.ones(2, 2, 8, 8)
    #     test_script_save(loss, test_input, test_input)


if __name__ == "__main__":
    unittest.main()
