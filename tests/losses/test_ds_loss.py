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

from monai.losses import DeepSupervisionLoss, DiceCELoss, DiceFocalLoss, DiceLoss
from tests.util import SkipIfBeforePyTorchVersion, test_script_save

TEST_CASES_DICECE = [
    [
        {"to_onehot_y": True},
        {},
        {
            "input": torch.tensor([[[[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]]]),
            "target": torch.tensor([[[[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]]]),
        },
        0.606557,
    ]
]

TEST_CASES_DICECE2 = [
    [
        {"to_onehot_y": True},
        {},
        {
            "input": [
                torch.tensor([[[[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]]]),
                torch.tensor([[[[1.0, 1.0], [0.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]]]]),
                torch.tensor([[[[1.0], [0.0]], [[1.0], [0.0]]]]),
            ],
            "target": torch.tensor([[[[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]]]),
        },
        1.78144,
    ],
    [
        {"to_onehot_y": True},
        {"weight_mode": "same"},
        {
            "input": [
                torch.tensor([[[[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]]]),
                torch.tensor([[[[1.0, 1.0], [0.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]]]]),
                torch.tensor([[[[1.0], [0.0]], [[1.0], [0.0]]]]),
            ],
            "target": torch.tensor([[[[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]]]),
        },
        3.5529,
    ],
    [
        {"to_onehot_y": True},
        {"weight_mode": "two"},
        {
            "input": [
                torch.tensor([[[[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]]]),
                torch.tensor([[[[1.0, 1.0], [0.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]]]]),
                torch.tensor([[[[1.0], [0.0]], [[1.0], [0.0]]]]),
            ],
            "target": torch.tensor([[[[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]]]),
        },
        2.07973,
    ],
    [
        {"to_onehot_y": True},
        {"weights": [0.1, 0.2, 0.3]},
        {
            "input": [
                torch.tensor([[[[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]]]),
                torch.tensor([[[[1.0, 1.0], [0.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]]]]),
                torch.tensor([[[[1.0], [0.0]], [[1.0], [0.0]]]]),
            ],
            "target": torch.tensor([[[[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]]]),
        },
        0.76924,
    ],
]

TEST_CASES_DICE = [
    [
        {"to_onehot_y": True},
        {
            "input": torch.tensor([[[[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]]]),
            "target": torch.tensor([[[[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]]]),
        },
        0.166666,  # the result equals to -1 + np.log(1 + np.exp(1))
    ],
    [
        {"to_onehot_y": True},
        {
            "input": [
                torch.tensor([[[[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]]]),
                torch.tensor([[[[1.0, 1.0], [0.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]]]]),
                torch.tensor([[[[1.0], [0.0]], [[1.0], [0.0]]]]),
            ],
            "target": torch.tensor([[[[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]]]),
        },
        0.666665,
    ],
]

TEST_CASES_DICEFOCAL = [
    [
        {"to_onehot_y": True},
        {
            "input": torch.tensor([[[[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]]]),
            "target": torch.tensor([[[[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]]]),
        },
        0.32124,  # the result equals to -1 + np.log(1 + np.exp(1))
    ],
    [
        {"to_onehot_y": True},
        {
            "input": [
                torch.tensor([[[[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]]]),
                torch.tensor([[[[1.0, 1.0], [0.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]]]]),
                torch.tensor([[[[1.0], [0.0]], [[1.0], [0.0]]]]),
            ],
            "target": torch.tensor([[[[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]]]),
        },
        1.06452,
    ],
]


class TestDSLossDiceCE(unittest.TestCase):
    @parameterized.expand(TEST_CASES_DICECE)
    def test_result(self, input_param, input_param2, input_data, expected_val):
        diceceloss = DeepSupervisionLoss(DiceCELoss(**input_param), **input_param2)
        result = diceceloss(**input_data)
        np.testing.assert_allclose(result.detach().cpu().numpy(), expected_val, atol=1e-4, rtol=1e-4)

    def test_ill_shape(self):
        loss = DeepSupervisionLoss(DiceCELoss())
        with self.assertRaisesRegex(ValueError, ""):
            loss(torch.ones((1, 2, 3)), torch.ones((1, 1, 2, 3)))

    def test_ill_reduction(self):
        with self.assertRaisesRegex(ValueError, ""):
            loss = DeepSupervisionLoss(DiceCELoss(reduction="none"))
            loss(torch.ones((1, 2, 3)), torch.ones((1, 1, 2, 3)))

    @SkipIfBeforePyTorchVersion((1, 10))
    def test_script(self):
        loss = DeepSupervisionLoss(DiceCELoss())
        test_input = torch.ones(2, 2, 8, 8)
        test_script_save(loss, test_input, test_input)


@SkipIfBeforePyTorchVersion((1, 11))
class TestDSLossDiceCE2(unittest.TestCase):
    @parameterized.expand(TEST_CASES_DICECE2)
    def test_result(self, input_param, input_param2, input_data, expected_val):
        diceceloss = DeepSupervisionLoss(DiceCELoss(**input_param), **input_param2)
        result = diceceloss(**input_data)
        np.testing.assert_allclose(result.detach().cpu().numpy(), expected_val, atol=1e-4, rtol=1e-4)


@SkipIfBeforePyTorchVersion((1, 11))
class TestDSLossDice(unittest.TestCase):
    @parameterized.expand(TEST_CASES_DICE)
    def test_result(self, input_param, input_data, expected_val):
        loss = DeepSupervisionLoss(DiceLoss(**input_param))
        result = loss(**input_data)
        np.testing.assert_allclose(result.detach().cpu().numpy(), expected_val, atol=1e-4, rtol=1e-4)


@SkipIfBeforePyTorchVersion((1, 11))
class TestDSLossDiceFocal(unittest.TestCase):
    @parameterized.expand(TEST_CASES_DICEFOCAL)
    def test_result(self, input_param, input_data, expected_val):
        loss = DeepSupervisionLoss(DiceFocalLoss(**input_param))
        result = loss(**input_data)
        np.testing.assert_allclose(result.detach().cpu().numpy(), expected_val, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
