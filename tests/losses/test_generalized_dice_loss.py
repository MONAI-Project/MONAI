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

from monai.losses import GeneralizedDiceLoss
from tests.util import test_script_save

TEST_CASES = [
    [  # shape: (1, 1, 2, 2), (1, 1, 2, 2)
        {"include_background": True, "sigmoid": True, "smooth_nr": 1e-6, "smooth_dr": 1e-6},
        {"input": torch.tensor([[[[1.0, -1.0], [-1.0, 1.0]]]]), "target": torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]])},
        0.307576,
    ],
    [  # shape: (2, 1, 2, 2), (2, 1, 2, 2)
        {"include_background": True, "sigmoid": True, "smooth_nr": 1e-4, "smooth_dr": 1e-4},
        {
            "input": torch.tensor([[[[1.0, -1.0], [-1.0, 1.0]]], [[[1.0, -1.0], [-1.0, 1.0]]]]),
            "target": torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]], [[[1.0, 0.0], [1.0, 0.0]]]]),
        },
        0.416597,
    ],
    [  # shape: (2, 1, 2, 2), (2, 1, 2, 2)
        {"include_background": True, "smooth_nr": 1e-4, "smooth_dr": 1e-4, "soft_label": True},
        {
            "input": torch.tensor([[[[0.3, 0.4], [0.7, 0.9]]], [[[1.0, 0.1], [0.5, 0.3]]]]),
            "target": torch.tensor([[[[0.3, 0.4], [0.7, 0.9]]], [[[1.0, 0.1], [0.5, 0.3]]]]),
        },
        0.0,
    ],
    [  # shape: (2, 1, 2, 2), (2, 1, 2, 2)
        {"include_background": True, "smooth_nr": 1e-4, "smooth_dr": 1e-4, "soft_label": False},
        {
            "input": torch.tensor([[[[0.3, 0.4], [0.7, 0.9]]], [[[1.0, 0.1], [0.5, 0.3]]]]),
            "target": torch.tensor([[[[0.3, 0.4], [0.7, 0.9]]], [[[1.0, 0.1], [0.5, 0.3]]]]),
        },
        0.307748,
    ],
    [  # shape: (2, 2, 3), (2, 1, 3)
        {"include_background": False, "to_onehot_y": True, "smooth_nr": 0.0, "smooth_dr": 0.0},
        {
            "input": torch.tensor([[[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]]),
            "target": torch.tensor([[[0.0, 0.0, 1.0]], [[0.0, 1.0, 0.0]]]),
        },
        0.0,
    ],
    [  # shape: (2, 2, 3), (2, 1, 3)
        {"include_background": True, "to_onehot_y": True, "sigmoid": True, "smooth_nr": 1e-4, "smooth_dr": 1e-4},
        {
            "input": torch.tensor([[[-1.0, 0.0, 1.0], [1.0, 0.0, -1.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
            "target": torch.tensor([[[1.0, 0.0, 0.0]], [[1.0, 1.0, 0.0]]]),
        },
        0.469964,
    ],
    [  # shape: (2, 2, 3), (2, 1, 3)
        {"include_background": True, "to_onehot_y": True, "softmax": True, "smooth_nr": 1e-4, "smooth_dr": 1e-4},
        {
            "input": torch.tensor([[[-1.0, 0.0, 1.0], [1.0, 0.0, -1.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
            "target": torch.tensor([[[1.0, 0.0, 0.0]], [[1.0, 1.0, 0.0]]]),
        },
        0.414507,
    ],
    [  # shape: (2, 2, 3), (2, 1, 3)
        {
            "include_background": True,
            "to_onehot_y": True,
            "softmax": True,
            "reduction": "sum",
            "smooth_nr": 1e-4,
            "smooth_dr": 1e-4,
        },
        {
            "input": torch.tensor([[[-1.0, 0.0, 1.0], [1.0, 0.0, -1.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
            "target": torch.tensor([[[1.0, 0.0, 0.0]], [[1.0, 1.0, 0.0]]]),
        },
        0.829015,
    ],
    [  # shape: (2, 2, 3), (2, 1, 3)
        {
            "include_background": True,
            "to_onehot_y": True,
            "softmax": True,
            "reduction": "none",
            "smooth_nr": 1e-4,
            "smooth_dr": 1e-4,
        },
        {
            "input": torch.tensor([[[-1.0, 0.0, 1.0], [1.0, 0.0, -1.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
            "target": torch.tensor([[[1.0, 0.0, 0.0]], [[1.0, 1.0, 0.0]]]),
        },
        [[[0.273476]], [[0.555539]]],
    ],
    [  # shape: (2, 2, 3), (2, 1, 3)
        {"include_background": False, "to_onehot_y": True, "smooth_nr": 1e-8, "smooth_dr": 1e-8},
        {
            "input": torch.tensor([[[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]]),
            "target": torch.tensor([[[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]]),
        },
        0.0,
    ],
    [  # shape: (1, 1, 2, 2), (1, 1, 2, 2)
        {"include_background": True, "sigmoid": True, "smooth_nr": 1e-6, "smooth_dr": 1e-6},
        {"input": torch.tensor([[[[1.0, -1.0], [-1.0, 1.0]]]]), "target": torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]])},
        0.307576,
    ],
    [  # shape: (1, 2, 4), (1, 1, 4)
        {
            "include_background": True,
            "to_onehot_y": True,
            "softmax": True,
            "w_type": "simple",
            "smooth_nr": 0,
            "smooth_dr": 0,
        },
        {
            "input": torch.tensor([[[0.0, 10.0, 10.0, 10.0], [10.0, 0.0, 0.0, 0.0]]]),
            "target": torch.tensor([[[1, 1, 0, 0]]]),
        },
        0.250023,
    ],
    [  # shape: (2, 1, 2, 2), (2, 1, 2, 2)
        {"include_background": True, "other_act": torch.tanh, "smooth_nr": 1e-4, "smooth_dr": 1e-4},
        {
            "input": torch.tensor([[[[1.0, -1.0], [-1.0, 1.0]]], [[[1.0, -1.0], [-1.0, 1.0]]]]),
            "target": torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]], [[[1.0, 0.0], [1.0, 0.0]]]]),
        },
        0.99970,
    ],
    [  # shape: (2, 2, 3), (2, 1, 3)
        {
            "include_background": True,
            "to_onehot_y": True,
            "other_act": lambda x: torch.log_softmax(x, dim=1),
            "smooth_nr": 1e-4,
            "smooth_dr": 1e-4,
        },
        {
            "input": torch.tensor([[[-1.0, 0.0, 1.0], [1.0, 0.0, -1.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
            "target": torch.tensor([[[1.0, 0.0, 0.0]], [[1.0, 1.0, 0.0]]]),
        },
        -0.097833,
    ],
]


class TestGeneralizedDiceLoss(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_shape(self, input_param, input_data, expected_val):
        result = GeneralizedDiceLoss(**input_param).forward(**input_data)
        np.testing.assert_allclose(result.detach().cpu().numpy(), expected_val, rtol=1e-5)

    def test_ill_shape(self):
        loss = GeneralizedDiceLoss()
        with self.assertRaisesRegex(AssertionError, ""):
            loss.forward(torch.ones((1, 2, 3)), torch.ones((4, 5, 6)))

    def test_ill_opts(self):
        with self.assertRaisesRegex(ValueError, ""):
            GeneralizedDiceLoss(sigmoid=True, softmax=True)
        chn_input = torch.ones((1, 1, 3))
        chn_target = torch.ones((1, 1, 3))
        with self.assertRaisesRegex(ValueError, ""):
            GeneralizedDiceLoss(reduction="unknown")(chn_input, chn_target)
        with self.assertRaisesRegex(ValueError, ""):
            GeneralizedDiceLoss(reduction=None)(chn_input, chn_target)

    def test_input_warnings(self):
        chn_input = torch.ones((1, 1, 3))
        chn_target = torch.ones((1, 1, 3))
        with self.assertWarns(Warning):
            loss = GeneralizedDiceLoss(include_background=False)
            loss.forward(chn_input, chn_target)
        with self.assertWarns(Warning):
            loss = GeneralizedDiceLoss(softmax=True)
            loss.forward(chn_input, chn_target)
        with self.assertWarns(Warning):
            loss = GeneralizedDiceLoss(to_onehot_y=True)
            loss.forward(chn_input, chn_target)

    def test_differentiability(self):
        prediction = torch.ones((1, 1, 1, 3))
        target = torch.ones((1, 1, 1, 3))
        prediction.requires_grad = True
        target.requires_grad = True

        generalized_dice_loss = GeneralizedDiceLoss()
        loss = generalized_dice_loss(prediction, target)
        self.assertIsNotNone(loss.grad_fn)

    def test_batch(self):
        prediction = torch.zeros(2, 3, 3, 3)
        target = torch.zeros(2, 3, 3, 3)
        prediction.requires_grad = True
        target.requires_grad = True

        generalized_dice_loss = GeneralizedDiceLoss(batch=True)
        loss = generalized_dice_loss(prediction, target)
        self.assertIsNotNone(loss.grad_fn)

    def test_script(self):
        loss = GeneralizedDiceLoss()
        test_input = torch.ones(2, 1, 8, 8)
        test_script_save(loss, test_input, test_input)


if __name__ == "__main__":
    unittest.main()
