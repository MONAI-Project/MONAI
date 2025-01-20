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

from monai.losses import FocalLoss, GeneralizedDiceFocalLoss, GeneralizedDiceLoss
from tests.util import test_script_save


class TestGeneralizedDiceFocalLoss(unittest.TestCase):
    def test_result_onehot_target_include_bg(self):
        size = [3, 3, 5, 5]
        label = torch.randint(low=0, high=2, size=size)
        pred = torch.randn(size)
        for reduction in ["sum", "mean", "none"]:
            common_params = {"include_background": True, "to_onehot_y": False, "reduction": reduction}
            for weight in [None, torch.tensor([1.0, 1.0, 2.0]), (3, 2.0, 1)]:
                for lambda_focal in [0.5, 1.0, 1.5]:
                    generalized_dice_focal = GeneralizedDiceFocalLoss(
                        weight=weight, gamma=1.0, lambda_focal=lambda_focal, **common_params
                    )
                    generalized_dice = GeneralizedDiceLoss(**common_params)
                    focal = FocalLoss(weight=weight, gamma=1.0, **common_params)
                    result = generalized_dice_focal(pred, label)
                    expected_val = generalized_dice(pred, label) + lambda_focal * focal(pred, label)
                    np.testing.assert_allclose(result, expected_val)

    def test_result_no_onehot_no_bg(self):
        size = [3, 3, 5, 5]
        label = torch.randint(low=0, high=2, size=size)
        label = torch.argmax(label, dim=1, keepdim=True)
        pred = torch.randn(size)
        for reduction in ["sum", "mean", "none"]:
            common_params = {"include_background": False, "to_onehot_y": True, "reduction": reduction}
            for weight in [2.0, torch.tensor([1.0, 2.0]), (2.0, 1)]:
                for lambda_focal in [0.5, 1.0, 1.5]:
                    generalized_dice_focal = GeneralizedDiceFocalLoss(
                        weight=weight, lambda_focal=lambda_focal, **common_params
                    )
                    generalized_dice = GeneralizedDiceLoss(**common_params)
                    focal = FocalLoss(weight=weight, **common_params)
                    result = generalized_dice_focal(pred, label)
                    expected_val = generalized_dice(pred, label) + lambda_focal * focal(pred, label)
                    np.testing.assert_allclose(result, expected_val)

    def test_ill_shape(self):
        loss = GeneralizedDiceFocalLoss()
        with self.assertRaises(AssertionError):
            loss.forward(torch.ones((1, 2, 3)), torch.ones((1, 2, 5)))

    def test_ill_shape2(self):
        loss = GeneralizedDiceFocalLoss()
        with self.assertRaises(ValueError):
            loss.forward(torch.ones((1, 2, 3)), torch.ones((1, 1, 2, 3)))

    def test_ill_shape3(self):
        loss = GeneralizedDiceFocalLoss()
        with self.assertRaises(ValueError):
            loss.forward(torch.ones((1, 3, 4, 4)), torch.ones((1, 2, 4, 4)))

    def test_ill_lambda(self):
        with self.assertRaisesRegex(ValueError, ""):
            GeneralizedDiceFocalLoss(lambda_gdl=-1.0)

    def test_script(self):
        loss = GeneralizedDiceFocalLoss()
        test_input = torch.ones(2, 1, 8, 8)
        test_script_save(loss, test_input, test_input)


if __name__ == "__main__":
    unittest.main()
