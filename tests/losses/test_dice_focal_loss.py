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

from monai.losses import DiceFocalLoss, DiceLoss, FocalLoss
from tests.util import test_script_save


class TestDiceFocalLoss(unittest.TestCase):
    def test_result_onehot_target_include_bg(self):
        size = [3, 3, 5, 5]
        label = torch.randint(low=0, high=2, size=size)
        pred = torch.randn(size)
        for reduction in ["sum", "mean", "none"]:
            for weight in [None, torch.tensor([1.0, 1.0, 2.0]), (3, 2.0, 1)]:
                common_params = {
                    "include_background": True,
                    "to_onehot_y": False,
                    "reduction": reduction,
                    "weight": weight,
                }
                for lambda_focal in [0.5, 1.0, 1.5]:
                    dice_focal = DiceFocalLoss(gamma=1.0, lambda_focal=lambda_focal, **common_params)
                    dice = DiceLoss(**common_params)
                    focal = FocalLoss(gamma=1.0, **common_params)
                    result = dice_focal(pred, label)
                    expected_val = dice(pred, label) + lambda_focal * focal(pred, label)
                    np.testing.assert_allclose(result, expected_val)

    @parameterized.expand([[[3, 3, 5, 5], True], [[3, 2, 5, 5], False]])
    def test_result_no_onehot_no_bg(self, size, onehot):
        label = torch.randint(low=0, high=size[1] - 1, size=size)
        if onehot:
            label = torch.argmax(label, dim=1, keepdim=True)
        pred = torch.randn(size)
        for reduction in ["sum", "mean", "none"]:
            for weight in [2.0] + [] if size[1] != 3 else [torch.tensor([1.0, 2.0]), (2.0, 1)]:
                for lambda_focal in [0.5, 1.0, 1.5]:
                    common_params = {
                        "include_background": False,
                        "softmax": True,
                        "to_onehot_y": onehot,
                        "reduction": reduction,
                        "weight": weight,
                    }
                    dice_focal = DiceFocalLoss(lambda_focal=lambda_focal, **common_params)
                    dice = DiceLoss(**common_params)
                    common_params.pop("softmax", None)
                    focal = FocalLoss(**common_params)
                    result = dice_focal(pred, label)
                    expected_val = dice(pred, label) + lambda_focal * focal(pred, label)
                    np.testing.assert_allclose(result, expected_val)

    def test_ill_shape(self):
        loss = DiceFocalLoss()
        with self.assertRaises(AssertionError):
            loss.forward(torch.ones((1, 2, 3)), torch.ones((1, 2, 5)))

    def test_ill_shape2(self):
        loss = DiceFocalLoss()
        with self.assertRaises(ValueError):
            loss.forward(torch.ones((1, 2, 3)), torch.ones((1, 1, 2, 3)))

    def test_ill_shape3(self):
        loss = DiceFocalLoss()
        with self.assertRaises(ValueError):
            loss.forward(torch.ones((1, 3, 4, 4)), torch.ones((1, 2, 4, 4)))

    def test_ill_lambda(self):
        with self.assertRaisesRegex(ValueError, ""):
            DiceFocalLoss(lambda_dice=-1.0)

    def test_script(self):
        loss = DiceFocalLoss()
        test_input = torch.ones(2, 1, 8, 8)
        test_script_save(loss, test_input, test_input)

    @parameterized.expand(
        [
            ("sum_None_0.5_0.25", "sum", None, 0.5, 0.25),
            ("sum_weight_0.5_0.25", "sum", torch.tensor([1.0, 1.0, 2.0]), 0.5, 0.25),
            ("sum_weight_tuple_0.5_0.25", "sum", (3, 2.0, 1), 0.5, 0.25),
            ("mean_None_0.5_0.25", "mean", None, 0.5, 0.25),
            ("mean_weight_0.5_0.25", "mean", torch.tensor([1.0, 1.0, 2.0]), 0.5, 0.25),
            ("mean_weight_tuple_0.5_0.25", "mean", (3, 2.0, 1), 0.5, 0.25),
            ("none_None_0.5_0.25", "none", None, 0.5, 0.25),
            ("none_weight_0.5_0.25", "none", torch.tensor([1.0, 1.0, 2.0]), 0.5, 0.25),
            ("none_weight_tuple_0.5_0.25", "none", (3, 2.0, 1), 0.5, 0.25),
        ]
    )
    def test_with_alpha(self, name, reduction, weight, lambda_focal, alpha):
        size = [3, 3, 5, 5]
        label = torch.randint(low=0, high=2, size=size)
        pred = torch.randn(size)

        common_params = {"include_background": True, "to_onehot_y": False, "reduction": reduction, "weight": weight}

        dice_focal = DiceFocalLoss(gamma=1.0, lambda_focal=lambda_focal, alpha=alpha, **common_params)
        dice = DiceLoss(**common_params)
        focal = FocalLoss(gamma=1.0, alpha=alpha, **common_params)

        result = dice_focal(pred, label)
        expected_val = dice(pred, label) + lambda_focal * focal(pred, label)

        np.testing.assert_allclose(result, expected_val, err_msg=f"Failed on case: {name}")


if __name__ == "__main__":
    unittest.main()
