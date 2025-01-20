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

from monai.losses import SURELoss


class TestSURELoss(unittest.TestCase):

    def test_real_value(self):
        """Test SURELoss with real-valued input: when the input is real value, the loss should be 0.0."""
        sure_loss_real = SURELoss(perturb_noise=torch.zeros(2, 1, 128, 128), eps=0.1)

        def operator(x):
            return x

        y_pseudo_gt = torch.randn(2, 1, 128, 128)
        x = torch.randn(2, 1, 128, 128)
        loss = sure_loss_real(operator, x, y_pseudo_gt, complex_input=False)
        self.assertAlmostEqual(loss.item(), 0.0)

    def test_complex_value(self):
        """Test SURELoss with complex-valued input: when the input is complex value, the loss should be 0.0."""

        def operator(x):
            return x

        sure_loss_complex = SURELoss(perturb_noise=torch.zeros(2, 2, 128, 128), eps=0.1)
        y_pseudo_gt = torch.randn(2, 2, 128, 128)
        x = torch.randn(2, 2, 128, 128)
        loss = sure_loss_complex(operator, x, y_pseudo_gt, complex_input=True)
        self.assertAlmostEqual(loss.item(), 0.0)

    def test_complex_general_input(self):
        """Test SURELoss with complex-valued input: when the input is general complex value, the loss should be 0.0."""

        def operator(x):
            return x

        perturb_noise_real = torch.randn(2, 1, 128, 128)
        perturb_noise_complex = torch.zeros(2, 2, 128, 128)
        perturb_noise_complex[:, 0, :, :] = perturb_noise_real.squeeze()
        y_pseudo_gt_real = torch.randn(2, 1, 128, 128)
        y_pseudo_gt_complex = torch.zeros(2, 2, 128, 128)
        y_pseudo_gt_complex[:, 0, :, :] = y_pseudo_gt_real.squeeze()
        x_real = torch.randn(2, 1, 128, 128)
        x_complex = torch.zeros(2, 2, 128, 128)
        x_complex[:, 0, :, :] = x_real.squeeze()

        sure_loss_real = SURELoss(perturb_noise=perturb_noise_real, eps=0.1)
        sure_loss_complex = SURELoss(perturb_noise=perturb_noise_complex, eps=0.1)

        loss_real = sure_loss_real(operator, x_real, y_pseudo_gt_real, complex_input=False)
        loss_complex = sure_loss_complex(operator, x_complex, y_pseudo_gt_complex, complex_input=True)
        self.assertAlmostEqual(loss_real.item(), loss_complex.abs().item(), places=5)


if __name__ == "__main__":
    unittest.main()
