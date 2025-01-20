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

from monai.losses.ssim_loss import SSIMLoss
from monai.utils import set_determinism

# from tests.utils import test_script_save


class TestSSIMLoss(unittest.TestCase):

    def test_shape(self):
        set_determinism(0)
        preds = torch.abs(torch.randn(2, 3, 16, 16))
        target = torch.abs(torch.randn(2, 3, 16, 16))
        preds = preds / preds.max()
        target = target / target.max()

        result = SSIMLoss(spatial_dims=2, data_range=1.0, kernel_type="gaussian", reduction="mean").forward(
            preds, target
        )
        expected_val = 0.9546
        np.testing.assert_allclose(result.detach().cpu().numpy(), expected_val, rtol=1e-4)

        result = SSIMLoss(spatial_dims=2, data_range=1.0, kernel_type="gaussian", reduction="sum").forward(
            preds, target
        )
        expected_val = 1.9092
        np.testing.assert_allclose(result.detach().cpu().numpy(), expected_val, rtol=1e-4)

        result = SSIMLoss(spatial_dims=2, data_range=1.0, kernel_type="gaussian", reduction="none").forward(
            preds, target
        )
        expected_val = [[0.9121], [0.9971]]
        np.testing.assert_allclose(result.detach().cpu().numpy(), expected_val, rtol=1e-4)

    # def test_script(self):
    #     loss = SSIMLoss(spatial_dims=2)
    #     test_input = torch.ones(2, 2, 16, 16)
    #     test_script_save(loss, test_input, test_input)


if __name__ == "__main__":
    unittest.main()
