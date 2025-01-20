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

from monai.losses.image_dissimilarity import LocalNormalizedCrossCorrelationLoss

device = "cuda" if torch.cuda.is_available() else "cpu"

TEST_CASES = [
    [
        {"spatial_dims": 1, "kernel_type": "rectangular", "reduction": "sum"},
        {
            "pred": torch.arange(0, 3).reshape(1, 1, -1).to(dtype=torch.float, device=device),
            "target": torch.arange(0, 3).reshape(1, 1, -1).to(dtype=torch.float, device=device),
        },
        -1.0 * 3,
    ],
    [
        {"spatial_dims": 1, "kernel_type": "rectangular"},
        {
            "pred": torch.arange(0, 3).reshape(1, 1, -1).to(dtype=torch.float, device=device),
            "target": torch.arange(0, 3).reshape(1, 1, -1).to(dtype=torch.float, device=device),
        },
        -1.0,
    ],
    [
        {"spatial_dims": 1, "kernel_type": "triangular", "smooth_dr": 0.1},
        {
            "pred": torch.zeros(1, 2, 3).reshape(1, 1, -1).to(dtype=torch.float, device=device),
            "target": torch.zeros(1, 2, 3).reshape(1, 1, -1).to(dtype=torch.float, device=device),
        },
        0.0,
    ],
    [
        {"spatial_dims": 2, "kernel_type": "rectangular"},
        {
            "pred": torch.arange(0, 3).reshape(1, 1, -1, 1).expand(1, 1, 3, 3).to(dtype=torch.float, device=device),
            "target": torch.arange(0, 3).reshape(1, 1, -1, 1).expand(1, 1, 3, 3).to(dtype=torch.float, device=device),
        },
        -1.0,
    ],
    [
        {"spatial_dims": 3, "kernel_type": "rectangular"},
        {
            "pred": torch.arange(0, 3)
            .reshape(1, 1, -1, 1, 1)
            .expand(1, 1, 3, 3, 3)
            .to(dtype=torch.float, device=device),
            "target": torch.arange(0, 3)
            .reshape(1, 1, -1, 1, 1)
            .expand(1, 1, 3, 3, 3)
            .to(dtype=torch.float, device=device),
        },
        -1.0,
    ],
    [
        {"spatial_dims": 3, "kernel_type": "rectangular"},
        {
            "pred": torch.arange(0, 3)
            .reshape(1, 1, -1, 1, 1)
            .expand(1, 3, 3, 3, 3)
            .to(dtype=torch.float, device=device),
            "target": torch.arange(0, 3)
            .reshape(1, 1, -1, 1, 1)
            .expand(1, 3, 3, 3, 3)
            .to(dtype=torch.float, device=device)
            ** 2,
        },
        -0.95801723,
    ],
    [
        {"spatial_dims": 3, "kernel_type": "triangular", "kernel_size": 5},
        {
            "pred": torch.arange(0, 5)
            .reshape(1, 1, -1, 1, 1)
            .expand(1, 3, 5, 5, 5)
            .to(dtype=torch.float, device=device),
            "target": torch.arange(0, 5)
            .reshape(1, 1, -1, 1, 1)
            .expand(1, 3, 5, 5, 5)
            .to(dtype=torch.float, device=device)
            ** 2,
        },
        -0.918672,
    ],
    [
        {"spatial_dims": 3, "kernel_type": "gaussian"},
        {
            "pred": torch.arange(0, 3)
            .reshape(1, 1, -1, 1, 1)
            .expand(1, 3, 3, 3, 3)
            .to(dtype=torch.float, device=device),
            "target": torch.arange(0, 3)
            .reshape(1, 1, -1, 1, 1)
            .expand(1, 3, 3, 3, 3)
            .to(dtype=torch.float, device=device)
            ** 2,
        },
        -0.95406944,
    ],
]


class TestLocalNormalizedCrossCorrelationLoss(unittest.TestCase):

    @parameterized.expand(TEST_CASES)
    def test_shape(self, input_param, input_data, expected_val):
        result = LocalNormalizedCrossCorrelationLoss(**input_param).forward(**input_data)
        np.testing.assert_allclose(result.detach().cpu().numpy(), expected_val, rtol=1e-5)

    def test_ill_shape(self):
        loss = LocalNormalizedCrossCorrelationLoss(spatial_dims=3)
        # spatial_dims unmatch
        with self.assertRaisesRegex(ValueError, ""):
            loss.forward(
                torch.ones((1, 3, 3, 3), dtype=torch.float, device=device),
                torch.ones((1, 3, 3, 3), dtype=torch.float, device=device),
            )
        # pred, target shape unmatch
        with self.assertRaisesRegex(ValueError, ""):
            loss.forward(
                torch.ones((1, 3, 3, 3, 3), dtype=torch.float, device=device),
                torch.ones((1, 3, 4, 4, 4), dtype=torch.float, device=device),
            )

    def test_ill_opts(self):
        pred = torch.ones((1, 3, 3, 3, 3), dtype=torch.float)
        target = torch.ones((1, 3, 3, 3, 3), dtype=torch.float)
        with self.assertRaisesRegex(ValueError, ""):
            LocalNormalizedCrossCorrelationLoss(kernel_type="unknown")(pred, target)
        with self.assertRaisesRegex(ValueError, ""):
            LocalNormalizedCrossCorrelationLoss(kernel_type=None)(pred, target)
        with self.assertRaisesRegex(ValueError, ""):
            LocalNormalizedCrossCorrelationLoss(kernel_size=4)(pred, target)
        with self.assertRaisesRegex(ValueError, ""):
            LocalNormalizedCrossCorrelationLoss(reduction="unknown")(pred, target)
        with self.assertRaisesRegex(ValueError, ""):
            LocalNormalizedCrossCorrelationLoss(reduction=None)(pred, target)


#     def test_script(self):
#         input_param, input_data, _ = TEST_CASES[0]
#         loss = LocalNormalizedCrossCorrelationLoss(**input_param)
#         test_script_save(loss, input_data["pred"], input_data["target"])

if __name__ == "__main__":
    unittest.main()
