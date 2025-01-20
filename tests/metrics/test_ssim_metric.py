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

from monai.metrics.regression import SSIMMetric, compute_ssim_and_cs
from monai.utils import set_determinism


class TestSSIMMetric(unittest.TestCase):

    def test2d_gaussian(self):
        set_determinism(0)
        preds = torch.abs(torch.randn(2, 3, 16, 16))
        target = torch.abs(torch.randn(2, 3, 16, 16))
        preds = preds / preds.max()
        target = target / target.max()

        metric = SSIMMetric(spatial_dims=2, data_range=1.0, kernel_type="gaussian")
        metric(preds, target)
        result = metric.aggregate()
        expected_value = 0.045415
        self.assertTrue(expected_value - result.item() < 0.000001)

    def test2d_uniform(self):
        set_determinism(0)
        preds = torch.abs(torch.randn(2, 3, 16, 16))
        target = torch.abs(torch.randn(2, 3, 16, 16))
        preds = preds / preds.max()
        target = target / target.max()

        metric = SSIMMetric(spatial_dims=2, data_range=1.0, kernel_type="uniform")
        metric(preds, target)
        result = metric.aggregate()
        expected_value = 0.050103
        self.assertTrue(expected_value - result.item() < 0.000001)

    def test3d_gaussian(self):
        set_determinism(0)
        preds = torch.abs(torch.randn(2, 3, 16, 16, 16))
        target = torch.abs(torch.randn(2, 3, 16, 16, 16))
        preds = preds / preds.max()
        target = target / target.max()

        metric = SSIMMetric(spatial_dims=3, data_range=1.0, kernel_type="gaussian")
        metric(preds, target)
        result = metric.aggregate()
        expected_value = 0.017644
        self.assertTrue(expected_value - result.item() < 0.000001)

    def input_ill_input_shape(self):
        with self.assertRaises(ValueError):
            metric = SSIMMetric(spatial_dims=3)
            metric(torch.randn(1, 1, 16, 16), torch.randn(1, 1, 16, 16))

        with self.assertRaises(ValueError):
            metric = SSIMMetric(spatial_dims=2)
            metric(torch.randn(1, 1, 16, 16, 16), torch.randn(1, 1, 16, 16, 16))

    def mismatch_y_pred_and_y(self):
        with self.assertRaises(ValueError):
            compute_ssim_and_cs(y_pred=torch.randn(1, 1, 16, 8), y=torch.randn(1, 1, 16, 16), spatial_dims=2)


if __name__ == "__main__":
    unittest.main()
