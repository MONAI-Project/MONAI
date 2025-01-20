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

from monai.metrics import MultiScaleSSIMMetric
from monai.utils import set_determinism


class TestMultiScaleSSIMMetric(unittest.TestCase):

    def test2d_gaussian(self):
        set_determinism(0)
        preds = torch.abs(torch.randn(1, 1, 64, 64))
        target = torch.abs(torch.randn(1, 1, 64, 64))
        preds = preds / preds.max()
        target = target / target.max()

        metric = MultiScaleSSIMMetric(spatial_dims=2, data_range=1.0, kernel_type="gaussian", weights=[0.5, 0.5])
        metric(preds, target)
        result = metric.aggregate()
        expected_value = 0.023176
        self.assertTrue(expected_value - result.item() < 0.000001)

    def test2d_uniform(self):
        set_determinism(0)
        preds = torch.abs(torch.randn(1, 1, 64, 64))
        target = torch.abs(torch.randn(1, 1, 64, 64))
        preds = preds / preds.max()
        target = target / target.max()

        metric = MultiScaleSSIMMetric(spatial_dims=2, data_range=1.0, kernel_type="uniform", weights=[0.5, 0.5])
        metric(preds, target)
        result = metric.aggregate()
        expected_value = 0.022655
        self.assertTrue(expected_value - result.item() < 0.000001)

    def test3d_gaussian(self):
        set_determinism(0)
        preds = torch.abs(torch.randn(1, 1, 64, 64, 64))
        target = torch.abs(torch.randn(1, 1, 64, 64, 64))
        preds = preds / preds.max()
        target = target / target.max()

        metric = MultiScaleSSIMMetric(spatial_dims=3, data_range=1.0, kernel_type="gaussian", weights=[0.5, 0.5])
        metric(preds, target)
        result = metric.aggregate()
        expected_value = 0.061796
        self.assertTrue(expected_value - result.item() < 0.000001)

    def input_ill_input_shape2d(self):
        metric = MultiScaleSSIMMetric(spatial_dims=3, weights=[0.5, 0.5])

        with self.assertRaises(ValueError):
            metric(torch.randn(1, 1, 64, 64), torch.randn(1, 1, 64, 64))

    def input_ill_input_shape3d(self):
        metric = MultiScaleSSIMMetric(spatial_dims=2, weights=[0.5, 0.5])

        with self.assertRaises(ValueError):
            metric(torch.randn(1, 1, 64, 64, 64), torch.randn(1, 1, 64, 64, 64))

    def small_inputs(self):
        metric = MultiScaleSSIMMetric(spatial_dims=2)

        with self.assertRaises(ValueError):
            metric(torch.randn(1, 1, 16, 16, 16), torch.randn(1, 1, 16, 16, 16))


if __name__ == "__main__":
    unittest.main()
