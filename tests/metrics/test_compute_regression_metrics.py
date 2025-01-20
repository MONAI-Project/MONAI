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
from functools import partial

import numpy as np
import torch

from monai.metrics import MAEMetric, MSEMetric, PSNRMetric, RMSEMetric
from monai.utils import set_determinism


# define a numpy flatten function that only preserves batch dimension
def flatten(data):
    return np.reshape(data, [data.shape[0], -1])


# define metrics computation truth functions to check our monai metrics against
def msemetric_np(y_pred, y):
    return np.mean((flatten(y_pred) - flatten(y)) ** 2)


def maemetric_np(y_pred, y):
    return np.mean(np.abs(flatten(y_pred) - flatten(y)))


def rmsemetric_np(y_pred, y):
    return np.mean(np.sqrt(np.mean((flatten(y_pred) - flatten(y)) ** 2, axis=1)))


def psnrmetric_np(max_val, y_pred, y):
    mse = np.mean((flatten(y_pred) - flatten(y)) ** 2, axis=1)
    return np.mean(20 * np.log10(max_val) - 10 * np.log10(mse))


class TestRegressionMetrics(unittest.TestCase):

    def test_shape_reduction(self):
        set_determinism(seed=123)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # regression metrics to check
        metrics = [MSEMetric, MAEMetric, RMSEMetric, partial(PSNRMetric, max_val=1.0)]

        # define variations in batch/base_dims/spatial_dims
        batch_dims = [1, 2, 4, 16]
        base_dims = [16, 32, 64]
        spatial_dims = [2, 3, 4]

        # iterate over all variations and check shapes for different reduction functions
        for batch in batch_dims:
            for spatial in spatial_dims:
                for base in base_dims:
                    # create random tensors
                    in_tensor = torch.rand((batch,) + (base,) * (spatial - 1)).to(device)

                    # iterate over regression metrics, check shape for diff. reduction func
                    for mt_fn in metrics:
                        mt = mt_fn(reduction="mean")
                        mt(in_tensor, in_tensor)
                        out_tensor = mt.aggregate()
                        self.assertEqual(len(out_tensor.shape), 1)

                        mt = mt_fn(reduction="sum")
                        mt(in_tensor, in_tensor)
                        out_tensor = mt.aggregate()
                        self.assertEqual(len(out_tensor.shape), 0)

                        mt = mt_fn(reduction="sum")  # test reduction arg overriding
                        mt(in_tensor, in_tensor)
                        out_tensor = mt.aggregate(reduction="mean_channel")
                        self.assertEqual(len(out_tensor.shape), 1)
                        self.assertEqual(out_tensor.shape[0], batch)

                        mt = mt_fn(reduction="sum_channel")
                        mt(in_tensor, in_tensor)
                        out_tensor = mt.aggregate()
                        self.assertEqual(len(out_tensor.shape), 1)
                        self.assertEqual(out_tensor.shape[0], batch)

    def test_compare_numpy(self):
        set_determinism(seed=123)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # regression metrics to check + truth metric function in numpy
        metrics = [MSEMetric, MAEMetric, RMSEMetric, partial(PSNRMetric, max_val=1.0)]
        metrics_np = [msemetric_np, maemetric_np, rmsemetric_np, partial(psnrmetric_np, max_val=1.0)]

        # define variations in batch/base_dims/spatial_dims
        batch_dims = [1, 2, 4, 16]
        base_dims = [16, 32, 64]
        spatial_dims = [2, 3, 4]

        # iterate over all variations and check shapes for different reduction functions
        for batch in batch_dims:
            for spatial in spatial_dims:
                for base in base_dims:
                    # create random tensors
                    in_tensor_a = torch.rand((batch,) + (base,) * (spatial - 1)).to(device)
                    in_tensor_b = torch.rand((batch,) + (base,) * (spatial - 1)).to(device)

                    # check metrics
                    for mt_fn, mt_fn_np in zip(metrics, metrics_np):
                        mt = mt_fn()
                        mt(y_pred=in_tensor_a, y=in_tensor_b)
                        out_tensor = mt.aggregate(reduction="mean")
                        out_np = mt_fn_np(y_pred=in_tensor_a.cpu().numpy(), y=in_tensor_b.cpu().numpy())

                        np.testing.assert_allclose(out_tensor.cpu().numpy(), out_np, atol=1e-4)

    def test_ill_shape(self):
        set_determinism(seed=123)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # regression metrics to check + truth metric function in numpy
        metrics = [MSEMetric, MAEMetric, RMSEMetric, partial(PSNRMetric, max_val=1.0)]
        basedim = 10

        # too small shape
        with self.assertRaises(ValueError):
            in_tensor = torch.rand((basedim,)).to(device)
            for mt_fn in metrics:
                mt_fn()(in_tensor, in_tensor)

        # different shape for pred/target
        with self.assertRaises(ValueError):
            in_tensor_a = torch.rand((basedim,)).to(device)
            in_tensor_b = torch.rand((basedim, basedim)).to(device)
            for mt_fn in metrics:
                mt_fn()(y_pred=in_tensor_a, y=in_tensor_b)

    def test_same_input(self):
        set_determinism(seed=123)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        metrics = [MSEMetric, MAEMetric, RMSEMetric, partial(PSNRMetric, max_val=1.0)]
        results = [0.0, 0.0, 0.0, float("inf")]

        # define variations in batch/base_dims/spatial_dims
        batch_dims = [1, 2, 4, 16]
        base_dims = [16, 32, 64]
        spatial_dims = [2, 3, 4]

        # iterate over all variations and check shapes for different reduction functions
        for batch in batch_dims:
            for spatial in spatial_dims:
                for base in base_dims:
                    # create random tensors
                    in_tensor = torch.rand((batch,) + (base,) * (spatial - 1)).to(device)

                    # check metrics
                    for mt_fn, rs in zip(metrics, results):
                        mt = mt_fn(reduction="mean")
                        mt(in_tensor, in_tensor)
                        out_tensor = mt.aggregate()
                        np.testing.assert_allclose(out_tensor.cpu(), rs, atol=1e-4)

    def test_diff_input(self):
        set_determinism(seed=123)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        metrics = [MSEMetric, MAEMetric, RMSEMetric, partial(PSNRMetric, max_val=1.0)]
        results = [1.0, 1.0, 1.0, 0.0]

        # define variations in batch/base_dims/spatial_dims
        batch_dims = [1, 2, 4, 16]
        base_dims = [16, 32, 64]
        spatial_dims = [2, 3, 4]

        # iterate over all variations and check shapes for different reduction functions
        for batch in batch_dims:
            for spatial in spatial_dims:
                for base in base_dims:
                    # create random tensors
                    in_tensor_a = torch.zeros((batch,) + (base,) * (spatial - 1)).to(device)
                    in_tensor_b = torch.ones((batch,) + (base,) * (spatial - 1)).to(device)

                    # check metrics
                    for mt_fn, rs in zip(metrics, results):
                        mt = mt_fn(reduction="mean")
                        mt(in_tensor_a, in_tensor_b)
                        out_tensor = mt.aggregate()
                        np.testing.assert_allclose(out_tensor.cpu(), rs, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
