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
from ignite.engine import Engine

from monai.handlers import MeanAbsoluteError, MeanSquaredError, PeakSignalToNoiseRatio, RootMeanSquaredError
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


class TestHandlerRegressionMetrics(unittest.TestCase):

    def test_compute(self):
        set_determinism(seed=123)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # regression metrics to check + truth metric function in numpy
        metrics = [
            MeanSquaredError,
            MeanAbsoluteError,
            RootMeanSquaredError,
            partial(PeakSignalToNoiseRatio, max_val=1.0),
        ]
        metrics_np = [msemetric_np, maemetric_np, rmsemetric_np, partial(psnrmetric_np, max_val=1.0)]

        # define variations in batch/base_dims/spatial_dims
        batch_dims = [1, 2, 4, 16]
        base_dims = [16, 32, 64]
        spatial_dims = [2, 3, 4]

        # iterate over all variations and check shapes for different reduction functions
        for mt_fn, mt_fn_np in zip(metrics, metrics_np):
            for batch in batch_dims:
                for spatial in spatial_dims:
                    for base in base_dims:
                        mt_fn_obj = mt_fn(**{"save_details": False})

                        # create random tensor
                        in_tensor_a1 = torch.rand((batch,) + (base,) * (spatial - 1)).to(device)
                        in_tensor_b1 = torch.rand((batch,) + (base,) * (spatial - 1)).to(device)
                        mt_fn_obj.update([in_tensor_a1, in_tensor_b1])
                        out_tensor_np1 = mt_fn_np(y_pred=in_tensor_a1.cpu().numpy(), y=in_tensor_b1.cpu().numpy())

                        in_tensor_a2 = torch.rand((batch,) + (base,) * (spatial - 1)).to(device)
                        in_tensor_b2 = torch.rand((batch,) + (base,) * (spatial - 1)).to(device)
                        mt_fn_obj.update([in_tensor_a2, in_tensor_b2])
                        out_tensor_np2 = mt_fn_np(y_pred=in_tensor_a2.cpu().numpy(), y=in_tensor_b2.cpu().numpy())

                        out_tensor = mt_fn_obj.compute()
                        out_tensor_np = (out_tensor_np1 + out_tensor_np2) / 2.0

                        np.testing.assert_allclose(out_tensor, out_tensor_np, atol=1e-4)

    def test_compute_engine(self):
        set_determinism(seed=123)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # regression metrics to check + truth metric function in numpy
        metrics_names = ["MSE", "MAE", "RMSE", "PSNR"]
        metrics = [
            MeanSquaredError,
            MeanAbsoluteError,
            RootMeanSquaredError,
            partial(PeakSignalToNoiseRatio, max_val=1.0),
        ]
        metrics_np = [msemetric_np, maemetric_np, rmsemetric_np, partial(psnrmetric_np, max_val=1.0)]

        def _val_func(engine, batch):
            pass

        # define variations in batch/base_dims/spatial_dims
        batch_dims = [1, 2, 4, 16]
        base_dims = [16, 32, 64]
        spatial_dims = [2, 3, 4]

        # iterate over all variations and check shapes for different reduction functions
        for mt_fn_name, mt_fn, mt_fn_np in zip(metrics_names, metrics, metrics_np):
            for batch in batch_dims:
                for spatial in spatial_dims:
                    for base in base_dims:
                        mt_fn_obj = mt_fn()  # 'save_details' == True
                        engine = Engine(_val_func)
                        mt_fn_obj.attach(engine, mt_fn_name)

                        # create random tensor
                        in_tensor_a1 = torch.rand((batch,) + (base,) * (spatial - 1)).to(device)
                        in_tensor_b1 = torch.rand((batch,) + (base,) * (spatial - 1)).to(device)
                        mt_fn_obj.update([in_tensor_a1, in_tensor_b1])
                        out_tensor_np1 = mt_fn_np(y_pred=in_tensor_a1.cpu().numpy(), y=in_tensor_b1.cpu().numpy())

                        in_tensor_a2 = torch.rand((batch,) + (base,) * (spatial - 1)).to(device)
                        in_tensor_b2 = torch.rand((batch,) + (base,) * (spatial - 1)).to(device)
                        mt_fn_obj.update([in_tensor_a2, in_tensor_b2])
                        out_tensor_np2 = mt_fn_np(y_pred=in_tensor_a2.cpu().numpy(), y=in_tensor_b2.cpu().numpy())

                        out_tensor = mt_fn_obj.compute()
                        out_tensor_np = (out_tensor_np1 + out_tensor_np2) / 2.0

                        np.testing.assert_allclose(out_tensor, out_tensor_np, atol=1e-4)

    def test_ill_shape(self):
        set_determinism(seed=123)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # regression metrics to check + truth metric function in numpy
        metrics = [
            MeanSquaredError,
            MeanAbsoluteError,
            RootMeanSquaredError,
            partial(PeakSignalToNoiseRatio, max_val=1.0),
        ]
        basedim = 10

        # different shape for pred/target
        with self.assertRaises((AssertionError, ValueError)):
            in_tensor_a = torch.rand((basedim,)).to(device)
            in_tensor_b = torch.rand((basedim, basedim)).to(device)
            for mt_fn in metrics:
                mt_fn().update([in_tensor_a, in_tensor_b])


if __name__ == "__main__":
    unittest.main()
