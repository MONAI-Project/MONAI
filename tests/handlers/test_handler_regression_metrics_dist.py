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
import torch.distributed as dist
from ignite.engine import Engine

from monai.handlers import MeanAbsoluteError, MeanSquaredError, PeakSignalToNoiseRatio, RootMeanSquaredError
from monai.utils import set_determinism
from tests.util import DistCall, DistTestCase


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


# define tensor size as (BATCH_SIZE, (BASE_DIM_SIZE,) * SPATIAL_DIM)
# One tensor with following shape takes 4*32*32*32*32/(8*1000) = 512 MB on a single GPU
# We have total of 2 tensors each on one GPU for following tests, so required GPU memory is 1024 MB on each GPU
# The required GPU memory can be lowered by changing BASE_DIM_SIZE to another value e.g. BASE_DIM_SIZE=16 will
# require 128 MB on each GPU
BATCH_SIZE = 4
BASE_DIM_SIZE = 32
SPATIAL_DIM = 3


class DistributedMeanSquaredError(DistTestCase):
    @DistCall(nnodes=1, nproc_per_node=2)
    def test_compute(self):
        set_determinism(123)
        self._compute()

    def _compute(self):
        device = f"cuda:{dist.get_rank()}" if torch.cuda.is_available() else "cpu"
        metric = MeanSquaredError()

        def _val_func(engine, batch):
            pass

        engine = Engine(_val_func)
        metric.attach(engine, "MSE")

        # get testing data
        batch = BATCH_SIZE
        base = BASE_DIM_SIZE
        spatial = SPATIAL_DIM
        in_tensor_a1 = torch.rand((batch,) + (base,) * (spatial - 1))
        in_tensor_b1 = torch.rand((batch,) + (base,) * (spatial - 1))

        in_tensor_a2 = torch.rand((batch,) + (base,) * (spatial - 1))
        in_tensor_b2 = torch.rand((batch,) + (base,) * (spatial - 1))

        if dist.get_rank() == 0:
            y_pred = in_tensor_a1.to(device)
            y = in_tensor_b1.to(device)
            metric.update([y_pred, y])

        if dist.get_rank() == 1:
            y_pred = in_tensor_a2.to(device)
            y = in_tensor_b2.to(device)
            metric.update([y_pred, y])

        out_tensor = metric.compute()

        # do numpy functions to get ground truth referece
        out_tensor_np1 = msemetric_np(y_pred=in_tensor_a1.cpu().numpy(), y=in_tensor_b1.cpu().numpy())
        out_tensor_np2 = msemetric_np(y_pred=in_tensor_a2.cpu().numpy(), y=in_tensor_b2.cpu().numpy())
        out_tensor_np = (out_tensor_np1 + out_tensor_np2) / 2.0

        np.testing.assert_allclose(out_tensor, out_tensor_np, rtol=1e-04, atol=1e-04)


class DistributedMeanAbsoluteError(DistTestCase):
    @DistCall(nnodes=1, nproc_per_node=2)
    def test_compute(self):
        set_determinism(123)
        self._compute()

    def _compute(self):
        device = f"cuda:{dist.get_rank()}" if torch.cuda.is_available() else "cpu"
        metric = MeanAbsoluteError()

        def _val_func(engine, batch):
            pass

        engine = Engine(_val_func)
        metric.attach(engine, "MAE")

        # get testing data
        batch = BATCH_SIZE
        base = BASE_DIM_SIZE
        spatial = SPATIAL_DIM
        in_tensor_a1 = torch.rand((batch,) + (base,) * (spatial - 1))
        in_tensor_b1 = torch.rand((batch,) + (base,) * (spatial - 1))

        in_tensor_a2 = torch.rand((batch,) + (base,) * (spatial - 1))
        in_tensor_b2 = torch.rand((batch,) + (base,) * (spatial - 1))

        if dist.get_rank() == 0:
            y_pred = in_tensor_a1.to(device)
            y = in_tensor_b1.to(device)
            metric.update([y_pred, y])

        if dist.get_rank() == 1:
            y_pred = in_tensor_a2.to(device)
            y = in_tensor_b2.to(device)
            metric.update([y_pred, y])

        out_tensor = metric.compute()

        # do numpy functions to get ground truth referece
        out_tensor_np1 = maemetric_np(y_pred=in_tensor_a1.cpu().numpy(), y=in_tensor_b1.cpu().numpy())
        out_tensor_np2 = maemetric_np(y_pred=in_tensor_a2.cpu().numpy(), y=in_tensor_b2.cpu().numpy())
        out_tensor_np = (out_tensor_np1 + out_tensor_np2) / 2.0

        np.testing.assert_allclose(out_tensor, out_tensor_np, rtol=1e-04, atol=1e-04)


class DistributedRootMeanSquaredError(DistTestCase):
    @DistCall(nnodes=1, nproc_per_node=2)
    def test_compute(self):
        set_determinism(123)
        self._compute()

    def _compute(self):
        device = f"cuda:{dist.get_rank()}" if torch.cuda.is_available() else "cpu"
        metric = RootMeanSquaredError()

        def _val_func(engine, batch):
            pass

        engine = Engine(_val_func)
        metric.attach(engine, "RMSE")

        # get testing data
        batch = BATCH_SIZE
        base = BASE_DIM_SIZE
        spatial = SPATIAL_DIM
        in_tensor_a1 = torch.rand((batch,) + (base,) * (spatial - 1))
        in_tensor_b1 = torch.rand((batch,) + (base,) * (spatial - 1))

        in_tensor_a2 = torch.rand((batch,) + (base,) * (spatial - 1))
        in_tensor_b2 = torch.rand((batch,) + (base,) * (spatial - 1))

        if dist.get_rank() == 0:
            y_pred = in_tensor_a1.to(device)
            y = in_tensor_b1.to(device)
            metric.update([y_pred, y])

        if dist.get_rank() == 1:
            y_pred = in_tensor_a2.to(device)
            y = in_tensor_b2.to(device)
            metric.update([y_pred, y])

        out_tensor = metric.compute()

        # do numpy functions to get ground truth referece
        out_tensor_np1 = rmsemetric_np(y_pred=in_tensor_a1.cpu().numpy(), y=in_tensor_b1.cpu().numpy())
        out_tensor_np2 = rmsemetric_np(y_pred=in_tensor_a2.cpu().numpy(), y=in_tensor_b2.cpu().numpy())
        out_tensor_np = (out_tensor_np1 + out_tensor_np2) / 2.0

        np.testing.assert_allclose(out_tensor, out_tensor_np, rtol=1e-04, atol=1e-04)


class DistributedPeakSignalToNoiseRatio(DistTestCase):
    @DistCall(nnodes=1, nproc_per_node=2)
    def test_compute(self):
        set_determinism(123)
        self._compute()

    def _compute(self):
        device = f"cuda:{dist.get_rank()}" if torch.cuda.is_available() else "cpu"
        max_val = 1.0
        metric = PeakSignalToNoiseRatio(max_val=max_val)

        def _val_func(engine, batch):
            pass

        engine = Engine(_val_func)
        metric.attach(engine, "PSNR")

        # get testing data
        batch = BATCH_SIZE
        base = BASE_DIM_SIZE
        spatial = SPATIAL_DIM
        in_tensor_a1 = torch.rand((batch,) + (base,) * (spatial - 1))
        in_tensor_b1 = torch.rand((batch,) + (base,) * (spatial - 1))

        in_tensor_a2 = torch.rand((batch,) + (base,) * (spatial - 1))
        in_tensor_b2 = torch.rand((batch,) + (base,) * (spatial - 1))

        if dist.get_rank() == 0:
            y_pred = in_tensor_a1.to(device)
            y = in_tensor_b1.to(device)
            metric.update([y_pred, y])

        if dist.get_rank() == 1:
            y_pred = in_tensor_a2.to(device)
            y = in_tensor_b2.to(device)
            metric.update([y_pred, y])

        out_tensor = metric.compute()

        # do numpy functions to get ground truth referece
        out_tensor_np1 = psnrmetric_np(max_val=max_val, y_pred=in_tensor_a1.cpu().numpy(), y=in_tensor_b1.cpu().numpy())
        out_tensor_np2 = psnrmetric_np(max_val=max_val, y_pred=in_tensor_a2.cpu().numpy(), y=in_tensor_b2.cpu().numpy())
        out_tensor_np = (out_tensor_np1 + out_tensor_np2) / 2.0

        np.testing.assert_allclose(out_tensor, out_tensor_np, rtol=1e-04, atol=1e-04)


if __name__ == "__main__":
    unittest.main()
