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
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from monai.data import create_test_image_2d
from monai.networks.layers import AffineTransform
from monai.utils import set_determinism
from tests.util import DistTestCase, TimedCall


class STNBenchmark(nn.Module):
    """
    adapted from https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
    """

    def __init__(self, is_ref=True, reverse_indexing=False):
        super().__init__()
        self.is_ref = is_ref
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
        )
        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(nn.Linear(10 * 3 * 3, 32), nn.ReLU(True), nn.Linear(32, 3 * 2))
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        if not self.is_ref:
            self.xform = AffineTransform(align_corners=False, normalized=True, reverse_indexing=reverse_indexing)

    # Spatial transformer network forward function
    def stn_ref(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        return x

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        x = self.xform(x, theta, spatial_size=x.size()[2:])
        return x

    def forward(self, x):
        if self.is_ref:
            return self.stn_ref(x)
        return self.stn(x)


def compare_2d(is_ref=True, device=None, reverse_indexing=False):
    batch_size = 32
    img_a = [create_test_image_2d(28, 28, 5, rad_max=6, noise_max=1)[0][None] for _ in range(batch_size)]
    img_b = [create_test_image_2d(28, 28, 5, rad_max=6, noise_max=1)[0][None] for _ in range(batch_size)]
    img_a = np.stack(img_a, axis=0)
    img_b = np.stack(img_b, axis=0)
    img_a = torch.as_tensor(img_a, device=device)
    img_b = torch.as_tensor(img_b, device=device)
    model = STNBenchmark(is_ref=is_ref, reverse_indexing=reverse_indexing).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    model.train()
    init_loss = None
    for _ in range(20):
        optimizer.zero_grad()
        output_a = model(img_a)
        loss = torch.mean((output_a - img_b) ** 2)
        if init_loss is None:
            init_loss = loss.item()
        loss.backward()
        optimizer.step()
    return model(img_a).detach().cpu().numpy(), loss.item(), init_loss


class TestSpatialTransformerCore(DistTestCase):
    def setUp(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu:0")

    def tearDown(self):
        set_determinism(seed=None)

    @TimedCall(seconds=100, skip_timing=not torch.cuda.is_available())
    def test_training(self):
        """
        check that the quality AffineTransform backpropagation
        """
        atol = 1e-5
        set_determinism(seed=0)
        out_ref, loss_ref, init_loss_ref = compare_2d(True, self.device)
        print(out_ref.shape, loss_ref, init_loss_ref)

        set_determinism(seed=0)
        out, loss, init_loss = compare_2d(False, self.device)
        print(out.shape, loss, init_loss)
        np.testing.assert_allclose(out_ref, out, atol=atol)
        np.testing.assert_allclose(init_loss_ref, init_loss, atol=atol)
        np.testing.assert_allclose(loss_ref, loss, atol=atol)

        set_determinism(seed=0)
        out, loss, init_loss = compare_2d(False, self.device, True)
        print(out.shape, loss, init_loss)
        np.testing.assert_allclose(out_ref, out, atol=atol)
        np.testing.assert_allclose(init_loss_ref, init_loss, atol=atol)
        np.testing.assert_allclose(loss_ref, loss, atol=atol)


if __name__ == "__main__":
    unittest.main()
