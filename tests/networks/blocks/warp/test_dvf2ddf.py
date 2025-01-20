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
from torch import nn
from torch.optim import SGD

from monai.networks.blocks.warp import DVF2DDF
from monai.utils import set_determinism

TEST_CASES = [
    [{"num_steps": 1}, {"dvf": torch.zeros(1, 2, 2, 2)}, torch.zeros(1, 2, 2, 2)],
    [
        {"num_steps": 1},
        {"dvf": torch.ones(1, 3, 2, 2, 2)},
        torch.tensor([[[1.0000, 0.7500], [0.7500, 0.6250]], [[0.7500, 0.6250], [0.6250, 0.5625]]])
        .reshape(1, 1, 2, 2, 2)
        .expand(-1, 3, -1, -1, -1),
    ],
    [
        {"num_steps": 2},
        {"dvf": torch.ones(1, 3, 2, 2, 2)},
        torch.tensor([[[0.9175, 0.6618], [0.6618, 0.5306]], [[0.6618, 0.5306], [0.5306, 0.4506]]])
        .reshape(1, 1, 2, 2, 2)
        .expand(-1, 3, -1, -1, -1),
    ],
]


class TestDVF2DDF(unittest.TestCase):

    def setUp(self):
        set_determinism(0)

    def tearDown(self):
        set_determinism(None)

    @parameterized.expand(TEST_CASES)
    def test_value(self, input_param, input_data, expected_val):
        layer = DVF2DDF(**input_param)
        result = layer(**input_data)
        np.testing.assert_allclose(result.cpu().numpy(), expected_val.cpu().numpy(), rtol=1e-4, atol=1e-4)

    def test_gradient(self):
        network = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=1)
        dvf2ddf = DVF2DDF(num_steps=1)
        optimizer = SGD(network.parameters(), lr=0.01)
        x = torch.ones((1, 1, 5, 5))
        x = network(x)
        x = dvf2ddf(x)
        loss = torch.sum(x)
        loss.backward()
        optimizer.step()
        np.testing.assert_allclose(network.weight.grad.cpu().numpy(), np.array([[[[22.471329]]], [[[22.552576]]]]))


if __name__ == "__main__":
    unittest.main()
