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

from monai.networks.layers import polyval

TEST_CASES = [
    [[1.0, 2.5, -4.2], 5.0, 33.3],
    [[2, 1, 0], 3.0, 21],
    [[2, 1, 0], [3.0, 3.0], [21, 21]],
    [torch.as_tensor([2, 1, 0]), [3.0, 3.0], [21, 21]],
    [torch.as_tensor([2, 1, 0]), torch.as_tensor([3.0, 3.0]), [21, 21]],
    [torch.as_tensor([2, 1, 0]), np.array([3.0, 3.0]), [21, 21]],
    [[], np.array([3.0, 3.0]), [0, 0]],
]


class TestPolyval(unittest.TestCase):

    @parameterized.expand(TEST_CASES)
    def test_floats(self, coef, x, expected):
        result = polyval(coef, x)
        np.testing.assert_allclose(result.cpu().numpy(), expected)

    @parameterized.expand(TEST_CASES)
    def test_gpu(self, coef, x, expected):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        x = torch.as_tensor(x, dtype=torch.float, device=device)
        x.requires_grad = True
        coef = torch.as_tensor(coef, dtype=torch.float, device=device)
        coef.requires_grad = True
        result = polyval(coef, x)
        if coef.shape[0] > 0:  # empty coef doesn't have grad
            result.mean().backward()
            np.testing.assert_allclose(coef.grad.shape, coef.shape)
        np.testing.assert_allclose(result.cpu().detach().numpy(), expected)


if __name__ == "__main__":
    unittest.main()
