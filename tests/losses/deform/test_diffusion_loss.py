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

from monai.losses.deform import DiffusionLoss

device = "cuda" if torch.cuda.is_available() else "cpu"

TEST_CASES = [
    # all first partials are zero, so the diffusion loss is also zero
    [{}, {"pred": torch.ones((1, 3, 5, 5, 5), device=device)}, 0.0],
    # all first partials are one, so the diffusion loss is also one
    [{}, {"pred": torch.arange(0, 5, device=device)[None, None, None, None, :].expand(1, 3, 5, 5, 5)}, 1.0],
    # before expansion, the first partials are 2, 4, 6, so the diffusion loss is (2^2 + 4^2 + 6^2) / 3 = 18.67
    [
        {"normalize": False},
        {"pred": torch.arange(0, 5, device=device)[None, None, None, None, :].expand(1, 3, 5, 5, 5) ** 2},
        56.0 / 3.0,
    ],
    # same as the previous case
    [
        {"normalize": False},
        {"pred": torch.arange(0, 5, device=device)[None, None, None, :].expand(1, 2, 5, 5) ** 2},
        56.0 / 3.0,
    ],
    # same as the previous case
    [{"normalize": False}, {"pred": torch.arange(0, 5, device=device)[None, None, :].expand(1, 1, 5) ** 2}, 56.0 / 3.0],
    # we have shown in the demo notebook that
    # diffusion loss is scale-invariant when the all axes have the same resolution
    [
        {"normalize": True},
        {"pred": torch.arange(0, 5, device=device)[None, None, None, None, :].expand(1, 3, 5, 5, 5) ** 2},
        56.0 / 3.0,
    ],
    [
        {"normalize": True},
        {"pred": torch.arange(0, 5, device=device)[None, None, None, :].expand(1, 2, 5, 5) ** 2},
        56.0 / 3.0,
    ],
    [{"normalize": True}, {"pred": torch.arange(0, 5, device=device)[None, None, :].expand(1, 1, 5) ** 2}, 56.0 / 3.0],
    # for the following case, consider the following 2D matrix:
    # tensor([[[[0, 1, 2],
    #           [1, 2, 3],
    #           [2, 3, 4],
    #           [3, 4, 5],
    #           [4, 5, 6]],
    #          [[0, 1, 2],
    #           [1, 2, 3],
    #           [2, 3, 4],
    #           [3, 4, 5],
    #           [4, 5, 6]]]])
    # the first partials wrt x are all ones, and so are the first partials wrt y
    # the diffusion loss, when normalization is not applied, is 1^2 + 1^2 = 2
    [{"normalize": False}, {"pred": torch.stack([torch.arange(i, i + 3) for i in range(5)]).expand(1, 2, 5, 3)}, 2.0],
    # consider the same matrix, this time with normalization applied, using the same notation as in the demo notebook,
    # the coefficients to be divided out are (1, 5/3) for partials wrt x and (3/5, 1) for partials wrt y
    # the diffusion loss is then (1/1)^2 + (1/(5/3))^2 + (1/(3/5))^2 + (1/1)^2 = (1 + 9/25 + 25/9 + 1) / 2 = 2.5689
    [
        {"normalize": True},
        {"pred": torch.stack([torch.arange(i, i + 3) for i in range(5)]).expand(1, 2, 5, 3)},
        (1.0 + 9.0 / 25.0 + 25.0 / 9.0 + 1.0) / 2.0,
    ],
]


class TestDiffusionLoss(unittest.TestCase):

    @parameterized.expand(TEST_CASES)
    def test_shape(self, input_param, input_data, expected_val):
        result = DiffusionLoss(**input_param).forward(**input_data)
        np.testing.assert_allclose(result.detach().cpu().numpy(), expected_val, rtol=1e-5)

    def test_ill_shape(self):
        loss = DiffusionLoss()
        # not in 3-d, 4-d, 5-d
        with self.assertRaisesRegex(ValueError, "Expecting 3-d, 4-d or 5-d"):
            loss.forward(torch.ones((1, 3), device=device))
        with self.assertRaisesRegex(ValueError, "Expecting 3-d, 4-d or 5-d"):
            loss.forward(torch.ones((1, 4, 5, 5, 5, 5), device=device))
        with self.assertRaisesRegex(ValueError, "All spatial dimensions"):
            loss.forward(torch.ones((1, 3, 2, 5, 5), device=device))
        with self.assertRaisesRegex(ValueError, "All spatial dimensions"):
            loss.forward(torch.ones((1, 3, 5, 2, 5)))
        with self.assertRaisesRegex(ValueError, "All spatial dimensions"):
            loss.forward(torch.ones((1, 3, 5, 5, 2)))

        # number of vector components unequal to number of spatial dims
        with self.assertRaisesRegex(ValueError, "Number of vector components"):
            loss.forward(torch.ones((1, 2, 5, 5, 5)))
        with self.assertRaisesRegex(ValueError, "Number of vector components"):
            loss.forward(torch.ones((1, 2, 5, 5, 5)))

    def test_ill_opts(self):
        pred = torch.rand(1, 3, 5, 5, 5).to(device=device)
        with self.assertRaisesRegex(ValueError, ""):
            DiffusionLoss(reduction="unknown")(pred)
        with self.assertRaisesRegex(ValueError, ""):
            DiffusionLoss(reduction=None)(pred)


if __name__ == "__main__":
    unittest.main()
