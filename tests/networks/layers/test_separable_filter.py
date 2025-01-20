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

from monai.networks.layers import separable_filtering


class SeparableFilterTestCase(unittest.TestCase):

    def test_1d(self):
        a = torch.tensor([[list(range(10))]], dtype=torch.float)
        out = separable_filtering(a, torch.tensor([-1, 0, 1]))
        expected = np.array([[[1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, -8.0]]])
        np.testing.assert_allclose(out.cpu().numpy(), expected, rtol=1e-4)
        if torch.cuda.is_available():
            out = separable_filtering(a.cuda(), torch.tensor([-1, 0, 1]).cuda())
            np.testing.assert_allclose(out.cpu().numpy(), expected, rtol=1e-4)

    def test_2d(self):
        a = torch.tensor([[[list(range(7)), list(range(7, 0, -1)), list(range(7))]]], dtype=torch.float)
        expected = np.array(
            [
                [28.0, 28.0, 28.0, 28.0, 28.0, 28.0],
                [30.0, 34.0, 38.0, 42.0, 46.0, 50.0],
                [28.0, 28.0, 28.0, 28.0, 28.0, 28.0],
            ]
        )
        expected = expected[None][None]
        out = separable_filtering(a, [torch.tensor([1, 1, 1]), torch.tensor([2, 2])])
        np.testing.assert_allclose(out.cpu().numpy(), expected, rtol=1e-4)
        if torch.cuda.is_available():
            out = separable_filtering(a.cuda(), [torch.tensor([1, 1, 1]).cuda(), torch.tensor([2, 2]).cuda()])
            np.testing.assert_allclose(out.cpu().numpy(), expected, rtol=1e-4)

    def test_3d(self):
        a = torch.tensor(
            [[list(range(7)), list(range(7)), list(range(7))], [list(range(7)), list(range(7)), list(range(7))]],
            dtype=torch.float,
        )
        a = a[None][None]
        a = a.expand(2, 3, -1, -1, -1)
        expected = np.array(
            [
                [
                    [4.0, 12.0, 24.0, 36.0, 48.0, 60.0, 44.0],
                    [6.0, 18.0, 36.0, 54.0, 72.0, 90.0, 66.0],
                    [4.0, 12.0, 24.0, 36.0, 48.0, 60.0, 44.0],
                ],
                [
                    [4.0, 12.0, 24.0, 36.0, 48.0, 60.0, 44.0],
                    [6.0, 18.0, 36.0, 54.0, 72.0, 90.0, 66.0],
                    [4.0, 12.0, 24.0, 36.0, 48.0, 60.0, 44.0],
                ],
            ]
        )
        # testing shapes
        k = torch.tensor([1, 1, 1])
        for kernel in (k, [k] * 3):
            out = separable_filtering(a, kernel)
            np.testing.assert_allclose(out.cpu().numpy()[1][2], expected, rtol=1e-4)
            if torch.cuda.is_available():
                out = separable_filtering(
                    a.cuda(), kernel.cuda() if isinstance(kernel, torch.Tensor) else [k.cuda() for k in kernel]
                )
                np.testing.assert_allclose(out.cpu().numpy()[0][1], expected, rtol=1e-4)

    def test_wrong_args(self):
        with self.assertRaisesRegex(TypeError, ""):
            separable_filtering(((1, 1, 1, 2, 3, 2)), torch.ones((2,)))


if __name__ == "__main__":
    unittest.main()
