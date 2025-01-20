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

from monai.networks.layers import apply_filter


class ApplyFilterTestCase(unittest.TestCase):

    def test_1d(self):
        a = torch.tensor([[list(range(10))]], dtype=torch.float)
        out = apply_filter(a, torch.tensor([-1, 0, 1]), stride=1)
        expected = np.array([[[1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, -8.0]]])
        np.testing.assert_allclose(out.cpu().numpy(), expected, rtol=1e-4)
        if torch.cuda.is_available():
            out = apply_filter(a.cuda(), torch.tensor([-1, 0, 1]).cuda())
            np.testing.assert_allclose(out.cpu().numpy(), expected, rtol=1e-4)

    def test_2d(self):
        a = torch.tensor([[[list(range(7)), list(range(7, 0, -1)), list(range(7))]]], dtype=torch.float)
        expected = np.array(
            [
                [14.0, 21.0, 21.0, 21.0, 21.0, 21.0, 14.0],
                [15.0, 24.0, 27.0, 30.0, 33.0, 36.0, 25.0],
                [14.0, 21.0, 21.0, 21.0, 21.0, 21.0, 14.0],
            ]
        )
        expected = expected[None][None]
        out = apply_filter(a, torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]]))
        np.testing.assert_allclose(out.cpu().numpy(), expected, rtol=1e-4)
        if torch.cuda.is_available():
            out = apply_filter(a.cuda(), torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]]).cuda())
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
                    [2.0, 6.0, 12.0, 18.0, 24.0, 30.0, 22.0],
                    [3.0, 9.0, 18.0, 27.0, 36.0, 45.0, 33.0],
                    [2.0, 6.0, 12.0, 18.0, 24.0, 30.0, 22.0],
                ],
                [
                    [2.0, 6.0, 12.0, 18.0, 24.0, 30.0, 22.0],
                    [3.0, 9.0, 18.0, 27.0, 36.0, 45.0, 33.0],
                    [2.0, 6.0, 12.0, 18.0, 24.0, 30.0, 22.0],
                ],
            ]
        )
        # testing shapes
        k = torch.tensor([[[1, 1, 1], [1, 1, 1], [1, 1, 1]]])
        for kernel in (k, k[None], k[None][None]):
            out = apply_filter(a, kernel)
            np.testing.assert_allclose(out.cpu().numpy()[1][2], expected, rtol=1e-4)
            if torch.cuda.is_available():
                out = apply_filter(a.cuda(), kernel.cuda())
                np.testing.assert_allclose(out.cpu().numpy()[0][1], expected, rtol=1e-4)

    def test_wrong_args(self):
        with self.assertRaisesRegex(ValueError, ""):
            apply_filter(torch.ones((1, 2, 3, 2)), torch.ones((2,)))
        with self.assertRaisesRegex(NotImplementedError, ""):
            apply_filter(torch.ones((1, 1, 1, 2, 3, 2)), torch.ones((2,)))
        with self.assertRaisesRegex(TypeError, ""):
            apply_filter(((1, 1, 1, 2, 3, 2)), torch.ones((2,)))


if __name__ == "__main__":
    unittest.main()
