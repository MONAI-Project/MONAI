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

from monai.networks.layers import GaussianFilter
from tests.util import SkipIfAtLeastPyTorchVersion, skip_if_quick

TEST_CASES = [[{"type": "erf", "gt": 2.0}], [{"type": "scalespace", "gt": 3.0}], [{"type": "sampled", "gt": 5.0}]]
TEST_CASES_GPU = [[{"type": "erf", "gt": 0.8, "device": "cuda"}], [{"type": "sampled", "gt": 5.0, "device": "cuda"}]]
TEST_CASES_3d = [
    [{"type": "scalespace", "gt": 0.5, "dims": (2, 3, 8, 9, 10), "lr": 0.01, "device": "cuda"}],
    [{"type": "erf", "gt": 3.8, "dims": (2, 3, 8, 9, 10), "lr": 0.1, "device": "cuda"}],
]
TEST_CASES_SLOW = [
    [{"type": "erf", "gt": 2.0, "dims": (2, 3, 8, 9, 10)}],
    [{"type": "scalespace", "gt": 3.0, "dims": (2, 3, 8, 9, 10), "device": "cuda"}],
    [{"type": "sampled", "gt": (0.5, 0.8, 3.0), "dims": (2, 3, 8, 9, 10), "lr": 0.1}],
    [{"type": "scalespace", "gt": 3.0, "device": "cuda"}],
]


@SkipIfAtLeastPyTorchVersion((2, 2, 0))  # https://github.com/Project-MONAI/MONAI/issues/7445
class TestGaussianFilterBackprop(unittest.TestCase):
    def code_to_run(self, input_args):
        input_dims = input_args.get("dims", (2, 3, 8))
        device = (
            torch.device("cuda")
            if (input_args.get("device") == "cuda" and torch.cuda.is_available())
            else torch.device("cpu:0")
        )

        base = torch.ones(*input_dims).to(device)
        gt = torch.tensor(input_args["gt"], requires_grad=False).to(device)
        g_type = input_args["type"]
        lr = input_args.get("lr", 0.1)
        init_sigma = input_args.get("init", 1.0)

        # static filter to generate a target
        spatial_dims = len(base.shape) - 2
        filtering = GaussianFilter(spatial_dims=spatial_dims, sigma=gt, approx=g_type, requires_grad=False)
        filtering.to(device)
        target = filtering(base)
        self.assertFalse(filtering.sigma[0].requires_grad)

        # build trainable
        init_sigma = torch.tensor(init_sigma).to(device)
        trainable = GaussianFilter(spatial_dims=spatial_dims, sigma=init_sigma, approx=g_type, requires_grad=True)
        trainable.to(device)
        self.assertTrue(trainable.sigma[0].requires_grad)

        # train
        optimizer = torch.optim.Adam(trainable.parameters(), lr=lr)
        for s in range(1000):
            optimizer.zero_grad()
            pred = trainable(base)
            loss = torch.pow(pred - target, 2).mean()
            loss.backward()
            if (s + 1) % 50 == 0:
                var = list(trainable.parameters())[0]
                print(f"step {s} loss {loss}")
                print(var, var.grad)
            if loss.item() < 1e-7:
                break
            optimizer.step()
        # check the result
        print(s, gt)
        for idx, s in enumerate(trainable.sigma):
            np.testing.assert_allclose(
                s.cpu().item(), gt.cpu() if len(gt.shape) == 0 else gt[idx].cpu().item(), rtol=1e-2
            )

    @parameterized.expand(TEST_CASES + TEST_CASES_GPU + TEST_CASES_3d)
    def test_train_quick(self, input_args):
        self.code_to_run(input_args)

    @parameterized.expand(TEST_CASES_SLOW)
    @skip_if_quick
    def test_train_slow(self, input_args):
        self.code_to_run(input_args)


@SkipIfAtLeastPyTorchVersion((2, 2, 0))  # https://github.com/Project-MONAI/MONAI/issues/7445
class GaussianFilterTestCase(unittest.TestCase):
    def test_1d(self):
        a = torch.ones(1, 8, 10)
        g = GaussianFilter(1, 3, 3).to(torch.device("cpu:0"))
        expected = np.array(
            [
                [
                    [
                        0.5654129,
                        0.68915915,
                        0.79146194,
                        0.8631974,
                        0.8998163,
                        0.8998163,
                        0.8631973,
                        0.79146194,
                        0.6891592,
                        0.5654129,
                    ]
                ]
            ]
        )
        expected = np.tile(expected, (1, 8, 1))
        np.testing.assert_allclose(g(a).cpu().numpy(), expected, rtol=1e-5)

    @skip_if_quick
    def test_2d(self):
        a = torch.ones(1, 1, 3, 3)
        g = GaussianFilter(2, 3, 3).to(torch.device("cpu:0"))
        expected = np.array(
            [
                [
                    [
                        [0.13239081, 0.13932934, 0.13239081],
                        [0.13932936, 0.14663152, 0.13932936],
                        [0.13239081, 0.13932934, 0.13239081],
                    ]
                ]
            ]
        )

        np.testing.assert_allclose(g(a).cpu().numpy(), expected, rtol=1e-5)
        if torch.cuda.is_available():
            g = GaussianFilter(2, 3, 3).to(torch.device("cuda:0"))
            np.testing.assert_allclose(g(a.cuda()).cpu().numpy(), expected, rtol=1e-2)

    def test_3d(self):
        a = torch.ones(1, 1, 4, 3, 4)
        g = GaussianFilter(3, 3, 3).to(torch.device("cpu:0"))

        expected = np.array(
            [
                [
                    [
                        [
                            [0.07189433, 0.07911152, 0.07911152, 0.07189433],
                            [0.07566228, 0.08325771, 0.08325771, 0.07566228],
                            [0.07189433, 0.07911152, 0.07911152, 0.07189433],
                        ],
                        [
                            [0.07911152, 0.08705322, 0.08705322, 0.07911152],
                            [0.08325771, 0.09161563, 0.09161563, 0.08325771],
                            [0.07911152, 0.08705322, 0.08705322, 0.07911152],
                        ],
                        [
                            [0.07911152, 0.08705322, 0.08705322, 0.07911152],
                            [0.08325771, 0.09161563, 0.09161563, 0.08325771],
                            [0.07911152, 0.08705322, 0.08705322, 0.07911152],
                        ],
                        [
                            [0.07189433, 0.07911152, 0.07911152, 0.07189433],
                            [0.07566228, 0.08325771, 0.08325771, 0.07566228],
                            [0.07189433, 0.07911152, 0.07911152, 0.07189433],
                        ],
                    ]
                ]
            ]
        )
        np.testing.assert_allclose(g(a).cpu().numpy(), expected, rtol=1e-5)

    def test_3d_sigmas(self):
        a = torch.ones(1, 1, 4, 3, 2)
        g = GaussianFilter(3, [3, 2, 1], 3).to(torch.device("cpu:0"))

        expected = np.array(
            [
                [
                    [
                        [[0.13690521, 0.13690521], [0.15181276, 0.15181276], [0.13690521, 0.13690521]],
                        [[0.1506486, 0.15064861], [0.16705267, 0.16705267], [0.1506486, 0.15064861]],
                        [[0.1506486, 0.15064861], [0.16705267, 0.16705267], [0.1506486, 0.15064861]],
                        [[0.13690521, 0.13690521], [0.15181276, 0.15181276], [0.13690521, 0.13690521]],
                    ]
                ]
            ]
        )
        np.testing.assert_allclose(g(a).cpu().numpy(), expected, rtol=1e-5)
        if torch.cuda.is_available():
            g = GaussianFilter(3, [3, 2, 1], 3).to(torch.device("cuda:0"))
            np.testing.assert_allclose(g(a.cuda()).cpu().numpy(), expected, rtol=1e-2)

    def test_wrong_args(self):
        with self.assertRaisesRegex(ValueError, ""):
            GaussianFilter(3, [3, 2], 3).to(torch.device("cpu:0"))
        GaussianFilter(3, [3, 2, 1], 3).to(torch.device("cpu:0"))  # test init


if __name__ == "__main__":
    unittest.main()
