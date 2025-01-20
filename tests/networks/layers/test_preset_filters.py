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

import torch
from parameterized import parameterized

from monai.networks.layers import ApplyFilter, EllipticalFilter, LaplaceFilter, MeanFilter, SharpenFilter

TEST_CASES_MEAN = [(3, 3, torch.ones(3, 3, 3)), (2, 5, torch.ones(5, 5))]

TEST_CASES_LAPLACE = [
    (
        3,
        3,
        torch.Tensor(
            [
                [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]],
                [[-1, -1, -1], [-1, 26, -1], [-1, -1, -1]],
                [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]],
            ]
        ),
    ),
    (2, 3, torch.Tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])),
]

TEST_CASES_ELLIPTICAL = [
    (
        3,
        3,
        torch.Tensor(
            [[[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 1, 0], [1, 1, 1], [0, 1, 0]], [[0, 0, 0], [0, 1, 0], [0, 0, 0]]]
        ),
    ),
    (2, 3, torch.Tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]])),
]

TEST_CASES_SHARPEN = [
    (
        3,
        3,
        torch.Tensor(
            [
                [[0, 0, 0], [0, -1, 0], [0, 0, 0]],
                [[0, -1, 0], [-1, 7, -1], [0, -1, 0]],
                [[0, 0, 0], [0, -1, 0], [0, 0, 0]],
            ]
        ),
    ),
    (2, 3, torch.Tensor([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])),
]


class _TestFilter:

    def test_init(self, spatial_dims, size, expected):
        test_filter = self.filter_class(spatial_dims=spatial_dims, size=size)
        torch.testing.assert_allclose(expected, test_filter.filter)
        self.assertIsInstance(test_filter, torch.nn.Module)

    def test_forward(self):
        test_filter = self.filter_class(spatial_dims=2, size=3)
        input = torch.ones(1, 1, 5, 5)
        _ = test_filter(input)


class TestApplyFilter(unittest.TestCase):

    def test_init_and_forward_2d(self):
        filter_2d = torch.ones(3, 3)
        image_2d = torch.ones(1, 3, 3)
        apply_filter_2d = ApplyFilter(filter_2d)
        out = apply_filter_2d(image_2d)
        self.assertEqual(out.shape, image_2d.shape)

    def test_init_and_forward_3d(self):
        filter_2d = torch.ones(3, 3, 3)
        image_2d = torch.ones(1, 3, 3, 3)
        apply_filter_2d = ApplyFilter(filter_2d)
        out = apply_filter_2d(image_2d)
        self.assertEqual(out.shape, image_2d.shape)


class MeanFilterTestCase(_TestFilter, unittest.TestCase):

    def setUp(self) -> None:
        self.filter_class = MeanFilter

    @parameterized.expand(TEST_CASES_MEAN)
    def test_init(self, spatial_dims, size, expected):
        super().test_init(spatial_dims, size, expected)


class LaplaceFilterTestCase(_TestFilter, unittest.TestCase):

    def setUp(self) -> None:
        self.filter_class = LaplaceFilter

    @parameterized.expand(TEST_CASES_LAPLACE)
    def test_init(self, spatial_dims, size, expected):
        super().test_init(spatial_dims, size, expected)


class EllipticalTestCase(_TestFilter, unittest.TestCase):

    def setUp(self) -> None:
        self.filter_class = EllipticalFilter

    @parameterized.expand(TEST_CASES_ELLIPTICAL)
    def test_init(self, spatial_dims, size, expected):
        super().test_init(spatial_dims, size, expected)


class SharpenTestCase(_TestFilter, unittest.TestCase):

    def setUp(self) -> None:
        self.filter_class = SharpenFilter

    @parameterized.expand(TEST_CASES_SHARPEN)
    def test_init(self, spatial_dims, size, expected):
        super().test_init(spatial_dims, size, expected)


if __name__ == "__main__":
    unittest.main()
