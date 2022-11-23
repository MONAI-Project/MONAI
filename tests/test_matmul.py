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

import unittest

import numpy as np
import torch
from parameterized import parameterized

from monai.transforms.meta_matrix import (
    Grid,
    Matrix,
    MatrixFactory,
    is_grid_shaped,
    is_matrix_shaped,
    matmul,
    matmul_grid_matrix,
    matmul_grid_matrix_slow,
    matmul_matrix_grid,
    matmul_matrix_matrix,
)
from monai.utils import TransformBackends


class TestMatmulFunctions(unittest.TestCase):
    def test_matrix_grid_and_grid_inv_matrix_are_equivalent(self):
        grid = torch.randn((3, 32, 32))

        matrix = torch.eye(3, 3)
        matrix[:, 0] = torch.FloatTensor([0, -1, 0])
        matrix[:, 1] = torch.FloatTensor([1, 0, 0])

        inv_matrix = torch.inverse(matrix)

        result1 = matmul_matrix_grid(matrix, grid)
        result2 = matmul_grid_matrix(grid, inv_matrix)
        self.assertTrue(torch.allclose(result1, result2))

    def test_matmul_grid_matrix_slow(self):
        grid = torch.randn((3, 32, 32))

        matrix = torch.eye(3, 3)
        matrix[:, 0] = torch.FloatTensor([0, -1, 0])
        matrix[:, 1] = torch.FloatTensor([1, 0, 0])

        result1 = matmul_grid_matrix_slow(grid, matrix)
        result2 = matmul_grid_matrix(grid, matrix)
        self.assertTrue(torch.allclose(result1, result2))

    MATMUL_TEST_CASES = [
        [np.eye(3, dtype=np.float32), np.eye(3, dtype=np.float32), np.ndarray],
        [np.eye(3, dtype=np.float32), torch.eye(3), torch.Tensor],
        [np.eye(3, dtype=np.float32), Matrix(torch.eye(3)), Matrix],
        [np.eye(3, dtype=np.float32), Grid(torch.randn((3, 8, 8))), Grid],
        [torch.eye(3), np.eye(3, dtype=np.float32), torch.Tensor],
        [torch.eye(3), torch.eye(3), torch.Tensor],
        [torch.eye(3), Matrix(torch.eye(3)), Matrix],
        [torch.eye(3), Grid(torch.randn((3, 8, 8))), Grid],
        [Matrix(torch.eye(3)), np.eye(3, dtype=np.float32), Matrix],
        [Matrix(torch.eye(3)), torch.eye(3), Matrix],
        [Matrix(torch.eye(3)), Matrix(torch.eye(3)), Matrix],
        [Matrix(torch.eye(3)), Grid(torch.randn((3, 8, 8))), Grid],
        [Grid(torch.randn((3, 8, 8))), np.eye(3, dtype=np.float32), Grid],
        [Grid(torch.randn((3, 8, 8))), torch.eye(3), Grid],
        [Grid(torch.randn((3, 8, 8))), Matrix(torch.eye(3)), Grid],
        [Grid(torch.randn((3, 8, 8))), Grid(torch.randn((3, 8, 8))), None],
    ]

    def _test_matmul_correct_return_type_impl(self, left, right, expected):
        if expected is None:
            with self.assertRaises(RuntimeError):
                result = matmul(left, right)
        else:
            result = matmul(left, right)
            self.assertIsInstance(result, expected)

    @parameterized.expand(MATMUL_TEST_CASES)
    def test_matmul_correct_return_type(self, left, right, expected):
        self._test_matmul_correct_return_type_impl(left, right, expected)

    # def test_all_matmul_correct_return_type(self):
    #     for case in self.MATMUL_TEST_CASES:
    #         with self.subTest(f"{case}"):
    #             self._test_matmul_correct_return_type_impl(*case)

    MATRIX_SHAPE_TESTCASES = [
        (torch.randn(2, 2), False),
        (torch.randn(3, 3), True),
        (torch.randn(4, 4), True),
        (torch.randn(5, 5), False),
        (torch.randn(3, 4), False),
        (torch.randn(4, 3), False),
        (torch.randn(3), False),
        (torch.randn(4), False),
        (torch.randn(5), False),
        (torch.randn(3, 3, 3), False),
        (torch.randn(4, 4, 4), False),
        (torch.randn(5, 5, 5), False),
    ]

    def _test_is_matrix_shaped_impl(self, matrix, expected):
        self.assertEqual(is_matrix_shaped(matrix), expected)

    @parameterized.expand(MATRIX_SHAPE_TESTCASES)
    def test_is_matrix_shaped(self, matrix, expected):
        self._test_is_matrix_shaped_impl(matrix, expected)

    # def test_all_is_matrix_shaped(self):
    #     for case in self.MATRIX_SHAPE_TESTCASES:
    #         with self.subTest(f"{case[0].shape}"):
    #             self._test_is_matrix_shaped_impl(*case)

    GRID_SHAPE_TESTCASES = [
        (torch.randn(1, 16, 32), False),
        (torch.randn(2, 16, 32), False),
        (torch.randn(3, 16, 32), True),
        (torch.randn(4, 16, 32), False),
        (torch.randn(5, 16, 32), False),
        (torch.randn(3, 16, 32, 64), False),
        (torch.randn(4, 16, 32, 64), True),
        (torch.randn(5, 16, 32, 64), False),
    ]

    def _test_is_grid_shaped_impl(self, grid, expected):
        self.assertEqual(is_grid_shaped(grid), expected)

    @parameterized.expand(GRID_SHAPE_TESTCASES)
    def test_is_grid_shaped(self, grid, expected):
        self._test_is_grid_shaped_impl(grid, expected)

    # def test_all_is_grid_shaped(self):
    #     for case in self.GRID_SHAPE_TESTCASES:
    #         with self.subTest(f"{case[0].shape}"):
    #             self._test_is_grid_shaped_impl(*case)


def get_matmul_2d_test_cases():
    f = MatrixFactory(2, TransformBackends.TORCH, "cpu")
    cases = [
        (
            f.rotate_euler(torch.pi / 4),
            f.scale((0.5, 0.5)),
            torch.FloatTensor([[0.35355339, -0.35355339, 0], [0.35355339, 0.35355339, 0], [0, 0, 1]]),
        ),
        (
            f.scale((0.5, 0.5)),
            f.rotate_euler(torch.pi / 4),
            torch.FloatTensor([[0.35355339, -0.35355339, 0], [0.35355339, 0.35355339, 0], [0, 0, 1]]),
        ),
        (f.translate((8, 8)), f.rotate_euler(torch.pi / 2), torch.FloatTensor([[0, -1, 8], [1, 0, 8], [0, 0, 1]])),
        (f.rotate_euler(torch.pi / 2), f.translate((8, 8)), torch.FloatTensor([[0, -1, -8], [1, 0, 8], [0, 0, 1]])),
    ]

    return cases


MATMUL_2D_TEST_CASES = get_matmul_2d_test_cases()


class TestMatmulOutputs(unittest.TestCase):
    def _test_matmul_outputs_impl(self, left, right, expected):
        actual = matmul(left, right)
        self.assertTrue(
            torch.allclose(actual.matrix.data, expected, atol=1e-7),
            msg=f"{actual.matrix.data} is not close to {expected}",
        )

    @parameterized.expand(MATMUL_2D_TEST_CASES)
    def test_matmul_outputs(self, left, right, expected):
        self._test_matmul_outputs_impl(left, right, expected)

    def test_all_matmul_outputs(self):
        cases = MATMUL_2D_TEST_CASES
        for case in cases:
            self._test_matmul_outputs_impl(*case)


class TestMatrixMatrixOutputs(unittest.TestCase):
    def _test_matrix_matrix_outputs_impl(self, left, right, expected):
        actual = matmul_matrix_matrix(left.matrix.data, right.matrix.data)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-7), msg=f"{actual} is not close to {expected}")

    @parameterized.expand(MATMUL_2D_TEST_CASES)
    def test_matrix_matrix_outputs(self, left, right, expected):
        self._test_matrix_matrix_outputs_impl(left, right, expected)

    def test_all_matrix_matrix_outputs(self):
        cases = MATMUL_2D_TEST_CASES
        for case in cases:
            self._test_matrix_matrix_outputs_impl(*case)


if __name__ == "__main__":
    unittest.main()
