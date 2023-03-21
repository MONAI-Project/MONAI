import unittest

import numpy as np

from monai.transforms.utils import create_rotate, create_scale, create_flip, create_rotate_90, create_shear
from monai.transforms.lazy.utils import matrix_to_eulers, check_matrix, check_axes, check_unit_translate, \
    get_scaling_factors

import torch


class TestMatrixToEulers(unittest.TestCase):

    def test_matrix_to_eulers(self):

        print(np.linalg.norm(np.asarray([1.0, 1.0, 0.0])))
        print(np.linalg.norm(np.asarray([0.0, 0.0, 0.0])))

        arr = np.asarray(np.eye(4))
        result = matrix_to_eulers(arr)
        print(result)

        arr = np.asarray(create_rotate(3, (0, 0, torch.pi)))
        result = matrix_to_eulers(arr)
        print(result)

        arr = np.asarray(create_rotate(3, (0, 0, torch.pi / 2)))
        result = matrix_to_eulers(arr)
        print(result)


class TestCheckMatrix(unittest.TestCase):

    TEST_CASES = [
        (np.eye(4), (True, True, True)),
        (create_scale(3, (1.0, 1.0, 1.0)), (True, True, True)),
        (create_scale(3, (1.1, 1.0, 1.0)), (True, False, True)),
        (create_scale(3, (1.0, 1.1, 1.0)), (True, False, True)),
        (create_scale(3, (1.0, 1.0, 1.1)), (True, False, True)),
        (create_scale(3, (0.9, 1.0, 1.0)), (True, False, True)),
        (create_scale(3, (1.0, 0.9, 1.0)), (True, False, True)),
        (create_scale(3, (1.0, 1.0, 0.9)), (True, False, True)),
        (create_scale(3, (0.9, 0.9, 0.9)), (True, False, True)),
        (create_scale(3, (1.1, 1.1, 1.1)), (True, False, True)),
        (create_flip(3, 0), (True, True, True)),
        (create_flip(3, 1), (True, True, True)),
        (create_flip(3, 2), (True, True, True)),
        (create_flip(3, (0, 1)), (True, True, True)),
        (create_flip(3, (0, 2)), (True, True, True)),
        (create_flip(3, (1, 2)), (True, True, True)),
        (create_flip(3, (0, 1, 2)), (True, True, True)),
        (create_rotate_90(3, (1, 2), 0), (True, True, True)),
        (create_rotate_90(3, (1, 2), 1), (True, True, True)),
        (create_rotate_90(3, (1, 2), 2), (True, True, True)),
        (create_rotate_90(3, (1, 2), 3), (True, True, True)),
        (create_rotate(3, (0, 0, torch.pi / 2)), (True, True, True)),
        (create_rotate(3, (0, 0, torch.pi)), (True, True, True)),
        (create_rotate(3, (0, 0, 3 * torch.pi / 2)), (True, True, True)),
        (create_rotate(3, (0, 0, 2 * torch.pi)), (True, True, True)),
        (create_rotate(3, (0, 0, torch.pi / 4)), (False, True, True)),
        (create_shear(3, 2, 0.5), (False, False, False))
    ]

    def test_check_matrix_cases(self):
        for i_c, c in enumerate(self.TEST_CASES):
            with self.subTest(i_c):
                self._test_check_matrix(*c)

    def _test_check_matrix(self, matrix, expected):
        self.assertEqual(check_matrix(matrix), expected)


def make_matrix(tvals, noise):
    matrix = np.eye(4)
    for i_t, t in enumerate(tvals):
        matrix[i_t, -1] = t + noise

    return matrix


class TestCheckUnitTranslate(unittest.TestCase):

    TEST_CASES = [
        (make_matrix((5, 4), 0.000001), (1, 16, 16), (1, 6, 6)),
        (make_matrix((5.5, 4.5), 0.000001), (1, 16, 16), (1, 5, 5)),
        (make_matrix((5.5, 4.5), 0.000001), (1, 17, 17), (1, 6, 6)),
        (make_matrix((5, 4), 0.000001), (1, 17, 17), (1, 5, 5)),
    ]

    def test_check_unit_translate_cases(self):
        for i_c, c in enumerate(self.TEST_CASES):
            with self.subTest(f"{i_c}"):
                self._test_check_unit_translate(*c)

    def _test_check_unit_translate(self, matrix, src_shape, dst_shape):
        actual = check_unit_translate(matrix, src_shape, dst_shape)
        print(actual)


class TestCheckAxes(unittest.TestCase):

    TEST_CASES = [
        (np.eye(4), (tuple(), (0, 1, 2))),
        (create_rotate(3, (0, 0, torch.pi / 2)), ((0,), (1, 0, 2))),
        (create_rotate(3, (0, torch.pi / 2, 0)), ((2,), (2, 1, 0))),
        (create_rotate(3, (torch.pi / 2, 0, 0)), ((1,), (0, 2, 1))),
        (create_rotate(3, (0, 0, torch.pi)), ((0, 1), (0, 1, 2))),
        (create_rotate(3, (0, torch.pi, 0)), ((0, 2), (0, 1, 2))),
        (create_rotate(3, (torch.pi, 0, 0)), ((1, 2), (0, 1, 2))),
    ]

    def test_check_axes_cases(self):
        for i_c, c in enumerate(self.TEST_CASES):
            with self.subTest(i_c):
                self._test_check_axes(*c)

    def _test_check_axes(self, matrix, expected):
        self.assertEqual(check_axes(matrix), expected)


class TestGetScalingFactors(unittest.TestCase):

    TEST_CASES = [
        (np.eye(4), (1, 1, 1)),
        (create_scale(3, (1.5, 2.5, 3.5)), (1.5, 2.5, 3.5)),
        (create_rotate_90(3, (0, 1), 1) @ create_scale(3, (1.5, 2.5, 3.5)), (2.5, -1.5, 3.5)),
        (create_rotate_90(3, (0, 1), 2) @ create_scale(3, (1.5, 2.5, 3.5)), (-1.5, -2.5, 3.5)),
        (create_rotate_90(3, (0, 1), 3) @ create_scale(3, (1.5, 2.5, 3.5)), (-2.5, 1.5, 3.5)),
        (create_scale(3, (1.5, 2.5, 3.5)) @ create_rotate_90(3, (0, 1), 1), (1.5, -2.5, 3.5)),
        (create_scale(3, (1.5, 2.5, 3.5)) @ create_rotate_90(3, (0, 1), 2), (-1.5, -2.5, 3.5)),
        (create_scale(3, (1.5, 2.5, 3.5)) @ create_rotate_90(3, (0, 1), 3), (-1.5, 2.5, 3.5)),
        (create_flip(3, tuple()) @ create_scale(3, (1.5, 2.5, 3.5)), (1.5, 2.5, 3.5)),
        (create_flip(3, (0,)) @ create_scale(3, (1.5, 2.5, 3.5)), (-1.5, 2.5, 3.5)),
        (create_flip(3, (1,)) @ create_scale(3, (1.5, 2.5, 3.5)), (1.5, -2.5, 3.5)),
        (create_flip(3, (2,)) @ create_scale(3, (1.5, 2.5, 3.5)), (1.5, 2.5, -3.5)),
        (create_flip(3, (0, 1)) @ create_scale(3, (1.5, 2.5, 3.5)), (-1.5, -2.5, 3.5)),
        (create_flip(3, (0, 2)) @ create_scale(3, (1.5, 2.5, 3.5)), (-1.5, 2.5, -3.5)),
        (create_flip(3, (1, 2)) @ create_scale(3, (1.5, 2.5, 3.5)), (1.5, -2.5, -3.5)),
        (create_flip(3, (0, 1, 2)) @ create_scale(3, (1.5, 2.5, 3.5)), (-1.5, -2.5, -3.5)),
        (create_scale(3, (1.5, 2.5, 3.5)) @ create_flip(3, tuple()), (1.5, 2.5, 3.5)),
        (create_scale(3, (1.5, 2.5, 3.5)) @ create_flip(3, (0,)), (-1.5, 2.5, 3.5)),
        (create_scale(3, (1.5, 2.5, 3.5)) @ create_flip(3, (1,)), (1.5, -2.5, 3.5)),
        (create_scale(3, (1.5, 2.5, 3.5)) @ create_flip(3, (2,)), (1.5, 2.5, -3.5)),
        (create_scale(3, (1.5, 2.5, 3.5)) @ create_flip(3, (0, 1)), (-1.5, -2.5, 3.5)),
        (create_scale(3, (1.5, 2.5, 3.5)) @ create_flip(3, (0, 2)), (-1.5, 2.5, -3.5)),
        (create_scale(3, (1.5, 2.5, 3.5)) @ create_flip(3, (1, 2)), (1.5, -2.5, -3.5)),
        (create_scale(3, (1.5, 2.5, 3.5)) @ create_flip(3, (0, 1, 2)), (-1.5, -2.5, -3.5)),
        (create_rotate(3, (0, 0, 0)) @ create_scale(3, (1.5, 2.5, 3.5)), (1.5, 2.5, 3.5)),
        (create_rotate(3, (0, 0, torch.pi / 2)) @ create_scale(3, (1.5, 2.5, 3.5)), (-2.5, 1.5, 3.5)),
        (create_rotate(3, (0, 0, torch.pi)) @ create_scale(3, (1.5, 2.5, 3.5)), (-1.5, -2.5, 3.5)),
        (create_rotate(3, (0, 0, 3 * torch.pi / 2)) @ create_scale(3, (1.5, 2.5, 3.5)), (2.5, -1.5, 3.5)),
    ]

    def test_get_scaling_factors(self):
        for i_c, c in enumerate(self.TEST_CASES):
            with self.subTest(i_c):
                self._test_get_scaling_factors(*c)

    def _test_get_scaling_factors(self, matrix, expected):
        actual = get_scaling_factors(matrix)
        print(actual, expected)
        self.assertEqual(actual, expected)

    def test_matrix_stuff(self):
        m1 = create_rotate_90(3, (0,1), 1)
        m2 = create_scale(3, (1.5, 2.5, 3.5))
        print(m1)
        print(m2)
        print(m2 @ m1)
