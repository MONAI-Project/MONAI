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

from monai.transforms import (
    create_control_grid,
    create_grid,
    create_rotate,
    create_scale,
    create_shear,
    create_translate,
)
from tests.util import assert_allclose, is_tf32_env


class TestCreateGrid(unittest.TestCase):
    def test_create_grid(self):
        with self.assertRaisesRegex(TypeError, ""):
            create_grid(None)
        with self.assertRaisesRegex(TypeError, ""):
            create_grid((1, 1), spacing=2.0)
        with self.assertRaisesRegex(TypeError, ""):
            create_grid((1, 1), spacing=2.0)

        test_assert(create_grid, ((1, 1),), np.array([[[0.0]], [[0.0]], [[1.0]]]))

        test_assert(create_grid, ((1, 1), None, False), np.array([[[0.0]], [[0.0]]]))

        test_assert(create_grid, ((1, 1), (1.2, 1.3)), np.array([[[0.0]], [[0.0]], [[1.0]]]))

        test_assert(create_grid, ((1, 1, 1), (1.2, 1.3, 1.0)), np.array([[[[0.0]]], [[[0.0]]], [[[0.0]]], [[[1.0]]]]))

        test_assert(create_grid, ((1, 1, 1), (1.2, 1.3, 1.0), False), np.array([[[[0.0]]], [[[0.0]]], [[[0.0]]]]))

        g = create_grid((1, 1, 1), spacing=(1.2, 1.3, 1.0), dtype=np.int32)
        np.testing.assert_equal(g.dtype, np.int32)

        g = create_grid((1, 1, 1), spacing=(1.2, 1.3, 1.0), dtype=torch.float64, backend="torch")
        np.testing.assert_equal(g.dtype, torch.float64)

        test_assert(
            create_grid,
            ((2, 2, 2),),
            np.array(
                [
                    [[[-0.5, -0.5], [-0.5, -0.5]], [[0.5, 0.5], [0.5, 0.5]]],
                    [[[-0.5, -0.5], [0.5, 0.5]], [[-0.5, -0.5], [0.5, 0.5]]],
                    [[[-0.5, 0.5], [-0.5, 0.5]], [[-0.5, 0.5], [-0.5, 0.5]]],
                    [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]],
                ]
            ),
        )

        test_assert(
            create_grid,
            ((2, 2, 2), (1.2, 1.3, 1.0)),
            np.array(
                [
                    [[[-0.6, -0.6], [-0.6, -0.6]], [[0.6, 0.6], [0.6, 0.6]]],
                    [[[-0.65, -0.65], [0.65, 0.65]], [[-0.65, -0.65], [0.65, 0.65]]],
                    [[[-0.5, 0.5], [-0.5, 0.5]], [[-0.5, 0.5], [-0.5, 0.5]]],
                    [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]],
                ]
            ),
        )

    def test_create_control_grid(self):
        with self.assertRaisesRegex(TypeError, ""):
            create_control_grid(None, None)
        with self.assertRaisesRegex(TypeError, ""):
            create_control_grid((1, 1), 2.0)

        test_assert(
            create_control_grid,
            ((1.0, 1.0), (1.0, 1.0)),
            np.array(
                [
                    [[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
                    [[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]],
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                ]
            ),
        )

        test_assert(
            create_control_grid,
            ((1.0, 1.0), (2.0, 2.0)),
            np.array(
                [
                    [[-2.0, -2.0, -2.0], [0.0, 0.0, 0.0], [2.0, 2.0, 2.0]],
                    [[-2.0, 0.0, 2.0], [-2.0, 0.0, 2.0], [-2.0, 0.0, 2.0]],
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                ]
            ),
        )

        test_assert(
            create_control_grid,
            ((2.0, 2.0), (1.0, 1.0)),
            np.array(
                [
                    [[-1.5, -1.5, -1.5, -1.5], [-0.5, -0.5, -0.5, -0.5], [0.5, 0.5, 0.5, 0.5], [1.5, 1.5, 1.5, 1.5]],
                    [[-1.5, -0.5, 0.5, 1.5], [-1.5, -0.5, 0.5, 1.5], [-1.5, -0.5, 0.5, 1.5], [-1.5, -0.5, 0.5, 1.5]],
                    [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
                ]
            ),
        )

        test_assert(
            create_control_grid,
            ((2.0, 2.0), (2.0, 2.0)),
            np.array(
                [
                    [[-3.0, -3.0, -3.0, -3.0], [-1.0, -1.0, -1.0, -1.0], [1.0, 1.0, 1.0, 1.0], [3.0, 3.0, 3.0, 3.0]],
                    [[-3.0, -1.0, 1.0, 3.0], [-3.0, -1.0, 1.0, 3.0], [-3.0, -1.0, 1.0, 3.0], [-3.0, -1.0, 1.0, 3.0]],
                    [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
                ]
            ),
        )

        test_assert(
            create_control_grid,
            ((1.0, 1.0, 1.0), (2.0, 2.0, 2.0), False),
            np.array(
                [
                    [
                        [[-2.0, -2.0, -2.0], [-2.0, -2.0, -2.0], [-2.0, -2.0, -2.0]],
                        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                        [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
                    ],
                    [
                        [[-2.0, -2.0, -2.0], [0.0, 0.0, 0.0], [2.0, 2.0, 2.0]],
                        [[-2.0, -2.0, -2.0], [0.0, 0.0, 0.0], [2.0, 2.0, 2.0]],
                        [[-2.0, -2.0, -2.0], [0.0, 0.0, 0.0], [2.0, 2.0, 2.0]],
                    ],
                    [
                        [[-2.0, 0.0, 2.0], [-2.0, 0.0, 2.0], [-2.0, 0.0, 2.0]],
                        [[-2.0, 0.0, 2.0], [-2.0, 0.0, 2.0], [-2.0, 0.0, 2.0]],
                        [[-2.0, 0.0, 2.0], [-2.0, 0.0, 2.0], [-2.0, 0.0, 2.0]],
                    ],
                ]
            ),
        )


def test_assert(func, params, expected):
    gpu_test = ("torch_gpu",) if torch.cuda.is_available() else ()
    for b in ("torch", "numpy") + gpu_test:
        if b == "torch_gpu":
            m = func(*params, device="cuda:0", backend="torch")
        else:
            m = func(*params, backend=b)
        assert_allclose(m, expected, type_test=False, rtol=1e-2 if is_tf32_env() else 1e-5, atol=1e-5)


class TestCreateAffine(unittest.TestCase):
    def test_create_rotate(self):
        with self.assertRaisesRegex(TypeError, ""):
            create_rotate(2, None)

        with self.assertRaisesRegex(ValueError, ""):
            create_rotate(5, 1)

        test_assert(
            create_rotate,
            (2, 1.1),
            np.array([[0.45359612, -0.89120736, 0.0], [0.89120736, 0.45359612, 0.0], [0.0, 0.0, 1.0]]),
        )
        test_assert(
            create_rotate,
            (3, 1.1),
            np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.45359612, -0.89120736, 0.0],
                    [0.0, 0.89120736, 0.45359612, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        )
        test_assert(
            create_rotate,
            (3, (1.1, 1)),
            np.array(
                [
                    [0.54030231, 0.0, 0.84147098, 0.0],
                    [0.74992513, 0.45359612, -0.48152139, 0.0],
                    [-0.38168798, 0.89120736, 0.24507903, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        )
        test_assert(
            create_rotate,
            (3, (1, 1, 1.1)),
            np.array(
                [
                    [0.24507903, -0.48152139, 0.84147098, 0.0],
                    [0.80270075, -0.38596121, -0.45464871, 0.0],
                    [0.54369824, 0.78687425, 0.29192658, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        )
        test_assert(
            create_rotate,
            (3, (0, 0, np.pi / 2)),
            np.array([[0.0, -1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
        )

    def test_create_shear(self):
        test_assert(create_shear, (2, 1.0), np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
        test_assert(create_shear, (2, (2.0, 3.0)), np.array([[1.0, 2.0, 0.0], [3.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
        test_assert(
            create_shear,
            (3, 1.0),
            np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
        )

    def test_create_scale(self):
        test_assert(create_scale, (2, 2), np.array([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
        test_assert(create_scale, (2, [2, 2, 2]), np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]]))
        test_assert(
            create_scale,
            (3, [1.5, 2.4]),
            np.array([[1.5, 0.0, 0.0, 0.0], [0.0, 2.4, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
        )
        test_assert(
            create_scale,
            (3, 1.5),
            np.array([[1.5, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
        )
        test_assert(
            create_scale,
            (3, [1, 2, 3, 4, 5]),
            np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 3.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
        )

    def test_create_translate(self):
        test_assert(create_translate, (2, 2), np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
        test_assert(create_translate, (2, [2, 2, 2]), np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]]))
        test_assert(
            create_translate,
            (3, [1.5, 2.4]),
            np.array([[1.0, 0.0, 0.0, 1.5], [0.0, 1.0, 0.0, 2.4], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
        )
        test_assert(
            create_translate,
            (3, 1.5),
            np.array([[1.0, 0.0, 0.0, 1.5], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
        )
        test_assert(
            create_translate,
            (3, [1, 2, 3, 4, 5]),
            np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 2.0], [0.0, 0.0, 1.0, 3.0], [0.0, 0.0, 0.0, 1.0]]),
        )


if __name__ == "__main__":
    unittest.main()
