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

from monai.transforms import RandAffine
from monai.utils.type_conversion import convert_data_type
from tests.lazy_transforms_utils import test_resampler_lazy
from tests.util import TEST_NDARRAYS_ALL, assert_allclose, is_tf32_env

_rtol = 1e-3 if is_tf32_env() else 1e-4

TESTS = []
for p in TEST_NDARRAYS_ALL:
    for device in [None, "cpu", "cuda"] if torch.cuda.is_available() else [None, "cpu"]:
        TESTS.append(
            [dict(device=device), {"img": p(torch.arange(27).reshape((3, 3, 3)))}, p(np.arange(27).reshape((3, 3, 3)))]
        )
        TESTS.append(
            [
                dict(device=device, spatial_size=-1),
                {"img": p(torch.arange(27).reshape((3, 3, 3)))},
                p(np.arange(27).reshape((3, 3, 3))),
            ]
        )
        TESTS.append(
            [
                dict(device=device),
                {"img": p(torch.arange(27).reshape((3, 3, 3))), "spatial_size": (2, 2)},
                p(np.array([[[2.0, 3.0], [5.0, 6.0]], [[11.0, 12.0], [14.0, 15.0]], [[20.0, 21.0], [23.0, 24.0]]])),
            ]
        )
        TESTS.append(
            [
                dict(device=device),
                {"img": p(torch.ones((1, 3, 3, 3))), "spatial_size": (2, 2, 2)},
                p(torch.ones((1, 2, 2, 2))),
            ]
        )
        TESTS.append(
            [
                dict(device=device, spatial_size=(2, 2, 2), cache_grid=True),
                {"img": p(torch.ones((1, 3, 3, 3)))},
                p(torch.ones((1, 2, 2, 2))),
            ]
        )
        TESTS.append(
            [
                dict(
                    prob=0.9,
                    rotate_range=(np.pi / 2,),
                    shear_range=[1, 2],
                    translate_range=[2, 1],
                    padding_mode="zeros",
                    spatial_size=(2, 2, 2),
                    device=device,
                ),
                {"img": p(torch.ones((1, 3, 3, 3))), "mode": "bilinear"},
                p(torch.tensor([[[[0.3658, 1.0000], [1.0000, 1.0000]], [[1.0000, 1.0000], [1.0000, 0.9333]]]])),
            ]
        )
        TESTS.append(
            [
                dict(
                    prob=0.9,
                    rotate_range=(np.pi / 2,),
                    shear_range=[1, 2],
                    translate_range=[2, 1],
                    padding_mode="zeros",
                    spatial_size=(2, 2, 2),
                    cache_grid=True,
                    device=device,
                ),
                {"img": p(torch.ones((1, 3, 3, 3))), "mode": "bilinear"},
                p(torch.tensor([[[[0.3658, 1.0000], [1.0000, 1.0000]], [[1.0000, 1.0000], [1.0000, 0.9333]]]])),
            ]
        )
        TESTS.append(
            [
                dict(
                    prob=0.9,
                    rotate_range=(np.pi / 2,),
                    shear_range=[1, 2],
                    translate_range=[2, 1],
                    scale_range=[0.1, 0.2],
                    device=device,
                ),
                {"img": p(torch.arange(64).reshape((1, 8, 8))), "spatial_size": (3, 3)},
                p(
                    torch.tensor(
                        [[[18.7362, 15.5820, 12.4278], [27.3988, 24.2446, 21.0904], [36.0614, 32.9072, 29.7530]]]
                    )
                ),
            ]
        )
        TESTS.append(
            [
                dict(
                    prob=0.9,
                    rotate_range=(np.pi / 2,),
                    shear_range=[1, 2],
                    translate_range=[2, 1],
                    scale_range=[0.1, 0.2],
                    spatial_size=(3, 3),
                    cache_grid=True,
                    device=device,
                ),
                {"img": p(torch.arange(64).reshape((1, 8, 8)))},
                p(
                    torch.tensor(
                        [[[18.7362, 15.5820, 12.4278], [27.3988, 24.2446, 21.0904], [36.0614, 32.9072, 29.7530]]]
                    )
                ),
            ]
        )

TEST_CASES_SKIPPED_CONSISTENCY = []
for p in TEST_NDARRAYS_ALL:
    for in_dtype in (np.int32, np.float32):
        TEST_CASES_SKIPPED_CONSISTENCY.append((p(np.arange(9 * 10).reshape(1, 9, 10)), in_dtype))

TEST_RANDOMIZE = []
for cache_grid in (False, True):
    for initial_randomize in (False, True):
        TEST_RANDOMIZE.append((initial_randomize, cache_grid))


class TestRandAffine(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_rand_affine(self, input_param, input_data, expected_val):
        g = RandAffine(**input_param)
        g.set_random_state(123)
        result = g(**input_data)
        g.rand_affine_grid.affine = torch.eye(4, dtype=torch.float64)  # reset affine
        test_resampler_lazy(g, result, input_param, input_data, seed=123, rtol=_rtol)
        if input_param.get("cache_grid", False):
            self.assertTrue(g._cached_grid is not None)
        assert_allclose(result, expected_val, rtol=_rtol, atol=1e-4, type_test="tensor")

    @parameterized.expand([(None,), ((1, 1, -1),)])
    def test_ill_cache(self, spatial_size):
        with self.assertWarns(UserWarning):
            RandAffine(cache_grid=True, spatial_size=spatial_size)

    @parameterized.expand(TEST_CASES_SKIPPED_CONSISTENCY)
    def test_skipped_transform_consistency(self, im, in_dtype):
        t1 = RandAffine(prob=0)
        t2 = RandAffine(prob=1, spatial_size=(10, 11))

        im, *_ = convert_data_type(im, dtype=in_dtype)

        out1 = t1(im)
        out2 = t2(im)

        # check same type
        self.assertEqual(type(out1), type(out2))
        # check matching dtype
        self.assertEqual(out1.dtype, out2.dtype)

    @parameterized.expand(TEST_RANDOMIZE)
    def test_no_randomize(self, initial_randomize, cache_grid):
        rand_affine = RandAffine(
            prob=1,
            rotate_range=(np.pi / 6, 0, 0),
            translate_range=((-2, 2), (-2, 2), (-2, 2)),
            scale_range=((-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1)),
            spatial_size=(16, 16, 16),
            cache_grid=cache_grid,
            padding_mode="zeros",
        )
        if initial_randomize:
            rand_affine.randomize(None)

        arr = torch.randn((1, 16, 16, 16)) * 100

        arr1 = rand_affine(arr, randomize=False)
        m1 = rand_affine.rand_affine_grid.get_transformation_matrix()

        arr2 = rand_affine(arr, randomize=False)
        m2 = rand_affine.rand_affine_grid.get_transformation_matrix()

        assert_allclose(m1, m2)
        assert_allclose(arr1, arr2)


if __name__ == "__main__":
    unittest.main()
