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
from parameterized import parameterized

from monai.data.meta_tensor import MetaTensor
from monai.transforms.croppad.dictionary import RandWeightedCropd
from monai.transforms.lazy.functional import apply_pending
from tests.util import TEST_NDARRAYS_ALL, NumpyImageTestCase2D, NumpyImageTestCase3D, assert_allclose


def get_data(ndim):
    im_gen = NumpyImageTestCase2D() if ndim == 2 else NumpyImageTestCase3D()
    im_gen.setUp()
    return im_gen.imt[0], im_gen.seg1[0], im_gen.segn[0]


IMT_2D, SEG1_2D, SEGN_2D = get_data(ndim=2)
IMT_3D, SEG1_3D, SEGN_3D = get_data(ndim=3)

TESTS = []
for p in TEST_NDARRAYS_ALL:
    for q in TEST_NDARRAYS_ALL:
        im = IMT_2D
        weight = np.zeros_like(im)
        weight[0, 30, 17] = 1.1
        weight[0, 40, 31] = 1
        weight[0, 80, 21] = 1
        TESTS.append(
            [
                "small roi 2d",
                dict(keys="img", w_key="w", spatial_size=(10, 12), num_samples=3),
                {"img": p(im), "w": q(weight)},
                (1, 10, 12),
                [[80, 21], [30, 17], [40, 31]],
            ]
        )

        weight = np.zeros_like(im)
        weight[0, 30, 17] = 1.1
        weight[0, 40, 31] = 1
        weight[0, 80, 21] = 1
        TESTS.append(
            [
                "default roi 2d",
                dict(keys="img", w_key="w", spatial_size=(10, -1), num_samples=3),
                {"img": p(im), "w": q(weight), "others": np.nan},
                (1, 10, 64),
                [[14, 32], [105, 32], [20, 32]],
            ]
        )

        weight = np.zeros_like(im)
        weight[0, 30, 17] = 1.1
        weight[0, 10, 1] = 1
        TESTS.append(
            [
                "large roi 2d",
                dict(keys=("img", "seg"), w_key="weight", spatial_size=(10000, 400), num_samples=3),
                {"img": p(im), "seg": p(SEGN_2D), "weight": q(weight)},
                (1, 128, 64),
                [[64, 32], [64, 32], [64, 32]],
            ]
        )

        weight = np.zeros_like(im)
        weight[0, 30, 17] = np.inf
        weight[0, 10, 1] = -np.inf
        weight[0, 10, 20] = -np.nan
        TESTS.append(
            [
                "bad w roi 2d",
                dict(keys=("img", "seg"), w_key="w", spatial_size=(20, 40), num_samples=3),
                {"img": p(im), "seg": p(SEGN_2D), "w": q(weight)},
                (1, 20, 40),
                [[63, 37], [31, 43], [66, 20]],
            ]
        )

        im = IMT_3D
        weight = np.zeros_like(im)
        weight[0, 5, 30, 17] = 1.1
        weight[0, 8, 40, 31] = 1
        weight[0, 11, 23, 21] = 1
        TESTS.append(
            [
                "small roi 3d",
                dict(keys="img", w_key="w", spatial_size=(8, 10, 12), num_samples=3),
                {"img": p(im), "w": q(weight)},
                (1, 8, 10, 12),
                [[11, 23, 21], [5, 30, 17], [8, 40, 31]],
            ]
        )

        weight = np.zeros_like(im)
        weight[0, 5, 30, 17] = 1.1
        weight[0, 8, 40, 31] = 1
        weight[0, 11, 23, 21] = 1
        TESTS.append(
            [
                "default roi 3d",
                dict(keys=("img", "seg"), w_key="w", spatial_size=(10, -1, -1), num_samples=3),
                {"img": p(im), "seg": p(SEGN_3D), "w": q(weight)},
                (1, 10, 64, 80),
                [[14, 32, 40], [41, 32, 40], [20, 32, 40]],
            ]
        )

        weight = np.zeros_like(im)
        weight[0, 30, 17, 20] = 1.1
        weight[0, 10, 1, 17] = 1
        TESTS.append(
            [
                "large roi 3d",
                dict(keys="img", w_key="w", spatial_size=(10000, 400, 80), num_samples=3),
                {"img": p(im), "w": q(weight)},
                (1, 48, 64, 80),
                [[24, 32, 40], [24, 32, 40], [24, 32, 40]],
            ]
        )

        weight = np.zeros_like(im)
        weight[0, 30, 17] = np.inf
        weight[0, 10, 1] = -np.inf
        weight[0, 10, 20] = -np.nan
        TESTS.append(
            [
                "bad w roi 3d",
                dict(keys=("img", "seg"), w_key="w", spatial_size=(48, 64, 80), num_samples=3),
                {"img": p(im), "seg": p(SEGN_3D), "w": q(weight)},
                (1, 48, 64, 80),
                [[24, 32, 40], [24, 32, 40], [24, 32, 40]],
            ]
        )


class TestRandWeightedCrop(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_rand_weighted_cropd(self, _, init_params, input_data, expected_shape, expected_centers):
        crop = RandWeightedCropd(**init_params)
        crop.set_random_state(10)
        result = crop(input_data)
        self.assertEqual(len(result), init_params["num_samples"])
        _len = len(tuple(input_data.keys()))
        self.assertTupleEqual(tuple(result[0].keys())[:_len], tuple(input_data.keys()))

    @parameterized.expand(TESTS)
    def test_pending_ops(self, _, input_param, input_data, expected_shape, expected_centers):
        crop = RandWeightedCropd(**input_param)
        # non-lazy
        crop.set_random_state(10)
        expected = crop(input_data)
        self.assertIsInstance(expected[0]["img"], MetaTensor)
        # lazy
        crop.set_random_state(10)
        crop.lazy = True
        pending_result = crop(input_data)
        for i, _pending_result in enumerate(pending_result):
            self.assertIsInstance(_pending_result["img"], MetaTensor)
            assert_allclose(_pending_result["img"].peek_pending_affine(), expected[i]["img"].affine)
            assert_allclose(_pending_result["img"].peek_pending_shape(), expected[i]["img"].shape[1:])
            # only support nearest
            result = apply_pending(_pending_result["img"], overrides={"mode": "nearest", "align_corners": False})[0]
            # compare
            assert_allclose(result, expected[i]["img"], rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
