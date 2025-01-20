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
from copy import deepcopy

import numpy as np
import torch
from parameterized import parameterized

from monai.data import MetaTensor, set_track_meta
from monai.transforms import Affine, Resize
from monai.transforms.lazy.functional import apply_pending
from monai.utils import optional_import
from tests.lazy_transforms_utils import test_resampler_lazy
from tests.util import TEST_NDARRAYS_ALL, assert_allclose, test_local_inversion

TESTS = []
for p in TEST_NDARRAYS_ALL:
    for device in [None, "cpu", "cuda"] if torch.cuda.is_available() else [None, "cpu"]:
        TESTS.append(
            [
                dict(padding_mode="zeros", device=device),
                {"img": p(np.arange(9).reshape((1, 3, 3))), "spatial_size": (-1, 0)},
                p(np.arange(9).reshape(1, 3, 3)),
            ]
        )
        TESTS.append(
            [
                dict(padding_mode="zeros", device=device, image_only=True),
                {"img": p(np.arange(9).reshape((1, 3, 3))), "spatial_size": (-1, 0)},
                p(np.arange(9).reshape(1, 3, 3)),
            ]
        )
        TESTS.append(
            [
                dict(padding_mode="zeros", device=device),
                {"img": p(np.arange(4).reshape((1, 2, 2)))},
                p(np.arange(4).reshape(1, 2, 2)),
            ]
        )
        TESTS.append(
            [
                dict(padding_mode="zeros", device=device),
                {"img": p(np.arange(4).reshape((1, 2, 2))), "spatial_size": (4, 4)},
                p(np.array([[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 2.0, 3.0, 0.0], [0.0, 0.0, 0.0, 0.0]]])),
            ]
        )
        TESTS.append(
            [
                dict(rotate_params=[np.pi / 2], padding_mode="zeros", device=device),
                {"img": p(np.arange(4).reshape((1, 2, 2))), "spatial_size": (4, 4)},
                p(np.array([[[0.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 3.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]]])),
            ]
        )
        TESTS.append(
            [
                dict(rotate_params=[np.pi / 2], padding_mode="zeros", device=device, align_corners=False),
                {"img": p(np.arange(4).reshape((1, 2, 2))), "spatial_size": (4, 4)},
                p(np.array([[[0.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 3.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]]])),
            ]
        )
        TESTS.append(
            [
                dict(
                    affine=p(torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])),
                    padding_mode="zeros",
                    device=device,
                ),
                {"img": p(np.arange(4).reshape((1, 2, 2))), "spatial_size": (4, 4)},
                p(np.array([[[0.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 3.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]]])),
            ]
        )
        TESTS.append(
            [
                dict(padding_mode="zeros", device=device),
                {"img": p(np.arange(27).reshape((1, 3, 3, 3))), "spatial_size": (-1, 0, 0)},
                p(np.arange(27).reshape(1, 3, 3, 3)),
            ]
        )
        TESTS.append(
            [
                dict(padding_mode="zeros", device=device),
                {"img": p(np.arange(8).reshape((1, 2, 2, 2))), "spatial_size": (4, 4, 4)},
                p(
                    np.array(
                        [
                            [
                                [
                                    [0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0],
                                ],
                                [
                                    [0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0, 0.0],
                                    [0.0, 2.0, 3.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0],
                                ],
                                [
                                    [0.0, 0.0, 0.0, 0.0],
                                    [0.0, 4.0, 5.0, 0.0],
                                    [0.0, 6.0, 7.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0],
                                ],
                                [
                                    [0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0],
                                ],
                            ]
                        ]
                    )
                ),
            ]
        )
        TESTS.append(
            [
                dict(rotate_params=[np.pi / 2], padding_mode="zeros", device=device),
                {"img": p(np.arange(8).reshape((1, 2, 2, 2))), "spatial_size": (4, 4, 4)},
                p(
                    np.array(
                        [
                            [
                                [
                                    [0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0],
                                ],
                                [
                                    [0.0, 0.0, 0.0, 0.0],
                                    [0.0, 2.0, 0.0, 0.0],
                                    [0.0, 3.0, 1.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0],
                                ],
                                [
                                    [0.0, 0.0, 0.0, 0.0],
                                    [0.0, 6.0, 4.0, 0.0],
                                    [0.0, 7.0, 5.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0],
                                ],
                                [
                                    [0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0],
                                ],
                            ]
                        ]
                    )
                ),
            ]
        )


class TestAffine(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_affine(self, input_param, input_data, expected_val):
        input_copy = deepcopy(input_data["img"])
        g = Affine(**input_param)
        result = g(**input_data)
        output_idx = None
        if isinstance(result, tuple):
            output_idx = 0
            result = result[output_idx]

        test_local_inversion(g, result, input_copy)
        assert_allclose(result, expected_val, rtol=1e-4, atol=1e-4, type_test=False)

        set_track_meta(False)
        result = g(**input_data)
        if isinstance(result, tuple):
            result = result[0]
        self.assertNotIsInstance(result, MetaTensor)
        self.assertIsInstance(result, torch.Tensor)
        set_track_meta(True)

        # test lazy
        lazy_input_param = input_param.copy()
        for align_corners in [True, False]:
            lazy_input_param["align_corners"] = align_corners
            resampler = Affine(**lazy_input_param)
            non_lazy_result = resampler(**input_data)
            test_resampler_lazy(resampler, non_lazy_result, lazy_input_param, input_data, output_idx=output_idx)


@unittest.skipUnless(optional_import("scipy")[1], "Requires scipy library.")
class TestAffineConsistency(unittest.TestCase):
    @parameterized.expand([[7], [8], [9]])
    def test_affine_resize(self, s):
        """s"""
        im = np.arange(4).reshape(1, 2, 2).astype(float)
        mat = np.array([[1 / s, 0, 0], [0, 1 / s, 0], [0, 0, 1]])
        sp_size = 2 * s

        def method_0(im, ac):
            xform = Affine(align_corners=ac, affine=mat, image_only=True, spatial_size=sp_size)
            xform.lazy = True
            out = xform(im)
            overrides = {"padding_mode": "border", "align_corners": ac}
            out = apply_pending(out, overrides=overrides)[0]
            return out

        def method_1(im, ac):
            xform = Affine(align_corners=ac, affine=mat, image_only=True, spatial_size=sp_size)
            xform.lazy = True
            out = xform(im)
            overrides = {"mode": 1, "padding_mode": "nearest", "align_corners": ac}
            out = apply_pending(out, overrides=overrides)[0]
            return out

        def method_2(im, ac):
            xform = Affine(align_corners=ac, affine=mat, padding_mode="border", image_only=True, spatial_size=sp_size)
            out = xform(im)
            return out

        def method_3(im, ac):
            xform = Affine(
                align_corners=ac, affine=mat, mode=1, padding_mode="nearest", image_only=True, spatial_size=sp_size
            )
            out = xform(im)
            return out

        for call in (method_0, method_1, method_2, method_3):
            for ac in (False, True):
                out = call(im, ac)
                ref = Resize(align_corners=ac, spatial_size=(sp_size, sp_size), mode="bilinear")(im)
                assert_allclose(out, ref, rtol=1e-4, atol=1e-4, type_test=False)


if __name__ == "__main__":
    unittest.main()
