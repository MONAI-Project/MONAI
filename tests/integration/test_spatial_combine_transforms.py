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

import monai.transforms as mt
from monai.data import create_test_image_2d, create_test_image_3d
from monai.data.meta_tensor import MetaTensor
from monai.transforms.lazy.functional import apply_pending
from monai.transforms.transform import MapTransform
from monai.utils import set_determinism
from tests.lazy_transforms_utils import get_apply_param
from tests.util import assert_allclose

TEST_2D = [
    [
        (2, 62, 61),
        [
            (mt.Spacing, {"pixdim": (1.2, 1.5), "padding_mode": "zeros", "dtype": torch.float32}),
            (mt.Orientation, {"axcodes": "RA"}),
            (mt.Resize, {"spatial_size": (64, 48), "mode": "bilinear"}),
            (mt.RandSpatialCrop, {"roi_size": (32, 32)}),
            (
                mt.RandAffine,
                {
                    "prob": 0.9,
                    "rotate_range": (np.pi / 2,),
                    "shear_range": [1, 2],
                    "translate_range": [2, 1],
                    "mode": "bilinear",
                    "padding_mode": "reflection",
                },
            ),
            (mt.RandFlip, {"prob": 0.9}),
            (mt.RandRotate, {"prob": 0.9, "range_x": np.pi / 4, "mode": "bilinear", "padding_mode": "reflection"}),
            (mt.CenterScaleCrop, {"roi_scale": (0.96, 0.8)}),
            (mt.RandZoom, {"prob": 0.9, "mode": "bilinear", "keep_size": False, "align_corners": False}),
        ],
    ],
    [
        (2, 63, 64),
        [
            (mt.CenterScaleCropd, {"roi_scale": (0.96, 0.8), "keys": "img"}),
            (mt.RandRotated, {"prob": 0.9, "range_x": np.pi / 4, "keys": "img"}),
            (
                mt.RandZoomd,
                {"prob": 0.9, "mode": "bilinear", "keep_size": False, "keys": "img", "align_corners": False},
            ),
            (mt.Spacingd, {"pixdim": (1.2, 1.5), "padding_mode": "zeros", "dtype": torch.float32, "keys": "img"}),
            (mt.RandFlipd, {"prob": 0.9, "keys": "img"}),
            (
                mt.RandAffined,
                {
                    "prob": 0.9,
                    "rotate_range": (np.pi / 2,),
                    "shear_range": [1, 2],
                    "translate_range": [2, 1],
                    "mode": "bilinear",
                    "keys": "img",
                },
            ),
            (mt.Orientationd, {"axcodes": "RA", "keys": "img"}),
            (mt.Resized, {"spatial_size": (48, 48), "mode": "bilinear", "keys": "img"}),
            (mt.RandScaleCropd, {"roi_scale": (0.4, 1.5), "random_size": False, "keys": "img"}),
        ],
    ],
]

TEST_3D = [
    [
        (2, 83, 100, 67),
        [
            (mt.Orientation, {"axcodes": "RAS"}),
            (mt.CenterScaleCrop, {"roi_scale": (1.2, 0.8, 1.0)}),
            (
                mt.RandAffine,
                {
                    "prob": 0.9,
                    "rotate_range": (np.pi / 2,),
                    "shear_range": [1, 2],
                    "translate_range": [2, 1],
                    "mode": "bilinear",
                },
            ),
            (mt.Spacing, {"pixdim": (0.9, 1.2, 1.0), "padding_mode": "zeros", "dtype": torch.float32}),
            (mt.RandSpatialCrop, {"roi_size": (36, 36, 38), "random_size": False}),
            (mt.RandZoom, {"prob": 0.9, "mode": "nearest", "keep_size": False}),
            (mt.Resize, {"spatial_size": (32, 32, 32), "mode": "nearest"}),
            (mt.RandFlip, {"prob": 0.9}),
            (mt.RandRotate, {"prob": 0.9, "range_x": np.pi / 4}),
        ],
    ],
    [
        (2, 62, 64, 72),
        [
            (mt.RandScaleCropd, {"roi_scale": (0.9, 0.7, 1.1), "random_size": False, "keys": "img"}),
            (mt.Spacingd, {"pixdim": (1.2, 1.5, 0.9), "padding_mode": "zeros", "dtype": torch.float32, "keys": "img"}),
            (mt.Orientationd, {"axcodes": "RAS", "keys": "img"}),
            (mt.Resized, {"spatial_size": (32, 32, 32), "mode": "nearest", "keys": "img"}),
            (mt.RandFlipd, {"prob": 0.9, "keys": "img"}),
            (mt.CenterScaleCropd, {"roi_scale": (0.96, 0.8, 1.25), "keys": "img"}),
            (mt.RandZoomd, {"prob": 0.9, "mode": "nearest", "keep_size": False, "keys": "img"}),
            (
                mt.RandAffined,
                {
                    "prob": 0.9,
                    "rotate_range": (np.pi / 2,),
                    "shear_range": [1, 2],
                    "translate_range": [2, 1],
                    "mode": "bilinear",
                    "keys": "img",
                },
            ),
            (mt.RandRotated, {"prob": 0.9, "range_x": np.pi / 4, "keys": "img"}),
        ],
    ],
]


class CombineLazyTest(unittest.TestCase):
    @parameterized.expand(TEST_2D + TEST_3D)
    def test_combine_transforms(self, input_shape, funcs):
        for device in ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]:
            for seed in [10, 100, 1000, 10000]:
                set_determinism(seed=seed)
                _funcs = []
                for _func, _params in funcs:
                    _funcs.append(_func(**_params))
                is_map = isinstance(_funcs[0], MapTransform)
                chns, sp_size = input_shape[0], input_shape[1:]
                imgs = []
                for _ in range(chns):
                    if len(sp_size) == 2:
                        imgs.append(create_test_image_2d(sp_size[0], sp_size[1])[0])
                    else:
                        imgs.append(create_test_image_3d(sp_size[0], sp_size[1], sp_size[2])[0])
                data = np.stack(imgs).astype(float)
                im = MetaTensor(data, meta={"a": "b", "affine": np.eye(len(input_shape))}).to(device)
                input_data = {"img": im} if is_map else im
                # non lazy
                non_lazy_result = input_data
                for _func in _funcs:
                    if isinstance(_func, mt.Randomizable):
                        _func.set_random_state(seed=seed)
                    non_lazy_result = _func(non_lazy_result)
                expected = non_lazy_result["img"] if is_map else non_lazy_result

                # lazy
                pending_result = input_data
                for _func in _funcs:
                    _func.lazy = True
                    if isinstance(_func, mt.Randomizable):
                        _func.set_random_state(seed=seed)
                    pending_result = _func(pending_result)
                pending_result = pending_result["img"] if is_map else pending_result

                assert_allclose(pending_result.peek_pending_affine(), expected.affine, atol=1e-7)
                assert_allclose(pending_result.peek_pending_shape(), expected.shape[1:4])

                # test final result
                init_param = funcs[-1][1]
                call_param = {}
                apply_param = get_apply_param(init_param, call_param)
                result = apply_pending(pending_result, overrides=apply_param)[0]

                match_ratio = np.sum(np.isclose(result.array, expected.array, atol=5e-1)) / np.prod(result.shape)
                self.assertGreater(match_ratio, 0.5)  # at least half of the images are very close


if __name__ == "__main__":
    unittest.main()
