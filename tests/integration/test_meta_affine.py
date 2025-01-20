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

import os
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np
from parameterized import parameterized

from monai.data.image_writer import ITKWriter
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    EnsureChannelFirstd,
    LoadImage,
    LoadImaged,
    MapTransform,
    Orientation,
    Orientationd,
    Randomizable,
    Spacing,
    Spacingd,
    Transform,
)
from monai.utils import convert_data_type, optional_import
from tests.util import assert_allclose, download_url_or_skip_test, testing_data_config

itk, has_itk = optional_import("itk")
TINY_DIFF = 1e-4

keys = ("img1", "img2")
key, key_1 = "ref_avg152T1_LR", "ref_avg152T1_RL"
TESTS_PATH = Path(__file__).parents[1]
FILE_PATH = os.path.join(TESTS_PATH, "testing_data", f"{key}.nii.gz")
FILE_PATH_1 = os.path.join(TESTS_PATH, "testing_data", f"{key_1}.nii.gz")

TEST_CASES_ARRAY = [
    [Compose([Spacing(pixdim=(1.0, 1.1, 1.2)), Orientation(axcodes="RAS")]), {}, TINY_DIFF],
    [Compose([Orientation(axcodes="RAS"), Spacing(pixdim=(1.0, 1.1, 1.2))]), {}, TINY_DIFF],
    ["CropForeground", {"k_divisible": 3}, TINY_DIFF],
    ["BorderPad", {"spatial_border": (2, 3, 4)}, TINY_DIFF],
    ["CenterScaleCrop", {"roi_scale": (0.6, 0.7, 0.8)}, TINY_DIFF],
    ["CenterSpatialCrop", {"roi_size": (30, 200, 52)}, TINY_DIFF],
    ["DivisiblePad", {"k": 16}, TINY_DIFF],
    ["RandScaleCrop", {"roi_scale": (0.3, 0.4, 0.5)}, TINY_DIFF],
    ["RandSpatialCrop", {"roi_size": (31, 32, 33)}, TINY_DIFF],
    ["ResizeWithPadOrCrop", {"spatial_size": (50, 80, 200)}, TINY_DIFF],
    ["Spacing", {"pixdim": (1.0, 1.1, 1.2)}, TINY_DIFF],
    ["Orientation", {"axcodes": "RAS"}, TINY_DIFF],
    ["Flip", {"spatial_axis": (0, 1)}, TINY_DIFF],
    ["Resize", {"spatial_size": (100, 201, 1)}, 30.0],
    ["Rotate", {"angle": (np.pi / 3, np.pi / 2, np.pi / 4), "mode": "bilinear"}, 20.0],
    ["Zoom", {"zoom": (0.8, 0.91, 1.2)}, 20.0],
    ["Rotate90", {"k": 3}, TINY_DIFF],
    ["RandRotate90", {"prob": 1.0, "max_k": 3}, TINY_DIFF],
    ["RandRotate", {"prob": 1.0, "range_x": np.pi / 3}, 20.0],
    ["RandFlip", {"prob": 1.0}, TINY_DIFF],
    ["RandAxisFlip", {"prob": 1.0}, TINY_DIFF],
    ["RandZoom", {"prob": 1.0, "mode": "trilinear"}, TINY_DIFF],
    [
        "RandAffine",
        {
            "prob": 1.0,
            "rotate_range": (np.pi / 4, np.pi / 3, np.pi / 2),
            "translate_range": (3, 4, 5),
            "scale_range": (-0.1, 0.2),
            "spatial_size": (30, 40, 50),
            "cache_grid": True,
            "mode": "bilinear",
        },
        20.0,
    ],
    [
        "Affine",
        {
            "rotate_params": (np.pi / 4, 0.0, 0.0),
            "translate_params": (3, 4, 5),
            "spatial_size": (30, 40, 50),
            "mode": "bilinear",
            "image_only": True,
        },
        20.0,
    ],
]

TEST_CASES_DICT = [
    [Compose([Spacingd(keys, pixdim=(1.0, 1.1, 1.2)), Orientationd(keys, axcodes="LAS")]), {}, TINY_DIFF],
    [Compose([Orientationd(keys, axcodes="LAS"), Spacingd(keys, pixdim=(1.0, 1.1, 1.2))]), {}, TINY_DIFF],
    ["CropForegroundd", {"k_divisible": 3, "source_key": "img1"}, TINY_DIFF],
]
for c in TEST_CASES_ARRAY[3:-1]:  # exclude CropForegroundd and Affined
    TEST_CASES_DICT.append(deepcopy(c))
    TEST_CASES_DICT[-1][0] = TEST_CASES_DICT[-1][0] + "d"  # type: ignore


def _create_itk_obj(array, affine):
    itk_img = deepcopy(array)
    itk_img = convert_data_type(itk_img, np.ndarray)[0]
    itk_obj = ITKWriter.create_backend_obj(itk_img, channel_dim=None, affine=affine, affine_lps=True)
    return itk_obj


def _resample_to_affine(itk_obj, ref_obj):
    """linear resample"""
    dim = itk_obj.GetImageDimension()
    transform = itk.IdentityTransform[itk.D, dim].New()
    interpolator = itk.LinearInterpolateImageFunction[type(itk_obj), itk.D].New()
    resampled = itk.resample_image_filter(
        Input=itk_obj, interpolator=interpolator, transform=transform, UseReferenceImage=True, ReferenceImage=ref_obj
    )
    return resampled


@unittest.skipUnless(has_itk, "Requires itk package.")
class TestAffineConsistencyITK(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        for k, n in ((key, FILE_PATH), (key_1, FILE_PATH_1)):
            config = testing_data_config("images", f"{k}")
            download_url_or_skip_test(filepath=n, **config)

    def run_transform(self, img, xform_cls, args_dict):
        if isinstance(xform_cls, Transform):
            xform = xform_cls
            output = xform(img, **args_dict)
        else:
            if isinstance(xform_cls, str):
                xform_cls, _ = optional_import("monai.transforms", name=xform_cls)
            if issubclass(xform_cls, MapTransform):
                args_dict.update({"keys": keys})
            xform = xform_cls(**args_dict)
            if isinstance(xform, Randomizable):
                xform.set_random_state(5)
            output = xform(img)
        return output

    @parameterized.expand(TEST_CASES_ARRAY)
    def test_linear_consistent(self, xform_cls, input_dict, atol):
        """xform cls testing itk consistency"""
        img = LoadImage(image_only=True, simple_keys=True)(FILE_PATH)
        img = EnsureChannelFirst()(img)
        ref_1 = _create_itk_obj(img[0], img.affine)
        output = self.run_transform(img, xform_cls, input_dict)
        ref_2 = _create_itk_obj(output[0], output.affine)
        assert_allclose(output.pixdim, np.asarray(ref_2.GetSpacing()), type_test=False)
        expected = _resample_to_affine(ref_1, ref_2)
        # compare ref_2 and expected results from itk
        diff = np.abs(itk.GetArrayFromImage(ref_2) - itk.GetArrayFromImage(expected))
        avg_diff = np.mean(diff)

        self.assertLess(avg_diff, atol, f"{xform_cls} avg_diff: {avg_diff}, tol: {atol}")

    @parameterized.expand(TEST_CASES_DICT)
    def test_linear_consistent_dict(self, xform_cls, input_dict, atol):
        """xform cls testing itk consistency"""
        img = LoadImaged(keys, image_only=True, simple_keys=True)({keys[0]: FILE_PATH, keys[1]: FILE_PATH_1})
        img = EnsureChannelFirstd(keys)(img)
        ref_1 = {k: _create_itk_obj(img[k][0], img[k].affine) for k in keys}
        output = self.run_transform(img, xform_cls, input_dict)
        ref_2 = {k: _create_itk_obj(output[k][0], output[k].affine) for k in keys}
        expected = {k: _resample_to_affine(ref_1[k], ref_2[k]) for k in keys}
        # compare ref_2 and expected results from itk
        diff = {k: np.abs(itk.GetArrayFromImage(ref_2[k]) - itk.GetArrayFromImage(expected[k])) for k in keys}
        avg_diff = {k: np.mean(diff[k]) for k in keys}
        for k in keys:
            self.assertLess(avg_diff[k], atol, f"{xform_cls} avg_diff: {avg_diff}, tol: {atol}")


if __name__ == "__main__":
    unittest.main()
