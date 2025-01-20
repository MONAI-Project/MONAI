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
from pathlib import Path

import numpy as np
import torch
from parameterized import parameterized
from torch.autograd import gradcheck

from monai.config.deviceconfig import USE_COMPILED
from monai.networks.blocks.warp import Warp
from monai.transforms import LoadImaged
from monai.utils import GridSampleMode, GridSamplePadMode
from tests.util import (
    SkipIfBeforePyTorchVersion,
    SkipIfNoModule,
    download_url_or_skip_test,
    skip_if_quick,
    testing_data_config,
)

LOW_POWER_TEST_CASES = [  # run with BUILD_MONAI=1 to test csrc/resample, BUILD_MONAI=0 to test native grid_sample
    [
        {"mode": "nearest", "padding_mode": "zeros"},
        {"image": torch.arange(4).reshape((1, 1, 2, 2)).to(dtype=torch.float), "ddf": torch.zeros(1, 2, 2, 2)},
        torch.arange(4).reshape((1, 1, 2, 2)),
    ],
    [
        {"mode": "bilinear", "padding_mode": "zeros"},
        {"image": torch.arange(4).reshape((1, 1, 2, 2)).to(dtype=torch.float), "ddf": torch.ones(1, 2, 2, 2)},
        torch.tensor([[[[3, 0], [0, 0]]]]),
    ],
    [
        {"mode": "bilinear", "padding_mode": "border"},
        {
            "image": torch.arange(8).reshape((1, 1, 2, 2, 2)).to(dtype=torch.float),
            "ddf": torch.ones(1, 3, 2, 2, 2) * -1,
        },
        torch.tensor([[[[[0, 0], [0, 0]], [[0, 0], [0, 0]]]]]),
    ],
    [
        {"mode": "bilinear", "padding_mode": "reflection"},
        {
            "image": torch.arange(8).reshape((1, 1, 2, 2, 2)).to(dtype=torch.float),
            "ddf": torch.ones(1, 3, 2, 2, 2) * -1,
        },
        torch.tensor([[[[[7.0, 6.0], [5.0, 4.0]], [[3.0, 2.0], [1.0, 0.0]]]]]),
    ],
]

CPP_TEST_CASES = [  # high order, BUILD_MONAI=1 to test csrc/resample
    [
        {"mode": 2, "padding_mode": "border"},
        {
            "image": torch.arange(8).reshape((1, 1, 2, 2, 2)).to(dtype=torch.float),
            "ddf": torch.ones(1, 3, 2, 2, 2) * -1,
        },
        torch.tensor([[[[[0.0000, 0.1250], [0.2500, 0.3750]], [[0.5000, 0.6250], [0.7500, 0.8750]]]]]),
    ],
    [
        {"mode": 2, "padding_mode": "reflection"},
        {
            "image": torch.arange(8).reshape((1, 1, 2, 2, 2)).to(dtype=torch.float),
            "ddf": torch.ones(1, 3, 2, 2, 2) * -1,
        },
        torch.tensor([[[[[5.2500, 4.7500], [4.2500, 3.7500]], [[3.2500, 2.7500], [2.2500, 1.7500]]]]]),
    ],
    [
        {"mode": 2, "padding_mode": "zeros"},
        {
            "image": torch.arange(8).reshape((1, 1, 2, 2, 2)).to(dtype=torch.float),
            "ddf": torch.ones(1, 3, 2, 2, 2) * -1,
        },
        torch.tensor([[[[[0.0000, 0.0020], [0.0039, 0.0410]], [[0.0078, 0.0684], [0.0820, 0.6699]]]]]),
    ],
    [
        {"mode": 2, "padding_mode": 7},
        {
            "image": torch.arange(8).reshape((1, 1, 2, 2, 2)).to(dtype=torch.float),
            "ddf": torch.ones(1, 3, 2, 2, 2) * -1,
        },
        torch.tensor([[[[[0.0000, 0.0020], [0.0039, 0.0410]], [[0.0078, 0.0684], [0.0820, 0.6699]]]]]),
    ],
    [
        {"mode": 3, "padding_mode": "reflection"},
        {"image": torch.arange(8).reshape((1, 1, 2, 2, 2)).to(dtype=torch.float), "ddf": torch.ones(1, 3, 2, 2, 2)},
        torch.tensor([[[[[4.6667, 4.3333], [4.0000, 3.6667]], [[3.3333, 3.0000], [2.6667, 2.3333]]]]]),
    ],
]

TEST_CASES = LOW_POWER_TEST_CASES
if USE_COMPILED:
    TEST_CASES += CPP_TEST_CASES


@skip_if_quick
class TestWarp(unittest.TestCase):
    def setUp(self):
        config = testing_data_config("images", "Prostate_T2W_AX_1")
        download_url_or_skip_test(
            url=config["url"],
            filepath=FILE_PATH,
            hash_val=config.get("hash_val"),
            hash_type=config.get("hash_type", "sha256"),
        )

    @SkipIfNoModule("itk")
    def test_itk_benchmark(self):
        img, ddf = load_img_and_sample_ddf()
        monai_result = monai_warp(img, ddf)
        itk_result = itk_warp(img, ddf)
        relative_diff = np.mean(
            np.divide(monai_result - itk_result, itk_result, out=np.zeros_like(itk_result), where=(itk_result != 0))
        )
        self.assertLess(relative_diff, 0.01)

    @parameterized.expand(TEST_CASES, skip_on_empty=True)
    def test_resample(self, input_param, input_data, expected_val):
        warp_layer = Warp(**input_param)
        result = warp_layer(**input_data)
        np.testing.assert_allclose(result.cpu().numpy(), expected_val.cpu().numpy(), rtol=1e-4, atol=1e-4)

    def test_ill_shape(self):
        warp_layer = Warp()
        with self.assertRaisesRegex(ValueError, ""):
            warp_layer(
                image=torch.arange(4).reshape((1, 1, 1, 2, 2)).to(dtype=torch.float), ddf=torch.zeros(1, 2, 2, 2)
            )
        with self.assertRaisesRegex(ValueError, ""):
            warp_layer(
                image=torch.arange(4).reshape((1, 1, 2, 2)).to(dtype=torch.float), ddf=torch.zeros(1, 2, 1, 2, 2)
            )
        with self.assertRaisesRegex(ValueError, ""):
            warp_layer(image=torch.arange(4).reshape((1, 1, 2, 2)).to(dtype=torch.float), ddf=torch.zeros(1, 2, 3, 3))

    @SkipIfBeforePyTorchVersion((1, 8))
    def test_grad(self):
        for b in GridSampleMode:
            for p in GridSamplePadMode:
                warp_layer = Warp(mode=b.value, padding_mode=p.value)
                input_image = torch.rand((2, 3, 20, 20), dtype=torch.float64) * 10.0
                ddf = torch.rand((2, 2, 20, 20), dtype=torch.float64) * 2.0
                input_image.requires_grad = True
                ddf.requires_grad = False  # Jacobian mismatch for output 0 with respect to input 1
                gradcheck(warp_layer, (input_image, ddf), atol=1e-2, eps=1e-2)


TESTS_PATH = Path(__file__).parents[1]
FILE_PATH = os.path.join(TESTS_PATH, "testing_data", "temp_" + "mri.nii")


def load_img_and_sample_ddf():
    # load image
    img = LoadImaged(keys="img")({"img": FILE_PATH})["img"]
    img = img.detach().numpy()
    # W, H, D -> D, H, W
    img = img.transpose((2, 1, 0)).copy()

    # randomly sample ddf such that maximum displacement in each direction equals to one-tenth of the image dimension in
    # that direction.
    ddf = np.random.random((3, *img.shape)).astype(np.float32)  # (3, D, H, W)
    ddf[0] = ddf[0] * img.shape[0] * 0.1
    ddf[1] = ddf[1] * img.shape[1] * 0.1
    ddf[2] = ddf[2] * img.shape[2] * 0.1
    return img, ddf


def itk_warp(img, ddf):
    """
    warping with python itk
    Args:
        img: numpy array of shape (D, H, W)
        ddf: numpy array of shape (3, D, H, W)

    Returns:
        warped_img: numpy arrap of shape (D, H, W)
    """
    import itk

    # 3, D, H, W -> D, H, W, 3
    ddf = ddf.transpose((1, 2, 3, 0))
    # x, y, z -> z, x, y
    ddf = ddf[..., ::-1]

    dimension = 3

    # initialise image
    pixel_type = itk.F  # float32
    image_type = itk.Image[pixel_type, dimension]
    itk_img = itk.PyBuffer[image_type].GetImageFromArray(img.astype(np.float32), is_vector=None)

    # initialise displacement field
    vector_component_type = itk.F
    vector_pixel_type = itk.Vector[vector_component_type, dimension]
    displacement_field_type = itk.Image[vector_pixel_type, dimension]
    displacement_field = itk.PyBuffer[displacement_field_type].GetImageFromArray(ddf.astype(np.float32), is_vector=True)

    # initialise warp_filter
    warp_filter = itk.WarpImageFilter[image_type, image_type, displacement_field_type].New()
    interpolator = itk.LinearInterpolateImageFunction[image_type, itk.D].New()
    warp_filter.SetInterpolator(interpolator)
    warp_filter.SetOutputSpacing(itk_img.GetSpacing())
    warp_filter.SetOutputOrigin(itk_img.GetOrigin())
    warp_filter.SetOutputDirection(itk_img.GetDirection())

    # warp
    warp_filter.SetDisplacementField(displacement_field)
    warp_filter.SetInput(itk_img)
    warp_filter.Update()
    warped_img = warp_filter.GetOutput()
    warped_img = np.asarray(warped_img)

    return warped_img


def monai_warp(img, ddf):
    """
    warp with MONAI
    Args:
        img: numpy array of shape (D, H, W)
        ddf: numpy array of shape (3, D, H, W)

    Returns:
        warped_img: numpy arrap of shape (D, H, W)
    """
    warp_layer = Warp(padding_mode="zeros")
    # turn to tensor and add channel dim
    monai_img = torch.tensor(img).unsqueeze(0)
    ddf = torch.tensor(ddf)
    # img -> batch -> img
    warped_img = warp_layer(monai_img.unsqueeze(0), ddf.unsqueeze(0)).squeeze(0)
    # remove channel dim
    warped_img = np.asarray(warped_img.squeeze(0))

    return warped_img


if __name__ == "__main__":
    unittest.main()
