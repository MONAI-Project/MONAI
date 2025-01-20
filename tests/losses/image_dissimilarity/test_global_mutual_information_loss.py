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

from monai import transforms
from monai.losses.image_dissimilarity import GlobalMutualInformationLoss
from tests.util import SkipIfBeforePyTorchVersion, download_url_or_skip_test, skip_if_quick, testing_data_config

device = "cuda:0" if torch.cuda.is_available() else "cpu"

TESTS_PATH = Path(__file__).parents[1]
FILE_PATH = os.path.join(TESTS_PATH, "testing_data", "temp_" + "mri.nii")

EXPECTED_VALUE = {
    "xyz_translation": [
        -1.5860257,
        -0.62433463,
        -0.38217825,
        -0.2905613,
        -0.23233329,
        -0.1961407,
        -0.16905619,
        -0.15100679,
        -0.13666219,
        -0.12635908,
    ],
    "xyz_rotation": [
        -1.5860257,
        -0.30265224,
        -0.18666176,
        -0.15887907,
        -0.1625064,
        -0.16603896,
        -0.19222091,
        -0.18158069,
        -0.167644,
        -0.16698098,
    ],
}


@skip_if_quick
class TestGlobalMutualInformationLoss(unittest.TestCase):
    def setUp(self):
        config = testing_data_config("images", "Prostate_T2W_AX_1")
        download_url_or_skip_test(
            url=config["url"],
            filepath=FILE_PATH,
            hash_val=config.get("hash_val"),
            hash_type=config.get("hash_type", "sha256"),
        )

    @SkipIfBeforePyTorchVersion((1, 9))
    def test_bspline(self):
        loss_fn = GlobalMutualInformationLoss(kernel_type="b-spline", num_bins=32, sigma_ratio=0.015)

        transform_params_dict = {
            "xyz_translation": [(i, i, i) for i in range(10)],
            "xyz_rotation": [(np.pi / 100 * i, np.pi / 100 * i, np.pi / 100 * i) for i in range(10)],
        }

        def transformation(translate_params=(0.0, 0.0, 0.0), rotate_params=(0.0, 0.0, 0.0)):
            """
            Read and transform Prostate_T2W_AX_1.nii
            Args:
                translate_params: a tuple of 3 floats, translation is in pixel/voxel relative to the center of the input
                        image. Defaults to no translation.
                rotate_params: a rotation angle in radians, a tuple of 3 floats for 3D.
                        Defaults to no rotation.
            Returns:
                numpy array of shape HWD
            """
            transform_list = [
                transforms.LoadImaged(keys="img", image_only=True),
                transforms.Affined(
                    keys="img",
                    translate_params=translate_params,
                    rotate_params=rotate_params,
                    device=None,
                    padding_mode="border",
                ),
                transforms.NormalizeIntensityd(keys=["img"]),
            ]
            transformation = transforms.Compose(transform_list)
            return transformation({"img": FILE_PATH})["img"]

        a1 = transformation()
        a1 = a1.clone().unsqueeze(0).unsqueeze(0).to(device)

        for mode in transform_params_dict:
            transform_params_list = transform_params_dict[mode]
            expected_value_list = EXPECTED_VALUE[mode]
            for transform_params, expected_value in zip(transform_params_list, expected_value_list):
                a2 = transformation(
                    translate_params=transform_params if "translation" in mode else (0.0, 0.0, 0.0),
                    rotate_params=transform_params if "rotation" in mode else (0.0, 0.0, 0.0),
                )
                a2 = a2.clone().unsqueeze(0).unsqueeze(0).to(device)
                result = loss_fn(a2, a1).detach().cpu().numpy()
                np.testing.assert_allclose(result, expected_value, rtol=0.08, atol=0.05)


class TestGlobalMutualInformationLossIll(unittest.TestCase):
    @parameterized.expand(
        [
            (torch.ones((1, 2), dtype=torch.float), torch.ones((1, 3), dtype=torch.float)),  # mismatched_simple_dims
            (
                torch.ones((1, 3, 3), dtype=torch.float),
                torch.ones((1, 3), dtype=torch.float),
            ),  # mismatched_advanced_dims
        ]
    )
    def test_ill_shape(self, input1, input2):
        loss = GlobalMutualInformationLoss()
        with self.assertRaises(ValueError):
            loss.forward(input1, input2)

    @parameterized.expand(
        [
            (0, "mean", ValueError, ""),  # num_bins_zero
            (-1, "mean", ValueError, ""),  # num_bins_negative
            (64, "unknown", ValueError, ""),  # reduction_unknown
            (64, None, ValueError, ""),  # reduction_none
        ]
    )
    def test_ill_opts(self, num_bins, reduction, expected_exception, expected_message):
        pred = torch.ones((1, 3, 3, 3, 3), dtype=torch.float, device=device)
        target = torch.ones((1, 3, 3, 3, 3), dtype=torch.float, device=device)
        with self.assertRaisesRegex(expected_exception, expected_message):
            GlobalMutualInformationLoss(num_bins=num_bins, reduction=reduction)(pred, target)


if __name__ == "__main__":
    unittest.main()
