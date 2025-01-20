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
import time
import unittest
from pathlib import Path

import nibabel
import numpy as np
import torch
from nibabel.processing import resample_to_output
from parameterized import parameterized

from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged, Orientationd, Spacingd

TESTS_PATH = Path(__file__).parents[1]
FILES = tuple(
    os.path.join(TESTS_PATH, "testing_data", filename) for filename in ("anatomical.nii", "reoriented_anat_moved.nii")
)


class TestLoadSpacingOrientation(unittest.TestCase):
    @staticmethod
    def load_image(filename):
        data = {"image": filename}
        t = Compose([LoadImaged(keys="image"), EnsureChannelFirstd(keys="image", channel_dim="no_channel")])
        return t(data)

    @parameterized.expand(FILES)
    def test_load_spacingd(self, filename):
        data_dict = self.load_image(filename)
        t = time.time()
        res_dict = Spacingd(keys="image", pixdim=(1, 0.2, 1), diagonal=True, padding_mode="zeros")(data_dict)
        t1 = time.time()
        print(f"time monai: {t1 - t}")
        anat = nibabel.Nifti1Image(np.asarray(data_dict["image"][0]), data_dict["image"].meta["original_affine"])
        ref = resample_to_output(anat, (1, 0.2, 1), order=1)
        t2 = time.time()
        print(f"time scipy: {t2 - t1}")
        self.assertGreaterEqual(t2, t1)
        np.testing.assert_allclose(res_dict["image"].affine, ref.affine)
        np.testing.assert_allclose(res_dict["image"].shape[1:], ref.shape)
        np.testing.assert_allclose(ref.get_fdata(), res_dict["image"][0], atol=0.05)

    @parameterized.expand(FILES)
    def test_load_spacingd_rotate(self, filename):
        data_dict = self.load_image(filename)
        affine = data_dict["image"].affine
        data_dict["image"].meta["original_affine"] = data_dict["image"].affine = (
            torch.tensor([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]], dtype=torch.float64) @ affine
        )
        t = time.time()
        res_dict = Spacingd(keys="image", pixdim=(1, 2, 3), diagonal=True, padding_mode="zeros")(data_dict)
        t1 = time.time()
        print(f"time monai: {t1 - t}")
        anat = nibabel.Nifti1Image(np.asarray(data_dict["image"][0]), data_dict["image"].meta["original_affine"])
        ref = resample_to_output(anat, (1, 2, 3), order=1)
        t2 = time.time()
        print(f"time scipy: {t2 - t1}")
        self.assertGreaterEqual(t2, t1)
        np.testing.assert_allclose(res_dict["image"].affine, ref.affine)
        if "anatomical" not in filename:
            np.testing.assert_allclose(res_dict["image"].shape[1:], ref.shape)
            np.testing.assert_allclose(ref.get_fdata(), res_dict["image"][0], atol=0.05)
        else:
            # different from the ref implementation (shape computed by round
            # instead of ceil)
            np.testing.assert_allclose(ref.get_fdata()[..., :-1], res_dict["image"][0], atol=0.05)

    def test_load_spacingd_non_diag(self):
        data_dict = self.load_image(FILES[1])
        affine = data_dict["image"].affine
        data_dict["image"].meta["original_affine"] = data_dict["image"].affine = (
            torch.tensor([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]], dtype=torch.float64) @ affine
        )
        res_dict = Spacingd(keys="image", pixdim=(1, 2, 3), diagonal=False, padding_mode="zeros")(data_dict)
        np.testing.assert_allclose(
            res_dict["image"].affine,
            np.array(
                [
                    [0.0, 0.0, 3.0, -27.599409],
                    [0.0, 2.0, 0.0, -47.977585],
                    [-1.0, 0.0, 0.0, 35.297897],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        )

    def test_load_spacingd_rotate_non_diag(self):
        data_dict = self.load_image(FILES[0])
        res_dict = Spacingd(keys="image", pixdim=(1, 2, 3), diagonal=False, padding_mode="border")(data_dict)
        np.testing.assert_allclose(
            res_dict["image"].affine,
            np.array([[-1.0, 0.0, 0.0, 32.0], [0.0, 2.0, 0.0, -40.0], [0.0, 0.0, 3.0, -16.0], [0.0, 0.0, 0.0, 1.0]]),
        )

    def test_load_spacingd_rotate_non_diag_ornt(self):
        data_dict = self.load_image(FILES[0])
        t = Compose(
            [
                Spacingd(keys="image", pixdim=(1, 2, 3), diagonal=False, padding_mode="border"),
                Orientationd(keys="image", axcodes="LPI"),
            ]
        )
        res_dict = t(data_dict)
        np.testing.assert_allclose(
            res_dict["image"].affine,
            np.array([[-1.0, 0.0, 0.0, 32.0], [0.0, -2.0, 0.0, 40.0], [0.0, 0.0, -3.0, 32.0], [0.0, 0.0, 0.0, 1.0]]),
        )

    def test_load_spacingd_non_diag_ornt(self):
        data_dict = self.load_image(FILES[1])
        affine = data_dict["image"].affine
        data_dict["image"].meta["original_affine"] = data_dict["image"].affine = (
            torch.tensor([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]], dtype=torch.float64) @ affine
        )
        t = Compose(
            [
                Spacingd(keys="image", pixdim=(1, 2, 3), diagonal=False, padding_mode="border"),
                Orientationd(keys="image", axcodes="LPI"),
            ]
        )
        res_dict = t(data_dict)
        np.testing.assert_allclose(
            res_dict["image"].affine,
            np.array(
                [
                    [-3.0, 0.0, 0.0, 56.4005909],
                    [0.0, -2.0, 0.0, 52.02241516],
                    [0.0, 0.0, -1.0, 35.29789734],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        )


if __name__ == "__main__":
    unittest.main()
