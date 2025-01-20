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

import itertools
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from parameterized import parameterized

import monai
import monai.transforms as mt
from monai.apps import download_url
from monai.data import ITKReader
from monai.data.itk_torch_bridge import (
    get_itk_image_center,
    itk_image_to_metatensor,
    itk_to_monai_affine,
    metatensor_to_itk_image,
    monai_to_itk_affine,
    monai_to_itk_ddf,
)
from monai.networks.blocks import Warp
from monai.transforms import Affine
from monai.utils import optional_import, set_determinism
from tests.util import assert_allclose, skip_if_downloading_fails, skip_if_quick, test_is_quick, testing_data_config

itk, has_itk = optional_import("itk")
_, has_nib = optional_import("nibabel")

TESTS = ["CT_2D_head_fixed.mha", "CT_2D_head_moving.mha"]
if not test_is_quick():
    TESTS += ["copd1_highres_INSP_STD_COPD_img.nii.gz", "copd1_highres_EXP_STD_COPD_img.nii.gz"]

RW_TESTS = TESTS + ["nrrd_example.nrrd"]


@unittest.skipUnless(has_itk, "Requires `itk` package.")
class TestITKTorchAffineMatrixBridge(unittest.TestCase):
    def setUp(self):
        set_determinism(seed=0)
        self.data_dir = Path(__file__).parents[1] / "testing_data"
        self.reader = ITKReader(pixel_type=itk.F)

        for file_name in RW_TESTS:
            path = os.path.join(self.data_dir, file_name)
            if not os.path.exists(path):
                with skip_if_downloading_fails():
                    data_spec = testing_data_config("images", f"{file_name.split('.', 1)[0]}")
                    download_url(
                        data_spec["url"], path, hash_val=data_spec["hash_val"], hash_type=data_spec["hash_type"]
                    )

    def tearDown(self):
        set_determinism(seed=None)

    def create_itk_affine_from_parameters(
        self, image, translation=None, rotation=None, scale=None, shear=None, center_of_rotation=None
    ):
        """
        Creates an affine transformation for an ITK image based on the provided parameters.

        Args:
            image: The ITK image.
            translation: The translation (shift) to apply to the image.
            rotation: The rotation to apply to the image, specified as angles in radians around the x, y, and z axes.
            scale: The scaling factor to apply to the image.
            shear: The shear to apply to the image.
            center_of_rotation: The center of rotation for the image. If not specified,
                                the center of the image is used.

        Returns:
            A tuple containing the affine transformation matrix and the translation vector.
        """
        itk_transform = itk.AffineTransform[itk.D, image.ndim].New()

        # Set center
        if center_of_rotation:
            itk_transform.SetCenter(center_of_rotation)
        else:
            itk_transform.SetCenter(get_itk_image_center(image))

        # Set parameters
        if rotation:
            if image.ndim == 2:
                itk_transform.Rotate2D(rotation[0])
            else:
                for i, angle_in_rads in enumerate(rotation):
                    if angle_in_rads != 0:
                        axis = [0, 0, 0]
                        axis[i] = 1
                        itk_transform.Rotate3D(axis, angle_in_rads)

        if scale:
            itk_transform.Scale(scale)

        if shear:
            itk_transform.Shear(*shear)

        if translation:
            itk_transform.Translate(translation)

        matrix = np.asarray(itk_transform.GetMatrix(), dtype=np.float64)

        return matrix, translation

    def itk_affine_resample(self, image, matrix, translation, center_of_rotation=None, reference_image=None):
        # Translation transform
        itk_transform = itk.AffineTransform[itk.D, image.ndim].New()

        # Set center
        if center_of_rotation:
            itk_transform.SetCenter(center_of_rotation)
        else:
            itk_transform.SetCenter(get_itk_image_center(image))

        # Set matrix and translation
        itk_transform.SetMatrix(itk.matrix_from_array(matrix))
        itk_transform.Translate(translation)

        # Interpolator
        image = image.astype(itk.D)
        interpolator = itk.LinearInterpolateImageFunction.New(image)

        if not reference_image:
            reference_image = image

        # Resample with ITK
        output_image = itk.resample_image_filter(
            image, interpolator=interpolator, transform=itk_transform, output_parameters_from_image=reference_image
        )

        return np.asarray(output_image, dtype=np.float32)

    def monai_affine_resample(self, metatensor, affine_matrix):
        affine = Affine(
            affine=affine_matrix, padding_mode="zeros", mode="bilinear", dtype=torch.float64, image_only=True
        )
        output_tensor = affine(metatensor)

        return output_tensor.squeeze().permute(*torch.arange(output_tensor.ndim - 2, -1, -1)).array

    def remove_border(self, image):
        """
        MONAI seems to have different behavior in the borders of the image than ITK.
        This helper function sets the border of the ITK image as 0 (padding but keeping
        the same image size) in order to allow numerical comparison between the
        result from resampling with ITK/Elastix and resampling with MONAI.
        To use: image[:] = remove_border(image)
        Args:
            image: The ITK image to be padded.

        Returns:
            The padded array of data.
        """
        return np.pad(image[1:-1, 1:-1, 1:-1] if image.ndim == 3 else image[1:-1, 1:-1], pad_width=1)

    def itk_warp(self, image, ddf):
        """
        Warping with python itk
        Args:
            image: itk image of array shape 2D: (H, W) or 3D: (D, H, W)
            ddf: numpy array of shape 2D: (2, H, W) or 3D: (3, D, H, W)
        Returns:
            warped_image: numpy array of shape (H, W) or (D, H, W)
        """
        # MONAI -> ITK ddf
        displacement_field = monai_to_itk_ddf(image, ddf)

        # Resample using the ddf
        interpolator = itk.LinearInterpolateImageFunction.New(image)
        warped_image = itk.warp_image_filter(
            image, interpolator=interpolator, displacement_field=displacement_field, output_parameters_from_image=image
        )

        return np.asarray(warped_image)

    def monai_warp(self, image_tensor, ddf_tensor):
        """
        Warping with MONAI
        Args:
            image_tensor: torch tensor of shape 2D: (1, 1, H, W) and 3D: (1, 1, D, H, W)
            ddf_tensor: torch tensor of shape 2D: (1, 2, H, W) and 3D: (1, 3, D, H, W)
        Returns:
            warped_image: numpy array of shape (H, W) or (D, H, W)
        """
        warp = Warp(mode="bilinear", padding_mode="zeros")
        warped_image = warp(image_tensor.to(torch.float64), ddf_tensor.to(torch.float64))

        return warped_image.to(torch.float32).squeeze().numpy()

    @parameterized.expand(TESTS)
    def test_setting_affine_parameters(self, filepath):
        # Read image
        image = self.reader.read(os.path.join(self.data_dir, filepath))
        image[:] = self.remove_border(image)
        ndim = image.ndim

        # Affine parameters
        translation = [65.2, -50.2, 33.9][:ndim]
        rotation = [0.78539816339, 1.0, -0.66][:ndim]
        scale = [2.0, 1.5, 3.2][:ndim]
        shear = [0, 1, 1.6]  # axis1, axis2, coeff

        # Spacing
        spacing = np.array([1.2, 1.5, 2.0])[:ndim]
        image.SetSpacing(spacing)

        # ITK
        matrix, translation = self.create_itk_affine_from_parameters(image, translation, rotation, scale, shear)
        output_array_itk = self.itk_affine_resample(image, matrix=matrix, translation=translation)

        # MONAI
        metatensor = itk_image_to_metatensor(image)
        affine_matrix_for_monai = itk_to_monai_affine(image, matrix, translation)
        output_array_monai = self.monai_affine_resample(metatensor, affine_matrix=affine_matrix_for_monai)

        # Make sure that the array conversion of the inputs is the same
        input_array_monai = metatensor.squeeze().permute(*torch.arange(metatensor.ndim - 2, -1, -1)).array
        np.testing.assert_array_equal(input_array_monai, np.asarray(image))

        # Compare outputs
        percentage = (
            100 * np.isclose(output_array_monai, output_array_itk).sum(dtype=np.float64) / output_array_itk.size
        )
        self.assertGreaterEqual(percentage, 99.0)

    @parameterized.expand(TESTS)
    def test_arbitary_center_of_rotation(self, filepath):
        # Read image
        image = self.reader.read(os.path.join(self.data_dir, filepath))
        image[:] = self.remove_border(image)
        ndim = image.ndim

        # ITK matrix (3x3 affine matrix)
        matrix = np.array(
            [
                [0.55915995, 0.50344867, 0.43208387],
                [0.01133669, 0.82088571, 0.86841365],
                [0.30478496, 0.94998986, 0.32742505],
            ]
        )[:ndim, :ndim]
        translation = [54.0, 2.7, -11.9][:ndim]

        # Spatial properties
        center_of_rotation = [-32.3, 125.1, 0.7][:ndim]
        origin = [1.6, 0.5, 2.0][:ndim]
        spacing = np.array([1.2, 1.5, 0.6])[:ndim]

        image.SetSpacing(spacing)
        image.SetOrigin(origin)

        # ITK
        output_array_itk = self.itk_affine_resample(image, matrix, translation, center_of_rotation)

        # MONAI
        metatensor = itk_image_to_metatensor(image)
        affine_matrix_for_monai = itk_to_monai_affine(image, matrix, translation, center_of_rotation)
        output_array_monai = self.monai_affine_resample(metatensor, affine_matrix=affine_matrix_for_monai)

        # Make sure that the array conversion of the inputs is the same
        input_array_monai = metatensor.squeeze().permute(*torch.arange(metatensor.ndim - 2, -1, -1)).array
        np.testing.assert_array_equal(input_array_monai, np.asarray(image))

        # Compare outputs
        percentage = (
            100 * np.isclose(output_array_monai, output_array_itk).sum(dtype=np.float64) / output_array_itk.size
        )
        self.assertGreaterEqual(percentage, 99.0)

    @parameterized.expand(TESTS)
    def test_monai_to_itk(self, filepath):
        # Read image
        image = self.reader.read(os.path.join(self.data_dir, filepath))
        image[:] = self.remove_border(image)
        ndim = image.ndim

        # MONAI affine matrix
        affine_matrix = torch.eye(ndim + 1, dtype=torch.float64)
        affine_matrix[:ndim, :ndim] = torch.tensor(
            [
                [0.55915995, 0.50344867, 0.43208387],
                [0.01133669, 0.82088571, 0.86841365],
                [0.30478496, 0.94998986, 0.32742505],
            ],
            dtype=torch.float64,
        )[:ndim, :ndim]

        affine_matrix[:ndim, ndim] = torch.tensor([54.0, 2.7, -11.9], dtype=torch.float64)[:ndim]

        # Spatial properties
        center_of_rotation = [-32.3, 125.1, 0.7][:ndim]
        origin = [1.6, 0.5, 2.0][:ndim]
        spacing = np.array([1.2, 1.5, 0.6])[:ndim]

        image.SetSpacing(spacing)
        image.SetOrigin(origin)

        # ITK
        matrix, translation = monai_to_itk_affine(image, affine_matrix, center_of_rotation)
        output_array_itk = self.itk_affine_resample(image, matrix, translation, center_of_rotation)

        # MONAI
        metatensor = itk_image_to_metatensor(image)
        output_array_monai = self.monai_affine_resample(metatensor, affine_matrix)

        # Make sure that the array conversion of the inputs is the same
        input_array_monai = metatensor.squeeze().permute(*torch.arange(metatensor.ndim - 2, -1, -1)).array
        np.testing.assert_array_equal(input_array_monai, np.asarray(image))

        # Compare outputs
        percentage = (
            100 * np.isclose(output_array_monai, output_array_itk).sum(dtype=np.float64) / output_array_itk.size
        )
        self.assertGreaterEqual(percentage, 99.0)

    @parameterized.expand(TESTS)
    def test_cyclic_conversion(self, filepath):
        image = self.reader.read(os.path.join(self.data_dir, filepath))
        image[:] = self.remove_border(image)
        ndim = image.ndim

        # ITK matrix (3x3 affine matrix)
        matrix = np.array(
            [
                [2.90971094, 1.18297296, 2.60008784],
                [0.29416137, 0.10294283, 2.82302616],
                [1.70578374, 1.39706003, 2.54652029],
            ]
        )[:ndim, :ndim]

        translation = [-29.05463245, 35.27116398, 48.58759597][:ndim]

        # Spatial properties
        center_of_rotation = [-27.84789587, -60.7871084, 42.73501932][:ndim]
        origin = [8.10416794, 5.4831944, 0.49211025][:ndim]
        spacing = np.array([0.7, 3.2, 1.3])[:ndim]

        direction = np.array(
            [
                [1.02895588, 0.22791448, 0.02429561],
                [0.21927512, 1.28632268, -0.14932226],
                [0.47455613, 0.38534345, 0.98505633],
            ],
            dtype=np.float64,
        )
        image.SetDirection(direction[:ndim, :ndim])

        image.SetSpacing(spacing)
        image.SetOrigin(origin)

        affine_matrix = itk_to_monai_affine(image, matrix, translation, center_of_rotation)
        matrix_result, translation_result = monai_to_itk_affine(image, affine_matrix, center_of_rotation)

        meta_tensor = itk_image_to_metatensor(image)
        image_result = metatensor_to_itk_image(meta_tensor)

        np.testing.assert_allclose(matrix, matrix_result)
        np.testing.assert_allclose(translation, translation_result)
        np.testing.assert_array_equal(image.shape, image_result.shape)
        np.testing.assert_array_equal(image, image_result)

    @parameterized.expand([(2,), (3,)])
    def test_random_array(self, ndim):
        # Create image/array with random size and pixel intensities
        s = torch.randint(low=2, high=20, size=(ndim,))
        img = 100 * torch.rand((1, 1, *s.tolist()), dtype=torch.float32)

        # Pad at the edges because ITK and MONAI have different behavior there
        # during resampling
        img = torch.nn.functional.pad(img, pad=ndim * (1, 1))
        ddf = 5 * torch.rand((1, ndim, *img.shape[-ndim:]), dtype=torch.float32) - 2.5

        # Warp with MONAI
        img_resampled = self.monai_warp(img, ddf)

        # Create ITK image
        itk_img = itk.GetImageFromArray(img.squeeze().numpy())

        # Set random spacing
        spacing = 3 * np.random.rand(ndim)
        itk_img.SetSpacing(spacing)

        # Set random direction
        direction = 5 * np.random.rand(ndim, ndim) - 5
        direction = itk.matrix_from_array(direction)
        itk_img.SetDirection(direction)

        # Set random origin
        origin = 100 * np.random.rand(ndim) - 100
        itk_img.SetOrigin(origin)

        # Warp with ITK
        itk_img_resampled = self.itk_warp(itk_img, ddf.squeeze().numpy())

        # Compare
        np.testing.assert_allclose(img_resampled, itk_img_resampled, rtol=1e-2, atol=1e-2)

    @parameterized.expand(TESTS)
    @skip_if_quick
    def test_real_data(self, filepath):
        # Read image
        image = self.reader.read(os.path.join(self.data_dir, filepath))
        image[:] = self.remove_border(image)
        ndim = image.ndim

        # Random ddf
        ddf = 10 * torch.rand((1, ndim, *image.shape), dtype=torch.float32) - 10

        # Warp with MONAI
        image_tensor = torch.tensor(itk.GetArrayFromImage(image), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        img_resampled = self.monai_warp(image_tensor, ddf)

        # Warp with ITK
        itk_img_resampled = self.itk_warp(image, ddf.squeeze().numpy())

        # Compare
        np.testing.assert_allclose(img_resampled, itk_img_resampled, rtol=1e-3, atol=1e-3)

    @parameterized.expand(zip(TESTS[::2], TESTS[1::2]))
    @skip_if_quick
    def test_use_reference_space(self, ref_filepath, filepath):
        # Read the images
        image = self.reader.read(os.path.join(self.data_dir, filepath))
        image[:] = self.remove_border(image)
        ndim = image.ndim

        ref_image = self.reader.read(os.path.join(self.data_dir, ref_filepath))

        # Set arbitary origin, spacing, direction for both of the images
        image.SetSpacing([1.2, 2.0, 1.7][:ndim])
        ref_image.SetSpacing([1.9, 1.5, 1.3][:ndim])

        direction = np.array(
            [
                [1.02895588, 0.22791448, 0.02429561],
                [0.21927512, 1.28632268, -0.14932226],
                [0.47455613, 0.38534345, 0.98505633],
            ],
            dtype=np.float64,
        )
        image.SetDirection(direction[:ndim, :ndim])

        ref_direction = np.array(
            [
                [1.26032417, -0.19243174, 0.54877414],
                [0.31958275, 0.9543068, 0.2720827],
                [-0.24106769, -0.22344502, 0.9143302],
            ],
            dtype=np.float64,
        )
        ref_image.SetDirection(ref_direction[:ndim, :ndim])

        image.SetOrigin([57.3, 102.0, -20.9][:ndim])
        ref_image.SetOrigin([23.3, -0.5, 23.7][:ndim])

        # Set affine parameters
        matrix = np.array(
            [
                [0.55915995, 0.50344867, 0.43208387],
                [0.01133669, 0.82088571, 0.86841365],
                [0.30478496, 0.94998986, 0.32742505],
            ]
        )[:ndim, :ndim]
        translation = [54.0, 2.7, -11.9][:ndim]
        center_of_rotation = [-32.3, 125.1, 0.7][:ndim]

        # Resample using ITK
        output_array_itk = self.itk_affine_resample(image, matrix, translation, center_of_rotation, ref_image)

        # MONAI
        metatensor = itk_image_to_metatensor(image)
        affine_matrix_for_monai = itk_to_monai_affine(image, matrix, translation, center_of_rotation, ref_image)
        output_array_monai = self.monai_affine_resample(metatensor, affine_matrix_for_monai)

        # Compare outputs
        np.testing.assert_allclose(output_array_monai, output_array_itk, rtol=1e-3, atol=1e-3)


@unittest.skipUnless(has_itk, "Requires `itk` package.")
@unittest.skipUnless(has_nib, "Requires `nibabel` package.")
@skip_if_quick
class TestITKTorchRW(unittest.TestCase):
    def setUp(self):
        TestITKTorchAffineMatrixBridge.setUp(self)

    def tearDown(self):
        TestITKTorchAffineMatrixBridge.setUp(self)

    @parameterized.expand(list(itertools.product(RW_TESTS, ["ITKReader", "NrrdReader"], [True, False])))
    def test_rw_itk(self, filepath, reader, flip):
        """reading and convert: filepath, reader, flip"""
        print(filepath, reader, flip)
        fname = os.path.join(self.data_dir, filepath)
        xform = mt.LoadImageD("img", image_only=True, ensure_channel_first=True, affine_lps_to_ras=flip, reader=reader)
        out = xform({"img": fname})["img"]
        itk_image = metatensor_to_itk_image(out, channel_dim=0, dtype=float)
        with tempfile.TemporaryDirectory() as tempdir:
            tname = os.path.join(tempdir, filepath) + (".nii.gz" if not filepath.endswith(".nii.gz") else "")
            itk.imwrite(itk_image, tname, True)
            ref = mt.LoadImage(image_only=True, ensure_channel_first=True, reader="NibabelReader")(tname)
        if out.meta["space"] != ref.meta["space"]:
            ref.affine = monai.data.utils.orientation_ras_lps(ref.affine)
        assert_allclose(
            out.affine, monai.data.utils.to_affine_nd(len(out.affine) - 1, ref.affine), rtol=1e-3, atol=1e-3
        )


if __name__ == "__main__":
    unittest.main()
