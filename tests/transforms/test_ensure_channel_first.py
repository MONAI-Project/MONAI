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
import tempfile
import unittest

import nibabel as nib
import numpy as np
import torch
from parameterized import parameterized
from PIL import Image

from monai.data.meta_tensor import MetaTensor
from monai.transforms import EnsureChannelFirst, LoadImage
from monai.utils import optional_import

itk, has_itk = optional_import("itk", allow_namespace_pkg=True)
ITKReader, _ = optional_import("monai.data", name="ITKReader", as_type="decorator")

TEST_CASE_1 = [{}, ["test_image.nii.gz"], None]

TEST_CASE_2 = [{}, ["test_image.nii.gz"], -1]

TEST_CASE_3 = [{}, ["test_image.nii.gz", "test_image2.nii.gz", "test_image3.nii.gz"], None]

TEST_CASE_4 = [{"reader": ITKReader() if has_itk else "itkreader"}, ["test_image.nii.gz"], None]

TEST_CASE_5 = [{"reader": ITKReader() if has_itk else "itkreader"}, ["test_image.nii.gz"], -1]

TEST_CASE_6 = [
    {"reader": ITKReader() if has_itk else "itkreader"},
    ["test_image.nii.gz", "test_image2.nii.gz", "test_image3.nii.gz"],
    None,
]


class TestEnsureChannelFirst(unittest.TestCase):

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4, TEST_CASE_5, TEST_CASE_6])
    @unittest.skipUnless(has_itk, "itk not installed")
    def test_load_nifti(self, input_param, filenames, original_channel_dim):
        # if original_channel_dim is None
        test_image = np.random.rand(8, 8, 8)

        if original_channel_dim == -1:
            test_image = np.random.rand(8, 8, 8, 1)

        with tempfile.TemporaryDirectory() as tempdir:
            for i, name in enumerate(filenames):
                filenames[i] = os.path.join(tempdir, name)
                nib.save(nib.Nifti1Image(test_image, np.eye(4)), filenames[i])

            result = LoadImage(image_only=True, **input_param)(filenames)
            result = EnsureChannelFirst()(result)
            self.assertEqual(result.shape[0], len(filenames))

    @unittest.skipUnless(has_itk, "itk not installed")
    def test_itk_dicom_series_reader(self):
        filenames = "tests/testing_data/CT_DICOM"
        itk.ProcessObject.SetGlobalWarningDisplay(False)
        result = LoadImage(image_only=True, reader=ITKReader(pixel_type=itk.UC))(filenames)
        result = EnsureChannelFirst()(result)
        self.assertEqual(result.shape[0], 1)

    def test_load_png(self):
        spatial_size = (6, 6, 3)
        test_image = np.random.randint(0, 6, size=spatial_size)
        with tempfile.TemporaryDirectory() as tempdir:
            filename = os.path.join(tempdir, "test_image.png")
            Image.fromarray(test_image.astype("uint8")).save(filename)
            result = LoadImage(image_only=True)(filename)
            result = EnsureChannelFirst()(result)
            self.assertEqual(result.shape[0], 3)
            result = EnsureChannelFirst(channel_dim=-1)(result)
            self.assertEqual(result.shape, (6, 3, 6))

    def test_check(self):
        im = torch.zeros(1, 2, 3)
        im_nodim = MetaTensor(im, meta={"original_channel_dim": None})

        with self.assertRaises(ValueError):  # not MetaTensor
            EnsureChannelFirst(channel_dim=None)(im)
        with self.assertRaises(ValueError):  # no meta
            EnsureChannelFirst(channel_dim=None)(MetaTensor(im))
        with self.assertRaises(ValueError):  # no meta channel
            EnsureChannelFirst()(im_nodim)

        with self.assertWarns(Warning):
            EnsureChannelFirst(strict_check=False, channel_dim=None)(im)

        with self.assertWarns(Warning):
            EnsureChannelFirst(strict_check=False, channel_dim=None)(im_nodim)

    def test_default_channel_first(self):
        im = torch.rand(4, 4)
        result = EnsureChannelFirst(channel_dim="no_channel")(im)

        self.assertEqual(result.shape, (1, 4, 4))


if __name__ == "__main__":
    unittest.main()
