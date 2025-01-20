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
import shutil
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from parameterized import parameterized
from PIL import Image

from monai.apps import download_and_extract
from monai.data import NibabelReader, PydicomReader
from monai.data.meta_obj import set_track_meta
from monai.data.meta_tensor import MetaTensor
from monai.transforms import LoadImage
from monai.utils import optional_import
from tests.utils import SkipIfNoModule, assert_allclose, skip_if_downloading_fails, testing_data_config

itk, has_itk = optional_import("itk", allow_namespace_pkg=True)
ITKReader, _ = optional_import("monai.data", name="ITKReader", as_type="decorator")
itk_uc, _ = optional_import("itk", name="UC", allow_namespace_pkg=True)


class _MiniReader:
    """a test case customised reader"""

    def __init__(self, is_compatible=False):
        self.is_compatible = is_compatible

    def verify_suffix(self, _name):
        return self.is_compatible

    def read(self, name):
        return name

    def get_data(self, _obj):
        return np.zeros((1, 1, 1)), {"name": "my test"}


TEST_CASE_1 = [{}, ["test_image.nii.gz"], (128, 128, 128)]

TEST_CASE_2 = [{}, ["test_image.nii.gz"], (128, 128, 128)]

TEST_CASE_3 = [{}, ["test_image.nii.gz", "test_image2.nii.gz", "test_image3.nii.gz"], (3, 128, 128, 128)]

TEST_CASE_3_1 = [  # .mgz format
    {"reader": "nibabelreader"},
    ["test_image.mgz", "test_image2.mgz", "test_image3.mgz"],
    (3, 128, 128, 128),
]

TEST_CASE_4 = [{}, ["test_image.nii.gz", "test_image2.nii.gz", "test_image3.nii.gz"], (3, 128, 128, 128)]

TEST_CASE_4_1 = [  # additional parameter
    {"mmap": False},
    ["test_image.nii.gz", "test_image2.nii.gz", "test_image3.nii.gz"],
    (3, 128, 128, 128),
]

TEST_CASE_5 = [{"reader": NibabelReader(mmap=False)}, ["test_image.nii.gz"], (128, 128, 128)]

TEST_CASE_GPU_1 = [{"reader": "nibabelreader", "to_gpu": True}, ["test_image.nii.gz"], (128, 128, 128)]

TEST_CASE_GPU_2 = [{"reader": "nibabelreader", "to_gpu": True}, ["test_image.nii"], (128, 128, 128)]

TEST_CASE_GPU_3 = [
    {"reader": "nibabelreader", "to_gpu": True},
    ["test_image.nii", "test_image2.nii", "test_image3.nii"],
    (3, 128, 128, 128),
]

TEST_CASE_GPU_4 = [
    {"reader": "nibabelreader", "to_gpu": True},
    ["test_image.nii.gz", "test_image2.nii.gz", "test_image3.nii.gz"],
    (3, 128, 128, 128),
]

TEST_CASE_6 = [{"reader": ITKReader() if has_itk else "itkreader"}, ["test_image.nii.gz"], (128, 128, 128)]

TEST_CASE_7 = [{"reader": ITKReader() if has_itk else "itkreader"}, ["test_image.nii.gz"], (128, 128, 128)]

TEST_CASE_8 = [
    {"reader": ITKReader() if has_itk else "itkreader"},
    ["test_image.nii.gz", "test_image2.nii.gz", "test_image3.nii.gz"],
    (3, 128, 128, 128),
]

TEST_CASE_8_1 = [
    {"reader": ITKReader(channel_dim=0) if has_itk else "itkreader"},
    ["test_image.nii.gz", "test_image2.nii.gz", "test_image3.nii.gz"],
    (384, 128, 128),
]

TEST_CASE_9 = [
    {"reader": ITKReader() if has_itk else "itkreader"},
    ["test_image.nii.gz", "test_image2.nii.gz", "test_image3.nii.gz"],
    (3, 128, 128, 128),
]

TEST_CASE_10 = [
    {"reader": ITKReader(pixel_type=itk_uc) if has_itk else "itkreader"},
    "tests/testing_data/CT_DICOM",
    (16, 16, 4),
    (16, 16, 4),
]

TEST_CASE_11 = [{"reader": "ITKReader", "pixel_type": itk_uc}, "tests/testing_data/CT_DICOM", (16, 16, 4), (16, 16, 4)]

TEST_CASE_12 = [
    {"reader": "ITKReader", "pixel_type": itk_uc, "reverse_indexing": True},
    "tests/testing_data/CT_DICOM",
    (16, 16, 4),
    (4, 16, 16),
]

TEST_CASE_13 = [{"reader": "nibabelreader", "channel_dim": 0}, "test_image.nii.gz", (3, 128, 128, 128)]

TEST_CASE_14 = [
    {"reader": "nibabelreader", "channel_dim": -1, "ensure_channel_first": True},
    "test_image.nii.gz",
    (128, 128, 128, 3),
]

TEST_CASE_15 = [{"reader": "nibabelreader", "channel_dim": 2}, "test_image.nii.gz", (128, 128, 3, 128)]

TEST_CASE_16 = [{"reader": "itkreader", "channel_dim": 0}, "test_image.nii.gz", (3, 128, 128, 128)]

TEST_CASE_17 = [{"reader": "monai.data.ITKReader", "channel_dim": -1}, "test_image.nii.gz", (128, 128, 128, 3)]

TEST_CASE_18 = [
    {"reader": "ITKReader", "channel_dim": 2, "ensure_channel_first": True},
    "test_image.nii.gz",
    (128, 128, 3, 128),
]

# test same dicom data with PydicomReader
TEST_CASE_19 = [{"reader": PydicomReader()}, "tests/testing_data/CT_DICOM", (16, 16, 4), (16, 16, 4)]

TEST_CASE_20 = [
    {"reader": "PydicomReader", "ensure_channel_first": True, "force": True},
    "tests/testing_data/CT_DICOM",
    (16, 16, 4),
    (1, 16, 16, 4),
]

TEST_CASE_21 = [
    {"reader": "PydicomReader", "affine_lps_to_ras": True, "defer_size": "2 MB", "force": True},
    "tests/testing_data/CT_DICOM",
    (16, 16, 4),
    (16, 16, 4),
]

# test reader consistency between PydicomReader and ITKReader on dicom data
TEST_CASE_22 = ["tests/testing_data/CT_DICOM"]

TESTS_META = []
for track_meta in (False, True):
    TESTS_META.append([{}, (128, 128, 128), track_meta])
    TESTS_META.append([{"reader": "ITKReader", "fallback_only": False}, (128, 128, 128), track_meta])


@unittest.skipUnless(has_itk, "itk not installed")
class TestLoadImage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super(__class__, cls).setUpClass()
        with skip_if_downloading_fails():
            cls.tmpdir = tempfile.mkdtemp()
            key = "DICOM_single"
            url = testing_data_config("images", key, "url")
            hash_type = testing_data_config("images", key, "hash_type")
            hash_val = testing_data_config("images", key, "hash_val")
            download_and_extract(
                url=url, output_dir=cls.tmpdir, hash_val=hash_val, hash_type=hash_type, file_type="zip"
            )
            cls.data_dir = os.path.join(cls.tmpdir, "CT_DICOM_SINGLE")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir)
        super(__class__, cls).tearDownClass()

    @parameterized.expand(
        [TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_3_1, TEST_CASE_4, TEST_CASE_4_1, TEST_CASE_5]
    )
    def test_nibabel_reader(self, input_param, filenames, expected_shape):
        test_image = np.random.rand(128, 128, 128)
        with tempfile.TemporaryDirectory() as tempdir:
            for i, name in enumerate(filenames):
                filenames[i] = os.path.join(tempdir, name)
                nib.save(nib.Nifti1Image(test_image, np.eye(4)), filenames[i])
            result = LoadImage(image_only=True, **input_param)(filenames)
            ext = "".join(Path(name).suffixes)
            self.assertEqual(result.meta["filename_or_obj"], os.path.join(tempdir, "test_image" + ext))
            self.assertEqual(result.meta["space"], "RAS")
            assert_allclose(result.affine, torch.eye(4))
            self.assertTupleEqual(result.shape, expected_shape)

    @SkipIfNoModule("nibabel")
    @SkipIfNoModule("cupy")
    @SkipIfNoModule("kvikio")
    @parameterized.expand([TEST_CASE_GPU_1, TEST_CASE_GPU_2, TEST_CASE_GPU_3, TEST_CASE_GPU_4])
    def test_nibabel_reader_gpu(self, input_param, filenames, expected_shape):
        if torch.__version__.endswith("nv24.8"):
            # related issue: https://github.com/Project-MONAI/MONAI/issues/8274
            # for this version, use randint test case to avoid the issue
            test_image = torch.randint(0, 256, (128, 128, 128), dtype=torch.uint8).numpy()
        else:
            test_image = np.random.rand(128, 128, 128)
        with tempfile.TemporaryDirectory() as tempdir:
            for i, name in enumerate(filenames):
                filenames[i] = os.path.join(tempdir, name)
                nib.save(nib.Nifti1Image(test_image, np.eye(4)), filenames[i])
            result = LoadImage(image_only=True, **input_param)(filenames)
            ext = "".join(Path(name).suffixes)
            self.assertEqual(result.meta["filename_or_obj"], os.path.join(tempdir, "test_image" + ext))
            self.assertEqual(result.meta["space"], "RAS")
            assert_allclose(result.affine, torch.eye(4))
            self.assertTupleEqual(result.shape, expected_shape)

            # verify gpu and cpu loaded data are the same
            input_param_cpu = input_param.copy()
            input_param_cpu["to_gpu"] = False
            result_cpu = LoadImage(image_only=True, **input_param_cpu)(filenames)
            assert_allclose(result_cpu, result.cpu(), atol=1e-6)

    @parameterized.expand([TEST_CASE_6, TEST_CASE_7, TEST_CASE_8, TEST_CASE_8_1, TEST_CASE_9])
    def test_itk_reader(self, input_param, filenames, expected_shape):
        test_image = np.random.rand(128, 128, 128)
        with tempfile.TemporaryDirectory() as tempdir:
            for i, name in enumerate(filenames):
                filenames[i] = os.path.join(tempdir, name)
                itk_np_view = itk.image_view_from_array(test_image)
                itk.imwrite(itk_np_view, filenames[i])
            result = LoadImage(image_only=True, **input_param)(filenames)
            self.assertEqual(result.meta["filename_or_obj"], os.path.join(tempdir, "test_image.nii.gz"))
            diag = torch.as_tensor(np.diag([-1, -1, 1, 1]))
            np.testing.assert_allclose(result.affine, diag)
            self.assertTupleEqual(result.shape, expected_shape)

    @parameterized.expand([TEST_CASE_10, TEST_CASE_11, TEST_CASE_12, TEST_CASE_19, TEST_CASE_20, TEST_CASE_21])
    def test_itk_dicom_series_reader(self, input_param, filenames, expected_shape, expected_np_shape):
        result = LoadImage(image_only=True, **input_param)(filenames)
        self.assertEqual(result.meta["filename_or_obj"], f"{Path(filenames)}")
        assert_allclose(
            result.affine,
            torch.tensor(
                [
                    [-0.488281, 0.0, 0.0, 125.0],
                    [0.0, -0.488281, 0.0, 128.100006],
                    [0.0, 0.0, 68.33333333, -99.480003],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        )
        self.assertTupleEqual(result.shape, expected_np_shape)

    def test_no_files(self):
        with self.assertRaisesRegex(RuntimeError, "list index out of range"):  # fname_regex excludes everything
            LoadImage(image_only=True, reader="PydicomReader", fname_regex=r"^(?!.*).*")("tests/testing_data/CT_DICOM")
        LoadImage(image_only=True, reader="PydicomReader", fname_regex=None)("tests/testing_data/CT_DICOM")

    def test_itk_dicom_series_reader_single(self):
        result = LoadImage(image_only=True, reader="ITKReader")(self.data_dir)
        self.assertEqual(result.meta["filename_or_obj"], f"{Path(self.data_dir)}")
        assert_allclose(
            result.affine,
            torch.tensor(
                [
                    [-0.488281, 0.0, 0.0, 125.0],
                    [0.0, -0.488281, 0.0, 128.100006],
                    [0.0, 0.0, 1.0, -99.480003],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        )
        self.assertTupleEqual(result.shape, (16, 16, 1))

    def test_itk_reader_multichannel(self):
        test_image = np.random.randint(0, 256, size=(256, 224, 3)).astype("uint8")
        with tempfile.TemporaryDirectory() as tempdir:
            filename = os.path.join(tempdir, "test_image.png")
            itk_np_view = itk.image_view_from_array(test_image, is_vector=True)
            itk.imwrite(itk_np_view, filename)
            for flag in (False, True):
                result = LoadImage(image_only=True, reader=ITKReader(reverse_indexing=flag))(Path(filename))
                test_image = test_image.transpose(1, 0, 2)
                np.testing.assert_allclose(result[:, :, 0], test_image[:, :, 0])
                np.testing.assert_allclose(result[:, :, 1], test_image[:, :, 1])
                np.testing.assert_allclose(result[:, :, 2], test_image[:, :, 2])

    @parameterized.expand([TEST_CASE_22])
    def test_dicom_reader_consistency(self, filenames):
        itk_param = {"reader": "ITKReader"}
        pydicom_param = {"reader": "PydicomReader"}
        for affine_flag in [True, False]:
            itk_param["affine_lps_to_ras"] = affine_flag
            pydicom_param["affine_lps_to_ras"] = affine_flag
            itk_result = LoadImage(image_only=True, **itk_param)(filenames)
            pydicom_result = LoadImage(image_only=True, **pydicom_param)(filenames)
            np.testing.assert_allclose(pydicom_result, itk_result)
            np.testing.assert_allclose(pydicom_result.affine, itk_result.affine)

    def test_dicom_reader_consistency_single(self):
        itk_param = {"reader": "ITKReader"}
        pydicom_param = {"reader": "PydicomReader"}
        for affine_flag in [True, False]:
            itk_param["affine_lps_to_ras"] = affine_flag
            pydicom_param["affine_lps_to_ras"] = affine_flag
            itk_result = LoadImage(image_only=True, **itk_param)(self.data_dir)
            pydicom_result = LoadImage(image_only=True, **pydicom_param)(self.data_dir)
            np.testing.assert_allclose(pydicom_result, itk_result.squeeze())
            np.testing.assert_allclose(pydicom_result.affine, itk_result.affine)

    def test_load_nifti_multichannel(self):
        test_image = np.random.randint(0, 256, size=(31, 64, 16, 2)).astype(np.float32)
        with tempfile.TemporaryDirectory() as tempdir:
            filename = os.path.join(tempdir, "test_image.nii.gz")
            itk_np_view = itk.image_view_from_array(test_image, is_vector=True)
            itk.imwrite(itk_np_view, filename)

            itk_img = LoadImage(image_only=True, reader=ITKReader())(Path(filename))
            self.assertTupleEqual(tuple(itk_img.shape), (16, 64, 31, 2))

            nib_image = LoadImage(image_only=True, reader=NibabelReader(squeeze_non_spatial_dims=True))(Path(filename))
            self.assertTupleEqual(tuple(nib_image.shape), (16, 64, 31, 2))

            np.testing.assert_allclose(itk_img, nib_image, atol=1e-3, rtol=1e-3)

    def test_load_png(self):
        spatial_size = (256, 224)
        test_image = np.random.randint(0, 256, size=spatial_size)
        with tempfile.TemporaryDirectory() as tempdir:
            filename = os.path.join(tempdir, "test_image.png")
            Image.fromarray(test_image.astype("uint8")).save(filename)
            result = LoadImage(image_only=True)(filename)
            self.assertTupleEqual(result.shape, spatial_size[::-1])
            np.testing.assert_allclose(result.T, test_image)

    def test_register(self):
        spatial_size = (32, 64, 128)
        test_image = np.random.rand(*spatial_size)
        with tempfile.TemporaryDirectory() as tempdir:
            filename = os.path.join(tempdir, "test_image.nii.gz")
            itk_np_view = itk.image_view_from_array(test_image)
            itk.imwrite(itk_np_view, filename)

            loader = LoadImage(image_only=True)
            loader.register(ITKReader())
            result = loader(filename)
            self.assertTupleEqual(result.shape, spatial_size[::-1])

    def test_kwargs(self):
        spatial_size = (32, 64, 128)
        test_image = np.random.rand(*spatial_size)
        with tempfile.TemporaryDirectory() as tempdir:
            filename = os.path.join(tempdir, "test_image.nii.gz")
            itk_np_view = itk.image_view_from_array(test_image)
            itk.imwrite(itk_np_view, filename)

            loader = LoadImage(image_only=True)
            reader = ITKReader(fallback_only=False)
            loader.register(reader)
            result = loader(filename)

            reader = ITKReader()
            img = reader.read(filename, fallback_only=False)
            result_raw = reader.get_data(img)
            result_raw = MetaTensor.ensure_torch_and_prune_meta(*result_raw)
            self.assertTupleEqual(result.shape, result_raw.shape)

    def test_my_reader(self):
        """test customised readers"""
        out = LoadImage(image_only=True, reader=_MiniReader, is_compatible=True)("test")
        self.assertEqual(out.meta["name"], "my test")
        out = LoadImage(image_only=True, reader=_MiniReader, is_compatible=False)("test")
        self.assertEqual(out.meta["name"], "my test")
        for item in (_MiniReader, _MiniReader(is_compatible=False)):
            out = LoadImage(image_only=True, reader=item)("test")
            self.assertEqual(out.meta["name"], "my test")
        out = LoadImage(image_only=True)("test", reader=_MiniReader(is_compatible=False))
        self.assertEqual(out.meta["name"], "my test")

    def test_itk_meta(self):
        """test metadata from a directory"""
        out = LoadImage(image_only=True, reader="ITKReader", pixel_type=itk_uc, series_meta=True)(
            "tests/testing_data/CT_DICOM"
        )
        idx = "0008|103e"
        label = itk.GDCMImageIO.GetLabelFromTag(idx, "")[1]
        val = out.meta[idx]
        expected = "Series Description=Routine Brain "
        self.assertEqual(f"{label}={val}", expected)

    @parameterized.expand([TEST_CASE_13, TEST_CASE_14, TEST_CASE_15, TEST_CASE_16, TEST_CASE_17, TEST_CASE_18])
    def test_channel_dim(self, input_param, filename, expected_shape):
        test_image = np.random.rand(*expected_shape)
        with tempfile.TemporaryDirectory() as tempdir:
            filename = os.path.join(tempdir, filename)
            nib.save(nib.Nifti1Image(test_image, np.eye(4)), filename)
            result = LoadImage(image_only=True, **input_param)(filename)  # with itk, meta has 'qto_xyz': itkMatrixF44

        self.assertTupleEqual(
            result.shape, (3, 128, 128, 128) if input_param.get("ensure_channel_first", False) else expected_shape
        )
        self.assertEqual(result.meta["original_channel_dim"], input_param["channel_dim"])


@unittest.skipUnless(has_itk, "itk not installed")
class TestLoadImageMeta(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super(__class__, cls).setUpClass()
        cls.tmpdir = tempfile.mkdtemp()
        test_image = nib.Nifti1Image(np.random.rand(128, 128, 128), np.eye(4))
        nib.save(test_image, os.path.join(cls.tmpdir, "im.nii.gz"))
        cls.test_data = os.path.join(cls.tmpdir, "im.nii.gz")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir)
        super(__class__, cls).tearDownClass()

    @parameterized.expand(TESTS_META)
    def test_correct(self, input_param, expected_shape, track_meta):
        set_track_meta(track_meta)
        r = LoadImage(image_only=True, prune_meta_pattern="glmax", prune_meta_sep="%", **input_param)(self.test_data)
        self.assertTupleEqual(r.shape, expected_shape)
        if track_meta:
            self.assertIsInstance(r, MetaTensor)
            self.assertTrue(hasattr(r, "affine"))
            self.assertIsInstance(r.affine, torch.Tensor)
            self.assertTrue("glmax" not in r.meta)
        else:
            self.assertIsInstance(r, torch.Tensor)
            self.assertNotIsInstance(r, MetaTensor)
            self.assertFalse(hasattr(r, "affine"))


if __name__ == "__main__":
    unittest.main()
