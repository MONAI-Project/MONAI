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
import shutil
import tempfile
import unittest

import numpy as np
import torch
from parameterized import parameterized

from monai.data.image_reader import ITKReader, NibabelReader, NrrdReader, PILReader
from monai.data.image_writer import ITKWriter, NibabelWriter, PILWriter, register_writer, resolve_writer
from monai.data.meta_tensor import MetaTensor
from monai.transforms import LoadImage, SaveImage, moveaxis
from monai.utils import MetaKeys, OptionalImportError, optional_import
from tests.util import TEST_NDARRAYS, assert_allclose

_, has_itk = optional_import("itk", allow_namespace_pkg=True)


@unittest.skipUnless(has_itk, "itk not installed")
class TestLoadSaveNifti(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def nifti_rw(self, test_data, reader, writer, dtype, resample=True):
        test_data = test_data.astype(dtype)
        ndim = len(test_data.shape) - 1
        for p in TEST_NDARRAYS:
            output_ext = ".nii.gz"
            filepath = f"testfile_{ndim}d"
            saver = SaveImage(
                output_dir=self.test_dir,
                output_ext=output_ext,
                output_dtype=None,
                resample=resample,
                separate_folder=False,
                writer=writer,
            )
            meta_dict = {
                "filename_or_obj": f"{filepath}.png",
                "affine": np.eye(4),
                "original_affine": np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
            }
            test_data = MetaTensor(p(test_data), meta=meta_dict)
            self.assertEqual(test_data.meta[MetaKeys.SPACE], "RAS")
            saver(test_data)
            saved_path = os.path.join(self.test_dir, filepath + "_trans" + output_ext)
            self.assertTrue(os.path.exists(saved_path))
            loader = LoadImage(image_only=True, reader=reader, squeeze_non_spatial_dims=True, dtype=None)
            data = loader(saved_path)
            self.assertIn(dtype.__name__, str(data.dtype))
            meta = data.meta
            if meta["original_channel_dim"] == -1:
                _test_data = moveaxis(test_data, 0, -1)
            else:
                _test_data = test_data[0]
            if resample:
                _test_data = moveaxis(_test_data, 0, 1)
            assert_allclose(meta["qform_code"], 1, type_test=False)
            assert_allclose(meta["sform_code"], 1, type_test=False)
            assert_allclose(data, torch.as_tensor(_test_data))

    @parameterized.expand(itertools.product([NibabelReader, ITKReader], [NibabelWriter, "ITKWriter"]))
    def test_2d(self, reader, writer):
        test_data = np.arange(48, dtype=np.uint8).reshape(1, 6, 8)
        self.nifti_rw(test_data, reader, writer, np.uint8)
        self.nifti_rw(test_data, reader, writer, np.float32)

    @parameterized.expand(itertools.product([NibabelReader, ITKReader], [NibabelWriter, ITKWriter]))
    def test_3d(self, reader, writer):
        test_data = np.arange(48, dtype=np.uint8).reshape(1, 2, 3, 8)
        self.nifti_rw(test_data, reader, writer, np.int16)
        self.nifti_rw(test_data, reader, writer, float, False)

    @parameterized.expand(itertools.product([NibabelReader, ITKReader], ["NibabelWriter", ITKWriter]))
    def test_4d(self, reader, writer):
        test_data = np.arange(48, dtype=np.uint8).reshape(2, 1, 3, 8)
        self.nifti_rw(test_data, reader, writer, np.float64)


@unittest.skipUnless(has_itk, "itk not installed")
class TestLoadSavePNG(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def png_rw(self, test_data, reader, writer, dtype, resample=True):
        test_data = test_data.astype(dtype)
        ndim = len(test_data.shape) - 1
        for p in TEST_NDARRAYS:
            output_ext = ".png"
            filepath = f"testfile_{ndim}d"
            saver = SaveImage(
                output_dir=self.test_dir, output_ext=output_ext, resample=resample, separate_folder=False, writer=writer
            )
            test_data = MetaTensor(p(test_data), meta={"filename_or_obj": f"{filepath}.png", "spatial_shape": (6, 8)})
            saver(test_data)
            saved_path = os.path.join(self.test_dir, filepath + "_trans" + output_ext)
            self.assertTrue(os.path.exists(saved_path))
            loader = LoadImage(image_only=True, reader=reader)
            data = loader(saved_path)
            meta = data.meta
            if meta["original_channel_dim"] == -1:
                _test_data = moveaxis(test_data, 0, -1)
            else:
                _test_data = test_data[0]
            assert_allclose(data, torch.as_tensor(_test_data))

    @parameterized.expand(itertools.product([PILReader, ITKReader], [PILWriter, ITKWriter]))
    def test_2d(self, reader, writer):
        test_data = np.arange(48, dtype=np.uint8).reshape(1, 6, 8)
        self.png_rw(test_data, reader, writer, np.uint8)

    @parameterized.expand(itertools.product([PILReader, ITKReader], ["monai.data.PILWriter", ITKWriter]))
    def test_rgb(self, reader, writer):
        test_data = np.arange(48, dtype=np.uint8).reshape(3, 2, 8)
        self.png_rw(test_data, reader, writer, np.uint8, False)


class TestRegRes(unittest.TestCase):
    def test_0_default(self):
        self.assertTrue(len(resolve_writer(".png")) > 0, "has png writer")
        self.assertTrue(len(resolve_writer(".nrrd")) > 0, "has nrrd writer")
        self.assertTrue(len(resolve_writer("unknown")) > 0, "has writer")
        register_writer("unknown1", lambda: (_ for _ in ()).throw(OptionalImportError))
        with self.assertRaises(OptionalImportError):
            resolve_writer("unknown1")

    def test_1_new(self):
        register_writer("new", lambda x: x + 1)
        register_writer("new2", lambda x: x + 1)
        self.assertEqual(resolve_writer("new")[0](0), 1)


@unittest.skipUnless(has_itk, "itk not installed")
class TestLoadSaveNrrd(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def nrrd_rw(self, test_data, reader, writer, dtype, resample=True):
        test_data = test_data.astype(dtype)
        ndim = len(test_data.shape)
        for p in TEST_NDARRAYS:
            output_ext = ".nrrd"
            filepath = f"testfile_{ndim}d"
            saver = SaveImage(
                output_dir=self.test_dir, output_ext=output_ext, resample=resample, separate_folder=False, writer=writer
            ).set_options(init_kwargs={"affine_lps_to_ras": True})
            test_data = MetaTensor(
                p(test_data), meta={"filename_or_obj": f"{filepath}{output_ext}", "spatial_shape": test_data.shape}
            )
            saver(test_data)
            saved_path = os.path.join(self.test_dir, filepath + "_trans" + output_ext)
            loader = LoadImage(image_only=True, reader=reader)
            data = loader(saved_path)
            assert_allclose(data, torch.as_tensor(test_data))

    @parameterized.expand(itertools.product([NrrdReader, ITKReader], [ITKWriter, ITKWriter]))
    def test_2d(self, reader, writer):
        test_data = np.random.randn(8, 8).astype(np.float32)
        self.nrrd_rw(test_data, reader, writer, np.float32)

    @parameterized.expand(itertools.product([NrrdReader, ITKReader], [ITKWriter, ITKWriter]))
    def test_3d(self, reader, writer):
        test_data = np.random.randn(8, 8, 8).astype(np.float32)
        self.nrrd_rw(test_data, reader, writer, np.float32)


if __name__ == "__main__":
    unittest.main()
