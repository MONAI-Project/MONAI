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
from typing import Any
from unittest import skipUnless

import numpy as np
import torch
from parameterized import parameterized

from monai.config import PathLike
from monai.data import DataLoader, Dataset
from monai.data.wsi_reader import WSIReader
from monai.transforms import Compose, LoadImaged, ToTensord
from monai.utils import first, optional_import
from monai.utils.enums import PostFix, WSIPatchKeys
from tests.util import assert_allclose, download_url_or_skip_test, skip_if_no_cuda, testing_data_config

cucim, has_cucim = optional_import("cucim")
has_cucim = has_cucim and hasattr(cucim, "CuImage")
openslide, has_osl = optional_import("openslide")
imwrite, has_tiff = optional_import("tifffile", name="imwrite")
_, has_codec = optional_import("imagecodecs")
has_tiff = has_tiff and has_codec

TESTS_PATH = Path(__file__).parents[1]
WSI_GENERIC_TIFF_KEY = "wsi_generic_tiff"
WSI_GENERIC_TIFF_PATH = os.path.join(TESTS_PATH, "testing_data", f"temp_{WSI_GENERIC_TIFF_KEY}.tiff")

WSI_APERIO_SVS_KEY = "wsi_aperio_svs"
WSI_APERIO_SVS_PATH = os.path.join(TESTS_PATH, "testing_data", f"temp_{WSI_APERIO_SVS_KEY}.svs")

WSI_GENERIC_TIFF_HEIGHT = 32914
WSI_GENERIC_TIFF_WIDTH = 46000

TEST_CASE_WHOLE_0 = [WSI_GENERIC_TIFF_PATH, 2, (3, WSI_GENERIC_TIFF_HEIGHT // 4, WSI_GENERIC_TIFF_WIDTH // 4)]

TEST_CASE_TRANSFORM_0 = [
    WSI_GENERIC_TIFF_PATH,
    4,
    (WSI_GENERIC_TIFF_HEIGHT // 16, WSI_GENERIC_TIFF_WIDTH // 16),
    (1, 3, WSI_GENERIC_TIFF_HEIGHT // 16, WSI_GENERIC_TIFF_WIDTH // 16),
]

# ----------------------------------------------------------------------------
# Test cases for reading patches
# ----------------------------------------------------------------------------

TEST_CASE_0 = [
    WSI_GENERIC_TIFF_PATH,
    {"level": 8, "dtype": None},
    {"location": (0, 0), "size": (2, 1)},
    np.array([[[242], [242]], [[242], [242]], [[242], [242]]], dtype=np.float64),
]

TEST_CASE_1 = [
    WSI_GENERIC_TIFF_PATH,
    {},
    {"location": (WSI_GENERIC_TIFF_HEIGHT // 2, WSI_GENERIC_TIFF_WIDTH // 2), "size": (2, 1), "level": 0},
    np.array([[[246], [246]], [[246], [246]], [[246], [246]]], dtype=np.uint8),
]

TEST_CASE_2 = [
    WSI_GENERIC_TIFF_PATH,
    {},
    {"location": (0, 0), "size": (2, 1), "level": 8},
    np.array([[[242], [242]], [[242], [242]], [[242], [242]]], dtype=np.uint8),
]

TEST_CASE_3 = [
    WSI_GENERIC_TIFF_PATH,
    {"channel_dim": -1},
    {"location": (WSI_GENERIC_TIFF_HEIGHT // 2, WSI_GENERIC_TIFF_WIDTH // 2), "size": (4, 1), "level": 0},
    np.moveaxis(
        np.array(
            [[[246], [246], [246], [246]], [[246], [246], [246], [246]], [[246], [246], [246], [246]]], dtype=np.uint8
        ),
        0,
        -1,
    ),
]

TEST_CASE_4 = [
    WSI_GENERIC_TIFF_PATH,
    {"channel_dim": 2},
    {"location": (0, 0), "size": (4, 1), "level": 8},
    np.moveaxis(
        np.array(
            [[[242], [242], [242], [242]], [[242], [242], [242], [242]], [[242], [242], [242], [242]]], dtype=np.uint8
        ),
        0,
        -1,
    ),
]

TEST_CASE_5 = [
    WSI_GENERIC_TIFF_PATH,
    {"level": 8},
    {"location": (0, 0), "size": (2, 1)},
    np.array([[[242], [242]], [[242], [242]], [[242], [242]]], dtype=np.uint8),
]

TEST_CASE_6 = [
    WSI_GENERIC_TIFF_PATH,
    {"level": 8, "dtype": np.int32},
    {"location": (0, 0), "size": (2, 1)},
    np.array([[[242], [242]], [[242], [242]], [[242], [242]]], dtype=np.int32),
]

TEST_CASE_7 = [
    WSI_GENERIC_TIFF_PATH,
    {"level": 8, "dtype": np.float32},
    {"location": (0, 0), "size": (2, 1)},
    np.array([[[242], [242]], [[242], [242]], [[242], [242]]], dtype=np.float32),
]

TEST_CASE_8 = [
    WSI_GENERIC_TIFF_PATH,
    {"level": 8, "dtype": torch.uint8},
    {"location": (0, 0), "size": (2, 1)},
    torch.tensor([[[242], [242]], [[242], [242]], [[242], [242]]], dtype=torch.uint8),
]

TEST_CASE_9 = [
    WSI_GENERIC_TIFF_PATH,
    {"level": 8, "dtype": torch.float32},
    {"location": (0, 0), "size": (2, 1)},
    torch.tensor([[[242], [242]], [[242], [242]], [[242], [242]]], dtype=torch.float32),
]

# exact mpp in get_data
TEST_CASE_10_MPP = [
    WSI_GENERIC_TIFF_PATH,
    {"mpp_atol": 0.0, "mpp_rtol": 0.0},
    {"location": (WSI_GENERIC_TIFF_HEIGHT // 2, WSI_GENERIC_TIFF_WIDTH // 2), "size": (2, 1), "mpp": 1000},
    np.array([[[246], [246]], [[246], [246]], [[246], [246]]], dtype=np.uint8),
    {"level": 0},
]

# exact mpp as default
TEST_CASE_11_MPP = [
    WSI_GENERIC_TIFF_PATH,
    {"mpp_atol": 0.0, "mpp_rtol": 0.0, "mpp": 1000},
    {"location": (WSI_GENERIC_TIFF_HEIGHT // 2, WSI_GENERIC_TIFF_WIDTH // 2), "size": (2, 1)},
    np.array([[[246], [246]], [[246], [246]], [[246], [246]]], dtype=np.uint8),
    {"level": 0},
]

# exact mpp as default (Aperio SVS)
TEST_CASE_12_MPP = [
    WSI_APERIO_SVS_PATH,
    {"mpp_atol": 0.0, "mpp_rtol": 0.0, "mpp": 0.499},
    {"location": (0, 0), "size": (2, 1)},
    np.array([[[239], [239]], [[239], [239]], [[239], [239]]], dtype=np.uint8),
    {"level": 0},
]
# acceptable mpp within default tolerances
TEST_CASE_13_MPP = [
    WSI_GENERIC_TIFF_PATH,
    {},
    {"location": (0, 0), "size": (2, 1), "mpp": 256000},
    np.array([[[242], [242]], [[242], [242]], [[242], [242]]], dtype=np.uint8),
    {"level": 8},
]

# acceptable mpp within default tolerances (Aperio SVS)
TEST_CASE_14_MPP = [
    WSI_APERIO_SVS_PATH,
    {"mpp": 8.0},
    {"location": (0, 0), "size": (2, 1)},
    np.array([[[238], [240]], [[239], [241]], [[240], [241]]], dtype=np.uint8),
    {"level": 2},
]

# acceptable mpp within absolute tolerance (Aperio SVS)
TEST_CASE_15_MPP = [
    WSI_APERIO_SVS_PATH,
    {"mpp": 7.0, "mpp_atol": 1.0, "mpp_rtol": 0.0},
    {"location": (0, 0), "size": (2, 1)},
    np.array([[[238], [240]], [[239], [241]], [[240], [241]]], dtype=np.uint8),
    {"level": 2},
]

# acceptable mpp within relative tolerance (Aperio SVS)
TEST_CASE_16_MPP = [
    WSI_APERIO_SVS_PATH,
    {"mpp": 7.8, "mpp_atol": 0.0, "mpp_rtol": 0.1},
    {"location": (0, 0), "size": (2, 1)},
    np.array([[[238], [240]], [[239], [241]], [[240], [241]]], dtype=np.uint8),
    {"level": 2},
]

# exact power
TEST_CASE_17_POWER = [
    WSI_APERIO_SVS_PATH,
    {"power_atol": 0.0, "power_rtol": 0.0},
    {"location": (0, 0), "size": (2, 1), "power": 20},
    np.array([[[239], [239]], [[239], [239]], [[239], [239]]], dtype=np.uint8),
    {"level": 0},
]

# exact power
TEST_CASE_18_POWER = [
    WSI_APERIO_SVS_PATH,
    {"power": 20, "power_atol": 0.0, "power_rtol": 0.0},
    {"location": (0, 0), "size": (2, 1)},
    np.array([[[239], [239]], [[239], [239]], [[239], [239]]], dtype=np.uint8),
    {"level": 0},
]

# acceptable power within default tolerances (Aperio SVS)
TEST_CASE_19_POWER = [
    WSI_APERIO_SVS_PATH,
    {},
    {"location": (0, 0), "size": (2, 1), "power": 1.25},
    np.array([[[238], [240]], [[239], [241]], [[240], [241]]], dtype=np.uint8),
    {"level": 2},
]

# acceptable power within absolute tolerance (Aperio SVS)
TEST_CASE_20_POWER = [
    WSI_APERIO_SVS_PATH,
    {"power_atol": 0.3, "power_rtol": 0.0},
    {"location": (0, 0), "size": (2, 1), "power": 1.0},
    np.array([[[238], [240]], [[239], [241]], [[240], [241]]], dtype=np.uint8),
    {"level": 2},
]

# acceptable power within relative tolerance (Aperio SVS)
TEST_CASE_21_POWER = [
    WSI_APERIO_SVS_PATH,
    {"power_atol": 0.0, "power_rtol": 0.3},
    {"location": (0, 0), "size": (2, 1), "power": 1.0},
    np.array([[[238], [240]], [[239], [241]], [[240], [241]]], dtype=np.uint8),
    {"level": 2},
]
# device tests
TEST_CASE_DEVICE_1 = [
    WSI_GENERIC_TIFF_PATH,
    {"level": 8, "dtype": torch.float32, "device": "cpu"},
    {"location": (0, 0), "size": (2, 1)},
    torch.tensor([[[242], [242]], [[242], [242]], [[242], [242]]], dtype=torch.float32),
    "cpu",
]

TEST_CASE_DEVICE_2 = [
    WSI_GENERIC_TIFF_PATH,
    {"level": 8, "dtype": torch.float32, "device": "cuda"},
    {"location": (0, 0), "size": (2, 1)},
    torch.tensor([[[242], [242]], [[242], [242]], [[242], [242]]], dtype=torch.float32),
    "cuda",
]

TEST_CASE_DEVICE_3 = [
    WSI_GENERIC_TIFF_PATH,
    {"level": 8, "dtype": np.float32, "device": "cpu"},
    {"location": (0, 0), "size": (2, 1)},
    np.array([[[242], [242]], [[242], [242]], [[242], [242]]], dtype=np.float32),
    "cpu",
]

TEST_CASE_DEVICE_4 = [
    WSI_GENERIC_TIFF_PATH,
    {"level": 8, "dtype": np.float32, "device": "cuda"},
    {"location": (0, 0), "size": (2, 1)},
    torch.tensor([[[242], [242]], [[242], [242]], [[242], [242]]], dtype=torch.float32),
    "cuda",
]

TEST_CASE_DEVICE_5 = [
    WSI_GENERIC_TIFF_PATH,
    {"level": 8, "device": "cuda"},
    {"location": (0, 0), "size": (2, 1)},
    torch.tensor([[[242], [242]], [[242], [242]], [[242], [242]]], dtype=torch.uint8),
    "cuda",
]

TEST_CASE_DEVICE_6 = [
    WSI_GENERIC_TIFF_PATH,
    {"level": 8},
    {"location": (0, 0), "size": (2, 1)},
    np.array([[[242], [242]], [[242], [242]], [[242], [242]]], dtype=np.uint8),
    "cpu",
]

TEST_CASE_DEVICE_7 = [
    WSI_GENERIC_TIFF_PATH,
    {"level": 8, "device": None},
    {"location": (0, 0), "size": (2, 1)},
    np.array([[[242], [242]], [[242], [242]], [[242], [242]]], dtype=np.uint8),
    "cpu",
]

TEST_CASE_MULTI_WSI = [
    [WSI_GENERIC_TIFF_PATH, WSI_GENERIC_TIFF_PATH],
    {"location": (0, 0), "size": (2, 1), "level": 8},
    np.concatenate(
        [
            np.array([[[242], [242]], [[242], [242]], [[242], [242]]]),
            np.array([[[242], [242]], [[242], [242]], [[242], [242]]]),
        ],
        axis=0,
    ),
]

TEST_CASE_RGB_0 = [np.ones((3, 2, 2), dtype=np.uint8)]  # CHW

TEST_CASE_RGB_1 = [np.ones((3, 100, 100), dtype=np.uint8)]  # CHW

TEST_CASE_ERROR_0C = [np.ones((16, 16), dtype=np.uint8)]  # no color channel
TEST_CASE_ERROR_1C = [np.ones((16, 16, 1), dtype=np.uint8)]  # one color channel
TEST_CASE_ERROR_2C = [np.ones((16, 16, 2), dtype=np.uint8)]  # two color channels
TEST_CASE_ERROR_3D = [np.ones((16, 16, 16, 3), dtype=np.uint8)]  # 3D + color

# mpp not within default
TEST_CASE_ERROR_0_MPP = [
    WSI_GENERIC_TIFF_PATH,
    {},
    {"location": (WSI_GENERIC_TIFF_HEIGHT // 2, WSI_GENERIC_TIFF_WIDTH // 2), "size": (2, 1), "mpp": 1200},
    ValueError,
]

# mpp is not exact (no tolerance)
TEST_CASE_ERROR_1_MPP = [
    WSI_APERIO_SVS_PATH,
    {"mpp_atol": 0.0, "mpp_rtol": 0.0},
    {"location": (0, 0), "size": (2, 1), "mpp": 8.0},
    ValueError,
]

# power not within default
TEST_CASE_ERROR_2_POWER = [WSI_APERIO_SVS_PATH, {}, {"location": (0, 0), "size": (2, 1), "power": 40}, ValueError]

# power is not exact (no tolerance)
TEST_CASE_ERROR_3_POWER = [
    WSI_APERIO_SVS_PATH,
    {"power_atol": 0.0, "power_rtol": 0.0},
    {"location": (0, 0), "size": (2, 1), "power": 1.25},
    ValueError,
]

TEST_CASE_MPP_0 = [WSI_GENERIC_TIFF_PATH, 0, (1000.0, 1000.0)]


def save_rgba_tiff(array: np.ndarray, filename: str, mode: str):
    """
    Save numpy array into a TIFF RGB/RGBA file

    Args:
        array: numpy ndarray with the shape of CxHxW and C==3 representing a RGB image
        filename: the filename to be used for the tiff file. '_RGB.tiff' or '_RGBA.tiff' will be appended to this filename.
        mode: RGB or RGBA
    """
    if mode == "RGBA":
        array = np.concatenate([array, 255 * np.ones_like(array[0])[np.newaxis]]).astype(np.uint8)

    img_rgb = array.transpose(1, 2, 0)
    imwrite(filename, img_rgb, shape=img_rgb.shape, tile=(16, 16))

    return filename


def save_gray_tiff(array: np.ndarray, filename: str):
    """
    Save numpy array into a TIFF file

    Args:
        array: numpy ndarray with any shape
        filename: the filename to be used for the tiff file.
    """
    img_gray = array
    imwrite(filename, img_gray, shape=img_gray.shape)

    return filename


@skipUnless(has_cucim or has_osl or has_tiff, "Requires cucim, openslide, or tifffile!")
def setUpModule():
    download_url_or_skip_test(
        testing_data_config("images", WSI_GENERIC_TIFF_KEY, "url"),
        WSI_GENERIC_TIFF_PATH,
        hash_type=testing_data_config("images", WSI_GENERIC_TIFF_KEY, "hash_type"),
        hash_val=testing_data_config("images", WSI_GENERIC_TIFF_KEY, "hash_val"),
    )
    download_url_or_skip_test(
        testing_data_config("images", WSI_APERIO_SVS_KEY, "url"),
        WSI_APERIO_SVS_PATH,
        hash_type=testing_data_config("images", WSI_APERIO_SVS_KEY, "hash_type"),
        hash_val=testing_data_config("images", WSI_APERIO_SVS_KEY, "hash_val"),
    )


class WSIReaderTests:
    class Tests(unittest.TestCase):
        backend = None

        @parameterized.expand([TEST_CASE_WHOLE_0])
        def test_read_whole_image(self, file_path, level, expected_shape):
            reader = WSIReader(self.backend, level=level)
            with reader.read(file_path) as img_obj:
                img, meta = reader.get_data(img_obj)
            self.assertTupleEqual(img.shape, expected_shape)
            self.assertEqual(meta["backend"], self.backend)
            self.assertEqual(meta[WSIPatchKeys.PATH].lower(), str(os.path.abspath(file_path)).lower())
            self.assertEqual(meta[WSIPatchKeys.LEVEL], level)
            assert_allclose(meta[WSIPatchKeys.SIZE], expected_shape[1:], type_test=False)
            assert_allclose(meta[WSIPatchKeys.LOCATION], (0, 0), type_test=False)

        @parameterized.expand(
            [
                TEST_CASE_0,
                TEST_CASE_1,
                TEST_CASE_2,
                TEST_CASE_3,
                TEST_CASE_4,
                TEST_CASE_5,
                TEST_CASE_6,
                TEST_CASE_7,
                TEST_CASE_8,
                TEST_CASE_9,
                TEST_CASE_10_MPP,
                TEST_CASE_11_MPP,
                TEST_CASE_12_MPP,
                TEST_CASE_13_MPP,
                TEST_CASE_14_MPP,
                TEST_CASE_15_MPP,
                TEST_CASE_16_MPP,
                TEST_CASE_17_POWER,
                TEST_CASE_18_POWER,
                TEST_CASE_19_POWER,
                TEST_CASE_20_POWER,
                TEST_CASE_21_POWER,
            ]
        )
        def test_read_region(self, file_path, reader_kwargs, patch_info, expected_img, *args):
            reader = WSIReader(self.backend, **reader_kwargs)
            level = patch_info.get("level", reader_kwargs.get("level"))
            # Skip mpp, power tests for TiffFile backend
            if self.backend == "tifffile" and (level is None or level < 2 or file_path == WSI_APERIO_SVS_PATH):
                return
            if level is None:
                level = args[0].get("level")
            with reader.read(file_path) as img_obj:
                # Read twice to check multiple calls
                img, meta = reader.get_data(img_obj, **patch_info)
                img2 = reader.get_data(img_obj, **patch_info)[0]
            self.assertTupleEqual(img.shape, img2.shape)
            assert_allclose(img, img2)
            self.assertTupleEqual(img.shape, expected_img.shape)
            assert_allclose(img, expected_img)
            self.assertEqual(img.dtype, expected_img.dtype)
            self.assertEqual(meta["backend"], self.backend)
            self.assertEqual(meta[WSIPatchKeys.PATH].lower(), str(os.path.abspath(file_path)).lower())
            self.assertEqual(meta[WSIPatchKeys.LEVEL], level)
            assert_allclose(meta[WSIPatchKeys.SIZE], patch_info["size"], type_test=False)
            assert_allclose(meta[WSIPatchKeys.LOCATION], patch_info["location"], type_test=False)

        @parameterized.expand([TEST_CASE_MULTI_WSI])
        def test_read_region_multi_wsi(self, file_path_list, patch_info, expected_img):
            kwargs = {"name": None, "offset": None} if self.backend == "tifffile" else {}
            reader = WSIReader(self.backend, **kwargs)
            img_obj_list = reader.read(file_path_list, **kwargs)
            # Read twice to check multiple calls
            img, meta = reader.get_data(img_obj_list, **patch_info)
            img2 = reader.get_data(img_obj_list, **patch_info)[0]
            for img_obj in img_obj_list:
                img_obj.close()
            self.assertTupleEqual(img.shape, img2.shape)
            assert_allclose(img, img2)
            self.assertTupleEqual(img.shape, expected_img.shape)
            assert_allclose(img, expected_img)
            self.assertEqual(meta["backend"], self.backend)
            self.assertEqual(meta[WSIPatchKeys.PATH][0].lower(), str(os.path.abspath(file_path_list[0])).lower())
            self.assertEqual(meta[WSIPatchKeys.LEVEL][0], patch_info["level"])
            assert_allclose(meta[WSIPatchKeys.SIZE][0], expected_img.shape[1:], type_test=False)
            assert_allclose(meta[WSIPatchKeys.LOCATION][0], patch_info["location"], type_test=False)

        @parameterized.expand([TEST_CASE_RGB_0, TEST_CASE_RGB_1])
        @skipUnless(has_tiff, "Requires tifffile.")
        def test_read_rgba(self, img_expected):
            # skip for OpenSlide since not working with images without tiles
            if self.backend == "openslide":
                return
            image = {}
            reader = WSIReader(self.backend)
            for mode in ["RGB", "RGBA"]:
                file_path = save_rgba_tiff(
                    img_expected, os.path.join(TESTS_PATH, "testing_data", f"temp_tiff_image_{mode}.tiff"), mode=mode
                )
                with reader.read(file_path) as img_obj:
                    image[mode], _ = reader.get_data(img_obj)

            assert_allclose(image["RGB"], img_expected)
            assert_allclose(image["RGBA"], img_expected)

        @parameterized.expand([TEST_CASE_ERROR_0C, TEST_CASE_ERROR_1C, TEST_CASE_ERROR_2C, TEST_CASE_ERROR_3D])
        @skipUnless(has_tiff, "Requires tifffile.")
        def test_read_malformats(self, img_expected):
            if self.backend == "cucim" and (len(img_expected.shape) < 3 or img_expected.shape[2] == 1):
                # Until cuCIM addresses https://github.com/rapidsai/cucim/issues/230
                return
            reader = WSIReader(self.backend)
            file_path = os.path.join(TESTS_PATH, "testing_data", "temp_tiff_image_gray.tiff")
            imwrite(file_path, img_expected, shape=img_expected.shape)
            with self.assertRaises((RuntimeError, ValueError, openslide.OpenSlideError if has_osl else ValueError)):
                with reader.read(file_path) as img_obj:
                    reader.get_data(img_obj)

        @parameterized.expand([TEST_CASE_TRANSFORM_0])
        def test_with_dataloader(
            self, file_path: PathLike, level: int, expected_spatial_shape: Any, expected_shape: tuple[int, ...]
        ):
            train_transform = Compose(
                [
                    LoadImaged(keys=["image"], reader=WSIReader, backend=self.backend, level=level, image_only=False),
                    ToTensord(keys=["image"]),
                ]
            )
            dataset = Dataset([{"image": file_path}], transform=train_transform)
            data_loader = DataLoader(dataset)
            data: dict = first(data_loader, {})
            for s in data[PostFix.meta("image")]["spatial_shape"]:
                assert_allclose(s, expected_spatial_shape, type_test=False)
            self.assertTupleEqual(data["image"].shape, expected_shape)

        @parameterized.expand([TEST_CASE_TRANSFORM_0])
        def test_with_dataloader_batch(
            self, file_path: PathLike, level: int, expected_spatial_shape: Any, expected_shape: tuple[int, ...]
        ):
            train_transform = Compose(
                [
                    LoadImaged(keys=["image"], reader=WSIReader, backend=self.backend, level=level, image_only=False),
                    ToTensord(keys=["image"]),
                ]
            )
            dataset = Dataset([{"image": file_path}, {"image": file_path}], transform=train_transform)
            batch_size = 2
            data_loader = DataLoader(dataset, batch_size=batch_size)
            data: dict = first(data_loader, {})
            for s in data[PostFix.meta("image")]["spatial_shape"]:
                assert_allclose(s, expected_spatial_shape, type_test=False)
            self.assertTupleEqual(data["image"].shape, (batch_size, *expected_shape[1:]))

        @parameterized.expand([TEST_CASE_WHOLE_0])
        def test_read_whole_image_multi_thread(self, file_path, level, expected_shape):
            if self.backend == "cucim":
                reader = WSIReader(self.backend, level=level, num_workers=4)
                with reader.read(file_path) as img_obj:
                    img, meta = reader.get_data(img_obj)
                self.assertTupleEqual(img.shape, expected_shape)
                self.assertEqual(meta["backend"], self.backend)
                self.assertEqual(meta[WSIPatchKeys.PATH].lower(), str(os.path.abspath(file_path)).lower())
                self.assertEqual(meta[WSIPatchKeys.LEVEL], level)
                assert_allclose(meta[WSIPatchKeys.SIZE], expected_shape[1:], type_test=False)
                assert_allclose(meta[WSIPatchKeys.LOCATION], (0, 0), type_test=False)

        @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4])
        def test_read_region_multi_thread(self, file_path, kwargs, patch_info, expected_img):
            if self.backend == "cucim":
                reader = WSIReader(backend=self.backend, num_workers=2, **kwargs)
                with reader.read(file_path) as img_obj:
                    # Read twice to check multiple calls
                    img, meta = reader.get_data(img_obj, **patch_info)
                    img2 = reader.get_data(img_obj, **patch_info)[0]
                    self.assertTupleEqual(img.shape, img2.shape)
                    assert_allclose(img, img2)
                    self.assertTupleEqual(img.shape, expected_img.shape)
                    assert_allclose(img, expected_img)
                    self.assertEqual(meta["backend"], self.backend)
                    self.assertEqual(meta[WSIPatchKeys.PATH].lower(), str(os.path.abspath(file_path)).lower())
                    self.assertEqual(meta[WSIPatchKeys.LEVEL], patch_info["level"])
                    assert_allclose(meta[WSIPatchKeys.SIZE], patch_info["size"], type_test=False)
                    assert_allclose(meta[WSIPatchKeys.LOCATION], patch_info["location"], type_test=False)

        @parameterized.expand([TEST_CASE_MPP_0])
        def test_resolution_mpp(self, file_path, level, expected_mpp):
            reader = WSIReader(self.backend, level=level)
            with reader.read(file_path) as img_obj:
                mpp = reader.get_mpp(img_obj, level)
            self.assertTupleEqual(mpp, expected_mpp)

        @parameterized.expand(
            [
                TEST_CASE_DEVICE_1,
                TEST_CASE_DEVICE_2,
                TEST_CASE_DEVICE_3,
                TEST_CASE_DEVICE_4,
                TEST_CASE_DEVICE_5,
                TEST_CASE_DEVICE_6,
                TEST_CASE_DEVICE_7,
            ]
        )
        @skip_if_no_cuda
        def test_read_region_device(self, file_path, kwargs, patch_info, expected_img, device):
            reader = WSIReader(self.backend, **kwargs)
            level = patch_info.get("level", kwargs.get("level"))
            if self.backend == "tifffile" and level < 2:
                return
            with reader.read(file_path) as img_obj:
                # Read twice to check multiple calls
                img, meta = reader.get_data(img_obj, **patch_info)
                img2 = reader.get_data(img_obj, **patch_info)[0]
            self.assertTupleEqual(img.shape, img2.shape)
            assert_allclose(img, img2)
            self.assertTupleEqual(img.shape, expected_img.shape)
            assert_allclose(img, expected_img)
            self.assertEqual(img.dtype, expected_img.dtype)
            if isinstance(img, torch.Tensor):
                self.assertEqual(img.device.type, device)
            else:
                self.assertEqual("cpu", device)
            self.assertEqual(meta["backend"], self.backend)
            self.assertEqual(meta[WSIPatchKeys.PATH].lower(), str(os.path.abspath(file_path)).lower())
            self.assertEqual(meta[WSIPatchKeys.LEVEL], level)
            assert_allclose(meta[WSIPatchKeys.SIZE], patch_info["size"], type_test=False)
            assert_allclose(meta[WSIPatchKeys.LOCATION], patch_info["location"], type_test=False)

        @parameterized.expand(
            [TEST_CASE_ERROR_0_MPP, TEST_CASE_ERROR_1_MPP, TEST_CASE_ERROR_2_POWER, TEST_CASE_ERROR_3_POWER]
        )
        def test_errors(self, file_path, reader_kwargs, patch_info, exception):
            with self.assertRaises(exception):
                reader = WSIReader(self.backend, **reader_kwargs)
                with reader.read(file_path) as img_obj:
                    reader.get_data(img_obj, **patch_info)


@skipUnless(has_cucim, "Requires cucim")
class TestCuCIM(WSIReaderTests.Tests):
    @classmethod
    def setUpClass(cls):
        cls.backend = "cucim"


@skipUnless(has_osl, "Requires openslide")
class TestOpenSlide(WSIReaderTests.Tests):
    @classmethod
    def setUpClass(cls):
        cls.backend = "openslide"


@skipUnless(has_tiff, "Requires tifffile")
class TestTiffFile(WSIReaderTests.Tests):
    @classmethod
    def setUpClass(cls):
        cls.backend = "tifffile"


if __name__ == "__main__":
    unittest.main()
