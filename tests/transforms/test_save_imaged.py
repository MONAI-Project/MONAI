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
from pathlib import Path

import torch
from parameterized import parameterized

from monai.config import PathLike
from monai.data.folder_layout import FolderLayoutBase
from monai.data.meta_tensor import MetaTensor
from monai.transforms import SaveImaged
from monai.utils import optional_import

_, has_itk = optional_import("itk", allow_namespace_pkg=True)

TEST_CASE_1 = [
    {"img": MetaTensor(torch.randint(0, 255, (1, 2, 3, 4)), meta={"filename_or_obj": "testfile0.nii.gz"})},
    ".nii.gz",
    False,
]

TEST_CASE_2 = [
    {
        "img": MetaTensor(torch.randint(0, 255, (1, 2, 3, 4)), meta={"filename_or_obj": "testfile0.nii.gz"}),
        "patch_index": 6,
    },
    ".nii.gz",
    False,
]

TEST_CASE_3 = [
    {
        "img": MetaTensor(torch.randint(0, 255, (1, 2, 3, 4)), meta={"filename_or_obj": "testfile0.nrrd"}),
        "patch_index": 6,
    },
    ".nrrd",
    False,
]


@unittest.skipUnless(has_itk, "itk not installed")
class TestSaveImaged(unittest.TestCase):

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3])
    def test_saved_content(self, test_data, output_ext, resample):
        with tempfile.TemporaryDirectory() as tempdir:
            trans = SaveImaged(
                keys=["img", "pred"],
                output_dir=tempdir,
                output_ext=output_ext,
                resample=resample,
                allow_missing_keys=True,
            )
            trans(test_data)

            patch_index = test_data["img"].meta.get("patch_index", None)
            patch_index = f"_{patch_index}" if patch_index is not None else ""
            filepath = os.path.join("testfile0", "testfile0" + "_trans" + patch_index + output_ext)
            self.assertTrue(os.path.exists(os.path.join(tempdir, filepath)))

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3])
    def test_custom_folderlayout(self, test_data, output_ext, resample):

        class TestFolderLayout(FolderLayoutBase):

            def __init__(self, basepath: Path, extension: str, makedirs: bool):
                self.basepath = basepath
                self.ext = extension
                self.makedirs = makedirs

            def filename(self, **kwargs) -> PathLike:
                p = self.basepath / str(kwargs["subdirectory"])
                if not p.exists() and self.makedirs:
                    p.mkdir()

                return p / (str(kwargs["filename"]) + self.ext)

        def name_formatter(metadict: dict, _) -> dict:
            # "[filename].[ext]"
            # quick and dirty split on .
            base_filename = metadict["filename_or_obj"].split(".")[0]

            return {"subdirectory": base_filename, "filename": "image"}

        with tempfile.TemporaryDirectory() as tempdir:
            trans = SaveImaged(
                keys=["img", "pred"],
                resample=resample,
                allow_missing_keys=True,
                output_name_formatter=name_formatter,
                folder_layout=TestFolderLayout(basepath=Path(tempdir), extension=output_ext, makedirs=True),
            )
            trans(test_data)

            filepath = os.path.join("testfile0", "image" + output_ext)
            self.assertTrue(os.path.exists(os.path.join(tempdir, filepath)))

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3])
    def test_includes_metadata(self, test_data, output_ext, resample):
        with tempfile.TemporaryDirectory() as tempdir:
            trans = SaveImaged(
                keys=["img", "pred"],
                output_dir=tempdir,
                output_ext=output_ext,
                resample=resample,
                allow_missing_keys=True,
                savepath_in_metadict=True,
            )
            trans(test_data)

            self.assertTrue("saved_to" in test_data["img"].meta.keys())
            self.assertTrue(os.path.exists(test_data["img"].meta["saved_to"]))


if __name__ == "__main__":
    unittest.main()
