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
import unittest
from pathlib import Path

from monai.apps import TciaDataset
from monai.apps.tcia import DCM_FILENAME_REGEX, TCIA_LABEL_DICT
from monai.data import MetaTensor
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged, ScaleIntensityd
from tests.util import skip_if_downloading_fails, skip_if_quick


class TestTciaDataset(unittest.TestCase):
    @skip_if_quick
    def test_values(self):
        testing_dir = Path(__file__).parents[1] / "testing_data"
        download_len = 1
        val_frac = 1.0
        collection = "QIN-PROSTATE-Repeatability"

        transform = Compose(
            [
                LoadImaged(
                    keys=["image", "seg"],
                    reader="PydicomReader",
                    fname_regex=DCM_FILENAME_REGEX,
                    label_dict=TCIA_LABEL_DICT[collection],
                ),
                EnsureChannelFirstd(keys="image", channel_dim="no_channel"),
                ScaleIntensityd(keys="image"),
            ]
        )

        def _test_dataset(dataset):
            self.assertEqual(len(dataset), int(download_len * val_frac))
            self.assertTrue("image" in dataset[0])
            self.assertTrue("seg" in dataset[0])
            self.assertTrue(isinstance(dataset[0]["image"], MetaTensor))
            self.assertTupleEqual(dataset[0]["image"].shape, (1, 256, 256, 24))
            self.assertTupleEqual(dataset[0]["seg"].shape, (256, 256, 24, 4))

        with skip_if_downloading_fails():
            data = TciaDataset(
                root_dir=testing_dir,
                collection=collection,
                transform=transform,
                section="validation",
                download=True,
                download_len=download_len,
                copy_cache=False,
                val_frac=val_frac,
            )

        _test_dataset(data)
        data = TciaDataset(
            root_dir=testing_dir,
            collection=collection,
            transform=transform,
            section="validation",
            download=False,
            val_frac=val_frac,
            runtime_cache=True,
        )
        _test_dataset(data)
        self.assertTrue(
            data[0]["image"].meta["filename_or_obj"].endswith("QIN-PROSTATE-Repeatability/PCAMPMRI-00015/1901/image")
        )
        self.assertTrue(
            data[0]["seg"].meta["filename_or_obj"].endswith("QIN-PROSTATE-Repeatability/PCAMPMRI-00015/1901/seg")
        )
        # test validation without transforms
        data = TciaDataset(
            root_dir=testing_dir, collection=collection, section="validation", download=False, val_frac=val_frac
        )
        self.assertTupleEqual(data[0]["image"].shape, (256, 256, 24))
        self.assertEqual(len(data), int(download_len * val_frac))
        data = TciaDataset(
            root_dir=testing_dir,
            collection=collection,
            section="validation",
            download=False,
            fname_regex=DCM_FILENAME_REGEX,
            val_frac=val_frac,
        )
        self.assertTupleEqual(data[0]["image"].shape, (256, 256, 24))
        self.assertEqual(len(data), download_len)
        with self.assertWarns(UserWarning):
            data = TciaDataset(
                root_dir=testing_dir,
                collection=collection,
                section="validation",
                fname_regex=".*",  # all files including 'LICENSE' is not a valid input
                download=False,
                val_frac=val_frac,
            )[0]

        shutil.rmtree(os.path.join(testing_dir, collection))
        with self.assertRaisesRegex(RuntimeError, "^Cannot find dataset directory"):
            TciaDataset(
                root_dir=testing_dir,
                collection=collection,
                transform=transform,
                section="validation",
                download=False,
                val_frac=val_frac,
            )


if __name__ == "__main__":
    unittest.main()
