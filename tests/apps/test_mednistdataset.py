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

from monai.apps import MedNISTDataset
from monai.data import MetaTensor
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged, ScaleIntensityd
from tests.util import skip_if_downloading_fails, skip_if_quick

MEDNIST_FULL_DATASET_LENGTH = 58954


class TestMedNISTDataset(unittest.TestCase):
    @skip_if_quick
    def test_values(self):
        testing_dir = Path(__file__).parents[1] / "testing_data"
        transform = Compose(
            [
                LoadImaged(keys="image"),
                EnsureChannelFirstd(keys="image", channel_dim="no_channel"),
                ScaleIntensityd(keys="image"),
            ]
        )

        def _test_dataset(dataset):
            self.assertEqual(len(dataset), int(MEDNIST_FULL_DATASET_LENGTH * dataset.test_frac))
            self.assertTrue("image" in dataset[0])
            self.assertTrue("label" in dataset[0])
            self.assertIsInstance(dataset[0]["image"], MetaTensor)
            self.assertTupleEqual(dataset[0]["image"].shape, (1, 64, 64))

        with skip_if_downloading_fails():
            data = MedNISTDataset(
                root_dir=testing_dir, transform=transform, section="test", download=True, copy_cache=False
            )

        _test_dataset(data)

        # testing from
        data = MedNISTDataset(
            root_dir=Path(testing_dir), transform=transform, section="test", download=False, runtime_cache=True
        )
        self.assertEqual(data.get_num_classes(), 6)
        _test_dataset(data)
        data = MedNISTDataset(root_dir=testing_dir, section="test", download=False)
        self.assertTupleEqual(data[0]["image"].shape, (64, 64))
        # test same dataset length with different random seed
        data = MedNISTDataset(root_dir=testing_dir, transform=transform, section="test", download=False, seed=42)
        _test_dataset(data)
        self.assertEqual(data[0]["class_name"], "AbdomenCT")
        self.assertEqual(data[0]["label"], 0)
        shutil.rmtree(os.path.join(testing_dir, "MedNIST"))
        with self.assertRaisesRegex(RuntimeError, "^Cannot find dataset directory"):
            MedNISTDataset(root_dir=testing_dir, transform=transform, section="test", download=False)


if __name__ == "__main__":
    unittest.main()
