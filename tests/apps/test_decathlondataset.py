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

from monai.apps import DecathlonDataset
from monai.data import MetaTensor
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged, ScaleIntensityd
from tests.util import skip_if_downloading_fails, skip_if_quick


class TestDecathlonDataset(unittest.TestCase):
    @skip_if_quick
    def test_values(self):
        testing_dir = Path(__file__).resolve().parents[1] / "testing_data"
        transform = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
                ScaleIntensityd(keys="image"),
            ]
        )

        def _test_dataset(dataset):
            self.assertEqual(len(dataset), 52)
            self.assertTrue("image" in dataset[0])
            self.assertTrue("label" in dataset[0])
            self.assertTrue(isinstance(dataset[0]["image"], MetaTensor))
            self.assertTupleEqual(dataset[0]["image"].shape, (1, 36, 47, 44))

        with skip_if_downloading_fails():
            data = DecathlonDataset(
                root_dir=testing_dir,
                task="Task04_Hippocampus",
                transform=transform,
                section="validation",
                download=True,
                copy_cache=False,
            )

        _test_dataset(data)
        data = DecathlonDataset(
            root_dir=testing_dir,
            task="Task04_Hippocampus",
            transform=transform,
            section="validation",
            download=False,
            runtime_cache=True,
        )
        _test_dataset(data)
        self.assertTrue(data[0]["image"].meta["filename_or_obj"].endswith("hippocampus_163.nii.gz"))
        self.assertTrue(data[0]["label"].meta["filename_or_obj"].endswith("hippocampus_163.nii.gz"))
        # test validation without transforms
        data = DecathlonDataset(root_dir=testing_dir, task="Task04_Hippocampus", section="validation", download=False)
        self.assertTupleEqual(data[0]["image"].shape, (36, 47, 44))
        self.assertEqual(len(data), 52)
        data = DecathlonDataset(root_dir=testing_dir, task="Task04_Hippocampus", section="training", download=False)
        self.assertTupleEqual(data[0]["image"].shape, (34, 56, 31))
        self.assertEqual(len(data), 208)

        # test dataset properties
        data = DecathlonDataset(
            root_dir=Path(testing_dir), task="Task04_Hippocampus", section="validation", download=False
        )
        properties = data.get_properties(keys="labels")
        self.assertDictEqual(properties["labels"], {"0": "background", "1": "Anterior", "2": "Posterior"})

        shutil.rmtree(os.path.join(testing_dir, "Task04_Hippocampus"))
        with self.assertRaisesRegex(RuntimeError, "^Cannot find dataset directory"):
            DecathlonDataset(
                root_dir=testing_dir,
                task="Task04_Hippocampus",
                transform=transform,
                section="validation",
                download=False,
            )


if __name__ == "__main__":
    unittest.main()
