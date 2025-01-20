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

import unittest
from pathlib import Path

from monai.apps import CrossValidation, DecathlonDataset
from monai.data import MetaTensor
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged, ScaleIntensityd
from tests.util import skip_if_downloading_fails, skip_if_quick


class TestCrossValidation(unittest.TestCase):
    @skip_if_quick
    def test_values(self):
        testing_dir = Path(__file__).parents[1] / "testing_data"
        train_transform = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
                ScaleIntensityd(keys="image"),
            ]
        )
        val_transform = LoadImaged(keys=["image", "label"])

        def _test_dataset(dataset):
            self.assertEqual(len(dataset), 52)
            self.assertTrue("image" in dataset[0])
            self.assertTrue("label" in dataset[0])
            self.assertTrue(isinstance(dataset[0]["image"], MetaTensor))
            self.assertTupleEqual(dataset[0]["image"].shape, (1, 34, 49, 41))

        cvdataset = CrossValidation(
            dataset_cls=DecathlonDataset,
            nfolds=5,
            seed=12345,
            root_dir=testing_dir,
            task="Task04_Hippocampus",
            section="validation",
            transform=train_transform,
            download=True,
        )

        with skip_if_downloading_fails():
            data = cvdataset.get_dataset(folds=0)

        _test_dataset(data)

        # test training data for fold [1, 2, 3, 4] of 5 splits
        data = cvdataset.get_dataset(folds=[1, 2, 3, 4])
        self.assertTupleEqual(data[0]["image"].shape, (1, 35, 52, 33))
        self.assertEqual(len(data), 208)
        # test train / validation for fold 4 of 5 splits
        data = cvdataset.get_dataset(folds=[4], transform=val_transform, download=False)
        # val_transform doesn't add the channel dim to shape
        self.assertTupleEqual(data[0]["image"].shape, (38, 53, 30))
        self.assertEqual(len(data), 52)
        data = cvdataset.get_dataset(folds=[0, 1, 2, 3])
        self.assertTupleEqual(data[0]["image"].shape, (1, 34, 49, 41))
        self.assertEqual(len(data), 208)


if __name__ == "__main__":
    unittest.main()
