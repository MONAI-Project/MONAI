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
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import torch

from monai.data import CacheDataset, DataLoader, create_test_image_2d
from monai.data.test_time_augmentation import TestTimeAugmentation
from monai.data.utils import pad_list_data_collate
from monai.losses import DiceLoss
from monai.networks.nets import UNet
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    CropForegroundd,
    DivisiblePadd,
    EnsureChannelFirstd,
    RandAffined,
    RandScaleIntensityd,
)
from monai.transforms.croppad.dictionary import SpatialPadd
from monai.transforms.spatial.dictionary import RandFlipd
from monai.utils import optional_import, set_determinism
from monai.utils.enums import PostFix
from tests.util import TEST_NDARRAYS

if TYPE_CHECKING:
    import tqdm

    has_tqdm = True
    has_nib = True
else:
    tqdm, has_tqdm = optional_import("tqdm")
    _, has_nib = optional_import("nibabel")

trange = partial(tqdm.trange, desc="training") if has_tqdm else range


class TestTestTimeAugmentation(unittest.TestCase):
    @staticmethod
    def get_data(num_examples, input_size, data_type=np.asarray, include_label=True):
        custom_create_test_image_2d = partial(
            create_test_image_2d, *input_size, rad_max=7, num_seg_classes=1, num_objs=1
        )
        data = []
        for i in range(num_examples):
            im, label = custom_create_test_image_2d()
            d = {"image": data_type(im[:, i:])}
            if include_label:
                d["label"] = data_type(label[:, i:])
                d[PostFix.meta("label")] = {"affine": np.eye(4)}
            data.append(d)
        return data[0] if num_examples == 1 else data

    def setUp(self) -> None:
        set_determinism(seed=0)

    def tearDown(self) -> None:
        set_determinism(None)

    def test_test_time_augmentation(self):
        input_size = (20, 40)  # test different input data shape to pad list collate
        keys = ["image", "label"]
        num_training_ims = 10

        train_data = self.get_data(num_training_ims, input_size)
        test_data = self.get_data(1, input_size)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        transforms = Compose(
            [
                EnsureChannelFirstd(keys, channel_dim="no_channel"),
                RandAffined(
                    keys,
                    prob=1.0,
                    spatial_size=(30, 30),
                    rotate_range=(np.pi / 3, np.pi / 3),
                    translate_range=(3, 3),
                    scale_range=((0.8, 1), (0.8, 1)),
                    padding_mode="zeros",
                    mode=("bilinear", "nearest"),
                ),
                CropForegroundd(keys, source_key="image"),
                DivisiblePadd(keys, 4),
            ]
        )

        train_ds = CacheDataset(train_data, transforms)
        # output might be different size, so pad so that they match
        train_loader = DataLoader(train_ds, batch_size=2, collate_fn=pad_list_data_collate)

        model = UNet(2, 1, 1, channels=(6, 6), strides=(2, 2)).to(device)
        loss_function = DiceLoss(sigmoid=True)
        optimizer = torch.optim.Adam(model.parameters(), 1e-3)

        num_epochs = 10
        for _ in trange(num_epochs):
            epoch_loss = 0

            for batch_data in train_loader:
                inputs, labels = (batch_data["image"].to(device), batch_data["label"].to(device))
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= len(train_loader)

        post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

        tt_aug = TestTimeAugmentation(
            transform=transforms,
            batch_size=5,
            num_workers=0,
            inferrer_fn=model,
            device=device,
            to_tensor=True,
            output_device="cpu",
            post_func=post_trans,
        )
        mode, mean, std, vvc = tt_aug(test_data)
        self.assertEqual(mode.shape, (1,) + input_size)
        self.assertEqual(mean.shape, (1,) + input_size)
        self.assertTrue(all(np.unique(mode) == (0, 1)))
        self.assertGreaterEqual(mean.min(), 0.0)
        self.assertLessEqual(mean.max(), 1.0)
        self.assertEqual(std.shape, (1,) + input_size)
        self.assertIsInstance(vvc, float)

    def test_warn_non_random(self):
        transforms = Compose([EnsureChannelFirstd("im", channel_dim="no_channel"), SpatialPadd("im", 1)])
        with self.assertWarns(UserWarning):
            TestTimeAugmentation(transforms, None, None, None)

    def test_warn_random_but_has_no_invertible(self):
        transforms = Compose(
            [
                EnsureChannelFirstd("image", channel_dim="no_channel"),
                RandFlipd("image", prob=1.0),
                RandScaleIntensityd("image", 0.1, prob=1.0),
            ]
        )
        with self.assertWarns(UserWarning):
            tta = TestTimeAugmentation(transforms, 5, 0, orig_key="image")
            tta(self.get_data(1, (20, 20), data_type=np.float32))

    def test_warn_random_but_all_not_invertible(self):
        """test with no invertible stack"""
        transforms = Compose(
            [EnsureChannelFirstd("image", channel_dim="no_channel"), RandScaleIntensityd("image", 0.1, prob=1.0)]
        )
        with self.assertWarns(UserWarning):
            tta = TestTimeAugmentation(transforms, 1, 0, orig_key="image")
            tta(self.get_data(1, (20, 20), data_type=np.float32))

    def test_single_transform(self):
        for p in TEST_NDARRAYS:
            transforms = RandFlipd(["image", "label"], prob=1.0)
            tta = TestTimeAugmentation(transforms, batch_size=5, num_workers=0, inferrer_fn=lambda x: x)
            tta(self.get_data(1, (20, 20), data_type=p))

    def test_image_no_label(self):
        transforms = RandFlipd(["image"], prob=1.0)
        tta = TestTimeAugmentation(transforms, batch_size=5, num_workers=0, inferrer_fn=lambda x: x, orig_key="image")
        tta(self.get_data(1, (20, 20), include_label=False))


if __name__ == "__main__":
    unittest.main()
