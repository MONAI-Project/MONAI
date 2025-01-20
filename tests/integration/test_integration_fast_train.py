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

import math
import os
import shutil
import tempfile
import time
import unittest
from glob import glob

import nibabel as nib
import numpy as np
import torch

import monai
from monai.data import CacheDataset, ThreadDataLoader, create_test_image_3d, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monai.optimizers import Novograd
from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    FgBgToIndicesd,
    LoadImaged,
    RandAffined,
    RandAxisFlipd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandRotate90d,
    RandRotated,
    RandStdShiftIntensityd,
    RandZoomd,
    ScaleIntensityd,
    Spacingd,
    ToDeviced,
)
from monai.utils import set_determinism
from tests.util import DistTestCase, TimedCall, skip_if_no_cuda, skip_if_quick


@skip_if_no_cuda
@skip_if_quick
class IntegrationFastTrain(DistTestCase):
    def setUp(self):
        set_determinism(seed=0)
        monai.config.print_config()

        self.data_dir = tempfile.mkdtemp()
        for i in range(41):
            im, seg = create_test_image_3d(128, 128, 128, num_seg_classes=1, channel_dim=-1)
            n = nib.Nifti1Image(im, np.eye(4))
            nib.save(n, os.path.join(self.data_dir, f"img{i:d}.nii.gz"))
            n = nib.Nifti1Image(seg, np.eye(4))
            nib.save(n, os.path.join(self.data_dir, f"seg{i:d}.nii.gz"))

    def tearDown(self):
        set_determinism(seed=None)
        shutil.rmtree(self.data_dir)

    # test the fast training speed is as expected
    @TimedCall(seconds=100, daemon=False, force_quit=False)
    def test_train_timing(self):
        images = sorted(glob(os.path.join(self.data_dir, "img*.nii.gz")))
        segs = sorted(glob(os.path.join(self.data_dir, "seg*.nii.gz")))
        train_files = [{"image": img, "label": seg} for img, seg in zip(images[:32], segs[:32])]
        val_files = [{"image": img, "label": seg} for img, seg in zip(images[-9:], segs[-9:])]

        device = torch.device("cuda:0")
        # define transforms for train and validation
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
                ScaleIntensityd(keys="image"),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                # pre-compute foreground and background indexes
                # and cache them to accelerate training
                FgBgToIndicesd(keys="label", fg_postfix="_fg", bg_postfix="_bg"),
                # move the data to GPU and cache to avoid CPU -> GPU sync in every epoch
                ToDeviced(keys=["image", "label"], device=device),
                # randomly crop out patch samples from big
                # image based on pos / neg ratio
                # the image centers of negative samples
                # must be in valid image area
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(64, 64, 64),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    fg_indices_key="label_fg",
                    bg_indices_key="label_bg",
                ),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[1, 2]),
                RandAxisFlipd(keys=["image", "label"], prob=0.5),
                RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(1, 2)),
                RandZoomd(keys=["image", "label"], prob=0.5, min_zoom=0.8, max_zoom=1.2, keep_size=True),
                RandRotated(
                    keys=["image", "label"],
                    prob=0.5,
                    range_x=np.pi / 4,
                    mode=("bilinear", "nearest"),
                    align_corners=True,
                    dtype=np.float64,
                ),
                RandAffined(keys=["image", "label"], prob=0.5, rotate_range=np.pi / 2, mode=("bilinear", "nearest")),
                RandGaussianNoised(keys="image", prob=0.5),
                RandStdShiftIntensityd(keys="image", prob=0.5, factors=0.05, nonzero=True),
            ]
        )

        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
                ScaleIntensityd(keys="image"),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                # move the data to GPU and cache to avoid CPU -> GPU sync in every epoch
                ToDeviced(keys=["image", "label"], device=device),
            ]
        )

        max_epochs = 5
        learning_rate = 2e-4
        val_interval = 1  # do validation for every epoch

        # set CacheDataset, ThreadDataLoader and DiceCE loss for MONAI fast training
        train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=8)
        val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, runtime_cache=True)
        # disable multi-workers because `ThreadDataLoader` works with multi-threads
        train_loader = ThreadDataLoader(train_ds, num_workers=0, batch_size=4, shuffle=True)
        val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1)

        loss_function = DiceCELoss(to_onehot_y=True, softmax=True, squared_pred=True, batch=True)
        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        ).to(device)

        # Novograd paper suggests to use a bigger LR than Adam,
        # because Adam does normalization by element-wise second moments
        optimizer = Novograd(model.parameters(), learning_rate * 10)
        scaler = torch.cuda.amp.GradScaler()

        post_pred = AsDiscrete(argmax=True, to_onehot=2)
        post_label = AsDiscrete(to_onehot=2)

        dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

        best_metric = -1
        total_start = time.time()
        for epoch in range(max_epochs):
            epoch_start = time.time()
            print("-" * 10)
            print(f"epoch {epoch + 1}/{max_epochs}")
            model.train()
            epoch_loss = 0
            step = 0
            for batch_data in train_loader:
                step_start = time.time()
                step += 1
                optimizer.zero_grad()
                # set AMP for training
                with torch.cuda.amp.autocast():
                    outputs = model(batch_data["image"])
                    loss = loss_function(outputs, batch_data["label"])
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.item()
                epoch_len = math.ceil(len(train_ds) / train_loader.batch_size)
                print(
                    f"{step}/{epoch_len}, train_loss: {loss.item():.4f}" f" step time: {(time.time() - step_start):.4f}"
                )
            epoch_loss /= step
            print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

            if (epoch + 1) % val_interval == 0:
                model.eval()
                with torch.no_grad():
                    for val_data in val_loader:
                        roi_size = (96, 96, 96)
                        sw_batch_size = 4
                        # set AMP for validation
                        with torch.cuda.amp.autocast():
                            val_outputs = sliding_window_inference(val_data["image"], roi_size, sw_batch_size, model)

                        val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                        val_labels = [post_label(i) for i in decollate_batch(val_data["label"])]
                        dice_metric(y_pred=val_outputs, y=val_labels)

                    metric = dice_metric.aggregate().item()
                    dice_metric.reset()
                    if metric > best_metric:
                        best_metric = metric
                    print(f"epoch: {epoch + 1} current mean dice: {metric:.4f}, best mean dice: {best_metric:.4f}")
            print(f"time consuming of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")

        total_time = time.time() - total_start
        print(f"train completed, best_metric: {best_metric:.4f} total time: {total_time:.4f}")
        # test expected metrics
        self.assertGreater(best_metric, 0.95)


if __name__ == "__main__":
    unittest.main()
