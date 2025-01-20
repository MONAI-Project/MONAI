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
from glob import glob

import nibabel as nib
import numpy as np
import torch

import monai
from monai.data import create_test_image_3d, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks import eval_mode
from monai.networks.nets import UNet
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandRotate90d,
    SaveImage,
    ScaleIntensityd,
    Spacingd,
)
from monai.utils import optional_import, set_determinism
from monai.visualize import plot_2d_or_3d_image
from tests.testing_data.integration_answers import test_integration_value
from tests.util import DistTestCase, TimedCall, skip_if_quick

SummaryWriter, _ = optional_import("torch.utils.tensorboard", name="SummaryWriter")

TASK = "integration_segmentation_3d"


def run_training_test(root_dir, device="cuda:0", cachedataset=0, readers=(None, None)):
    monai.config.print_config()
    images = sorted(glob(os.path.join(root_dir, "img*.nii.gz")))
    segs = sorted(glob(os.path.join(root_dir, "seg*.nii.gz")))
    train_files = [{"img": img, "seg": seg} for img, seg in zip(images[:20], segs[:20])]
    val_files = [{"img": img, "seg": seg} for img, seg in zip(images[-20:], segs[-20:])]

    # define transforms for image and segmentation
    train_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"], reader=readers[0]),
            EnsureChannelFirstd(keys=["img", "seg"]),
            # resampling with align_corners=True or dtype=float64 will generate
            # slight different results between PyTorch 1.5 an 1.6
            Spacingd(keys=["img", "seg"], pixdim=[1.2, 0.8, 0.7], mode=["bilinear", "nearest"], dtype=np.float32),
            ScaleIntensityd(keys="img"),
            RandCropByPosNegLabeld(
                keys=["img", "seg"], label_key="seg", spatial_size=[96, 96, 96], pos=1, neg=1, num_samples=4
            ),
            RandRotate90d(keys=["img", "seg"], prob=0.8, spatial_axes=[0, 2]),
        ]
    )
    train_transforms.set_random_state(1234)
    val_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"], reader=readers[1]),
            EnsureChannelFirstd(keys=["img", "seg"]),
            # resampling with align_corners=True or dtype=float64 will generate
            # slight different results between PyTorch 1.5 an 1.6
            Spacingd(keys=["img", "seg"], pixdim=[1.2, 0.8, 0.7], mode=["bilinear", "nearest"], dtype=np.float32),
            ScaleIntensityd(keys="img"),
        ]
    )

    # create a training data loader
    if cachedataset == 2:
        train_ds = monai.data.CacheDataset(
            data=train_files, transform=train_transforms, cache_rate=0.8, runtime_cache="process"
        )
    elif cachedataset == 3:
        train_ds = monai.data.LMDBDataset(data=train_files, transform=train_transforms, cache_dir=root_dir)
    else:
        train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    # use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
    train_loader = monai.data.DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4)
    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = monai.data.DataLoader(val_ds, batch_size=1, num_workers=4)
    val_post_tran = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

    # create UNet, DiceLoss and Adam optimizer
    model = monai.networks.nets.UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    loss_function = monai.losses.DiceLoss(sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), 5e-4)

    # start a typical PyTorch training
    val_interval = 2
    best_metric, best_metric_epoch = -1, -1
    epoch_loss_values = []
    metric_values = []
    writer = SummaryWriter(log_dir=os.path.join(root_dir, "runs"))
    model_filename = os.path.join(root_dir, "best_metric_model.pth")
    for epoch in range(6):
        print("-" * 10)
        print(f"Epoch {epoch + 1}/{6}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["img"].to(device), batch_data["seg"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss:{loss.item():0.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss:{epoch_loss:0.4f}")

        if (epoch + 1) % val_interval == 0:
            with eval_mode(model):
                val_images = None
                val_labels = None
                val_outputs = None
                for val_data in val_loader:
                    val_images, val_labels = (val_data["img"].to(device), val_data["seg"].to(device))
                    sw_batch_size, roi_size = 4, (96, 96, 96)
                    val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
                    # decollate prediction into a list and execute post processing for every item
                    val_outputs = [val_post_tran(i) for i in decollate_batch(val_outputs)]
                    # compute metrics
                    dice_metric(y_pred=val_outputs, y=val_labels)

                metric = dice_metric.aggregate().item()
                dice_metric.reset()
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), model_filename)
                    print("saved new best metric model")
                print(
                    f"current epoch {epoch + 1} current mean dice: {metric:0.4f} "
                    f"best mean dice: {best_metric:0.4f} at epoch {best_metric_epoch}"
                )
                writer.add_scalar("val_mean_dice", metric, epoch + 1)
                # plot the last model output as GIF image in TensorBoard with the corresponding image and label
                plot_2d_or_3d_image(val_images, epoch + 1, writer, index=0, tag="image")
                plot_2d_or_3d_image(val_labels, epoch + 1, writer, index=0, tag="label")
                plot_2d_or_3d_image(val_outputs, epoch + 1, writer, index=0, tag="output")
    print(f"train completed, best_metric: {best_metric:0.4f}  at epoch: {best_metric_epoch}")
    writer.close()
    return epoch_loss_values, best_metric


def run_inference_test(root_dir, device="cuda:0"):
    images = sorted(glob(os.path.join(root_dir, "im*.nii.gz")))
    segs = sorted(glob(os.path.join(root_dir, "seg*.nii.gz")))
    val_files = [{"img": img, "seg": seg} for img, seg in zip(images, segs)]

    saver = SaveImage(
        output_dir=os.path.join(root_dir, "output"),
        dtype=np.float32,
        output_ext=".nii.gz",
        output_postfix="seg",
        mode="bilinear",
    )
    # define transforms for image and segmentation
    val_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            EnsureChannelFirstd(keys=["img", "seg"]),
            # resampling with align_corners=True or dtype=float64 will generate
            # slight different results between PyTorch 1.5 an 1.6
            Spacingd(keys=["img", "seg"], pixdim=[1.2, 0.8, 0.7], mode=["bilinear", "nearest"], dtype=np.float32),
            ScaleIntensityd(keys="img"),
        ]
    )
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    # sliding window inference need to input 1 image in every iteration
    val_loader = monai.data.DataLoader(val_ds, batch_size=1, num_workers=4)
    val_post_tran = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5), saver])
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    model_filename = os.path.join(root_dir, "best_metric_model.pth")
    model.load_state_dict(torch.load(model_filename))
    with eval_mode(model):
        # resampling with align_corners=True or dtype=float64 will generate
        # slight different results between PyTorch 1.5 an 1.6
        for val_data in val_loader:
            val_images, val_labels = (val_data["img"].to(device), val_data["seg"].to(device))
            # define sliding window size and batch size for windows inference
            sw_batch_size, roi_size = 4, (96, 96, 96)
            val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
            # decollate prediction into a list
            val_outputs = [val_post_tran(i) for i in decollate_batch(val_outputs)]
            # compute metrics
            dice_metric(y_pred=val_outputs, y=val_labels)

    return dice_metric.aggregate().item()


@skip_if_quick
class IntegrationSegmentation3D(DistTestCase):
    def setUp(self):
        set_determinism(seed=0)

        self.data_dir = tempfile.mkdtemp()
        for i in range(40):
            im, seg = create_test_image_3d(128, 128, 128, num_seg_classes=1, channel_dim=-1)
            n = nib.Nifti1Image(im, np.eye(4))
            nib.save(n, os.path.join(self.data_dir, f"img{i:d}.nii.gz"))
            n = nib.Nifti1Image(seg, np.eye(4))
            nib.save(n, os.path.join(self.data_dir, f"seg{i:d}.nii.gz"))

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu:0"

    def tearDown(self):
        set_determinism(seed=None)
        shutil.rmtree(self.data_dir)

    def train_and_infer(self, idx=0):
        results = []
        set_determinism(0)
        _readers = (None, None)
        if idx == 1:
            _readers = ("itkreader", "itkreader")
        elif idx == 2:
            _readers = ("itkreader", "nibabelreader")
        losses, best_metric = run_training_test(self.data_dir, device=self.device, cachedataset=idx, readers=_readers)
        infer_metric = run_inference_test(self.data_dir, device=self.device)

        # check training properties
        print("losses", losses)
        print("best metric", best_metric)
        print("infer metric", infer_metric)
        self.assertTrue(len(glob(os.path.join(self.data_dir, "runs"))) > 0)
        model_file = os.path.join(self.data_dir, "best_metric_model.pth")
        self.assertTrue(os.path.exists(model_file))

        # check inference properties
        output_files = sorted(glob(os.path.join(self.data_dir, "output", "img*", "*.nii.gz")))
        print([np.mean(nib.load(output).get_fdata()) for output in output_files])
        results.extend(losses)
        results.append(best_metric)
        results.append(infer_metric)
        for output in output_files:
            ave = np.mean(nib.load(output).get_fdata())
            results.append(ave)
        self.assertTrue(test_integration_value(TASK, key="losses", data=results[:6], rtol=1e-3))
        self.assertTrue(test_integration_value(TASK, key="best_metric", data=results[6], rtol=1e-2))
        self.assertTrue(test_integration_value(TASK, key="infer_metric", data=results[7], rtol=1e-2))
        self.assertTrue(test_integration_value(TASK, key="output_sums", data=results[8:], rtol=5e-2))
        return results

    def test_training(self):
        repeated = []
        for i in range(4):
            results = self.train_and_infer(i)
            repeated.append(results)
        np.testing.assert_allclose(repeated[0], repeated[1])
        np.testing.assert_allclose(repeated[0], repeated[2])
        np.testing.assert_allclose(repeated[0], repeated[3])

    @TimedCall(seconds=360, daemon=False)
    def test_timing(self):
        self.train_and_infer(idx=3)


if __name__ == "__main__":
    unittest.main()
