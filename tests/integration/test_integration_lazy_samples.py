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
import monai.transforms as mt
from monai.data import create_test_image_3d, decollate_batch
from monai.transforms.utils import has_status_keys
from monai.utils import TraceStatusKeys, set_determinism
from tests.util import HAS_CUPY, DistTestCase, SkipIfBeforePyTorchVersion, skip_if_quick


def _no_op(x):
    return x


def run_training_test(root_dir, device="cuda:0", cachedataset=0, readers=(None, None), num_workers=4, lazy=True):
    print(f"test case: {locals()}")
    images = sorted(glob(os.path.join(root_dir, "img*.nii.gz")))
    segs = sorted(glob(os.path.join(root_dir, "seg*.nii.gz")))
    train_files = [{"img": img, "seg": seg} for img, seg in zip(images[:20], segs[:20])]
    device = "cuda:0" if HAS_CUPY and torch.cuda.is_available() else "cpu"  # mode 0 and cuda requires CUPY
    num_workers = 0 if torch.cuda.is_available() else num_workers

    # define transforms for image and segmentation
    lazy_kwargs = {
        "img": {"mode": "bilinear", "device": device, "padding_mode": "border", "dtype": torch.float32},
        "seg": {"mode": 0, "device": device, "padding_mode": "nearest", "dtype": torch.uint8},
    }
    train_transforms = mt.Compose(
        [
            mt.LoadImaged(keys=["img", "seg"], reader=readers[0], image_only=True),
            mt.EnsureChannelFirstd(keys=["img", "seg"]),
            mt.Spacingd(
                keys=["img", "seg"],
                pixdim=[1.2, 0.8, 0.7],
                mode=["bilinear", 0],
                padding_mode=("border", "nearest"),
                dtype=np.float32,
            ),
            mt.Orientationd(keys=["img", "seg"], axcodes="ARS"),
            mt.RandRotate90d(keys=["img", "seg"], prob=1.0, spatial_axes=(1, 2)),
            mt.ScaleIntensityd(keys="img"),
            mt.ApplyPendingd(keys=["seg"]),
            mt.RandCropByPosNegLabeld(
                keys=["img", "seg"], label_key="seg", spatial_size=[76, 82, 80], pos=1, neg=1, num_samples=4
            ),
            mt.RandRotate90d(keys=["img", "seg"], prob=0.8, spatial_axes=(0, 2)),
            mt.RandZoomd(
                keys=["img", "seg"], prob=1.0, min_zoom=1.0, max_zoom=1.0, mode=("trilinear", 0), keep_size=True
            ),
            mt.ResizeWithPadOrCropD(keys=["img", "seg"], spatial_size=[80, 72, 80]),
            mt.Rotated(keys=["img", "seg"], angle=[np.pi / 2, np.pi / 2, 0], mode="nearest", keep_size=False),
            mt.Lambdad(keys=["img"], func=_no_op),
        ],
        lazy=lazy,
        overrides=lazy_kwargs,
        log_stats=num_workers > 0,
    )

    # create a training data loader
    if cachedataset == 2:
        train_ds = monai.data.CacheDataset(
            data=train_files, transform=train_transforms, cache_rate=0.8, runtime_cache=False, num_workers=0
        )
    elif cachedataset == 3:
        train_ds = monai.data.LMDBDataset(data=train_files, transform=train_transforms, cache_dir=root_dir)
    else:
        train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)

    # create UNet, DiceLoss and Adam optimizer
    model = monai.networks.nets.UNet(
        spatial_dims=3, in_channels=1, out_channels=1, channels=(2, 2, 2, 2), strides=(2, 2, 2), num_res_units=2
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), 5e-4)
    loss_function = monai.losses.DiceLoss(sigmoid=True)

    saver = mt.SaveImage(
        output_dir=os.path.join(root_dir, "output"),
        dtype=np.float32,
        output_ext=".nii.gz",
        output_postfix=f"seg_{lazy}_{num_workers}",
        mode="bilinear",
        resample=False,
        separate_folder=False,
        print_log=False,
    )
    inverter = mt.Invertd(
        keys="seg", orig_keys="img", transform=mt.Compose(train_transforms.transforms[-5:]), to_tensor=True
    )

    # use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
    _g = torch.Generator()
    _g.manual_seed(0)
    set_determinism(0)
    train_loader = monai.data.DataLoader(
        train_ds, batch_size=1, shuffle=True, num_workers=num_workers, generator=_g, persistent_workers=num_workers > 0
    )
    all_coords = set()
    batch_data = None
    for epoch in range(5):
        print("-" * 10)
        print(f"Epoch {epoch + 1}/5")
        for step, batch_data in enumerate(train_loader, start=1):
            inputs, labels = batch_data["img"].to(device), batch_data["seg"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss:{loss.item():0.4f}")

            for item, in_img, in_seg in zip(outputs, inputs, labels):  # this decollates the batch, pt 1.9+
                item.copy_meta_from(in_img)
                np.testing.assert_array_equal(item.pending_operations, [])
                np.testing.assert_array_equal(in_seg.pending_operations, [])
                ops = [0]
                if len(item.applied_operations) > 1:
                    found = False
                    for idx, n in enumerate(item.applied_operations):  # noqa
                        if n["class"] == "RandCropByPosNegLabel":
                            found = True
                            break
                    if found:
                        ops = item.applied_operations[idx]["extra_info"]["extra_info"]["cropped"]
                img_name = os.path.basename(item.meta["filename_or_obj"])
                coords = f"{img_name} - {ops}"
                print(coords)
                # np.testing.assert_allclose(coords in all_coords, False)
                all_coords.add(coords)
                saver(item)  # just testing the saving
                saver(in_img)
                saver(in_seg)
    invertible, reasons = has_status_keys(batch_data, TraceStatusKeys.PENDING_DURING_APPLY)
    inverted = [inverter(b_data) for b_data in decollate_batch(batch_data)]  # expecting no error

    return ops


@skip_if_quick
@SkipIfBeforePyTorchVersion((1, 11))
class IntegrationLazyResampling(DistTestCase):
    def setUp(self):
        monai.config.print_config()
        set_determinism(seed=0)

        self.data_dir = tempfile.mkdtemp()
        for i in range(3):
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
        _readers = (None, None)
        _w = 2
        if idx == 1:
            _readers = ("itkreader", "itkreader")
            _w = 1
        elif idx == 2:
            _readers = ("itkreader", "nibabelreader")
            _w = 0

        results_expected = run_training_test(
            self.data_dir, device=self.device, cachedataset=0, readers=_readers, num_workers=_w, lazy=False
        )
        results = run_training_test(
            self.data_dir, device=self.device, cachedataset=idx, readers=_readers, num_workers=_w, lazy=True
        )
        self.assertFalse(np.allclose(results, [0]))
        self.assertFalse(np.allclose(results_expected, [0]))
        np.testing.assert_allclose(results, results_expected)
        lazy_files = glob(os.path.join(self.data_dir, "output", "*_True_*.nii.gz"))
        regular_files = glob(os.path.join(self.data_dir, "output", "*_False_*.nii.gz"))
        diffs = []
        for a, b in zip(sorted(lazy_files), sorted(regular_files)):
            img_lazy = mt.LoadImage(image_only=True)(a)
            img_regular = mt.LoadImage(image_only=True)(b)
            diff = np.size(img_lazy) - np.sum(np.isclose(img_lazy, img_regular, atol=1e-4))
            diff_rate = diff / np.size(img_lazy)
            diffs.append(diff_rate)
            np.testing.assert_allclose(diff_rate, 0.0, atol=0.03)
        print("volume diff:", diffs)

    def test_training(self):
        for i in range(4):
            self.train_and_infer(i)


if __name__ == "__main__":
    unittest.main()
