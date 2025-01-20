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

import nibabel as nib
import numpy as np
import torch
from ignite.engine import Engine, Events
from torch.utils.data import DataLoader

from monai.data import ImageDataset, create_test_image_3d
from monai.inferers import sliding_window_inference
from monai.networks import eval_mode, predict_segmentation
from monai.networks.nets import UNet
from monai.transforms import EnsureChannelFirst, SaveImage
from monai.utils import pytorch_after, set_determinism
from tests.util import DistTestCase, TimedCall, make_nifti_image, skip_if_quick


def run_test(batch_size, img_name, seg_name, output_dir, device="cuda:0"):
    ds = ImageDataset(
        [img_name],
        [seg_name],
        transform=EnsureChannelFirst(channel_dim="no_channel"),
        seg_transform=EnsureChannelFirst(channel_dim="no_channel"),
        image_only=True,
    )
    loader = DataLoader(ds, batch_size=1, pin_memory=torch.cuda.is_available())

    net = UNet(
        spatial_dims=3, in_channels=1, out_channels=1, channels=(4, 8, 16, 32), strides=(2, 2, 2), num_res_units=2
    ).to(device)
    roi_size = (16, 32, 48)
    sw_batch_size = batch_size

    saver = SaveImage(output_dir=output_dir, output_ext=".nii.gz", output_postfix="seg")

    def _sliding_window_processor(_engine, batch):
        img = batch[0]  # first item from ImageDataset is the input image
        with eval_mode(net):
            seg_probs = sliding_window_inference(img.to(device), roi_size, sw_batch_size, net, device=device)
            return predict_segmentation(seg_probs)

    def save_func(engine):
        if pytorch_after(1, 9, 1):
            for m in engine.state.output:
                saver(m)
        else:
            saver(engine.state.output[0])

    infer_engine = Engine(_sliding_window_processor)
    infer_engine.add_event_handler(Events.ITERATION_COMPLETED, save_func)
    infer_engine.run(loader)

    basename = os.path.basename(img_name)[: -len(".nii.gz")]
    saved_name = os.path.join(output_dir, basename, f"{basename}_seg.nii.gz")
    return saved_name


@skip_if_quick
class TestIntegrationSlidingWindow(DistTestCase):
    def setUp(self):
        set_determinism(seed=0)

        im, seg = create_test_image_3d(28, 25, 63, rad_max=10, noise_max=1, num_objs=4, num_seg_classes=1)
        self.img_name = make_nifti_image(im)
        self.seg_name = make_nifti_image(seg)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu:0")

    def tearDown(self):
        set_determinism(seed=None)
        if os.path.exists(self.img_name):
            os.remove(self.img_name)
        if os.path.exists(self.seg_name):
            os.remove(self.seg_name)

    @TimedCall(seconds=20)
    def test_training(self):
        set_determinism(seed=0)
        with tempfile.TemporaryDirectory() as tempdir:
            output_file = run_test(
                batch_size=2, img_name=self.img_name, seg_name=self.seg_name, output_dir=tempdir, device=self.device
            )
            output_image = nib.load(output_file).get_fdata()
            np.testing.assert_allclose(np.sum(output_image), 33621)
            np.testing.assert_allclose(output_image.shape, (28, 25, 63))


if __name__ == "__main__":
    unittest.main()
