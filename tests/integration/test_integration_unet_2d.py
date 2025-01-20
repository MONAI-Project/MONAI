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

import numpy as np
import torch
from ignite.engine import create_supervised_trainer
from torch.utils.data import DataLoader, Dataset

from monai.data import create_test_image_2d
from monai.losses import DiceLoss
from monai.networks.nets import BasicUNet, UNet
from tests.util import DistTestCase, TimedCall, skip_if_quick


def run_test(net_name="basicunet", batch_size=64, train_steps=100, device="cuda:0"):
    class _TestBatch(Dataset):
        def __getitem__(self, _unused_id):
            im, seg = create_test_image_2d(128, 128, noise_max=1, num_objs=4, num_seg_classes=1)
            return im[None], seg[None].astype(np.float32)

        def __len__(self):
            return train_steps

    net = None
    if net_name == "basicunet":
        net = BasicUNet(spatial_dims=2, in_channels=1, out_channels=1, features=(4, 8, 8, 16, 16, 32))
    elif net_name == "unet":
        net = UNet(
            spatial_dims=2, in_channels=1, out_channels=1, channels=(4, 8, 16, 32), strides=(2, 2, 2), num_res_units=2
        )
    net.to(device)

    loss = DiceLoss(sigmoid=True)
    opt = torch.optim.Adam(net.parameters(), 1e-4)
    src = DataLoader(_TestBatch(), batch_size=batch_size)

    trainer = create_supervised_trainer(net, opt, loss, device, False)

    trainer.run(src, 1)
    loss = trainer.state.output
    return loss


@skip_if_quick
class TestIntegrationUnet2D(DistTestCase):
    @TimedCall(seconds=20, daemon=False)
    def test_unet_training(self):
        for n in ["basicunet", "unet"]:
            loss = run_test(net_name=n, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu:0"))
            print(loss)
            self.assertGreaterEqual(0.85, loss)


if __name__ == "__main__":
    unittest.main()
