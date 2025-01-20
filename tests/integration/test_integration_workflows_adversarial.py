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

import numpy as np
import torch

import monai
from monai.data import create_test_image_2d
from monai.engines import AdversarialTrainer
from monai.handlers import CheckpointSaver, StatsHandler, TensorBoardStatsHandler
from monai.networks.nets import AutoEncoder, Discriminator
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged, RandFlipd, ScaleIntensityd
from monai.utils import AdversarialKeys as Keys
from monai.utils import CommonKeys, optional_import, set_determinism
from tests.util import DistTestCase, TimedCall, skip_if_quick

nib, has_nibabel = optional_import("nibabel")


def run_training_test(root_dir, device="cuda:0"):
    learning_rate = 2e-4
    real_label = 1
    fake_label = 0

    real_images = sorted(glob(os.path.join(root_dir, "img*.nii.gz")))
    train_files = [{CommonKeys.IMAGE: img, CommonKeys.LABEL: img} for img in zip(real_images)]

    # prepare real data
    train_transforms = Compose(
        [
            LoadImaged(keys=[CommonKeys.IMAGE, CommonKeys.LABEL]),
            EnsureChannelFirstd(keys=[CommonKeys.IMAGE, CommonKeys.LABEL], channel_dim=2),
            ScaleIntensityd(keys=[CommonKeys.IMAGE]),
            RandFlipd(keys=[CommonKeys.IMAGE, CommonKeys.LABEL], prob=0.5),
        ]
    )
    train_ds = monai.data.CacheDataset(data=train_files, transform=train_transforms, cache_rate=0.5)
    train_loader = monai.data.DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4)

    # Create Discriminator
    discriminator_net = Discriminator(
        in_shape=(1, 64, 64), channels=(8, 16, 32, 64, 1), strides=(2, 2, 2, 2, 1), num_res_units=1, kernel_size=5
    ).to(device)
    discriminator_opt = torch.optim.Adam(discriminator_net.parameters(), learning_rate)
    discriminator_loss_criterion = torch.nn.BCELoss()

    def discriminator_loss(real_logits, fake_logits):
        real_target = real_logits.new_full((real_logits.shape[0], 1), real_label)
        fake_target = fake_logits.new_full((fake_logits.shape[0], 1), fake_label)
        real_loss = discriminator_loss_criterion(real_logits, real_target)
        fake_loss = discriminator_loss_criterion(fake_logits.detach(), fake_target)
        return torch.div(torch.add(real_loss, fake_loss), 2)

    # Create Generator
    generator_network = AutoEncoder(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(8, 16, 32, 64),
        strides=(2, 2, 2, 2),
        num_res_units=1,
        num_inter_units=1,
    )
    generator_network = generator_network.to(device)
    generator_optimiser = torch.optim.Adam(generator_network.parameters(), learning_rate)
    generator_loss_criterion = torch.nn.MSELoss()

    def reconstruction_loss(recon_images, real_images):
        return generator_loss_criterion(recon_images, real_images)

    def generator_loss(fake_logits):
        fake_target = fake_logits.new_full((fake_logits.shape[0], 1), real_label)
        recon_loss = discriminator_loss_criterion(fake_logits.detach(), fake_target)
        return recon_loss

    key_train_metric = None

    train_handlers = [
        StatsHandler(
            name="training_loss",
            output_transform=lambda x: {
                Keys.RECONSTRUCTION_LOSS: x[Keys.RECONSTRUCTION_LOSS],
                Keys.DISCRIMINATOR_LOSS: x[Keys.DISCRIMINATOR_LOSS],
                Keys.GENERATOR_LOSS: x[Keys.GENERATOR_LOSS],
            },
        ),
        TensorBoardStatsHandler(
            log_dir=root_dir,
            tag_name="training_loss",
            output_transform=lambda x: {
                Keys.RECONSTRUCTION_LOSS: x[Keys.RECONSTRUCTION_LOSS],
                Keys.DISCRIMINATOR_LOSS: x[Keys.DISCRIMINATOR_LOSS],
                Keys.GENERATOR_LOSS: x[Keys.GENERATOR_LOSS],
            },
        ),
        CheckpointSaver(
            save_dir=root_dir,
            save_dict={"g_net": generator_network, "d_net": discriminator_net},
            save_interval=2,
            epoch_level=True,
        ),
    ]

    num_epochs = 5

    trainer = AdversarialTrainer(
        device=device,
        max_epochs=num_epochs,
        train_data_loader=train_loader,
        g_network=generator_network,
        g_optimizer=generator_optimiser,
        g_loss_function=generator_loss,
        recon_loss_function=reconstruction_loss,
        d_network=discriminator_net,
        d_optimizer=discriminator_opt,
        d_loss_function=discriminator_loss,
        non_blocking=True,
        key_train_metric=key_train_metric,
        train_handlers=train_handlers,
    )
    trainer.run()

    return trainer.state


@skip_if_quick
@unittest.skipUnless(has_nibabel, "Requires nibabel library.")
class IntegrationWorkflowsAdversarialTrainer(DistTestCase):
    def setUp(self):
        set_determinism(seed=0)

        self.data_dir = tempfile.mkdtemp()
        for i in range(40):
            im, _ = create_test_image_2d(64, 64, num_objs=3, rad_max=14, num_seg_classes=1, channel_dim=-1)
            n = nib.Nifti1Image(im, np.eye(4))
            nib.save(n, os.path.join(self.data_dir, f"img{i:d}.nii.gz"))

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu:0")
        monai.config.print_config()

    def tearDown(self):
        set_determinism(seed=None)
        shutil.rmtree(self.data_dir)

    @TimedCall(seconds=200, daemon=False)
    def test_training(self):
        torch.manual_seed(0)

        finish_state = run_training_test(self.data_dir, device=self.device)

        # Assert AdversarialTrainer training finished
        self.assertEqual(finish_state.iteration, 100)
        self.assertEqual(finish_state.epoch, 5)


if __name__ == "__main__":
    unittest.main()
