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
from monai.data import create_test_image_2d
from monai.engines import GanTrainer
from monai.handlers import CheckpointSaver, StatsHandler, TensorBoardStatsHandler
from monai.networks import normal_init
from monai.networks.nets import Discriminator, Generator
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged, RandFlipd, ScaleIntensityd
from monai.utils import GanKeys as Keys
from monai.utils import set_determinism
from tests.util import DistTestCase, TimedCall, skip_if_quick


def run_training_test(root_dir, device="cuda:0"):
    real_images = sorted(glob(os.path.join(root_dir, "img*.nii.gz")))
    train_files = [{"reals": img} for img in zip(real_images)]

    # prepare real data
    train_transforms = Compose(
        [
            LoadImaged(keys=["reals"]),
            EnsureChannelFirstd(keys=["reals"], channel_dim=-1),
            ScaleIntensityd(keys=["reals"]),
            RandFlipd(keys=["reals"], prob=0.5),
        ]
    )
    train_ds = monai.data.CacheDataset(data=train_files, transform=train_transforms, cache_rate=0.5)
    train_loader = monai.data.DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4)

    learning_rate = 2e-4
    betas = (0.5, 0.999)
    real_label = 1
    fake_label = 0

    # create discriminator
    disc_net = Discriminator(
        in_shape=(1, 64, 64), channels=(8, 16, 32, 64, 1), strides=(2, 2, 2, 2, 1), num_res_units=1, kernel_size=5
    ).to(device)
    disc_net.apply(normal_init)
    disc_opt = torch.optim.Adam(disc_net.parameters(), learning_rate, betas=betas)
    disc_loss_criterion = torch.nn.BCELoss()

    def discriminator_loss(gen_images, real_images):
        real = real_images.new_full((real_images.shape[0], 1), real_label)
        gen = gen_images.new_full((gen_images.shape[0], 1), fake_label)
        realloss = disc_loss_criterion(disc_net(real_images), real)
        genloss = disc_loss_criterion(disc_net(gen_images.detach()), gen)
        return torch.div(torch.add(realloss, genloss), 2)

    # create generator
    latent_size = 64
    gen_net = Generator(
        latent_shape=latent_size, start_shape=(latent_size, 8, 8), channels=[32, 16, 8, 1], strides=[2, 2, 2, 1]
    )
    gen_net.apply(normal_init)
    gen_net.conv.add_module("activation", torch.nn.Sigmoid())
    gen_net = gen_net.to(device)
    gen_opt = torch.optim.Adam(gen_net.parameters(), learning_rate, betas=betas)
    gen_loss_criterion = torch.nn.BCELoss()

    def generator_loss(gen_images):
        output = disc_net(gen_images)
        cats = output.new_full(output.shape, real_label)
        return gen_loss_criterion(output, cats)

    key_train_metric = None

    train_handlers = [
        StatsHandler(
            name="training_loss", output_transform=lambda x: {Keys.GLOSS: x[Keys.GLOSS], Keys.DLOSS: x[Keys.DLOSS]}
        ),
        TensorBoardStatsHandler(
            log_dir=root_dir,
            tag_name="training_loss",
            output_transform=lambda x: {Keys.GLOSS: x[Keys.GLOSS], Keys.DLOSS: x[Keys.DLOSS]},
        ),
        CheckpointSaver(
            save_dir=root_dir, save_dict={"g_net": gen_net, "d_net": disc_net}, save_interval=2, epoch_level=True
        ),
    ]

    disc_train_steps = 2
    num_epochs = 5

    trainer = GanTrainer(
        device,
        num_epochs,
        train_loader,
        gen_net,
        gen_opt,
        generator_loss,
        disc_net,
        disc_opt,
        discriminator_loss,
        d_train_steps=disc_train_steps,
        latent_shape=latent_size,
        key_train_metric=key_train_metric,
        train_handlers=train_handlers,
        to_kwargs={"memory_format": torch.preserve_format, "dtype": torch.float32},
    )
    trainer.run()

    return trainer.state


@skip_if_quick
class IntegrationWorkflowsGAN(DistTestCase):
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

        # assert GAN training finished
        self.assertEqual(finish_state.iteration, 100)
        self.assertEqual(finish_state.epoch, 5)


if __name__ == "__main__":
    unittest.main()
