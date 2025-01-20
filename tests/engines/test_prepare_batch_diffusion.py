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

import torch
from parameterized import parameterized

from monai.engines import SupervisedEvaluator
from monai.engines.utils import DiffusionPrepareBatch
from monai.inferers import DiffusionInferer
from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers import DDPMScheduler

TEST_CASES = [
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "channels": [8],
            "norm_num_groups": 8,
            "attention_levels": [True],
            "num_res_blocks": 1,
            "num_head_channels": 8,
        },
        (2, 1, 8, 8),
    ],
    [
        {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 1,
            "channels": [8],
            "norm_num_groups": 8,
            "attention_levels": [True],
            "num_res_blocks": 1,
            "num_head_channels": 8,
        },
        (2, 1, 8, 8, 8),
    ],
]


class TestPrepareBatchDiffusion(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_output_sizes(self, input_args, image_size):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataloader = [{"image": torch.randn(image_size).to(device)}]
        scheduler = DDPMScheduler(num_train_timesteps=20)
        inferer = DiffusionInferer(scheduler=scheduler)
        network = DiffusionModelUNet(**input_args).to(device)
        evaluator = SupervisedEvaluator(
            device=device,
            val_data_loader=dataloader,
            epoch_length=1,
            network=network,
            inferer=inferer,
            non_blocking=True,
            prepare_batch=DiffusionPrepareBatch(num_train_timesteps=20),
            decollate=False,
        )
        evaluator.run()
        output = evaluator.state.output
        # check shapes are the same
        self.assertEqual(output["pred"].shape, image_size)
        self.assertEqual(output["label"].shape, output["image"].shape)

    @parameterized.expand(TEST_CASES)
    def test_conditioning(self, input_args, image_size):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataloader = [{"image": torch.randn(image_size).to(device), "context": torch.randn((2, 4, 3)).to(device)}]
        scheduler = DDPMScheduler(num_train_timesteps=20)
        inferer = DiffusionInferer(scheduler=scheduler)
        network = DiffusionModelUNet(**input_args, with_conditioning=True, cross_attention_dim=3).to(device)
        evaluator = SupervisedEvaluator(
            device=device,
            val_data_loader=dataloader,
            epoch_length=1,
            network=network,
            inferer=inferer,
            non_blocking=True,
            prepare_batch=DiffusionPrepareBatch(num_train_timesteps=20, condition_name="context"),
            decollate=False,
        )
        evaluator.run()
        output = evaluator.state.output
        # check shapes are the same
        self.assertEqual(output["pred"].shape, image_size)
        self.assertEqual(output["label"].shape, output["image"].shape)


if __name__ == "__main__":
    unittest.main()
