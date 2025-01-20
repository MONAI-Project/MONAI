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

import pickle
import random
import sys
import unittest
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader

from monai.apps import MedNISTDataset
from monai.networks.nets import DenseNet
from monai.optimizers import LearningRateFinder
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged, ScaleIntensityd, ToTensord
from monai.utils import optional_import, set_determinism
from monai.utils.misc import MONAIEnvVars
from tests.util import skip_if_downloading_fails

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

    has_matplotlib = True
    has_pil = True
else:
    plt, has_matplotlib = optional_import("matplotlib.pyplot")
    _, has_pil = optional_import("PIL.Image")

RAND_SEED = 42
random.seed(RAND_SEED)
set_determinism(seed=RAND_SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"


@unittest.skipUnless(sys.platform == "linux", "requires linux")
@unittest.skipUnless(has_pil, "requires PIL")
class TestLRFinder(unittest.TestCase):
    def setUp(self):
        self.root_dir = MONAIEnvVars.data_dir()
        if not self.root_dir:
            self.root_dir = Path(__file__).parents[1] / "testing_data"

        self.transforms = Compose(
            [
                LoadImaged(keys="image"),
                EnsureChannelFirstd(keys="image", channel_dim="no_channel"),
                ScaleIntensityd(keys="image"),
                ToTensord(keys="image"),
            ]
        )

    def test_lr_finder(self):
        # 0.001 gives 54 examples
        with skip_if_downloading_fails():
            train_ds = MedNISTDataset(
                root_dir=self.root_dir,
                transform=self.transforms,
                section="validation",
                val_frac=0.001,
                download=True,
                num_workers=2,
            )
        train_loader = DataLoader(train_ds, batch_size=300, shuffle=True, num_workers=2)
        num_classes = train_ds.get_num_classes()

        model = DenseNet(
            spatial_dims=2, in_channels=1, out_channels=num_classes, init_features=2, growth_rate=2, block_config=(2,)
        )
        loss_function = torch.nn.CrossEntropyLoss()
        learning_rate = 1e-5
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)

        lr_finder = LearningRateFinder(
            model=model,
            optimizer=optimizer,
            criterion=loss_function,
            device=device,
            pickle_module=pickle,
            pickle_protocol=4,
        )
        lr_finder.range_test(train_loader, val_loader=train_loader, end_lr=10.0, num_iter=5)
        print(lr_finder.get_steepest_gradient(0, 0)[0])

        if has_matplotlib:
            ax = plt.subplot()
            plt.show(block=False)
            lr_finder.plot(0, 0, ax=ax)  # to inspect the loss-learning rate graph
            plt.pause(3)
            plt.close()

        lr_finder.reset()  # to reset the model and optimizer to their initial state


if __name__ == "__main__":
    unittest.main()
