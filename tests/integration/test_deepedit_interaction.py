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

from monai.apps.deepedit.interaction import Interaction
from monai.apps.deepedit.transforms import (
    AddGuidanceSignalDeepEditd,
    AddInitialSeedPointMissingLabelsd,
    AddRandomGuidanceDeepEditd,
    FindAllValidSlicesMissingLabelsd,
    FindDiscrepancyRegionsDeepEditd,
    SplitPredsLabeld,
)
from monai.data import DataLoader, Dataset
from monai.engines import SupervisedTrainer
from monai.engines.utils import IterationEvents
from monai.losses import DiceCELoss
from monai.transforms import Activationsd, AsDiscreted, Compose, ToTensord


def add_one(engine):
    if engine.state.best_metric == -1:
        engine.state.best_metric = 0
    else:
        engine.state.best_metric = engine.state.best_metric + 1


class TestInteractions(unittest.TestCase):

    def run_interaction(self, train):
        label_names = {"spleen": 1, "background": 0}
        np.random.seed(0)
        data = [
            {
                "image": np.random.randint(0, 256, size=(1, 15, 15, 15)).astype(np.float32),
                "label": np.random.randint(0, 2, size=(1, 15, 15, 15)),
                "label_names": label_names,
            }
            for _ in range(5)
        ]
        network = torch.nn.Conv3d(3, len(label_names), 1)
        lr = 1e-3
        opt = torch.optim.Adam(network.parameters(), lr)
        loss = DiceCELoss(to_onehot_y=True, softmax=True)
        pre_transforms = Compose(
            [
                FindAllValidSlicesMissingLabelsd(keys="label", sids="sids"),
                AddInitialSeedPointMissingLabelsd(keys="label", guidance="guidance", sids="sids"),
                AddGuidanceSignalDeepEditd(keys="image", guidance="guidance", number_intensity_ch=1),
                ToTensord(keys=("image", "label")),
            ]
        )
        dataset = Dataset(data, transform=pre_transforms)
        data_loader = DataLoader(dataset, batch_size=5)

        iteration_transforms = [
            FindDiscrepancyRegionsDeepEditd(keys="label", pred="pred", discrepancy="discrepancy"),
            AddRandomGuidanceDeepEditd(
                keys="NA", guidance="guidance", discrepancy="discrepancy", probability="probability"
            ),
            AddGuidanceSignalDeepEditd(keys="image", guidance="guidance", number_intensity_ch=1),
            ToTensord(keys=("image", "label")),
        ]
        post_transforms = [
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(keys=("pred", "label"), argmax=(True, False), to_onehot=len(label_names)),
            SplitPredsLabeld(keys="pred"),
            ToTensord(keys=("image", "label")),
        ]
        iteration_transforms = Compose(iteration_transforms)
        post_transforms = Compose(post_transforms)

        i = Interaction(
            deepgrow_probability=1.0,
            transforms=iteration_transforms,
            click_probability_key="probability",
            train=train,
            label_names=label_names,
        )
        self.assertEqual(len(i.transforms.transforms), 4, "Mismatch in expected transforms")

        # set up engine
        engine = SupervisedTrainer(
            device="cpu",
            max_epochs=1,
            train_data_loader=data_loader,
            network=network,
            optimizer=opt,
            loss_function=loss,
            postprocessing=post_transforms,
            iteration_update=i,
        )
        engine.add_event_handler(IterationEvents.INNER_ITERATION_STARTED, add_one)
        engine.add_event_handler(IterationEvents.INNER_ITERATION_COMPLETED, add_one)

        engine.run()
        self.assertIsNotNone(engine.state.batch[0].get("guidance"), "guidance is missing")
        self.assertEqual(engine.state.best_metric, 1)

    def test_train_interaction(self):
        self.run_interaction(train=True)

    def test_val_interaction(self):
        self.run_interaction(train=False)


if __name__ == "__main__":
    unittest.main()
