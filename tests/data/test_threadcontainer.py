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
import time
import unittest
from pathlib import Path

import torch

from monai.data import DataLoader
from monai.utils import optional_import, set_determinism
from monai.utils.enums import CommonKeys
from tests.util import SkipIfNoModule

try:
    _, has_ignite = optional_import("ignite")

    from monai.engines import SupervisedTrainer
    from monai.handlers import MetricLogger
    from monai.utils import ThreadContainer
except ImportError:
    has_ignite = False

compare_images, _ = optional_import("matplotlib.testing.compare", name="compare_images")


class TestThreadContainer(unittest.TestCase):
    @SkipIfNoModule("ignite")
    def test_container(self):
        net = torch.nn.Conv2d(1, 1, 3, padding=1)

        opt = torch.optim.Adam(net.parameters())

        img = torch.rand(1, 16, 16)
        data = {CommonKeys.IMAGE: img, CommonKeys.LABEL: img}
        loader = DataLoader([data for _ in range(10)])

        trainer = SupervisedTrainer(
            device=torch.device("cpu"),
            max_epochs=1,
            train_data_loader=loader,
            network=net,
            optimizer=opt,
            loss_function=torch.nn.L1Loss(),
        )

        con = ThreadContainer(trainer)
        con.start()
        time.sleep(1)  # wait for trainer to start

        self.assertTrue(con.is_alive)
        self.assertIsNotNone(con.status())
        self.assertGreater(len(con.status_dict), 0)

        con.join()

    @SkipIfNoModule("ignite")
    @SkipIfNoModule("matplotlib")
    def test_plot(self):
        set_determinism(0)
        test_dir = Path(__file__).parents[1]
        testing_dir = os.path.join(test_dir, "testing_data")

        net = torch.nn.Conv2d(1, 1, 3, padding=1)

        opt = torch.optim.Adam(net.parameters())

        img = torch.rand(1, 16, 16)

        # a third non-image key is added to test that this is correctly ignored when plotting
        data = {CommonKeys.IMAGE: img, CommonKeys.LABEL: img, "Not Image Data": ["This isn't an image"]}

        loader = DataLoader([data] * 20, batch_size=2)

        trainer = SupervisedTrainer(
            device=torch.device("cpu"),
            max_epochs=1,
            train_data_loader=loader,
            network=net,
            optimizer=opt,
            loss_function=torch.nn.L1Loss(),
        )

        logger = MetricLogger()
        logger.attach(trainer)

        con = ThreadContainer(trainer)
        con.start()
        con.join()

        fig = con.plot_status(logger)

        with tempfile.TemporaryDirectory() as tempdir:
            tempimg = f"{tempdir}/threadcontainer_plot_test.png"
            fig.savefig(tempimg)
            comp = compare_images(f"{testing_dir}/threadcontainer_plot_test.png", tempimg, 5e-2)

            self.assertIsNone(comp, comp)  # None indicates test passed


if __name__ == "__main__":
    unittest.main()
