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

import tempfile
import unittest

import torch
import torch.optim as optim
from ignite.engine import Engine, Events

from monai.handlers import CheckpointLoader, CheckpointSaver
from tests.util import assert_allclose


class TestHandlerCheckpointLoader(unittest.TestCase):
    def test_one_save_one_load(self):
        net1 = torch.nn.PReLU()
        data1 = net1.state_dict()
        data1["weight"] = torch.tensor([0.1])
        net1.load_state_dict(data1)
        net2 = torch.nn.PReLU()
        data2 = net2.state_dict()
        data2["weight"] = torch.tensor([0.2])
        net2.load_state_dict(data2)
        with tempfile.TemporaryDirectory() as tempdir:
            engine1 = Engine(lambda e, b: None)
            CheckpointSaver(save_dir=tempdir, save_dict={"net": net1, "eng": engine1}, save_final=True).attach(engine1)
            engine1.run([0] * 8, max_epochs=5)
            path = tempdir + "/checkpoint_final_iteration=40.pt"
            engine2 = Engine(lambda e, b: None)
            CheckpointLoader(load_path=path, load_dict={"net": net2, "eng": engine2}, strict=True).attach(engine2)

            @engine2.on(Events.STARTED)
            def check_epoch(engine: Engine):
                self.assertEqual(engine.state.epoch, 5)

            engine2.run([0] * 8, max_epochs=8)
            assert_allclose(net2.state_dict()["weight"], torch.tensor([0.1]))

            # test bad case with max_epochs smaller than current epoch
            engine3 = Engine(lambda e, b: None)
            CheckpointLoader(load_path=path, load_dict={"net": net2, "eng": engine3}, strict=True).attach(engine3)

            try:
                engine3.run([0] * 8, max_epochs=3)
            except ValueError:
                self.assertEqual(engine3.state.epoch, 5)
                self.assertEqual(engine3.state.max_epochs, 5)

    def test_two_save_one_load(self):
        net1 = torch.nn.PReLU()
        optimizer = optim.SGD(net1.parameters(), lr=0.02)
        data1 = net1.state_dict()
        data1["weight"] = torch.tensor([0.1])
        net1.load_state_dict(data1)
        net2 = torch.nn.PReLU()
        data2 = net2.state_dict()
        data2["weight"] = torch.tensor([0.2])
        net2.load_state_dict(data2)
        with tempfile.TemporaryDirectory() as tempdir:
            engine = Engine(lambda e, b: None)
            save_dict = {"net": net1, "opt": optimizer}
            CheckpointSaver(save_dir=tempdir, save_dict=save_dict, save_final=True).attach(engine)
            engine.run([0] * 8, max_epochs=5)
            path = tempdir + "/checkpoint_final_iteration=40.pt"
            engine = Engine(lambda e, b: None)
            CheckpointLoader(load_path=path, load_dict={"net": net2}, strict=True).attach(engine)
            engine.run([0] * 8, max_epochs=1)
            assert_allclose(net2.state_dict()["weight"], torch.tensor([0.1]))

    def test_save_single_device_load_multi_devices(self):
        net1 = torch.nn.PReLU()
        data1 = net1.state_dict()
        data1["weight"] = torch.tensor([0.1])
        net1.load_state_dict(data1)
        net2 = torch.nn.PReLU()
        data2 = net2.state_dict()
        data2["weight"] = torch.tensor([0.2])
        net2.load_state_dict(data2)
        net2 = torch.nn.DataParallel(net2)
        with tempfile.TemporaryDirectory() as tempdir:
            engine = Engine(lambda e, b: None)
            CheckpointSaver(save_dir=tempdir, save_dict={"net": net1}, save_final=True).attach(engine)
            engine.run([0] * 8, max_epochs=5)
            path = tempdir + "/net_final_iteration=40.pt"
            engine = Engine(lambda e, b: None)
            CheckpointLoader(load_path=path, load_dict={"net": net2}, strict=True).attach(engine)
            engine.run([0] * 8, max_epochs=1)
            assert_allclose(net2.state_dict()["module.weight"].cpu(), torch.tensor([0.1]))

    def test_partial_under_load(self):
        net1 = torch.nn.Sequential(*[torch.nn.PReLU(), torch.nn.PReLU()])
        data1 = net1.state_dict()
        data1["0.weight"] = torch.tensor([0.1])
        data1["1.weight"] = torch.tensor([0.2])
        net1.load_state_dict(data1)

        net2 = torch.nn.Sequential(*[torch.nn.PReLU()])
        data2 = net2.state_dict()
        data2["0.weight"] = torch.tensor([0.3])
        net2.load_state_dict(data2)

        with tempfile.TemporaryDirectory() as tempdir:
            engine = Engine(lambda e, b: None)
            CheckpointSaver(save_dir=tempdir, save_dict={"net": net1}, save_final=True).attach(engine)
            engine.run([0] * 8, max_epochs=5)
            path = tempdir + "/net_final_iteration=40.pt"
            engine = Engine(lambda e, b: None)
            CheckpointLoader(load_path=path, load_dict={"net": net2}, strict=False).attach(engine)
            engine.run([0] * 8, max_epochs=1)
            assert_allclose(net2.state_dict()["0.weight"].cpu(), torch.tensor([0.1]))

    def test_partial_over_load(self):
        net1 = torch.nn.Sequential(*[torch.nn.PReLU()])
        data1 = net1.state_dict()
        data1["0.weight"] = torch.tensor([0.1])
        net1.load_state_dict(data1)

        net2 = torch.nn.Sequential(*[torch.nn.PReLU(), torch.nn.PReLU()])
        data2 = net2.state_dict()
        data2["0.weight"] = torch.tensor([0.2])
        data2["1.weight"] = torch.tensor([0.3])
        net2.load_state_dict(data2)

        with tempfile.TemporaryDirectory() as tempdir:
            engine = Engine(lambda e, b: None)
            CheckpointSaver(save_dir=tempdir, save_dict={"net": net1}, save_final=True).attach(engine)
            engine.run([0] * 8, max_epochs=5)
            path = tempdir + "/net_final_iteration=40.pt"
            engine = Engine(lambda e, b: None)
            CheckpointLoader(load_path=path, load_dict={"net": net2}, strict=False).attach(engine)
            engine.run([0] * 8, max_epochs=1)
            assert_allclose(net2.state_dict()["0.weight"].cpu(), torch.tensor([0.1]))

    def test_strict_shape(self):
        net1 = torch.nn.Sequential(*[torch.nn.PReLU(num_parameters=5)])
        data1 = net1.state_dict()
        data1["0.weight"] = torch.tensor([1, 2, 3, 4, 5])
        data1["new"] = torch.tensor(0.1)
        net1.load_state_dict(data1, strict=False)
        opt1 = optim.SGD(net1.parameters(), lr=0.02)

        net2 = torch.nn.Sequential(*[torch.nn.PReLU(), torch.nn.PReLU()])
        data2 = net2.state_dict()
        data2["0.weight"] = torch.tensor([0.2])
        data2["1.weight"] = torch.tensor([0.3])
        net2.load_state_dict(data2)
        opt2 = optim.SGD(net2.parameters(), lr=0.02)

        with tempfile.TemporaryDirectory() as tempdir:
            engine = Engine(lambda e, b: None)
            CheckpointSaver(save_dir=tempdir, save_dict={"net": net1, "opt": opt1}, save_final=True).attach(engine)
            engine.run([0] * 8, max_epochs=5)
            path = tempdir + "/checkpoint_final_iteration=40.pt"
            engine = Engine(lambda e, b: None)
            CheckpointLoader(
                load_path=path,
                # expect to print a warning because it loads not only `net` but also `opt` with `strict_shape=False`
                load_dict={"net": net2, "opt": opt2},
                strict=False,
                strict_shape=False,
            ).attach(engine)
            engine.run([0] * 8, max_epochs=1)
            assert_allclose(net2.state_dict()["0.weight"].cpu(), torch.tensor([0.2]))
            # test whether `opt2` had been skipped when loading with `strict_shape=False`,
            # it should have 2 items in `params`(0.weight and 1.weight) while the checkpoint has 1 item(0.weight)
            self.assertEqual(len(opt1.state_dict()["param_groups"][0]["params"]), 1)
            self.assertEqual(len(opt2.state_dict()["param_groups"][0]["params"]), 2)


if __name__ == "__main__":
    unittest.main()
