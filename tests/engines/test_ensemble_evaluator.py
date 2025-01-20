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
from ignite.engine import EventEnum, Events
from parameterized import parameterized

from monai.engines import EnsembleEvaluator
from tests.util import assert_allclose

TEST_CASE_1 = [["pred_0", "pred_1", "pred_2", "pred_3", "pred_4"]]

TEST_CASE_2 = [None]


class TestEnsembleEvaluator(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    def test_content(self, pred_keys):
        device = torch.device("cpu:0")

        class TestDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 8

            def __getitem__(self, index):
                return {"image": torch.tensor([index]), "label": torch.zeros(1)}

        val_loader = torch.utils.data.DataLoader(TestDataset())

        class TestNet(torch.nn.Module):
            def __init__(self, func):
                super().__init__()
                self.func = func

            def forward(self, x):
                return self.func(x)

        net0 = TestNet(lambda x: x + 1)
        net1 = TestNet(lambda x: x + 2)
        net2 = TestNet(lambda x: x + 3)
        net3 = TestNet(lambda x: x + 4)
        net4 = TestNet(lambda x: x + 5)

        class CustomEvents(EventEnum):
            FOO_EVENT = "foo_event"
            BAR_EVENT = "bar_event"

        val_engine = EnsembleEvaluator(
            device=device,
            val_data_loader=val_loader,
            networks=[net0, net1, net2, net3, net4],
            pred_keys=pred_keys,
            event_names=["bwd_event", "opt_event", CustomEvents],
            event_to_attr={CustomEvents.FOO_EVENT: "foo", "opt_event": "opt"},
        )

        @val_engine.on(Events.ITERATION_COMPLETED)
        def run_transform(engine):
            for i in range(5):
                expected_value = engine.state.iteration + i
                assert_allclose(engine.state.output[0][f"pred_{i}"].item(), expected_value)

        @val_engine.on(Events.EPOCH_COMPLETED)
        def trigger_custom_event():
            val_engine.fire_event(CustomEvents.FOO_EVENT)
            val_engine.fire_event(CustomEvents.BAR_EVENT)
            val_engine.fire_event("bwd_event")
            val_engine.fire_event("opt_event")

        @val_engine.on(CustomEvents.FOO_EVENT)
        def do_foo_op():
            self.assertEqual(val_engine.state.foo, 0)

        @val_engine.on("opt_event")
        def do_bar_op():
            self.assertEqual(val_engine.state.opt, 0)

        val_engine.run()


if __name__ == "__main__":
    unittest.main()
