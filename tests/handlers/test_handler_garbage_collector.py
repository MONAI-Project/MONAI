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

import gc
import unittest
from unittest import skipUnless

import torch
from ignite.engine import Engine
from parameterized import parameterized

from monai.data import Dataset
from monai.handlers import GarbageCollector
from monai.utils import IgniteInfo, min_version, optional_import

Events, has_ignite = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Events")

TEST_CASE_0 = [[0, 1, 2], "epoch"]

TEST_CASE_1 = [[0, 1, 2], "iteration"]

TEST_CASE_2 = [[0, 1, 2], Events.EPOCH_COMPLETED]


class TestHandlerGarbageCollector(unittest.TestCase):

    @skipUnless(has_ignite, "Requires ignite")
    @parameterized.expand([TEST_CASE_0, TEST_CASE_1, TEST_CASE_2])
    def test_content(self, data, trigger_event):
        # set up engine
        gb_count_dict = {}

        def _train_func(engine, batch):
            # store garbage collection counts
            if trigger_event == Events.EPOCH_COMPLETED or trigger_event.lower() == "epoch":
                if engine.state.iteration % engine.state.epoch_length == 1:
                    gb_count_dict[engine.state.epoch] = gc.get_count()
            elif trigger_event.lower() == "iteration":
                gb_count_dict[engine.state.iteration] = gc.get_count()

        engine = Engine(_train_func)

        # set up testing handler
        dataset = Dataset(data, transform=None)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1)
        GarbageCollector(trigger_event=trigger_event, log_level=30).attach(engine)

        engine.run(data_loader, max_epochs=5)

        first_count = 0
        for iter, gb_count in gb_count_dict.items():
            # At least one zero-generation object is collected
            # self.assertGreaterEqual(gb_count[0], 0)
            if iter > 1:
                # Since we are collecting all objects from all generations manually at each call,
                # starting from the second call, there shouldn't be any 1st and 2nd
                # generation objects available to collect.
                self.assertEqual(gb_count[1], first_count)
                self.assertEqual(gb_count[2], first_count)


if __name__ == "__main__":
    unittest.main()
