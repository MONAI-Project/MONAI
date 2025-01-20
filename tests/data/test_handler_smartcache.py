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

import sys
import unittest

import torch
from ignite.engine import Engine

from monai.data import SmartCacheDataset
from monai.handlers import SmartCacheHandler


class TestHandlerSmartCache(unittest.TestCase):

    def test_content(self):
        data = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        expected = [[0, 1, 2, 3, 4], [1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7], [4, 5, 6, 7, 8]]

        # set up engine
        def _train_func(engine, batch):
            self.assertListEqual(batch.tolist(), expected[engine.state.epoch - 1])

        engine = Engine(_train_func)

        # set up testing handler
        dataset = SmartCacheDataset(data, transform=None, replace_rate=0.2, cache_num=5, shuffle=False)
        workers = 2 if sys.platform == "linux" else 0
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=5, num_workers=workers, persistent_workers=False)
        SmartCacheHandler(dataset).attach(engine)

        engine.run(data_loader, max_epochs=5)


if __name__ == "__main__":
    unittest.main()
