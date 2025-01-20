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

import glob
import tempfile
import unittest

import numpy as np
import torch
from ignite.engine import Engine, Events
from parameterized import parameterized

from monai.data import decollate_batch
from monai.handlers import TensorBoardImageHandler
from monai.utils import optional_import
from tests.util import SkipIfBeforePyTorchVersion

_, has_tb = optional_import("torch.utils.tensorboard", name="SummaryWriter")

TEST_CASES = [[[20, 20]], [[2, 20, 20]], [[3, 20, 20]], [[20, 20, 20]], [[2, 20, 20, 20]], [[2, 2, 20, 20, 20]]]


@unittest.skipUnless(has_tb, "Requires SummaryWriter installation")
@SkipIfBeforePyTorchVersion((1, 13))  # issue 6683
class TestHandlerTBImage(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_tb_image_shape(self, shape):
        with tempfile.TemporaryDirectory() as tempdir:
            # set up engine
            def _train_func(engine, batch):
                engine.state.batch = decollate_batch(list(batch))
                return [torch.zeros((1, 10, 10))]

            engine = Engine(_train_func)

            # set up testing handler
            stats_handler = TensorBoardImageHandler(log_dir=tempdir)
            engine.add_event_handler(Events.ITERATION_COMPLETED, stats_handler)

            data = zip(
                torch.as_tensor(np.random.normal(size=(10, 4, *shape))),
                torch.as_tensor(np.random.normal(size=(10, 4, *shape))),
            )
            engine.run(data, epoch_length=10, max_epochs=1)
            stats_handler.close()

            self.assertTrue(len(glob.glob(tempdir)) > 0)


if __name__ == "__main__":
    unittest.main()
