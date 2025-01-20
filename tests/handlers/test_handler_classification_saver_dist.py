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

import csv
import os
import tempfile
import unittest

import numpy as np
import torch
import torch.distributed as dist
from ignite.engine import Engine

from monai.data import decollate_batch
from monai.handlers import ClassificationSaver
from tests.util import DistCall, DistTestCase


class DistributedHandlerClassificationSaver(DistTestCase):
    @DistCall(nnodes=1, nproc_per_node=2)
    def test_saved_content(self):
        with tempfile.TemporaryDirectory() as tempdir:
            rank = dist.get_rank()

            # set up engine
            def _train_func(engine, batch):
                engine.state.batch = decollate_batch(batch)
                return [torch.zeros(1) for _ in range(8 + rank * 2)]

            engine = Engine(_train_func)

            # set up testing handler
            saver = ClassificationSaver(output_dir=tempdir, filename="predictions.csv", save_rank=1)
            saver.attach(engine)

            # rank 0 has 8 images, rank 1 has 10 images
            data = [
                {
                    "filename_or_obj": ["testfile" + str(i) for i in range(8 * rank, (8 + rank) * (rank + 1))],
                    "data_shape": torch.ones((8 + rank * 2, 1, 1)),
                }
            ]
            # rank 1 has more iterations
            if rank == 1:
                data.append(
                    {
                        "filename_or_obj": ["testfile" + str(i) for i in range(18, 28)],
                        "data_shape": torch.ones((10, 1, 1)),
                    }
                )

            engine.run(data, max_epochs=1)
            filepath = os.path.join(tempdir, "predictions.csv")
            if rank == 1:
                self.assertTrue(os.path.exists(filepath))
                with open(filepath) as f:
                    reader = csv.reader(f)
                    i = 0
                    for row in reader:
                        self.assertEqual(row[0], "testfile" + str(i))
                        self.assertEqual(np.array(row[1:]).astype(np.float32), 0.0)
                        i += 1
                    self.assertEqual(i, 28)


if __name__ == "__main__":
    unittest.main()
