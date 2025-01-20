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
from ignite.engine import Engine

from monai.data import decollate_batch
from monai.data.csv_saver import CSVSaver
from monai.handlers import ClassificationSaver


class TestHandlerClassificationSaver(unittest.TestCase):

    def test_saved_content(self):
        with tempfile.TemporaryDirectory() as tempdir:
            # set up engine
            def _train_func(engine, batch):
                engine.state.batch = decollate_batch(batch)
                return [torch.zeros(1) for _ in range(8)]

            engine = Engine(_train_func)

            # set up testing handler
            saver = CSVSaver(output_dir=tempdir, filename="predictions2.csv", delimiter="\t")
            ClassificationSaver(output_dir=tempdir, filename="predictions1.csv", delimiter="\t").attach(engine)
            ClassificationSaver(saver=saver).attach(engine)

            data = [{"filename_or_obj": ["testfile" + str(i) for i in range(8)]}]
            engine.run(data, max_epochs=1)

            def _test_file(filename):
                filepath = os.path.join(tempdir, filename)
                self.assertTrue(os.path.exists(filepath))
                with open(filepath) as f:
                    reader = csv.reader(f, delimiter="\t")
                    i = 0
                    for row in reader:
                        self.assertEqual(row[0], "testfile" + str(i))
                        self.assertEqual(np.array(row[1:]).astype(np.float32), 0.0)
                        i += 1
                    self.assertEqual(i, 8)

            _test_file("predictions1.csv")
            _test_file("predictions2.csv")


if __name__ == "__main__":
    unittest.main()
