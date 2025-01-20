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

from monai.data import CSVSaver


class TestCSVSaver(unittest.TestCase):

    def test_saved_content(self):
        with tempfile.TemporaryDirectory() as tempdir:
            saver = CSVSaver(output_dir=tempdir, filename="predictions.csv", delimiter="\t")
            meta_data = {"filename_or_obj": ["testfile" + str(i) for i in range(8)]}
            saver.save_batch(torch.zeros(8), meta_data)
            saver.finalize()
            filepath = os.path.join(tempdir, "predictions.csv")
            self.assertTrue(os.path.exists(filepath))
            with open(filepath) as f:
                reader = csv.reader(f, delimiter="\t")
                i = 0
                for row in reader:
                    self.assertEqual(row[0], "testfile" + str(i))
                    self.assertEqual(np.array(row[1:]).astype(np.float32), 0.0)
                    i += 1
                self.assertEqual(i, 8)


if __name__ == "__main__":
    unittest.main()
