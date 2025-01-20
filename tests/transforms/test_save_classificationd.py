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
from pathlib import Path

import numpy as np
import torch

from monai.data import CSVSaver, decollate_batch
from monai.transforms import Compose, CopyItemsd, SaveClassificationd
from monai.utils.enums import PostFix


class TestSaveClassificationd(unittest.TestCase):

    def test_saved_content(self):
        with tempfile.TemporaryDirectory() as tempdir:
            data = [
                {
                    "pred": torch.zeros(8),
                    PostFix.meta("image"): {"filename_or_obj": ["testfile" + str(i) for i in range(8)]},
                },
                {
                    "pred": torch.zeros(8),
                    PostFix.meta("image"): {"filename_or_obj": ["testfile" + str(i) for i in range(8, 16)]},
                },
                {
                    "pred": torch.zeros(8),
                    PostFix.meta("image"): {"filename_or_obj": ["testfile" + str(i) for i in range(16, 24)]},
                },
            ]

            saver = CSVSaver(
                output_dir=Path(tempdir), filename="predictions2.csv", overwrite=False, flush=False, delimiter="\t"
            )
            # set up test transforms
            post_trans = Compose(
                [
                    CopyItemsd(keys=PostFix.meta("image"), times=1, names=PostFix.meta("pred")),
                    # 1st saver saves data into CSV file
                    SaveClassificationd(
                        keys="pred",
                        saver=None,
                        meta_keys=None,
                        output_dir=Path(tempdir),
                        filename="predictions1.csv",
                        delimiter="\t",
                        overwrite=True,
                    ),
                    # 2rd saver only saves data into the cache, manually finalize later
                    SaveClassificationd(keys="pred", saver=saver, meta_key_postfix=PostFix.meta()),
                ]
            )
            # simulate inference 2 iterations
            d = decollate_batch(data[0])
            for i in d:
                post_trans(i)
            d = decollate_batch(data[1])
            for i in d:
                post_trans(i)
            # write into CSV file
            saver.finalize()

            # 3rd saver will not delete previous data due to `overwrite=False`
            trans2 = SaveClassificationd(
                keys="pred",
                saver=None,
                meta_keys=PostFix.meta("image"),  # specify meta key, so no need to copy anymore
                output_dir=tempdir,
                filename="predictions1.csv",
                delimiter="\t",
                overwrite=False,
            )
            d = decollate_batch(data[2])
            for i in d:
                trans2(i)

            def _test_file(filename, count):
                filepath = os.path.join(tempdir, filename)
                self.assertTrue(os.path.exists(filepath))
                with open(filepath) as f:
                    reader = csv.reader(f, delimiter="\t")
                    i = 0
                    for row in reader:
                        self.assertEqual(row[0], "testfile" + str(i))
                        self.assertEqual(np.array(row[1:]).astype(np.float32), 0.0)
                        i += 1
                    self.assertEqual(i, count)

            _test_file("predictions1.csv", 24)
            _test_file("predictions2.csv", 16)


if __name__ == "__main__":
    unittest.main()
