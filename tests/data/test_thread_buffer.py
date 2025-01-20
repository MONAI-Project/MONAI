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
import time
import unittest

import torch

from monai.data import DataLoader, Dataset, ThreadBuffer, ThreadDataLoader
from monai.transforms import Compose, SimulateDelayd
from monai.utils import PerfContext, set_determinism
from tests.util import assert_allclose


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        super().setUp()

        self.datalist = [
            {"image": "spleen_19.nii.gz", "label": "spleen_label_19.nii.gz"},
            {"image": "spleen_31.nii.gz", "label": "spleen_label_31.nii.gz"},
        ]

        self.transform = Compose([SimulateDelayd(keys=["image", "label"], delay_time=0.1)])

    def test_values(self):
        dataset = Dataset(data=self.datalist, transform=self.transform)
        dataloader = DataLoader(dataset=dataset, batch_size=2, num_workers=0)

        tbuffer = ThreadBuffer(dataloader)

        for d in tbuffer:
            self.assertEqual(d["image"][0], "spleen_19.nii.gz")
            self.assertEqual(d["image"][1], "spleen_31.nii.gz")
            self.assertEqual(d["label"][0], "spleen_label_19.nii.gz")
            self.assertEqual(d["label"][1], "spleen_label_31.nii.gz")

    def test_dataloader(self):
        dataset = Dataset(data=self.datalist, transform=self.transform)
        dataloader = ThreadDataLoader(dataset=dataset, batch_size=2, num_workers=0)

        for d in dataloader:
            self.assertEqual(d["image"][0], "spleen_19.nii.gz")
            self.assertEqual(d["image"][1], "spleen_31.nii.gz")

        for d in dataloader:
            self.assertEqual(d["label"][0], "spleen_label_19.nii.gz")
            self.assertEqual(d["label"][1], "spleen_label_31.nii.gz")

    def test_deterministic(self):
        set_determinism(0)
        res_1 = list(ThreadDataLoader(torch.arange(5), batch_size=2, buffer_size=2, shuffle=True, num_workers=0))

        set_determinism(0)
        num_workers = 2 if sys.platform == "linux" else 1
        res_2 = list(
            ThreadDataLoader(torch.arange(5), batch_size=2, buffer_size=3, shuffle=True, num_workers=num_workers)
        )

        set_determinism(None)
        assert_allclose(torch.cat(res_1), torch.cat(res_2), type_test=False)

    def test_time(self):
        dataset = Dataset(data=self.datalist * 2, transform=self.transform)  # contains data for 2 batches
        dataloader = DataLoader(dataset=dataset, batch_size=2, num_workers=0)

        tbuffer = ThreadBuffer(dataloader)

        with PerfContext() as pc:
            for _ in dataloader:
                time.sleep(0.5)  # each batch takes 0.8 s to generate on top of this time

        unbuffered_time = pc.total_time

        with PerfContext() as pc:
            for _ in tbuffer:
                time.sleep(0.5)  # while "computation" is happening the next batch is being generated, saving 0.4 s

        buffered_time = pc.total_time
        if sys.platform == "darwin":  # skip macOS measure
            print(f"darwin: Buffered time {buffered_time} vs unbuffered time {unbuffered_time}")
        else:
            self.assertTrue(
                buffered_time < unbuffered_time,
                f"Buffered time {buffered_time} should be less than unbuffered time {unbuffered_time}",
            )

    def test_dataloader_repeats(self):
        dataset = Dataset(data=self.datalist, transform=self.transform)
        dataloader = ThreadDataLoader(dataset=dataset, batch_size=2, num_workers=0, repeats=2)

        previous_batch = None

        for d in dataloader:
            self.assertEqual(d["image"][0], "spleen_19.nii.gz")
            self.assertEqual(d["image"][1], "spleen_31.nii.gz")

            if previous_batch is None:
                previous_batch = d
            else:
                self.assertTrue(previous_batch is d, "Batch object was not repeated")
                previous_batch = None

    def test_thread_workers(self):
        dataset = Dataset(data=self.datalist, transform=self.transform)
        dataloader = ThreadDataLoader(dataset=dataset, batch_size=2, num_workers=2, use_thread_workers=True)

        for d in dataloader:
            self.assertEqual(d["image"][0], "spleen_19.nii.gz")
            self.assertEqual(d["image"][1], "spleen_31.nii.gz")
            self.assertEqual(d["label"][0], "spleen_label_19.nii.gz")
            self.assertEqual(d["label"][1], "spleen_label_31.nii.gz")


if __name__ == "__main__":
    unittest.main()
