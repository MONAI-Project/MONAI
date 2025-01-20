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

import torch
import torch.distributed as dist
from ignite.engine import Engine, Events

from monai.handlers import MetricsSaver
from monai.utils import evenly_divisible_all_gather
from monai.utils.enums import PostFix
from tests.util import DistCall, DistTestCase


class DistributedMetricsSaver(DistTestCase):
    @DistCall(nnodes=1, nproc_per_node=2)
    def test_content(self):
        with tempfile.TemporaryDirectory() as tempdir:
            self._run(tempdir)

    def _run(self, tempdir):
        my_rank = dist.get_rank()
        fnames = ["aaa" * 300, "bbb" * 301, "ccc" * 302]

        metrics_saver = MetricsSaver(
            save_dir=tempdir,
            metrics=["metric1", "metric2"],
            metric_details=["metric3", "metric4"],
            batch_transform=lambda x: x[PostFix.meta("image")],
            summary_ops="*",
            delimiter="\t",
        )

        def _val_func(engine, batch):
            pass

        engine = Engine(_val_func)

        # define here to ensure symbol always exists regardless of the following if conditions
        data = [{PostFix.meta("image"): {"filename_or_obj": [fnames[0]]}}]

        if my_rank == 0:

            @engine.on(Events.EPOCH_COMPLETED)
            def _save_metrics0(engine):
                engine.state.metrics = {"metric1": 1, "metric2": 2}
                engine.state.metric_details = {"metric3": torch.tensor([[1, 2]]), "metric4": torch.tensor([[5, 6]])}

        if my_rank == 1:
            # different ranks have different data length
            data = [
                {PostFix.meta("image"): {"filename_or_obj": [fnames[1]]}},
                {PostFix.meta("image"): {"filename_or_obj": [fnames[2]]}},
            ]

            @engine.on(Events.EPOCH_COMPLETED)
            def _save_metrics1(engine):
                engine.state.metrics = {"metric1": 1, "metric2": 2}
                engine.state.metric_details = {
                    "metric3": torch.tensor([[2, 3], [3, 4]]),
                    "metric4": torch.tensor([[6, 7], [7, 8]]),
                }

        @engine.on(Events.EPOCH_COMPLETED)
        def _all_gather(engine):
            scores = engine.state.metric_details["metric3"]
            engine.state.metric_details["metric3"] = evenly_divisible_all_gather(data=scores, concat=True)
            scores = engine.state.metric_details["metric4"]
            engine.state.metric_details["metric4"] = evenly_divisible_all_gather(data=scores, concat=True)

        metrics_saver.attach(engine)
        engine.run(data, max_epochs=1)

        if my_rank == 0:
            # check the metrics.csv and content
            self.assertTrue(os.path.exists(os.path.join(tempdir, "metrics.csv")))
            with open(os.path.join(tempdir, "metrics.csv")) as f:
                f_csv = csv.reader(f)
                for i, row in enumerate(f_csv):
                    self.assertEqual(row, [f"metric{i + 1}\t{i + 1}"])
            self.assertTrue(os.path.exists(os.path.join(tempdir, "metric3_raw.csv")))
            # check the metric_raw.csv and content
            with open(os.path.join(tempdir, "metric3_raw.csv")) as f:
                f_csv = csv.reader(f)
                for i, row in enumerate(f_csv):
                    if i > 0:
                        expected = [f"{fnames[i - 1]}\t{float(i):.4f}\t{float(i + 1):.4f}\t{i + 0.5:.4f}"]
                        self.assertEqual(row, expected)
            self.assertTrue(os.path.exists(os.path.join(tempdir, "metric3_summary.csv")))
            # check the metric_summary.csv and content
            with open(os.path.join(tempdir, "metric3_summary.csv")) as f:
                f_csv = csv.reader(f)
                for i, row in enumerate(f_csv):
                    if i == 1:
                        self.assertEqual(row, ["class0\t2.0000\t2.0000\t3.0000\t1.0000\t2.8000\t0.8165\t3.0000"])
                    elif i == 2:
                        self.assertEqual(row, ["class1\t3.0000\t3.0000\t4.0000\t2.0000\t3.8000\t0.8165\t3.0000"])
                    elif i == 3:
                        self.assertEqual(row, ["mean\t2.5000\t2.5000\t3.5000\t1.5000\t3.3000\t0.8165\t3.0000"])
            self.assertTrue(os.path.exists(os.path.join(tempdir, "metric4_raw.csv")))
            self.assertTrue(os.path.exists(os.path.join(tempdir, "metric4_summary.csv")))
        dist.barrier()


if __name__ == "__main__":
    unittest.main()
