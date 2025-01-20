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
from ignite.engine import Engine, Events

from monai.handlers import MetricsSaver
from monai.utils.enums import PostFix


class TestHandlerMetricsSaver(unittest.TestCase):

    def test_content(self):
        with tempfile.TemporaryDirectory() as tempdir:
            metrics_saver = MetricsSaver(
                save_dir=tempdir,
                metrics=["metric1", "metric2"],
                metric_details=["metric3", "metric4"],
                batch_transform=lambda x: x[PostFix.meta("image")],
                summary_ops=["mean", "median", "max", "5percentile", "95percentile", "notnans"],
                delimiter="\t",
            )
            # set up engine
            data = [
                {PostFix.meta("image"): {"filename_or_obj": ["filepath1"]}},
                {PostFix.meta("image"): {"filename_or_obj": ["filepath2"]}},
            ]

            def _val_func(engine, batch):
                pass

            engine = Engine(_val_func)

            @engine.on(Events.EPOCH_COMPLETED)
            def _save_metrics(engine):
                engine.state.metrics = {"metric1": 1, "metric2": 2}
                engine.state.metric_details = {
                    "metric3": torch.tensor([[1, 2], [2, 3]]),
                    "metric4": torch.tensor([[5, 6], [7, torch.tensor(float("nan"))]]),
                }

            metrics_saver.attach(engine)
            engine.run(data, max_epochs=1)

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
                        self.assertEqual(row, [f"filepath{i}\t{float(i):.4f}\t{float(i + 1):.4f}\t{i + 0.5:.4f}"])
            self.assertTrue(os.path.exists(os.path.join(tempdir, "metric3_summary.csv")))
            # check the metric_summary.csv and content
            with open(os.path.join(tempdir, "metric4_summary.csv")) as f:
                f_csv = csv.reader(f)
                for i, row in enumerate(f_csv):
                    if i == 1:
                        self.assertEqual(row, ["class0\t6.0000\t6.0000\t7.0000\t5.1000\t6.9000\t2.0000"])
                    elif i == 2:
                        self.assertEqual(row, ["class1\t6.0000\t6.0000\t6.0000\t6.0000\t6.0000\t1.0000"])
                    elif i == 3:
                        self.assertEqual(row, ["mean\t6.2500\t6.2500\t7.0000\t5.5750\t6.9250\t2.0000"])
            self.assertTrue(os.path.exists(os.path.join(tempdir, "metric4_raw.csv")))
            self.assertTrue(os.path.exists(os.path.join(tempdir, "metric3_summary.csv")))


if __name__ == "__main__":
    unittest.main()
