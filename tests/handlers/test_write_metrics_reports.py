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

import torch

from monai.handlers.utils import write_metrics_reports


class TestWriteMetricsReports(unittest.TestCase):

    def test_content(self):
        with tempfile.TemporaryDirectory() as tempdir:
            write_metrics_reports(
                save_dir=Path(tempdir),
                images=["filepath1", "filepath2"],
                metrics={"metric1": 1, "metric2": 2},
                metric_details={"metric3": torch.tensor([[1, 2], [2, 3]]), "metric4": torch.tensor([[5, 6], [7, 8]])},
                summary_ops=["mean", "median", "max", "90percentile"],
                deli="\t",
                output_type="csv",
            )

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
            with open(os.path.join(tempdir, "metric3_summary.csv")) as f:
                f_csv = csv.reader(f)
                for i, row in enumerate(f_csv):
                    if i == 1:
                        self.assertEqual(row, ["class0\t1.5000\t1.5000\t2.0000\t1.9000"])
                    elif i == 2:
                        self.assertEqual(row, ["class1\t2.5000\t2.5000\t3.0000\t2.9000"])
                    elif i == 3:
                        self.assertEqual(row, ["mean\t2.0000\t2.0000\t2.5000\t2.4000"])
            self.assertTrue(os.path.exists(os.path.join(tempdir, "metric4_raw.csv")))
            self.assertTrue(os.path.exists(os.path.join(tempdir, "metric4_summary.csv")))


if __name__ == "__main__":
    unittest.main()
