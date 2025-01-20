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

import unittest

import numpy as np
import torch
from parameterized import parameterized

from monai.metrics import CumulativeAverage

TEST_CASE_1 = []
TEST_CASE_1.append([{"vals": [1, 2, 3], "avg": 2}])
TEST_CASE_1.append([{"vals": [[1, 1, 1], [2, 2, 2], [3, 6, 9]], "avg": [2, 3, 4]}])

TEST_CASE_1.append([{"vals": [2, 4, 6], "counts": [2, 1, 2], "avg": 4}])
TEST_CASE_1.append(
    [{"vals": [[3, 2, 1], [2, 3, 2], [0, 0, 9]], "counts": [[4, 4, 4], [4, 4, 4], [2, 2, 2]], "avg": [2, 2, 3]}]
)

TEST_CASE_1.append([{"vals": [1, 2, float("nan")], "avg": 1.5}])


class TestAverageMeter(unittest.TestCase):

    @parameterized.expand(TEST_CASE_1)
    def test_value_all(self, data):
        # test orig
        self.run_test(data)

        # test in numpy
        data["vals"] = np.array(data["vals"])
        data["avg"] = np.array(data["avg"])
        self.run_test(data)

        # test as Tensors
        data["vals"] = torch.tensor(data["vals"])
        data["avg"] = torch.tensor(data["avg"], dtype=torch.float)
        self.run_test(data)

        if torch.cuda.is_available():
            data["vals"] = data["vals"].cuda()
            self.run_test(data)

    def run_test(self, data):
        vals = data["vals"]
        avg = data["avg"]

        counts = data.get("counts", None)
        if counts is not None and not isinstance(counts, list) and isinstance(vals, list):
            counts = [counts] * len(vals)

        avg_meter = CumulativeAverage()
        for i in range(len(vals)):
            if counts is not None:
                avg_meter.append(vals[i], counts[i])
            else:
                avg_meter.append(vals[i])

        np.testing.assert_equal(avg_meter.aggregate(), avg)


if __name__ == "__main__":
    unittest.main()
