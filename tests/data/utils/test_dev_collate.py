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

import logging
import unittest

import numpy as np
import torch
from parameterized import parameterized

from monai.data.utils import dev_collate

TEST_CASES = [
    [
        [
            {"img": 2, "meta": {"shape": [torch.tensor(1.0)]}},
            {"img": 3, "meta": {"shape": [np.asarray(1.0)]}},
            {"img": 4, "meta": {"shape": [torch.tensor(1.0)]}},
        ],
        "got numpy.ndarray",
    ],
    [[["img", np.array([2])], ["img", np.array([3, 4])], ["img", np.array([4])]], "size"],
    [[["img", [2]], ["img", [3, 4]], ["img", 4]], "type"],
    [[["img", [2, 2]], ["img", [3, 4]], ["img", 4]], "type"],
]


class DevCollateTest(unittest.TestCase):

    @parameterized.expand(TEST_CASES)
    def test_dev_collate(self, inputs, msg):
        with self.assertLogs(level=logging.CRITICAL) as log:
            dev_collate(inputs)
            self.assertRegex(" ".join(log.output), f"{msg}")


if __name__ == "__main__":
    unittest.main()
