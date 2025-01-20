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
from parameterized import parameterized

from monai.data import partition_dataset_classes

TEST_CASE_1 = [
    {
        "data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        "classes": [2, 0, 2, 1, 3, 2, 2, 0, 2, 0, 3, 3, 1, 3],
        "ratios": [2, 1],
        "num_partitions": None,
        "shuffle": False,
        "seed": 0,
        "drop_last": False,
        "even_divisible": False,
    },
    [[2, 8, 4, 1, 3, 6, 5, 11, 12], [10, 13, 7, 9, 14]],
]

TEST_CASE_2 = [
    {
        "data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        "classes": [2, 0, 2, 1, 3, 2, 2, 0, 2, 0, 3, 3, 1, 3],
        "ratios": None,
        "num_partitions": 2,
        "shuffle": False,
        "seed": 0,
        "drop_last": False,
        "even_divisible": False,
    },
    [[2, 10, 4, 1, 6, 9, 5, 12], [8, 13, 3, 7, 11, 14]],
]

TEST_CASE_3 = [
    {
        "data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        "classes": [2, 0, 2, 1, 3, 2, 2, 0, 2, 0, 3, 3, 1, 3],
        "ratios": None,
        "num_partitions": 2,
        "shuffle": False,
        "seed": 0,
        "drop_last": False,
        "even_divisible": True,
    },
    [[2, 10, 4, 1, 6, 9, 5, 12], [8, 2, 13, 3, 7, 1, 11, 14]],
]

TEST_CASE_4 = [
    {
        "data": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
        "classes": np.array([2, 0, 2, 1, 3, 2, 2, 0, 2, 0, 3, 3, 1, 3]),
        "ratios": [1, 2],
        "num_partitions": None,
        "shuffle": True,
        "seed": 123,
        "drop_last": False,
        "even_divisible": False,
    },
    [[13, 7, 14, 2, 3], [6, 8, 1, 5, 12, 11, 4, 9, 10]],
]


class TestPartitionDatasetClasses(unittest.TestCase):

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4])
    def test_value(self, input_param, result):
        self.assertListEqual(partition_dataset_classes(**input_param), result)


if __name__ == "__main__":
    unittest.main()
