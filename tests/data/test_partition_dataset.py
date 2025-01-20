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

from parameterized import parameterized

from monai.data import partition_dataset

TEST_CASE_1 = [
    {
        "data": [1, 2, 3, 4],
        "num_partitions": 2,
        "shuffle": False,
        "seed": 0,
        "drop_last": False,
        "even_divisible": False,
    },
    [[1, 3], [2, 4]],
]

TEST_CASE_2 = [
    {
        "data": [1, 2, 3, 4],
        "num_partitions": 2,
        "shuffle": True,
        "seed": 123,
        "drop_last": False,
        "even_divisible": False,
    },
    [[4, 2], [1, 3]],
]

TEST_CASE_3 = [
    {
        "data": [1, 2, 3, 4, 5],
        "num_partitions": 2,
        "shuffle": False,
        "seed": 0,
        "drop_last": False,
        "even_divisible": False,
    },
    [[1, 3, 5], [2, 4]],
]

TEST_CASE_4 = [
    {
        "data": [1, 2, 3, 4, 5],
        "num_partitions": 2,
        "shuffle": False,
        "seed": 0,
        "drop_last": False,
        "even_divisible": True,
    },
    [[1, 3, 5], [2, 4, 1]],
]

TEST_CASE_5 = [
    {
        "data": [1, 2, 3, 4, 5],
        "num_partitions": 2,
        "shuffle": False,
        "seed": 0,
        "drop_last": True,
        "even_divisible": True,
    },
    [[1, 3], [2, 4]],
]

TEST_CASE_6 = [
    {
        "data": [1, 2, 3, 4, 5],
        "ratios": [3, 2],
        "num_partitions": None,
        "shuffle": False,
        "seed": 0,
        "drop_last": True,
        "even_divisible": True,
    },
    [[1, 2, 3], [4, 5]],
]

TEST_CASE_7 = [
    {
        "data": [1, 2, 3, 4, 5],
        "ratios": [2, 1],
        "num_partitions": None,
        "shuffle": False,
        "seed": 0,
        "drop_last": True,
        "even_divisible": True,
    },
    [[1, 2, 3], [4, 5]],
]

TEST_CASE_8 = [
    {
        "data": [1, 2, 3, 4, 5],
        "ratios": [2, 1],
        "num_partitions": None,
        "shuffle": True,
        "seed": 123,
        "drop_last": True,
        "even_divisible": True,
    },
    [[2, 4, 5], [1, 3]],
]


class TestPartitionDataset(unittest.TestCase):

    @parameterized.expand(
        [TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4, TEST_CASE_5, TEST_CASE_6, TEST_CASE_7, TEST_CASE_8]
    )
    def test_value(self, input_param, result):
        self.assertListEqual(partition_dataset(**input_param), result)


if __name__ == "__main__":
    unittest.main()
