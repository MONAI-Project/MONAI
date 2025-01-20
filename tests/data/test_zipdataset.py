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

import torch
from parameterized import parameterized

from monai.data import ZipDataset


class Dataset_(torch.utils.data.Dataset):

    def __init__(self, length, index_only=True):
        self.len = length
        self.index_only = index_only

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if self.index_only:
            return index
        return 1, 2, index


TEST_CASE_1 = [[Dataset_(5), Dataset_(5), Dataset_(5)], None, (0, 0, 0), 5]

TEST_CASE_2 = [[Dataset_(3), Dataset_(4), Dataset_(5)], None, (0, 0, 0), 3]

TEST_CASE_3 = [[Dataset_(3), Dataset_(4, index_only=False), Dataset_(5)], None, (0, 1, 2, 0, 0), 3]

TEST_CASE_4 = [
    [Dataset_(3), Dataset_(4, index_only=False), Dataset_(5)],
    lambda x: [i + 1 for i in x],
    (1, 2, 3, 1, 1),
    3,
]


class TestZipDataset(unittest.TestCase):

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4])
    def test_value(self, datasets, transform, expected_output, expected_length):
        test_dataset = ZipDataset(datasets=datasets, transform=transform)
        self.assertEqual(test_dataset[0], expected_output)
        self.assertEqual(len(test_dataset), expected_length)

    def test_slicing(self):
        test_dataset = ZipDataset(datasets=[Dataset_(5), Dataset_(5), Dataset_(5)], transform=None)
        subset = test_dataset[0:2]
        self.assertEqual(subset[-1], (1, 1, 1))
        self.assertEqual(len(subset), 2)

    def test_sequence(self):
        test_dataset = ZipDataset(datasets=[Dataset_(5), Dataset_(5), Dataset_(5)], transform=None)
        subset = test_dataset[[1, 3, 4]]
        self.assertEqual(subset[-1], (4, 4, 4))
        self.assertEqual(len(subset), 3)


if __name__ == "__main__":
    unittest.main()
