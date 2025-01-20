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

from monai.data import MetaTensor, list_data_collate

a = {"image": np.array([1, 2, 3]), "label": MetaTensor([4, 5, 6])}
b = {"image": np.array([7, 8, 9]), "label": MetaTensor([10, 11, 12])}
c = {"image": np.array([13, 14, 15]), "label": MetaTensor([16, 7, 18])}
d = {"image": np.array([19, 20, 21]), "label": MetaTensor([22, 23, 24])}
TEST_CASE_1 = [[[a, b], [c, d]], dict, torch.Size([4, 3])]  # dataset returns a list of dictionary data

e = (np.array([1, 2, 3]), MetaTensor([4, 5, 6]))
f = (np.array([7, 8, 9]), MetaTensor([10, 11, 12]))
g = (np.array([13, 14, 15]), MetaTensor([16, 7, 18]))
h = (np.array([19, 20, 21]), MetaTensor([22, 23, 24]))
TEST_CASE_2 = [[[e, f], [g, h]], list, torch.Size([4, 3])]  # dataset returns a list of tuple data

g_m = (np.array([13, 14, 15]), MetaTensor([16, 7, 18], meta={"key1": 0}))
h_m = (np.array([19, 20, 21]), MetaTensor([22, 23, 24], meta={"key2": 1}))
TEST_CASE_3 = [[[g_m], [h_m]], list, torch.Size([2, 3])]


class TestListDataCollate(unittest.TestCase):

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3])
    def test_type_shape(self, input_data, expected_type, expected_shape):
        result = list_data_collate(input_data)
        self.assertIsInstance(result, expected_type)
        if isinstance(result, dict):
            image = result["image"]
            label = result["label"]
        else:
            image = result[0]
            label = result[1]
        self.assertEqual(image.shape, expected_shape)
        self.assertEqual(label.shape, expected_shape)
        self.assertTrue(isinstance(label, MetaTensor))
        self.assertTrue(label.is_batch, True)


if __name__ == "__main__":
    unittest.main()
