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

from monai.data import json_hashing, pickle_hashing
from monai.utils import set_determinism


class TestPickleHashing(unittest.TestCase):

    def test_pickle(self):
        set_determinism(0)
        data1 = np.random.rand(10)
        data2 = np.random.rand(10)
        set_determinism(0)
        data3 = np.random.rand(10)
        data4 = np.random.rand(10)
        set_determinism(None)

        h1 = pickle_hashing(data1)
        h2 = pickle_hashing(data3)
        self.assertEqual(h1, h2)

        data_dict1 = {"b": data2, "a": data1}
        data_dict2 = {"a": data3, "b": data4}

        h1 = pickle_hashing(data_dict1)
        h2 = pickle_hashing(data_dict2)
        self.assertEqual(h1, h2)

        with self.assertRaises(TypeError):
            json_hashing(data_dict1)


class TestJSONHashing(unittest.TestCase):

    def test_json(self):
        data_dict1 = {"b": "str2", "a": "str1"}
        data_dict2 = {"a": "str1", "b": "str2"}

        h1 = json_hashing(data_dict1)
        h2 = json_hashing(data_dict2)
        self.assertEqual(h1, h2)


if __name__ == "__main__":
    unittest.main()
