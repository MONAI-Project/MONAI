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

from monai.utils import list_to_dict

TEST_CASE_1 = [["a=1", "b=2", "c=3", "d=4"], {"a": 1, "b": 2, "c": 3, "d": 4}]

TEST_CASE_2 = [["a=a", "b=b", "c=c", "d=d"], {"a": "a", "b": "b", "c": "c", "d": "d"}]

TEST_CASE_3 = [["a=0.1", "b=0.2", "c=0.3", "d=0.4"], {"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.4}]

TEST_CASE_4 = [["a=True", "b=TRUE", "c=false", "d=FALSE"], {"a": True, "b": True, "c": False, "d": False}]

TEST_CASE_5 = [
    ["a='1'", "b=2 ", " c = 3", "d='test'", "'e'=0", "f", "g=None"],
    {"a": 1, "b": 2, "c": 3, "d": "test", "e": 0, "f": None, "g": None},
]


class TestListToDict(unittest.TestCase):

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4, TEST_CASE_5])
    def test_value_shape(self, input, output):
        result = list_to_dict(input)
        self.assertDictEqual(result, output)


if __name__ == "__main__":
    unittest.main()
