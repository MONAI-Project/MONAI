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

import sys
import time
import unittest

from parameterized import parameterized

from monai.transforms import SelectItemsd

TEST_CASE_1 = [{"keys": [str(i) for i in range(30)]}, 30]


class TestSelectItemsd(unittest.TestCase):

    @parameterized.expand([TEST_CASE_1])
    def test_memory(self, input_param, expected_key_size):
        input_data = {}
        for i in range(50):
            input_data[str(i)] = [time.time()] * 100000
        result = SelectItemsd(**input_param)(input_data)
        self.assertEqual(len(result.keys()), expected_key_size)
        self.assertSetEqual(set(result.keys()), set(input_param["keys"]))
        self.assertGreaterEqual(
            sys.getsizeof(input_data) * float(expected_key_size) / len(input_data), sys.getsizeof(result)
        )


if __name__ == "__main__":
    unittest.main()
