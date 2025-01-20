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

from monai.transforms import DeleteItemsd
from monai.utils.enums import PostFix

TEST_CASE_1 = [{"keys": [str(i) for i in range(30)]}, 20]

TEST_CASE_2 = [{"keys": ["image/" + str(i) for i in range(30)], "sep": "/"}, 20]

TEST_CASE_3 = [{"keys": "meta_dict%0008\\|[0-9]", "sep": "%", "use_re": True}]


class TestDeleteItemsd(unittest.TestCase):

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    def test_memory(self, input_param, expected_key_size):
        input_data = {"image": {}} if "sep" in input_param else {}
        for i in range(50):
            if "sep" in input_param:
                input_data["image"][str(i)] = [time.time()] * 100000
            else:
                input_data[str(i)] = [time.time()] * 100000
        result = DeleteItemsd(**input_param)(input_data)
        if "sep" in input_param:
            self.assertEqual(len(result["image"].keys()), expected_key_size)
        else:
            self.assertEqual(len(result.keys()), expected_key_size)
        self.assertGreaterEqual(
            sys.getsizeof(input_data) * float(expected_key_size) / len(input_data), sys.getsizeof(result)
        )

    @parameterized.expand([TEST_CASE_3])
    def test_re(self, input_param):
        input_data = {"image": [1, 2, 3], PostFix.meta(): {"0008|0005": 1, "0008|1050": 2, "0008test": 3}}
        result = DeleteItemsd(**input_param)(input_data)
        self.assertEqual(result[PostFix.meta()]["0008test"], 3)
        self.assertEqual(len(result[PostFix.meta()]), 1)


if __name__ == "__main__":
    unittest.main()
