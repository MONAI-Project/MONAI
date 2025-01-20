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

from monai.data import resample_datalist

TEST_CASE_1 = [
    {"data": [1, 2, 3, 4, 5], "factor": 2.5, "random_pick": True, "seed": 123},
    [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 4, 5],
]

TEST_CASE_2 = [
    {"data": [1, 2, 3, 4, 5], "factor": 2.5, "random_pick": False, "seed": 0},
    [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3],
]

TEST_CASE_3 = [{"data": [1, 2, 3, 4, 5], "factor": 0.6, "random_pick": True, "seed": 123}, [2, 4, 5]]


class TestResampleDatalist(unittest.TestCase):

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3])
    def test_value_shape(self, input_param, expected):
        result = resample_datalist(**input_param)
        np.testing.assert_allclose(result, expected)


if __name__ == "__main__":
    unittest.main()
