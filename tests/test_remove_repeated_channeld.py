# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np
from parameterized import parameterized

from monai.transforms import RemoveRepeatedChanneld

TEST_CASE_1 = [
    {"keys": ["img"], "repeats": 2},
    {"img": np.array([[1, 2], [1, 2], [3, 4], [3, 4]]), "seg": np.array([[1, 2], [1, 2], [3, 4], [3, 4]])},
    (2, 2),
]


class TestRemoveRepeatedChanneld(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1])
    def test_shape(self, input_param, input_data, expected_shape):
        result = RemoveRepeatedChanneld(**input_param)(input_data)
        self.assertEqual(result["img"].shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
