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

from monai.transforms import ConvertToMultiChannelBasedOnBratsClassesd

TEST_CASE = [
    {"keys": "label"},
    {"label": np.array([[0, 1, 2], [1, 2, 4], [0, 1, 4]])},
    np.array([[[0, 1, 0], [1, 0, 1], [0, 1, 1]], [[0, 1, 1], [1, 1, 1], [0, 1, 1]], [[0, 0, 0], [0, 0, 1], [0, 0, 1]]]),
]


class TestConvertToMultiChanneld(unittest.TestCase):

    @parameterized.expand([TEST_CASE])
    def test_type_shape(self, keys, data, expected_result):
        result = ConvertToMultiChannelBasedOnBratsClassesd(**keys)(data)
        np.testing.assert_equal(result["label"], expected_result)


if __name__ == "__main__":
    unittest.main()
