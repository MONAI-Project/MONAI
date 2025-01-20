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

from monai.apps.pathology.transforms.post.dictionary import GenerateInstanceTyped
from tests.util import TEST_NDARRAYS, assert_allclose

y, x = np.ogrid[0:30, 0:30]

TEST_CASE_1 = [
    (x - 2) ** 2 + (y - 2) ** 2 <= 2**2,
    (x - 2) ** 2 + (y - 3) ** 2 <= 2**2,
    np.array([[0, 5, 0, 5]]),
    [1, 0.6666666111111158],
]

TEST_CASE_2 = [
    (x - 8) ** 2 / 3**2 + (y - 8) ** 2 / 2**2 <= 1,
    (x - 7) ** 2 / 3**2 + (y - 7) ** 2 / 2**2 <= 1,
    np.array([[6, 11, 5, 12]]),
    [1, 0.7058823114186875],
]
TEST_CASE = []
for p in TEST_NDARRAYS:
    TEST_CASE.append([p, *TEST_CASE_1])
    TEST_CASE.append([p, *TEST_CASE_2])


class TestGenerateInstanceTyped(unittest.TestCase):
    @parameterized.expand(TEST_CASE)
    def test_shape(self, in_type, type_pred, seg_pred, bbox, expected):
        test_data = {"type_pred": in_type(type_pred[None]), "seg": in_type(seg_pred[None]), "bbox": bbox, "id": 1}
        result = GenerateInstanceTyped(keys="type_pred")(test_data)
        assert_allclose(result["type_info"]["inst_type"], expected[0])
        assert_allclose(result["type_info"]["type_prob"], expected[1])


if __name__ == "__main__":
    unittest.main()
