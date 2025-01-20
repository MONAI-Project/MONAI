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

from monai.apps.pathology.transforms.post.array import GenerateInstanceContour
from monai.transforms import BoundingRect
from monai.utils import min_version, optional_import
from tests.util import TEST_NDARRAYS, assert_allclose

_, has_skimage = optional_import("skimage", "0.19.3", min_version)

y, x = np.ogrid[0:30, 0:30]
get_bbox = BoundingRect()

TEST_CASE_1 = [(x - 2) ** 2 + (y - 2) ** 2 <= 2**2, 3, [0, 0], [[2, 0], [0, 2], [2, 4], [4, 2]]]

TEST_CASE_2 = [(x - 8) ** 2 + (y - 8) ** 2 <= 2**2, 3, [8, 8], [[10, 8], [8, 10], [10, 12], [12, 10]]]

TEST_CASE_3 = [
    (x - 5) ** 2 / 3**2 + (y - 5) ** 2 / 2**2 <= 1,
    3,
    [2, 3],
    [[5, 3], [4, 4], [3, 4], [2, 5], [3, 6], [4, 6], [5, 7], [6, 6], [7, 6], [8, 5], [7, 4], [6, 4]],
]

TEST_CASE = []
for p in TEST_NDARRAYS:
    TEST_CASE.append([p, *TEST_CASE_1])
    TEST_CASE.append([p, *TEST_CASE_2])
    TEST_CASE.append([p, *TEST_CASE_3])


@unittest.skipUnless(has_skimage, "Requires scikit-image library.")
class TestGenerateInstanceContour(unittest.TestCase):
    @parameterized.expand(TEST_CASE)
    def test_shape(self, in_type, test_data, min_num_points, offset, expected):
        inst_bbox = get_bbox(test_data[None])
        inst_map = test_data[inst_bbox[0][0] : inst_bbox[0][1], inst_bbox[0][2] : inst_bbox[0][3]]
        result = GenerateInstanceContour(min_num_points=min_num_points)(in_type(inst_map[None]), offset=offset)
        assert_allclose(result, expected, type_test=False)


if __name__ == "__main__":
    unittest.main()
