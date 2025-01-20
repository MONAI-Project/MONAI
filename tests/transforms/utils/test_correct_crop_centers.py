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

import torch
from parameterized import parameterized

from monai.transforms.utils import correct_crop_centers
from tests.util import assert_allclose

TESTS = [[[1, 5, 0], [2, 2, 2], [10, 10, 10]], [[4, 4, 4], [2, 2, 1], [10, 10, 10]]]


class TestCorrectCropCenters(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_torch(self, spatial_size, centers, label_spatial_shape):
        result1 = correct_crop_centers(centers, spatial_size, label_spatial_shape)
        centers = [torch.tensor(i) for i in centers]
        result2 = correct_crop_centers(centers, spatial_size, label_spatial_shape)
        assert_allclose(result1, result2)
        self.assertEqual(type(result1[0]), type(result2[0]))


if __name__ == "__main__":
    unittest.main()
