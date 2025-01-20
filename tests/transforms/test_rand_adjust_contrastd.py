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

from monai.transforms import RandAdjustContrastd
from tests.util import TEST_NDARRAYS, NumpyImageTestCase2D, assert_allclose

TEST_CASE_1 = [(0.5, 4.5)]

TEST_CASE_2 = [1.5]


class TestRandAdjustContrastd(NumpyImageTestCase2D):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    def test_correct_results(self, gamma):
        adjuster = RandAdjustContrastd("img", prob=1.0, gamma=gamma)
        for p in TEST_NDARRAYS:
            result = adjuster({"img": p(self.imt)})
            epsilon = 1e-7
            img_min = self.imt.min()
            img_range = self.imt.max() - img_min
            expected = (
                np.power(((self.imt - img_min) / float(img_range + epsilon)), adjuster.adjuster.gamma_value) * img_range
                + img_min
            )
            assert_allclose(result["img"], expected, rtol=1e-05, type_test="tensor")


if __name__ == "__main__":
    unittest.main()
