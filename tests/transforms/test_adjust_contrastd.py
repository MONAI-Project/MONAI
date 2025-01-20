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

from monai.transforms import AdjustContrastd
from tests.util import TEST_NDARRAYS, NumpyImageTestCase2D, assert_allclose

TESTS = []
for invert_image in (True, False):
    for retain_stats in (True, False):
        TEST_CASE_1 = [1.0, invert_image, retain_stats]
        TEST_CASE_2 = [0.5, invert_image, retain_stats]
        TEST_CASE_3 = [4.5, invert_image, retain_stats]

        TESTS.extend([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3])


class TestAdjustContrastd(NumpyImageTestCase2D):
    @parameterized.expand(TESTS)
    def test_correct_results(self, gamma, invert_image, retain_stats):
        adjuster = AdjustContrastd("img", gamma=gamma, invert_image=invert_image, retain_stats=retain_stats)
        for p in TEST_NDARRAYS:
            result = adjuster({"img": p(self.imt)})
            if invert_image:
                self.imt = -self.imt

            if retain_stats:
                mn = self.imt.mean()
                sd = self.imt.std()

            epsilon = 1e-7
            img_min = self.imt.min()
            img_range = self.imt.max() - img_min

            expected = np.power(((self.imt - img_min) / float(img_range + epsilon)), gamma) * img_range + img_min

            if retain_stats:
                # zero mean and normalize
                expected = expected - expected.mean()
                expected = expected / (expected.std() + 1e-8)
                # restore old mean and standard deviation
                expected = sd * expected + mn

            if invert_image:
                expected = -expected
            assert_allclose(result["img"], expected, atol=1e-05, type_test="tensor")


if __name__ == "__main__":
    unittest.main()
