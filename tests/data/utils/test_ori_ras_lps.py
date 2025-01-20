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

from monai.data.utils import orientation_ras_lps
from tests.util import TEST_NDARRAYS, assert_allclose

TEST_CASES_AFFINE = []
for p in TEST_NDARRAYS:
    case_1d = p([[1.0, 0.0], [1.0, 1.0]]), p([[-1.0, 0.0], [1.0, 1.0]])
    TEST_CASES_AFFINE.append(case_1d)
    case_2d_1 = (p([[1.0, 0.0, 1.0], [1.0, 1.0, 1.0]]), p([[-1.0, 0.0, -1.0], [1.0, 1.0, 1.0]]))
    TEST_CASES_AFFINE.append(case_2d_1)
    case_2d_2 = (
        p([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
        p([[-1.0, 0.0, -1.0], [0.0, -1.0, -1.0], [1.0, 1.0, 1.0]]),
    )
    TEST_CASES_AFFINE.append(case_2d_2)
    case_3d = (
        p([[1.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 2.0], [1.0, 1.0, 1.0, 3.0]]),
        p([[-1.0, 0.0, -1.0, -1.0], [0.0, -1.0, -1.0, -2.0], [1.0, 1.0, 1.0, 3.0]]),
    )
    TEST_CASES_AFFINE.append(case_3d)
    case_4d = p(np.ones((5, 5))), p([[-1] * 5, [-1] * 5, [1] * 5, [1] * 5, [1] * 5])
    TEST_CASES_AFFINE.append(case_4d)


class TestITKWriter(unittest.TestCase):
    @parameterized.expand(TEST_CASES_AFFINE)
    def test_ras_to_lps(self, param, expected):
        assert_allclose(orientation_ras_lps(param), expected)


if __name__ == "__main__":
    unittest.main()
