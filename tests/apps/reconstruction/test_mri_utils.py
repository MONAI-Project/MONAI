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

from parameterized import parameterized

from monai.apps.reconstruction.mri_utils import root_sum_of_squares
from tests.util import TEST_NDARRAYS, assert_allclose

# root_sum_of_squares
im = [[3.0, 4.0], [3.0, 4.0]]
res = [5.0, 5.0]
TESTS = []
for p in TEST_NDARRAYS:
    TESTS.append((p(im), p(res)))


class TestMRIUtils(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_rss(self, test_data, res_data):
        result = root_sum_of_squares(test_data, spatial_dim=1)
        assert_allclose(result, res_data, type_test=False)


if __name__ == "__main__":
    unittest.main()
