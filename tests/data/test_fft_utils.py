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

from monai.data.fft_utils import fftn_centered, ifftn_centered
from tests.util import TEST_NDARRAYS, assert_allclose

#
im = [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]
res = [
    [[[0.0, 0.0], [0.0, 3.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]]
]
TESTS = []
for p in TEST_NDARRAYS:
    TESTS.append((p(im), p(res)))

#
TESTS_CONSISTENCY = []
for p in TEST_NDARRAYS:
    TESTS_CONSISTENCY.append(p(im))

#
im_complex = [
    [[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]]
]
TESTS_CONSISTENCY_COMPLEX = []
for p in TEST_NDARRAYS:
    TESTS_CONSISTENCY_COMPLEX.append(p(im_complex))


class TestFFT(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test(self, test_data, res_data):
        result = fftn_centered(test_data, spatial_dims=2, is_complex=False)
        assert_allclose(result, res_data, type_test=True)

    @parameterized.expand(TESTS_CONSISTENCY)
    def test_consistency(self, test_data):
        result = fftn_centered(test_data, spatial_dims=2, is_complex=False)
        result = ifftn_centered(result, spatial_dims=2, is_complex=True)
        result = (result[..., 0] ** 2 + result[..., 1] ** 2) ** 0.5
        assert_allclose(result, test_data, type_test=False)

    @parameterized.expand(TESTS_CONSISTENCY_COMPLEX)
    def test_consistency_complex(self, test_data):
        result = fftn_centered(test_data, spatial_dims=2)
        result = ifftn_centered(result, spatial_dims=2)
        assert_allclose(result, test_data, type_test=False)


if __name__ == "__main__":
    unittest.main()
