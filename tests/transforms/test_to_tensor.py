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

from monai.transforms import ToTensor
from tests.util import HAS_CUPY, TEST_NDARRAYS, assert_allclose, optional_import

cp, _ = optional_import("cupy")

im = [[1, 2], [3, 4]]

TESTS = [(im, (2, 2))]
for p in TEST_NDARRAYS:
    TESTS.append((p(im), (2, 2)))

TESTS_SINGLE = [[5]]
for p in TEST_NDARRAYS:
    TESTS_SINGLE.append([p(5)])


class TestToTensor(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_array_input(self, test_data, expected_shape):
        result = ToTensor(dtype=torch.float32, device="cpu", wrap_sequence=True)(test_data)
        self.assertTrue(isinstance(result, torch.Tensor))
        assert_allclose(result, test_data, type_test=False)
        self.assertTupleEqual(result.shape, expected_shape)

    @parameterized.expand(TESTS_SINGLE)
    def test_single_input(self, test_data):
        result = ToTensor(track_meta=True)(test_data)
        self.assertTrue(isinstance(result, torch.Tensor))
        assert_allclose(result, test_data, type_test=False)
        self.assertEqual(result.ndim, 0)

    @unittest.skipUnless(HAS_CUPY, "CuPy is required.")
    def test_cupy(self):
        test_data = [[1, 2], [3, 4]]
        cupy_array = cp.ascontiguousarray(cp.asarray(test_data))
        result = ToTensor()(cupy_array)
        self.assertTrue(isinstance(result, torch.Tensor))
        assert_allclose(result, test_data, type_test=False)


if __name__ == "__main__":
    unittest.main()
