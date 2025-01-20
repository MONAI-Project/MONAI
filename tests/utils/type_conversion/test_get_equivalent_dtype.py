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
import torch
from parameterized import parameterized

from monai.utils.type_conversion import get_equivalent_dtype, get_numpy_dtype_from_string, get_torch_dtype_from_string
from tests.util import TEST_NDARRAYS

DTYPES = [torch.float32, np.float32, np.dtype(np.float32)]

TESTS = []
for p in TEST_NDARRAYS:
    for im_dtype in DTYPES:
        TESTS.append((p(np.array(1.0, dtype=np.float32)), im_dtype))


class TestGetEquivalentDtype(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_get_equivalent_dtype(self, im, input_dtype):
        out_dtype = get_equivalent_dtype(input_dtype, type(im))
        self.assertEqual(out_dtype, im.dtype)

    def test_native_type(self):
        """the get_equivalent_dtype currently doesn't change the build-in type"""
        n_type = [float, int, bool]
        for n in n_type:
            for im_dtype in DTYPES:
                out_dtype = get_equivalent_dtype(n, type(im_dtype))
                self.assertEqual(out_dtype, n)

    @parameterized.expand(
        [
            ["float", np.float64],
            ["float32", np.float32],
            ["np.float32", np.float32],
            ["float64", np.float64],
            ["torch.float64", np.float64],
        ]
    )
    def test_from_string(self, dtype_str, expected_np):
        expected_pt = get_equivalent_dtype(expected_np, torch.Tensor)
        # numpy
        dtype = get_numpy_dtype_from_string(dtype_str)
        self.assertEqual(dtype, expected_np)
        # torch
        dtype = get_torch_dtype_from_string(dtype_str)
        self.assertEqual(dtype, expected_pt)


if __name__ == "__main__":
    unittest.main()
