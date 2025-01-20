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

from monai.utils import optional_import
from monai.utils.type_conversion import get_equivalent_dtype, safe_dtype_range
from tests.util import HAS_CUPY, TEST_NDARRAYS_ALL, assert_allclose

cp, _ = optional_import("cupy")

TESTS: list[tuple] = []
for in_type in TEST_NDARRAYS_ALL + (int, float):
    TESTS.append((in_type(np.array(1.0)), in_type(np.array(1.0)), None))  # type: ignore
    if in_type is not float:
        TESTS.append((in_type(np.array(256)), in_type(np.array(255)), np.uint8))  # type: ignore
        TESTS.append((in_type(np.array(-12)), in_type(np.array(0)), np.uint8))  # type: ignore
for in_type in TEST_NDARRAYS_ALL:
    TESTS.append((in_type(np.array([[256, 255], [-12, 0]])), in_type(np.array([[255, 255], [0, 0]])), np.uint8))

TESTS_LIST: list[tuple] = []
for in_type in TEST_NDARRAYS_ALL + (int, float):
    TESTS_LIST.append(
        (
            [in_type(np.array(1.0)), in_type(np.array(1.0))],  # type: ignore
            [in_type(np.array(1.0)), in_type(np.array(1.0))],  # type: ignore
            None,
        )
    )
    if in_type is not float:
        TESTS_LIST.append(
            (
                [in_type(np.array(257)), in_type(np.array(-12))],  # type: ignore
                [in_type(np.array(255)), in_type(np.array(0))],  # type: ignore
                np.uint8,
            )
        )

TESTS_CUPY = [[np.array(1.0), np.array(1.0), None], [np.array([-12]), np.array([0]), np.uint8]]


class TesSafeDtypeRange(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_safe_dtype_range(self, in_image, im_out, out_dtype):
        result = safe_dtype_range(in_image, out_dtype)
        # check type is unchanged
        self.assertEqual(type(in_image), type(result))
        # check dtype is unchanged
        if isinstance(in_image, (np.ndarray, torch.Tensor)):
            self.assertEqual(in_image.dtype, result.dtype)
        # check output
        assert_allclose(result, im_out)

    @parameterized.expand(TESTS_LIST)
    def test_safe_dtype_range_list(self, in_image, im_out, out_dtype):
        output_type = type(im_out[0])
        result = safe_dtype_range(in_image, dtype=out_dtype)
        # check type is unchanged
        self.assertEqual(type(result), type(im_out))
        # check output
        for i, _result in enumerate(result):
            assert_allclose(_result, im_out[i])
        # check dtype is unchanged
        if isinstance(in_image, (np.ndarray, torch.Tensor)):
            if out_dtype is None:
                self.assertEqual(result[0].dtype, im_out[0].dtype)
            else:
                _out_dtype = get_equivalent_dtype(out_dtype, output_type)
                self.assertEqual(result[0].dtype, _out_dtype)

    @parameterized.expand(TESTS_CUPY)
    @unittest.skipUnless(HAS_CUPY, "Requires CuPy")
    def test_type_cupy(self, in_image, im_out, out_dtype):
        in_image = cp.asarray(in_image)
        result = safe_dtype_range(in_image, dtype=out_dtype)
        # check type is unchanged
        self.assertEqual(type(in_image), type(result))
        # check dtype is unchanged
        self.assertEqual(result.dtype, in_image.dtype)
        # check output
        self.assertEqual(result, cp.asarray(im_out))


if __name__ == "__main__":
    unittest.main()
