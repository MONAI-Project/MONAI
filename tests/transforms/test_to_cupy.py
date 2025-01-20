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
from unittest import skipUnless

import numpy as np
import torch

from monai.transforms import ToCupy
from monai.utils import optional_import
from tests.util import HAS_CUPY, skip_if_no_cuda

cp, _ = optional_import("cupy")


@skipUnless(HAS_CUPY, "CuPy is required.")
class TestToCupy(unittest.TestCase):
    def test_cupy_input(self):
        test_data = cp.array([[1, 2], [3, 4]], dtype=cp.float32)
        test_data = cp.rot90(test_data)
        self.assertFalse(test_data.flags["C_CONTIGUOUS"])
        result = ToCupy()(test_data)
        self.assertTrue(result.dtype == cp.float32)
        self.assertTrue(isinstance(result, cp.ndarray))
        self.assertTrue(result.flags["C_CONTIGUOUS"])
        cp.testing.assert_allclose(result, test_data)

    def test_cupy_input_dtype(self):
        test_data = cp.array([[1, 2], [3, 4]], dtype=cp.float32)
        test_data = cp.rot90(test_data)
        self.assertFalse(test_data.flags["C_CONTIGUOUS"])
        result = ToCupy(cp.uint8)(test_data)
        self.assertTrue(result.dtype == cp.uint8)
        self.assertTrue(isinstance(result, cp.ndarray))
        self.assertTrue(result.flags["C_CONTIGUOUS"])
        cp.testing.assert_allclose(result, test_data)

    def test_numpy_input(self):
        test_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
        test_data = np.rot90(test_data)
        self.assertFalse(test_data.flags["C_CONTIGUOUS"])
        result = ToCupy()(test_data)
        self.assertTrue(result.dtype == cp.float32)
        self.assertTrue(isinstance(result, cp.ndarray))
        self.assertTrue(result.flags["C_CONTIGUOUS"])
        cp.testing.assert_allclose(result, test_data)

    def test_numpy_input_dtype(self):
        test_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
        test_data = np.rot90(test_data)
        self.assertFalse(test_data.flags["C_CONTIGUOUS"])
        result = ToCupy(np.uint8)(test_data)
        self.assertEqual(result.dtype, cp.uint8)
        self.assertIsInstance(result, cp.ndarray)
        self.assertTrue(result.flags["C_CONTIGUOUS"])
        cp.testing.assert_allclose(result, test_data)

    def test_tensor_input(self):
        test_data = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
        test_data = test_data.rot90()
        self.assertFalse(test_data.is_contiguous())
        result = ToCupy()(test_data)
        self.assertEqual(result.dtype, cp.float32)
        self.assertIsInstance(result, cp.ndarray)
        self.assertTrue(result.flags["C_CONTIGUOUS"])
        cp.testing.assert_allclose(result, test_data)

    @skip_if_no_cuda
    def test_tensor_cuda_input(self):
        test_data = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32).cuda()
        test_data = test_data.rot90()
        self.assertFalse(test_data.is_contiguous())
        result = ToCupy()(test_data)
        self.assertEqual(result.dtype, cp.float32)
        self.assertIsInstance(result, cp.ndarray)
        self.assertTrue(result.flags["C_CONTIGUOUS"])
        cp.testing.assert_allclose(result, test_data)

    @skip_if_no_cuda
    def test_tensor_cuda_input_dtype(self):
        test_data = torch.tensor([[1, 2], [3, 4]], dtype=torch.uint8).cuda()
        test_data = test_data.rot90()
        self.assertFalse(test_data.is_contiguous())

        result = ToCupy(dtype="float32")(test_data)
        self.assertEqual(result.dtype, cp.float32)
        self.assertIsInstance(result, cp.ndarray)
        self.assertTrue(result.flags["C_CONTIGUOUS"])
        cp.testing.assert_allclose(result, test_data)

    def test_list_tuple(self):
        test_data = [[1, 2], [3, 4]]
        result = ToCupy(wrap_sequence=True)(test_data)
        cp.testing.assert_allclose(result, cp.asarray(test_data))
        test_data = ((1, 2), (3, 4))
        result = ToCupy(wrap_sequence=True)(test_data)
        cp.testing.assert_allclose(result, cp.asarray(test_data))


if __name__ == "__main__":
    unittest.main()
