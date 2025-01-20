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

from monai.transforms import ToNumpyd
from monai.utils import optional_import
from tests.util import HAS_CUPY, assert_allclose, skip_if_no_cuda

cp, _ = optional_import("cupy")


class TestToNumpyd(unittest.TestCase):
    @skipUnless(HAS_CUPY, "CuPy is required.")
    def test_cupy_input(self):
        test_data = cp.array([[1, 2], [3, 4]])
        test_data = cp.rot90(test_data)
        self.assertFalse(test_data.flags["C_CONTIGUOUS"])
        result = ToNumpyd(keys="img")({"img": test_data})["img"]
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertTrue(result.flags["C_CONTIGUOUS"])
        assert_allclose(result, test_data.get(), type_test=False)

    def test_numpy_input(self):
        test_data = np.array([[1, 2], [3, 4]])
        test_data = np.rot90(test_data)
        self.assertFalse(test_data.flags["C_CONTIGUOUS"])
        result = ToNumpyd(keys="img")({"img": test_data})["img"]
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertTrue(result.flags["C_CONTIGUOUS"])
        assert_allclose(result, test_data, type_test=False)

    def test_tensor_input(self):
        test_data = torch.tensor([[1, 2], [3, 4]])
        test_data = test_data.rot90()
        self.assertFalse(test_data.is_contiguous())
        result = ToNumpyd(keys="img")({"img": test_data})["img"]
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertTrue(result.flags["C_CONTIGUOUS"])
        assert_allclose(result, test_data, type_test=False)

    @skip_if_no_cuda
    def test_tensor_cuda_input(self):
        test_data = torch.tensor([[1, 2], [3, 4]]).cuda()
        test_data = test_data.rot90()
        self.assertFalse(test_data.is_contiguous())
        result = ToNumpyd(keys="img")({"img": test_data})["img"]
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertTrue(result.flags["C_CONTIGUOUS"])
        assert_allclose(result, test_data, type_test=False)


if __name__ == "__main__":
    unittest.main()
