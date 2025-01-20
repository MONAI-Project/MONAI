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

from monai.transforms import ToTensord
from tests.util import HAS_CUPY, TEST_NDARRAYS, assert_allclose, optional_import

cp, _ = optional_import("cupy")

im = [[1, 2], [3, 4]]

TESTS = [(im, (2, 2))]
for p in TEST_NDARRAYS:
    TESTS.append((p(im), (2, 2)))

TESTS_SINGLE = [[5]]
for p in TEST_NDARRAYS:
    TESTS_SINGLE.append([p(5)])


class TestToTensord(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_array_input(self, test_data, expected_shape):
        test_data = {"img": test_data}
        to_tensord = ToTensord(keys="img", dtype=torch.float32, device="cpu", wrap_sequence=True)
        result = to_tensord(test_data)
        out_img = result["img"]
        self.assertTrue(isinstance(out_img, torch.Tensor))
        assert_allclose(out_img, test_data["img"], type_test=False)
        self.assertTupleEqual(out_img.shape, expected_shape)

        # test inverse
        inv_data = to_tensord.inverse(result)
        self.assertTrue(isinstance(inv_data["img"], np.ndarray))
        assert_allclose(test_data["img"], inv_data["img"], type_test=False)

    @parameterized.expand(TESTS_SINGLE)
    def test_single_input(self, test_data):
        test_data = {"img": test_data}
        result = ToTensord(keys="img", track_meta=True)(test_data)
        out_img = result["img"]
        self.assertTrue(isinstance(out_img, torch.Tensor))
        assert_allclose(out_img, test_data["img"], type_test=False)
        self.assertEqual(out_img.ndim, 0)

    @unittest.skipUnless(HAS_CUPY, "CuPy is required.")
    def test_cupy(self):
        test_data = [[1, 2], [3, 4]]
        cupy_array = cp.ascontiguousarray(cp.asarray(test_data))
        result = ToTensord(keys="img")({"img": cupy_array})
        self.assertTrue(isinstance(result["img"], torch.Tensor))
        assert_allclose(result["img"], test_data, type_test=False)


if __name__ == "__main__":
    unittest.main()
