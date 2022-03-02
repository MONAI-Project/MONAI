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

import unittest
from copy import deepcopy

import numpy as np
from parameterized import parameterized

from monai.transforms import LoadImaged
from monai.transforms.utility.dictionary import SplitDimd
from tests.utils import TEST_NDARRAYS, make_nifti_image, make_rand_affine

TESTS = []
for p in TEST_NDARRAYS:
    for keepdim in (True, False):
        for update_meta in (True, False):
            TESTS.append((keepdim, p, update_meta))


class TestSplitDimd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        arr = np.random.rand(2, 10, 8, 7)
        affine = make_rand_affine()
        data = {"i": make_nifti_image(arr, affine)}

        cls.data = LoadImaged("i")(data)

    @parameterized.expand(TESTS)
    def test_correct_shape(self, keepdim, im_type, update_meta):
        data = deepcopy(self.data)
        data["i"] = im_type(data["i"])
        arr = data["i"]
        for dim in range(arr.ndim):
            out = SplitDimd("i", dim=dim, keepdim=keepdim, update_meta=update_meta)(data)
            self.assertIsInstance(out, dict)
            num_new_keys = 2 if update_meta else 1
            self.assertEqual(len(out.keys()), len(data.keys()) + num_new_keys * arr.shape[dim])
            out = out["i_0"]
            expected_ndim = arr.ndim if keepdim else arr.ndim - 1
            self.assertEqual(out.ndim, expected_ndim)
            # assert is a shallow copy
            arr[0, 0, 0, 0] *= 2
            self.assertEqual(arr.flatten()[0], out.flatten()[0])

    def test_error(self):
        """Should fail because splitting along singleton dimension"""
        shape = (2, 1, 8, 7)
        for p in TEST_NDARRAYS:
            arr = p(np.random.rand(*shape))
            with self.assertRaises(RuntimeError):
                _ = SplitDimd("i", dim=1)({"i": arr})


if __name__ == "__main__":
    unittest.main()
