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

from numpy import ndarray
from torch import Tensor

from monai.data.meta_tensor import MetaTensor
from monai.transforms.utility.array import Lambda
from monai.utils.type_conversion import convert_to_numpy, convert_to_tensor
from tests.util import TEST_NDARRAYS, NumpyImageTestCase2D, assert_allclose


class TestLambda(NumpyImageTestCase2D):
    def test_lambda_identity(self):
        for p in TEST_NDARRAYS:
            img = p(self.imt)

            def identity_func(x):
                return x

            lambd = Lambda(func=identity_func)
            assert_allclose(identity_func(img), lambd(img), type_test=False)

    def test_lambda_slicing(self):
        for p in TEST_NDARRAYS:
            img = p(self.imt)

            def slice_func(x):
                return x[:, :, :6, ::2]

            lambd = Lambda(func=slice_func)
            out = lambd(img)
            assert_allclose(slice_func(img), out, type_test=False)
            self.assertIsInstance(out, MetaTensor)
            self.assertEqual(len(out.applied_operations), 1)
            out = lambd.inverse(out)
            self.assertEqual(len(out.applied_operations), 0)

    def test_lambda_track_meta_false(self):
        for p in TEST_NDARRAYS:
            img = p(self.imt)

            def to_numpy(x):
                return convert_to_numpy(x)

            lambd = Lambda(func=to_numpy, track_meta=False)
            out = lambd(img)
            self.assertIsInstance(out, ndarray)

            def to_tensor(x):
                return convert_to_tensor(x)

            lambd = Lambda(func=to_tensor, track_meta=False)
            out = lambd(img)
            self.assertIsInstance(out, Tensor)


if __name__ == "__main__":
    unittest.main()
