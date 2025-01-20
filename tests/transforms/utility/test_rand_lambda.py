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
from copy import deepcopy

import numpy as np
from parameterized import parameterized

from monai.data.meta_tensor import MetaTensor
from monai.transforms.transform import Randomizable
from monai.transforms.utility.array import RandLambda
from tests.util import TEST_NDARRAYS, assert_allclose


class RandTest(Randomizable):
    """
    randomisable transform for testing.
    """

    def randomize(self, data=None):
        self._a = self.R.random()

    def __call__(self, data):
        self.randomize()
        return deepcopy(data) + self._a


class TestRandLambda(unittest.TestCase):
    def check(self, tr: RandLambda, img, img_orig_type, out, expected=None):
        # input shouldn't change
        self.assertIsInstance(img, img_orig_type)
        if isinstance(img, MetaTensor):
            self.assertEqual(len(img.applied_operations), 0)
        # output data matches expected
        assert_allclose(expected, out, type_test=False)
        # output type is MetaTensor with 1 appended operation
        self.assertIsInstance(out, MetaTensor)
        self.assertEqual(len(out.applied_operations), 1)

        # inverse
        inv = tr.inverse(out)
        # after inverse, input image remains unchanged
        self.assertIsInstance(img, img_orig_type)
        if isinstance(img, MetaTensor):
            self.assertEqual(len(img.applied_operations), 0)
        # after inverse, output is MetaTensor with 0 applied operations
        self.assertIsInstance(inv, MetaTensor)
        self.assertEqual(len(inv.applied_operations), 0)

    @parameterized.expand([[p] for p in TEST_NDARRAYS])
    def test_rand_lambdad_identity(self, t):
        img = t(np.zeros((10, 10)))
        img_t = type(img)

        test_func = RandTest()
        test_func.set_random_state(seed=134)
        expected = test_func(img)
        test_func.set_random_state(seed=134)

        # default prob
        tr = RandLambda(func=test_func)
        ret = tr(img)
        self.check(tr, img, img_t, ret, expected)

        tr = RandLambda(func=test_func, prob=0.0)
        ret = tr(img)
        self.check(tr, img, img_t, ret, expected=img)

        trans = RandLambda(func=test_func, prob=0.5)
        trans.set_random_state(seed=123)
        ret = trans(img)
        self.check(trans, img, img_t, ret, expected=img)


if __name__ == "__main__":
    unittest.main()
