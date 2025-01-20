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
from monai.transforms.utility.dictionary import RandLambdad
from tests.util import TEST_NDARRAYS, assert_allclose


class RandTest(Randomizable):
    """
    randomisable transform for testing.
    """

    def randomize(self, data=None):
        self._a = self.R.random()

    def __call__(self, data):
        self.randomize()
        return data + self._a


class TestRandLambdad(unittest.TestCase):
    def check(self, tr: RandLambdad, input: dict, out: dict, expected: dict):
        if isinstance(input["img"], MetaTensor):
            self.assertEqual(len(input["img"].applied_operations), 0)
        self.assertIsInstance(out["img"], MetaTensor)
        self.assertEqual(len(out["img"].applied_operations), 1)
        assert_allclose(expected["img"], out["img"], type_test=False)
        assert_allclose(expected["prop"], out["prop"], type_test=False)
        inv = tr.inverse(out)
        self.assertIsInstance(inv["img"], MetaTensor)
        self.assertEqual(len(inv["img"].applied_operations), 0)  # type: ignore

    @parameterized.expand([[p] for p in TEST_NDARRAYS])
    def test_rand_lambdad_identity(self, t):
        img = t(np.zeros((10, 10)))
        data = {"img": img, "prop": 1.0}

        test_func = RandTest()
        test_func.set_random_state(seed=134)
        expected = {"img": test_func(data["img"]), "prop": 1.0}
        test_func.set_random_state(seed=134)

        # default prob
        tr = RandLambdad(keys=["img", "prop"], func=test_func, overwrite=[True, False])
        ret = tr(deepcopy(data))
        self.check(tr, data, ret, expected)

        tr = RandLambdad(keys=["img", "prop"], func=test_func, prob=0.0)
        ret = tr(deepcopy(data))
        self.check(tr, data, ret, expected=data)

        trans = RandLambdad(keys=["img", "prop"], func=test_func, prob=0.5)
        trans.set_random_state(seed=123)
        ret = trans(deepcopy(data))
        self.check(trans, data, ret, expected=data)


if __name__ == "__main__":
    unittest.main()
