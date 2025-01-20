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
from parameterized import parameterized

from monai.data import MetaTensor
from monai.transforms import SqueezeDim
from tests.util import TEST_NDARRAYS, assert_allclose

TESTS, TESTS_FAIL = [], []
for p in TEST_NDARRAYS:
    TESTS.append([{"dim": None}, p(np.random.rand(1, 2, 1, 3)), (2, 3)])
    TESTS.append([{"dim": 2}, p(np.random.rand(1, 2, 1, 8, 16)), (1, 2, 8, 16)])
    TESTS.append([{"dim": -1}, p(np.random.rand(1, 1, 16, 8, 1)), (1, 1, 16, 8)])
    TESTS.append([{}, p(np.random.rand(1, 2, 1, 3)), (2, 1, 3)])

    TESTS_FAIL.append([ValueError, {"dim": -2}, p(np.random.rand(1, 1, 16, 8, 1))])
    TESTS_FAIL.append([TypeError, {"dim": 0.5}, p(np.random.rand(1, 1, 16, 8, 1))])


class TestSqueezeDim(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_shape(self, input_param, test_data, expected_shape):
        result = SqueezeDim(**input_param)(test_data)
        self.assertTupleEqual(result.shape, expected_shape)
        if "dim" in input_param and input_param["dim"] == 2 and isinstance(result, MetaTensor):
            assert_allclose(result.affine.shape, [3, 3])

    @parameterized.expand(TESTS_FAIL)
    def test_invalid_inputs(self, exception, input_param, test_data):
        with self.assertRaises(exception):
            SqueezeDim(**input_param)(test_data)

    def test_affine_ill_inputs(self):
        img = MetaTensor(
            np.random.rand(1, 2, 1, 3),
            affine=[
                [-0.7422, 0.0, 0.0, 186.3210],
                [0.0, 0.0, -3.0, 70.6580],
                [0.0, -0.7422, 0.0, 189.4130],
                [0.0, 0.0, 0.0, 1.0],
            ],
        )
        with self.assertWarns(UserWarning):
            SqueezeDim(dim=2)(img)


if __name__ == "__main__":
    unittest.main()
