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
from monai.transforms import SqueezeDimd
from tests.util import TEST_NDARRAYS, assert_allclose

TESTS, TESTS_FAIL = [], []
for p in TEST_NDARRAYS:
    TESTS.append(
        [
            {"keys": ["img", "seg"], "dim": None},
            {"img": p(np.random.rand(1, 2, 1, 3)), "seg": p(np.random.randint(0, 2, size=[1, 2, 1, 3]))},
            (2, 3),
        ]
    )

    TESTS.append(
        [
            {"keys": ["img", "seg"], "dim": 2},
            {"img": p(np.random.rand(1, 2, 1, 8, 16)), "seg": p(np.random.randint(0, 2, size=[1, 2, 1, 8, 16]))},
            (1, 2, 8, 16),
        ]
    )

    TESTS.append(
        [
            {"keys": ["img", "seg"], "dim": -1},
            {"img": p(np.random.rand(1, 1, 16, 8, 1)), "seg": p(np.random.randint(0, 2, size=[1, 1, 16, 8, 1]))},
            (1, 1, 16, 8),
        ]
    )

    TESTS.append(
        [
            {"keys": ["img", "seg"]},
            {"img": p(np.random.rand(1, 2, 1, 3)), "seg": p(np.random.randint(0, 2, size=[1, 2, 1, 3]))},
            (2, 1, 3),
        ]
    )

    TESTS.append(
        [
            {"keys": ["img", "seg"], "dim": 0},
            {"img": p(np.random.rand(1, 2, 1, 3)), "seg": p(np.random.randint(0, 2, size=[1, 2, 1, 3]))},
            (2, 1, 3),
        ]
    )

    TESTS_FAIL.append(
        [
            ValueError,
            {"keys": ["img", "seg"], "dim": -2},
            {"img": p(np.random.rand(1, 1, 16, 8, 1)), "seg": p(np.random.randint(0, 2, size=[1, 1, 16, 8, 1]))},
        ]
    )

    TESTS_FAIL.append(
        [
            TypeError,
            {"keys": ["img", "seg"], "dim": 0.5},
            {"img": p(np.random.rand(1, 1, 16, 8, 1)), "seg": p(np.random.randint(0, 2, size=[1, 1, 16, 8, 1]))},
        ]
    )


class TestSqueezeDim(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_shape(self, input_param, test_data, expected_shape):
        result = SqueezeDimd(**input_param)(test_data)
        self.assertTupleEqual(result["img"].shape, expected_shape)
        self.assertTupleEqual(result["seg"].shape, expected_shape)
        if "dim" in input_param and isinstance(result["img"], MetaTensor) and input_param["dim"] == 2:
            assert_allclose(result["img"].affine.shape, [3, 3])

    @parameterized.expand(TESTS_FAIL)
    def test_invalid_inputs(self, exception, input_param, test_data):
        with self.assertRaises(exception):
            SqueezeDimd(**input_param)(test_data)


if __name__ == "__main__":
    unittest.main()
