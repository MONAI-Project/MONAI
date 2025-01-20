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

import torch
from parameterized import parameterized

from monai.transforms import Activationsd
from tests.util import TEST_NDARRAYS, assert_allclose

TEST_CASES = []
for p in TEST_NDARRAYS:
    TEST_CASES.append(
        [
            {"keys": ["pred", "label"], "sigmoid": False, "softmax": [True, False], "other": None, "dim": 0},
            {"pred": p([[[0.0, 1.0]], [[2.0, 3.0]]]), "label": p([[[0.0, 1.0]], [[2.0, 3.0]]])},
            {"pred": p([[[0.1192, 0.1192]], [[0.8808, 0.8808]]]), "label": p([[[0.0, 1.0]], [[2.0, 3.0]]])},
            (2, 1, 2),
        ]
    )

    TEST_CASES.append(
        [
            {"keys": ["pred", "label"], "sigmoid": False, "softmax": False, "other": [torch.tanh, None]},
            {"pred": p([[[0.0, 1.0], [2.0, 3.0]]]), "label": p([[[0.0, 1.0], [2.0, 3.0]]])},
            {"pred": p([[[0.0000, 0.7616], [0.9640, 0.9951]]]), "label": p([[[0.0, 1.0], [2.0, 3.0]]])},
            (1, 2, 2),
        ]
    )

    TEST_CASES.append(
        [
            {"keys": "pred", "sigmoid": False, "softmax": False, "other": torch.tanh},
            {"pred": p([[[0.0, 1.0], [2.0, 3.0]]])},
            {"pred": p([[[0.0000, 0.7616], [0.9640, 0.9951]]])},
            (1, 2, 2),
        ]
    )


class TestActivationsd(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_value_shape(self, input_param, test_input, output, expected_shape):
        result = Activationsd(**input_param)(test_input)
        assert_allclose(result["pred"], output["pred"], rtol=1e-3, type_test="tensor")
        self.assertTupleEqual(result["pred"].shape, expected_shape)
        if "label" in result:
            assert_allclose(result["label"], output["label"], rtol=1e-3, type_test="tensor")
            self.assertTupleEqual(result["label"].shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
