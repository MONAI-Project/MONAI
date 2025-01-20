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

from monai.networks import eval_mode
from monai.networks.layers import SkipConnection

TEST_CASES_3D = []
for type_1 in ("cat", "add", "mul"):
    input_shape = (16, 10, 32, 24, 48)
    if type_1 == "cat":
        result_shape = (input_shape[0] * 2, *input_shape[1:])
    else:
        result_shape = input_shape
    test_case = [{"dim": 0, "mode": type_1}, input_shape, result_shape]
    TEST_CASES_3D.append(test_case)


class TestSkipConnection(unittest.TestCase):

    @parameterized.expand(TEST_CASES_3D)
    def test_shape(self, input_param, input_shape, expected_shape):
        net = SkipConnection(submodule=torch.nn.Softmax(dim=1), **input_param)
        with eval_mode(net):
            result = net(torch.randn(input_shape))
            self.assertEqual(result.shape, expected_shape)

    def test_wrong_mode(self):
        with self.assertRaises(ValueError):
            SkipConnection(torch.nn.Softmax(dim=1), mode="test")(torch.randn(10, 10))


if __name__ == "__main__":
    unittest.main()
