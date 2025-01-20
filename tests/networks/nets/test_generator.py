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
from monai.networks.nets import Generator
from tests.util import test_script_save

TEST_CASE_0 = [
    {"latent_shape": (64,), "start_shape": (8, 8, 8), "channels": (8, 4, 1), "strides": (2, 2, 2), "num_res_units": 0},
    torch.rand(16, 64),
    (16, 1, 64, 64),
]

TEST_CASE_1 = [
    {"latent_shape": (64,), "start_shape": (8, 8, 8), "channels": (8, 4, 1), "strides": (2, 2, 2), "num_res_units": 2},
    torch.rand(16, 64),
    (16, 1, 64, 64),
]

TEST_CASE_2 = [
    {"latent_shape": (64,), "start_shape": (8, 8, 8), "channels": (8, 1), "strides": (2, 2), "num_res_units": 2},
    torch.rand(16, 64),
    (16, 1, 32, 32),
]

CASES = [TEST_CASE_0, TEST_CASE_1, TEST_CASE_2]


class TestGenerator(unittest.TestCase):
    @parameterized.expand(CASES)
    def test_shape(self, input_param, input_data, expected_shape):
        net = Generator(**input_param)
        with eval_mode(net):
            result = net.forward(input_data)
            self.assertEqual(result.shape, expected_shape)

    def test_script(self):
        net = Generator(latent_shape=(64,), start_shape=(8, 8, 8), channels=(8, 1), strides=(2, 2), num_res_units=2)
        test_data = torch.rand(16, 64)
        test_script_save(net, test_data)


if __name__ == "__main__":
    unittest.main()
