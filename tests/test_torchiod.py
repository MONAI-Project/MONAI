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
from unittest import skipUnless

import torch
from parameterized import parameterized

from monai.transforms import TorchIOd
from monai.utils import optional_import
from tests.util import assert_allclose

_, has_torchio = optional_import("torchio")

TEST_DIMS = [3, 128, 160, 160]
TEST_TENSOR = torch.rand(TEST_DIMS)
TEST_PARAMS = [
    [
        {"keys": "img", "name": "RescaleIntensity", "out_min_max": (0, 42)},
        {"img": TEST_TENSOR},
        ((TEST_TENSOR - TEST_TENSOR.min()) / (TEST_TENSOR.max() - TEST_TENSOR.min())) * 42,
    ]
]


@skipUnless(has_torchio, "Requires torchio")
class TestTorchIOd(unittest.TestCase):
    @parameterized.expand(TEST_PARAMS)
    def test_value(self, input_param, input_data, expected_value):
        result = TorchIOd(**input_param)(input_data)
        assert_allclose(result["img"], expected_value, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
