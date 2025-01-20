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

import numpy as np
import torch
from parameterized import parameterized

from monai.transforms import RandTorchIOd
from monai.utils import optional_import, set_determinism
from tests.util import assert_allclose

_, has_torchio = optional_import("torchio")

TEST_DIMS = [3, 128, 160, 160]
TEST_TENSOR = torch.rand(TEST_DIMS)
TEST_PARAMS = [[{"keys": ["img1", "img2"], "name": "RandomAffine"}, {"img1": TEST_TENSOR, "img2": TEST_TENSOR}]]


@skipUnless(has_torchio, "Requires torchio")
class TestRandTorchIOd(unittest.TestCase):
    @parameterized.expand(TEST_PARAMS)
    def test_random_transform(self, input_param, input_data):
        set_determinism(seed=0)
        result = RandTorchIOd(**input_param)(input_data)
        self.assertFalse(np.allclose(input_data["img1"], result["img1"], atol=1e-6, rtol=1e-6))
        assert_allclose(result["img1"], result["img2"], atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
