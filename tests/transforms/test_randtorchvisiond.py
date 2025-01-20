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

from monai.transforms import RandomizableTrait, RandTorchVisiond
from monai.utils import set_determinism
from tests.util import assert_allclose

TEST_CASE_1 = [
    {"keys": "img", "name": "ColorJitter"},
    {"img": torch.tensor([[[0.0, 1.0], [1.0, 2.0]], [[0.0, 1.0], [1.0, 2.0]], [[0.0, 1.0], [1.0, 2.0]]])},
    torch.tensor([[[0.0, 1.0], [1.0, 2.0]], [[0.0, 1.0], [1.0, 2.0]], [[0.0, 1.0], [1.0, 2.0]]]),
]

TEST_CASE_2 = [
    {"keys": "img", "name": "ColorJitter", "brightness": 0.5, "contrast": 0.5, "saturation": [0.1, 0.8], "hue": 0.5},
    {"img": torch.tensor([[[0.0, 1.0], [1.0, 2.0]], [[0.0, 1.0], [1.0, 2.0]], [[0.0, 1.0], [1.0, 2.0]]])},
    torch.tensor(
        [
            [[0.1090, 0.6193], [0.6193, 0.9164]],
            [[0.1090, 0.6193], [0.6193, 0.9164]],
            [[0.1090, 0.6193], [0.6193, 0.9164]],
        ]
    ),
]

TEST_CASE_3 = [
    {"keys": "img", "name": "Pad", "padding": [1, 1, 1, 1]},
    {"img": torch.tensor([[[0.0, 1.0], [1.0, 2.0]], [[0.0, 1.0], [1.0, 2.0]], [[0.0, 1.0], [1.0, 2.0]]])},
    torch.tensor(
        [
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 2.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 2.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 2.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        ]
    ),
]


class TestRandTorchVisiond(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3])
    def test_value(self, input_param, input_data, expected_value):
        set_determinism(seed=0)
        transform = RandTorchVisiond(**input_param)
        result = transform(input_data)
        self.assertTrue(isinstance(transform, RandomizableTrait))
        assert_allclose(result["img"], expected_value, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
