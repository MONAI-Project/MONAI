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
import torch
from parameterized import parameterized

from monai.transforms import CastToTyped
from monai.utils import optional_import
from tests.util import HAS_CUPY

cp, _ = optional_import("cupy")

TEST_CASE_1 = [
    {"keys": ["img"], "dtype": np.float64},
    {"img": np.array([[0, 1], [1, 2]], dtype=np.float32), "seg": np.array([[0, 1], [1, 2]], dtype=np.int8)},
    {"img": np.float64, "seg": np.int8},
]

TEST_CASE_2 = [
    {"keys": ["img"], "dtype": torch.float64},
    {
        "img": torch.tensor([[0, 1], [1, 2]], dtype=torch.float32),
        "seg": torch.tensor([[0, 1], [1, 2]], dtype=torch.int8),
    },
    {"img": torch.float64, "seg": torch.int8},
]

TESTS_CUPY = [
    [
        {"keys": "image", "dtype": np.uint8},
        {"image": np.array([[0, 1], [1, 2]], dtype=np.float32), "label": np.array([[0, 1], [1, 1]], dtype=np.float32)},
        {"image": np.uint8, "label": np.float32},
    ],
    [
        {"keys": ["image", "label"], "dtype": np.float32},
        {"image": np.array([[0, 1], [1, 2]], dtype=np.uint8), "label": np.array([[0, 1], [1, 1]], dtype=np.uint8)},
        {"image": np.float32, "label": np.float32},
    ],
]


class TestCastToTyped(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    def test_type(self, input_param, input_data, expected_type):
        result = CastToTyped(**input_param)(input_data)
        for k, v in result.items():
            self.assertEqual(v.dtype, expected_type[k])

    @parameterized.expand(TESTS_CUPY)
    @unittest.skipUnless(HAS_CUPY, "Requires CuPy")
    def test_type_cupy(self, input_param, input_data, expected_type):
        input_data = {k: cp.asarray(v) for k, v in input_data.items()}

        result = CastToTyped(**input_param)(input_data)
        for k, v in result.items():
            self.assertTrue(isinstance(v, cp.ndarray))
            self.assertEqual(v.dtype, expected_type[k])


if __name__ == "__main__":
    unittest.main()
