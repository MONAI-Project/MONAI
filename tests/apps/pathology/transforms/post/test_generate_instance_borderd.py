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

from monai.apps.pathology.transforms.post.dictionary import GenerateInstanceBorderd
from tests.util import TEST_NDARRAYS

EXCEPTION_TESTS = []
TESTS = []

np.random.RandomState(123)

for p in TEST_NDARRAYS:
    EXCEPTION_TESTS.append(
        [{"mask_key": "mask", "kernel_size": 3}, p(np.random.rand(1, 5, 5, 5)), p(np.random.rand(2, 5, 5)), ValueError]
    )
    EXCEPTION_TESTS.append(
        [{"mask_key": "mask", "kernel_size": 3}, p(np.random.rand(1, 5, 5)), p(np.random.rand(1, 5, 5)), ValueError]
    )
    EXCEPTION_TESTS.append(
        [{"mask_key": "mask", "kernel_size": 3}, p(np.random.rand(2, 5, 5)), p(np.random.rand(2, 5, 5)), ValueError]
    )

    TESTS.append(
        [{"mask_key": "mask", "kernel_size": 3}, p(np.random.rand(1, 5, 5)), p(np.random.rand(2, 5, 5)), (1, 5, 5)]
    )
    TESTS.append(
        [{"mask_key": "mask", "kernel_size": 3}, p(np.random.rand(1, 5, 5)), p(np.random.rand(2, 5, 5)), (1, 5, 5)]
    )


class TestGenerateInstanceBorderd(unittest.TestCase):
    @parameterized.expand(EXCEPTION_TESTS)
    def test_value(self, arguments, mask, hover_map, exception_type):
        with self.assertRaises(exception_type):
            GenerateInstanceBorderd(**arguments)({"mask": mask, "hover_map": hover_map})

    @parameterized.expand(TESTS)
    def test_value2(self, arguments, mask, hover_map, expected_shape):
        result = GenerateInstanceBorderd(**arguments)({"mask": mask, "hover_map": hover_map})
        self.assertEqual(result["border"].shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
