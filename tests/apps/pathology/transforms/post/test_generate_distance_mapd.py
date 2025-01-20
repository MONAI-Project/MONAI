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

from monai.apps.pathology.transforms.post.dictionary import GenerateDistanceMapd
from monai.transforms.intensity.array import GaussianSmooth
from tests.util import TEST_NDARRAYS

EXCEPTION_TESTS = []
TESTS = []

np.random.RandomState(123)

for p in TEST_NDARRAYS:
    EXCEPTION_TESTS.append(
        [
            {"mask_key": "mask", "border_key": "border"},
            p(np.random.rand(2, 5, 5)),
            p(np.random.rand(1, 5, 5)),
            ValueError,
        ]
    )
    EXCEPTION_TESTS.append(
        [
            {"mask_key": "mask", "border_key": "border"},
            p(np.random.rand(1, 5, 5)),
            p(np.random.rand(2, 5, 5)),
            ValueError,
        ]
    )

    TESTS.append([{}, p(np.random.rand(1, 5, 5)), p(np.random.rand(1, 5, 5)), (1, 5, 5)])
    TESTS.append(
        [
            {"mask_key": "mask", "border_key": "border", "smooth_fn": GaussianSmooth(sigma=0.4)},
            p(np.random.rand(1, 5, 5)),
            p(np.random.rand(1, 5, 5)),
            (1, 5, 5),
        ]
    )


class TestGenerateDistanceMapd(unittest.TestCase):
    @parameterized.expand(EXCEPTION_TESTS)
    def test_value(self, arguments, mask, border_map, exception_type):
        with self.assertRaises(exception_type):
            GenerateDistanceMapd(**arguments)({"mask": mask, "border": border_map})

    @parameterized.expand(TESTS)
    def test_value2(self, arguments, mask, border_map, expected_shape):
        result = GenerateDistanceMapd(**arguments)({"mask": mask, "border": border_map})
        self.assertEqual(result["dist_map"].shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
