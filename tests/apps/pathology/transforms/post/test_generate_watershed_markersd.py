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

from monai.apps.pathology.transforms.post.dictionary import GenerateWatershedMarkersd
from monai.utils import min_version, optional_import
from tests.util import TEST_NDARRAYS

_, has_skimage = optional_import("skimage", "0.19.3", min_version)
_, has_scipy = optional_import("scipy", "1.8.1", min_version)

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
    EXCEPTION_TESTS.append(
        [
            {"mask_key": "mask", "border_key": "border", "markers_key": "old_markers"},
            p(np.random.rand(1, 5, 5)),
            p(np.random.rand(1, 5, 5)),
            KeyError,
        ]
    )

    TESTS.append([{}, p(np.random.rand(1, 5, 5)), p(np.random.rand(1, 5, 5)), (1, 5, 5)])
    TESTS.append(
        [
            {"threshold": 0.4, "radius": 1, "min_object_size": 0},
            p(np.random.rand(1, 5, 5)),
            p(np.random.rand(1, 5, 5)),
            (1, 5, 5),
        ]
    )


@unittest.skipUnless(has_skimage, "Requires scikit-image library.")
@unittest.skipUnless(has_scipy, "Requires scipy library.")
class TestGenerateWatershedMarkersd(unittest.TestCase):
    @parameterized.expand(EXCEPTION_TESTS)
    def test_value(self, arguments, mask, border_map, exception_type):
        with self.assertRaises(exception_type):
            GenerateWatershedMarkersd(**arguments)({"mask": mask, "border": border_map, "old_markers": 1})

    @parameterized.expand(TESTS)
    def test_value2(self, arguments, mask, border_map, expected_shape):
        result = GenerateWatershedMarkersd(**arguments)({"mask": mask, "border": border_map})
        self.assertEqual(result["markers"].shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
