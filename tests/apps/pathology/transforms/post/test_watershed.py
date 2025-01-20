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

from monai.apps.pathology.transforms.post.array import (
    GenerateDistanceMap,
    GenerateInstanceBorder,
    GenerateWatershedMarkers,
    GenerateWatershedMask,
    Watershed,
)
from monai.utils import min_version, optional_import
from tests.util import TEST_NDARRAYS

_, has_skimage = optional_import("skimage", "0.19.3", min_version)
_, has_scipy = optional_import("scipy", "1.8.1", min_version)

np.random.RandomState(123)

TESTS = []
params = {"connectivity": 1}
for p in TEST_NDARRAYS:
    image = p(np.random.rand(1, 10, 10))
    hover_map = p(np.random.rand(2, 10, 10))

    TESTS.append([params, image, hover_map, (1, 10, 10)])


@unittest.skipUnless(has_skimage, "Requires scikit-image library.")
@unittest.skipUnless(has_scipy, "Requires scipy library.")
class TestWatershed(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_output(self, args, image, hover_map, expected_shape):
        mask = GenerateWatershedMask()(image)
        border_map = GenerateInstanceBorder(kernel_size=3)(mask, hover_map)
        distance_map = GenerateDistanceMap()(mask, border_map)
        markers = GenerateWatershedMarkers()(mask, border_map)

        calculate_instance_seg = Watershed(**args)
        output = calculate_instance_seg(distance_map, mask, markers)

        self.assertTupleEqual(output.shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
