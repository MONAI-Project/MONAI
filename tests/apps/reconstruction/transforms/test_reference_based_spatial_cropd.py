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

from monai.apps.reconstruction.transforms.dictionary import ReferenceBasedSpatialCropd
from tests.util import TEST_NDARRAYS

# see test_spatial_cropd for typical tests (like roi_start,
# roi_slices, etc.)
# here, we test TargetBasedSpatialCropd's functionality
# which focuses on automatic input crop based on target image's shape.

TESTS = []
for p in TEST_NDARRAYS:
    # 2D
    TESTS.append(
        [
            {"keys": ["kspace_masked_ifft"], "ref_key": "target"},
            {"kspace_masked_ifft": p(np.ones([10, 20, 20])), "target": p(np.ones([5, 8, 8]))},
            (8, 8),  # expected shape
        ]
    )

    # 3D
    TESTS.append(
        [
            {"keys": ["kspace_masked_ifft"], "ref_key": "target"},
            {"kspace_masked_ifft": p(np.ones([10, 20, 20, 16])), "target": p(np.ones([5, 8, 8, 6]))},
            (8, 8, 6),  # expected shape
        ]
    )


class TestTargetBasedSpatialCropd(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_shape(self, args, data, expected_shape):
        cropper = ReferenceBasedSpatialCropd(keys=args["keys"], ref_key=args["ref_key"])
        res_data = cropper(data)
        self.assertTupleEqual(res_data[args["keys"][0]].shape[1:], expected_shape)


if __name__ == "__main__":
    unittest.main()
