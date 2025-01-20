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

from parameterized import parameterized

from monai.transforms import SpatialCropd
from tests.croppers import CropTest

TESTS = [
    [
        {"keys": ["img"], "roi_center": [1, 1], "roi_size": [2, 2]},
        (1, 3, 3),
        (1, 2, 2),
        (slice(None), slice(None, 2), slice(None, 2)),
    ],
    [
        {"keys": ["img"], "roi_center": [1, 1, 1], "roi_size": [2, 2, 2]},
        (3, 3, 3, 3),
        (3, 2, 2, 2),
        (slice(None), slice(None, 2), slice(None, 2), slice(None, 2)),
    ],
    [
        {"keys": ["img"], "roi_start": [0, 0, 0], "roi_end": [2, 2, 2]},
        (3, 3, 3, 3),
        (3, 2, 2, 2),
        (slice(None), slice(None, 2), slice(None, 2), slice(None, 2)),
    ],
    [
        {"keys": ["img"], "roi_start": [0, 0], "roi_end": [2, 2]},
        (3, 3, 3, 3),
        (3, 2, 2, 3),
        (slice(None), slice(None, 2), slice(None, 2), slice(None)),
    ],
    [
        {"keys": ["img"], "roi_start": [0, 0, 0, 0, 0], "roi_end": [2, 2, 2, 2, 2]},
        (3, 3, 3, 3),
        (3, 2, 2, 2),
        (slice(None), slice(None, 2), slice(None, 2), slice(None, 2)),
    ],
    [
        {"keys": ["img"], "roi_slices": [slice(s, e) for s, e in zip([-1, -2, 0], [None, None, 2])]},
        (3, 3, 3, 3),
        (3, 1, 2, 2),
        (slice(None), slice(-1, None), slice(-2, None), slice(0, 2)),
    ],
]


class TestSpatialCropd(CropTest):
    Cropper = SpatialCropd

    @parameterized.expand(TESTS)
    def test_shape(self, input_param, input_shape, expected_shape, same_area):
        self.crop_test(input_param, input_shape, expected_shape, same_area)

    @parameterized.expand(TESTS)
    def test_pending_ops(self, input_param, input_shape, _expected_shape, _same_area):
        self.crop_test_pending_ops(input_param, input_shape)


if __name__ == "__main__":
    unittest.main()
