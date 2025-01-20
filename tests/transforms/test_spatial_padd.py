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

from monai.transforms import SpatialPadd
from tests.padders import PadTest

TESTS = [
    [{"keys": ["img"], "spatial_size": [15, 8, 8], "method": "symmetric"}, (3, 8, 8, 5), (3, 15, 8, 8)],
    [{"keys": ["img"], "spatial_size": [15, 8, 8], "method": "end"}, (3, 8, 8, 5), (3, 15, 8, 8)],
    [{"keys": ["img"], "spatial_size": [15, 8, 8], "method": "end"}, (3, 8, 8, 5), (3, 15, 8, 8)],
    [{"keys": ["img"], "spatial_size": [15, 8, -1], "method": "end"}, (3, 8, 5, 4), (3, 15, 8, 4)],
]


class TestSpatialPadd(PadTest):
    Padder = SpatialPadd

    @parameterized.expand(TESTS)
    def test_pad(self, input_param, input_shape, expected_shape):
        modes = ["constant", {"constant"}]
        self.pad_test(input_param, input_shape, expected_shape, modes)

    @parameterized.expand(TESTS)
    def test_pending_ops(self, input_param, input_shape, _):
        self.pad_test_pending_ops(input_param, input_shape)


if __name__ == "__main__":
    unittest.main()
