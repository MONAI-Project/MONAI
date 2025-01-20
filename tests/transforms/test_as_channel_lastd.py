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

from monai.transforms import AsChannelLastd
from tests.util import TEST_NDARRAYS

TESTS = []
for p in TEST_NDARRAYS:
    TESTS.append([p, {"keys": ["image", "label", "extra"], "channel_dim": 0}, (2, 3, 4, 1)])
    TESTS.append([p, {"keys": ["image", "label", "extra"], "channel_dim": 1}, (1, 3, 4, 2)])
    TESTS.append([p, {"keys": ["image", "label", "extra"], "channel_dim": 3}, (1, 2, 3, 4)])


class TestAsChannelLastd(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_shape(self, in_type, input_param, expected_shape):
        test_data = {
            "image": in_type(np.random.randint(0, 2, size=[1, 2, 3, 4])),
            "label": in_type(np.random.randint(0, 2, size=[1, 2, 3, 4])),
            "extra": in_type(np.random.randint(0, 2, size=[1, 2, 3, 4])),
        }
        result = AsChannelLastd(**input_param)(test_data)
        self.assertTupleEqual(result["image"].shape, expected_shape)
        self.assertTupleEqual(result["label"].shape, expected_shape)
        self.assertTupleEqual(result["extra"].shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
