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

from monai.transforms import TorchVision
from monai.utils import set_determinism
from tests.util import TEST_NDARRAYS, assert_allclose

TESTS = []
for p in TEST_NDARRAYS:
    TESTS.extend(
        [
            [
                {"name": "ColorJitter"},
                p([[[0.0, 1.0], [1.0, 2.0]], [[0.0, 1.0], [1.0, 2.0]], [[0.0, 1.0], [1.0, 2.0]]]),
                p([[[0.0, 1.0], [1.0, 2.0]], [[0.0, 1.0], [1.0, 2.0]], [[0.0, 1.0], [1.0, 2.0]]]),
            ],
            [
                {"name": "ColorJitter", "brightness": 0.5, "contrast": 0.5, "saturation": [0.1, 0.8], "hue": 0.5},
                p([[[0.0, 1.0], [1.0, 2.0]], [[0.0, 1.0], [1.0, 2.0]], [[0.0, 1.0], [1.0, 2.0]]]),
                p(
                    [
                        [[0.1090, 0.6193], [0.6193, 0.9164]],
                        [[0.1090, 0.6193], [0.6193, 0.9164]],
                        [[0.1090, 0.6193], [0.6193, 0.9164]],
                    ]
                ),
            ],
            [
                {"name": "Pad", "padding": [1, 1, 1, 1]},
                p([[[0.0, 1.0], [1.0, 2.0]], [[0.0, 1.0], [1.0, 2.0]], [[0.0, 1.0], [1.0, 2.0]]]),
                p(
                    [
                        [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 2.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                        [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 2.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                        [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 2.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                    ]
                ),
            ],
        ]
    )


class TestTorchVision(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_value(self, input_param, input_data, expected_value):
        set_determinism(seed=0)
        result = TorchVision(**input_param)(input_data)
        assert_allclose(result, expected_value, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
