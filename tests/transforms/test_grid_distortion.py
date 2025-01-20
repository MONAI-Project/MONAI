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

from monai.transforms import GridDistortion
from tests.util import TEST_NDARRAYS_ALL, assert_allclose

TESTS = []
for p in TEST_NDARRAYS_ALL:
    TESTS.append(
        [
            dict(num_cells=3, distort_steps=[(1.5,) * 4] * 2, mode="nearest", padding_mode="zeros"),
            p(np.indices([6, 6]).astype(np.float32)),
            p(
                np.array(
                    [
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [3.0, 3.0, 3.0, 0.0, 0.0, 0.0],
                            [3.0, 3.0, 3.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, 3.0, 3.0, 0.0, 0.0, 0.0],
                            [0.0, 3.0, 3.0, 0.0, 0.0, 0.0],
                            [0.0, 3.0, 3.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        ],
                    ]
                ).astype(np.float32)
            ),
        ]
    )
    num_cells = (2, 2)
    distort_steps = [(1.5,) * (1 + num_cells[0]), (1.0,) * (1 + num_cells[1])]
    TESTS.append(
        [
            dict(num_cells=num_cells, distort_steps=distort_steps, mode="bilinear", padding_mode="reflection"),
            p(np.indices([6, 6]).astype(np.float32)),
            p(
                np.array(
                    [
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [2.25, 2.25, 2.25, 2.25, 2.25, 2.25],
                            [4.5, 4.5, 4.5, 4.5, 4.5, 4.5],
                            [4.5, 4.5, 4.5, 4.5, 4.5, 4.5],
                            [4.2500, 4.2500, 4.2500, 4.2500, 4.2500, 4.2500],
                            [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                        ],
                        [
                            [0.0, 1.5, 3.0, 3.0, 4.5, 5.0],
                            [0.0, 1.5, 3.0, 3.0, 4.5, 5.0],
                            [0.0, 1.5, 3.0, 3.0, 4.5, 5.0],
                            [0.0, 1.5, 3.0, 3.0, 4.5, 5.0],
                            [0.0, 1.5, 3.0, 3.0, 4.5, 5.0],
                            [0.0, 1.5, 3.0, 3.0, 4.5, 5.0],
                        ],
                    ]
                ).astype(np.float32)
            ),
        ]
    )
    TESTS.append(
        [
            dict(num_cells=2, distort_steps=[(1.26,) * 3] * 3, mode="nearest", padding_mode="zeros"),
            p(np.indices([3, 3, 3])[:1].astype(np.float32)),
            p(
                np.array(
                    [
                        [
                            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                            [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
                            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                        ]
                    ]
                ).astype(np.float32)
            ),
        ]
    )


class TestGridDistortion(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_grid_distortion(self, input_param, input_data, expected_val):
        g = GridDistortion(**input_param)
        result = g(input_data)
        if input_param["padding_mode"] != "reflection":
            assert_allclose(result, expected_val, type_test=False, rtol=1e-4, atol=1e-4)
        else:
            assert_allclose(result.shape, expected_val.shape, type_test=False, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
