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

from monai.transforms import RandGridDistortion
from tests.util import TEST_NDARRAYS_ALL, assert_allclose

TESTS = []
for p in TEST_NDARRAYS_ALL:
    seed = 0
    TESTS.append(
        [
            dict(num_cells=2, prob=1.0, distort_limit=0.5, mode="nearest", padding_mode="zeros"),
            seed,
            p(np.indices([6, 6]).astype(np.float32)),
            p(
                np.array(
                    [
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [2.0, 2.0, 2.0, 2.0, 2.0, 0.0],
                            [4.0, 4.0, 4.0, 4.0, 4.0, 0.0],
                            [4.0, 4.0, 4.0, 4.0, 4.0, 0.0],
                            [5.0, 5.0, 5.0, 5.0, 5.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, 1.0, 3.0, 3.0, 4.0, 0.0],
                            [0.0, 1.0, 3.0, 3.0, 4.0, 0.0],
                            [0.0, 1.0, 3.0, 3.0, 4.0, 0.0],
                            [0.0, 1.0, 3.0, 3.0, 4.0, 0.0],
                            [0.0, 1.0, 3.0, 3.0, 4.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        ],
                    ]
                ).astype(np.float32)
            ),
        ]
    )
    seed = 1
    TESTS.append(
        [
            dict(num_cells=(2, 2), prob=1.0, distort_limit=0.1, mode="bilinear", padding_mode="reflection"),
            seed,
            p(np.indices([6, 6]).astype(np.float32)),
            p(
                np.array(
                    [
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [1.5660975, 1.5660975, 1.5660975, 1.5660975, 1.5660974, 1.5660975],
                            [3.132195, 3.132195, 3.132195, 3.132195, 3.132195, 3.132195],
                            [3.132195, 3.132195, 3.132195, 3.132195, 3.132195, 3.132195],
                            [4.482229, 4.482229, 4.482229, 4.482229, 4.482229, 4.482229],
                            [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                        ],
                        [
                            [0.0, 1.3940268, 2.7880535, 2.7880535, 4.1657553, 5.0],
                            [0.0, 1.3940268, 2.7880535, 2.7880535, 4.1657553, 5.0],
                            [0.0, 1.3940268, 2.7880535, 2.7880535, 4.1657553, 5.0],
                            [0.0, 1.3940268, 2.7880535, 2.7880535, 4.1657553, 5.0],
                            [0.0, 1.3940268, 2.7880535, 2.7880535, 4.1657553, 5.0],
                            [0.0, 1.3940266, 2.7880538, 2.7880538, 4.1657557, 5.0],
                        ],
                    ]
                ).astype(np.float32)
            ),
        ]
    )


class TestRandGridDistortion(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_rand_grid_distortion(self, input_param, seed, input_data, expected_val):
        g = RandGridDistortion(**input_param)
        g.set_random_state(seed=seed)
        result = g(input_data)
        if input_param["padding_mode"] != "reflection":
            assert_allclose(result, expected_val, type_test="tensor", rtol=1e-4, atol=1e-4)
        else:
            assert_allclose(result.shape, expected_val.shape, type_test=False, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
