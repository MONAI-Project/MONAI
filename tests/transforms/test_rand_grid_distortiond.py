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

from monai.transforms import RandGridDistortiond
from tests.util import TEST_NDARRAYS_ALL, assert_allclose

TESTS = []
num_cells = 2
seed = 0
for p in TEST_NDARRAYS_ALL:
    img = np.indices([6, 6]).astype(np.float32)
    TESTS.append(
        [
            dict(
                keys=["img", "mask"],
                num_cells=num_cells,
                prob=1.0,
                distort_limit=(-0.1, 0.1),
                mode=["bilinear", "nearest"],
                padding_mode="zeros",
            ),
            seed,
            {"img": p(img), "mask": p(np.ones_like(img[:1]))},
            p(
                np.array(
                    [
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [1.5645568, 1.5645568, 1.5645568, 1.5645568, 1.5645568, 0.0],
                            [3.1291137, 3.1291137, 3.1291137, 3.1291137, 3.1291137, 0.0],
                            [3.1291137, 3.1291137, 3.1291137, 3.1291137, 3.1291137, 0.0],
                            [4.6599426, 4.6599426, 4.6599426, 4.6599426, 4.6599426, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, 1.4770963, 2.9541926, 2.9541926, 4.497961, 0.0],
                            [0.0, 1.4770963, 2.9541926, 2.9541926, 4.497961, 0.0],
                            [0.0, 1.4770963, 2.9541926, 2.9541926, 4.497961, 0.0],
                            [0.0, 1.4770963, 2.9541926, 2.9541926, 4.497961, 0.0],
                            [0.0, 1.4770963, 2.9541926, 2.9541926, 4.497961, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        ],
                    ]
                ).astype(np.float32)
            ),
            p(
                np.array(
                    [
                        [
                            [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                            [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                            [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                            [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                            [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ]
                )
            ),
        ]
    )


class TestRandGridDistortiond(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_rand_grid_distortiond(self, input_param, seed, input_data, expected_val_img, expected_val_mask):
        g = RandGridDistortiond(**input_param)
        g.set_random_state(seed=seed)
        result = g(input_data)
        assert_allclose(result["img"], expected_val_img, type_test=False, rtol=1e-4, atol=1e-4)
        assert_allclose(result["mask"], expected_val_mask, type_test=False, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
