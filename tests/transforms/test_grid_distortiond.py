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

from monai.transforms import GridDistortiond
from tests.util import TEST_NDARRAYS_ALL, assert_allclose

TESTS = []
num_cells = (2, 2)
distort_steps = [(1.5,) * (1 + n_c) for n_c in num_cells]
for p in TEST_NDARRAYS_ALL:
    img = np.indices([6, 6]).astype(np.float32)
    TESTS.append(
        [
            dict(
                keys=["img", "mask"],
                num_cells=num_cells,
                distort_steps=distort_steps,
                mode=["bilinear", "nearest"],
                padding_mode=["reflection", "zeros"],
            ),
            {"img": p(img), "mask": p(np.ones_like(img[:1]))},
            p(
                np.array(
                    [
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [2.25, 2.25, 2.25, 2.25, 2.25, 2.25],
                            [4.5, 4.5, 4.5, 4.5, 4.5, 4.5],
                            [4.5, 4.5, 4.5, 4.5, 4.5, 4.5],
                            [4.2500, 4.2500, 4.2500, 4.2500, 4.2500, 4.2500],
                            [2.0000, 2.0000, 2.0000, 2.0000, 2.0000, 2.0000],
                        ],
                        [
                            [0.0000, 2.2500, 4.5000, 4.5000, 4.2500, 2.0000],
                            [0.0000, 2.2500, 4.5000, 4.5000, 4.2500, 2.0000],
                            [0.0000, 2.2500, 4.5000, 4.5000, 4.2500, 2.0000],
                            [0.0000, 2.2500, 4.5000, 4.5000, 4.2500, 2.0000],
                            [0.0000, 2.2500, 4.5000, 4.5000, 4.2500, 2.0000],
                            [0.0000, 2.2500, 4.5000, 4.5000, 4.2500, 2.0000],
                        ],
                    ]
                ).astype(np.float32)
            ),
            p(
                np.array(
                    [
                        [
                            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ]
                ).astype(np.float32)
            ),
        ]
    )


class TestGridDistortiond(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_grid_distortiond(self, input_param, input_data, expected_val_img, expected_val_mask):
        g = GridDistortiond(**input_param)
        result = g(input_data)
        assert_allclose(result["mask"], expected_val_mask, type_test=False, rtol=1e-4, atol=1e-4)
        assert_allclose(result["img"].shape, expected_val_img.shape, type_test=False, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
