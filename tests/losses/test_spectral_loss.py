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
import torch
from parameterized import parameterized

from monai.losses import JukeboxLoss
from tests.util import test_script_save

TEST_CASES = [
    [
        {"spatial_dims": 2},
        {
            "input": torch.tensor([[[[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]]]),
            "target": torch.tensor([[[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]]]),
        },
        0.070648,
    ],
    [
        {"spatial_dims": 2, "reduction": "sum"},
        {
            "input": torch.tensor([[[[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]]]),
            "target": torch.tensor([[[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]]]),
        },
        0.8478,
    ],
    [
        {"spatial_dims": 3},
        {
            "input": torch.tensor(
                [
                    [
                        [[[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]],
                        [[[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]],
                    ]
                ]
            ),
            "target": torch.tensor(
                [
                    [
                        [[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]],
                        [[[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]],
                    ]
                ]
            ),
        },
        0.03838,
    ],
]


class TestJukeboxLoss(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_results(self, input_param, input_data, expected_val):
        results = JukeboxLoss(**input_param).forward(**input_data)
        np.testing.assert_allclose(results.detach().cpu().numpy(), expected_val, rtol=1e-4)

    def test_2d_shape(self):
        results = JukeboxLoss(spatial_dims=2, reduction="none").forward(**TEST_CASES[0][1])
        self.assertEqual(results.shape, (1, 2, 2, 3))

    def test_3d_shape(self):
        results = JukeboxLoss(spatial_dims=3, reduction="none").forward(**TEST_CASES[2][1])
        self.assertEqual(results.shape, (1, 2, 2, 2, 3))

    def test_script(self):
        loss = JukeboxLoss(spatial_dims=2)
        test_input = torch.ones(2, 1, 8, 8)
        test_script_save(loss, test_input, test_input)


if __name__ == "__main__":
    unittest.main()
