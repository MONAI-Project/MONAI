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

from monai.networks.blocks import ADN
from tests.util import TorchImageTestCase2D, TorchImageTestCase3D

TEST_CASES_2D = [
    [{"act": None}],
    [{"norm_dim": 2}],
    [{"norm_dim": 2, "act": "relu", "dropout": 0.8, "ordering": "DA"}],
    [{"dropout_dim": 1, "dropout": 0.8, "ordering": "DA"}],
    [{"norm": "BATCH", "norm_dim": 2, "in_channels": 1, "dropout_dim": 1, "dropout": 0.8, "ordering": "NDA"}],
    [{"norm": "BATCH", "norm_dim": 2, "in_channels": 1, "dropout_dim": 1, "dropout": 0.8, "ordering": "AND"}],
    [{"norm": "INSTANCE", "norm_dim": 2, "dropout_dim": 1, "dropout": 0.8, "ordering": "AND"}],
    [
        {
            "norm": ("GROUP", {"num_groups": 1, "affine": False}),
            "in_channels": 1,
            "norm_dim": 2,
            "dropout_dim": 1,
            "dropout": 0.8,
            "ordering": "AND",
        }
    ],
    [{"norm": ("localresponse", {"size": 4}), "norm_dim": 2, "dropout_dim": 1, "dropout": 0.8, "ordering": "AND"}],
]

TEST_CASES_3D = [
    [{"norm_dim": 3}],
    [{"act": "prelu", "dropout_dim": 1, "dropout": 0.8, "ordering": "DA"}],
    [{"dropout_dim": 1, "dropout": 0.8, "ordering": "DA"}],
    [{"norm": "BATCH", "norm_dim": 3, "in_channels": 1, "dropout_dim": 1, "dropout": 0.8, "ordering": "NDA"}],
    [{"norm": "BATCH", "norm_dim": 3, "in_channels": 1, "dropout_dim": 1, "dropout": 0.8, "ordering": "AND"}],
    [{"norm": "INSTANCE", "norm_dim": 3, "dropout_dim": 1, "dropout": 0.8, "ordering": "AND"}],
    [
        {
            "norm": ("layer", {"normalized_shape": (48, 80)}),
            "norm_dim": 3,
            "dropout_dim": 1,
            "dropout": 0.8,
            "ordering": "AND",
        }
    ],
]


class TestADN2D(TorchImageTestCase2D):
    @parameterized.expand(TEST_CASES_2D)
    def test_adn_2d(self, args):
        adn = ADN(**args)
        print(adn)
        out = adn(self.imt)
        expected_shape = (1, self.input_channels, self.im_shape[0], self.im_shape[1])
        self.assertEqual(out.shape, expected_shape)

    def test_no_input(self):
        with self.assertRaises(ValueError):
            ADN(norm="instance")


class TestADN3D(TorchImageTestCase3D):
    @parameterized.expand(TEST_CASES_3D)
    def test_adn_3d(self, args):
        adn = ADN(**args)
        print(adn)
        out = adn(self.imt)
        expected_shape = (1, self.input_channels, self.im_shape[0], self.im_shape[1], self.im_shape[2])
        self.assertEqual(out.shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
