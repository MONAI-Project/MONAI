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

import torch
from parameterized import parameterized

from monai.networks.nets import Unet
from monai.optimizers import generate_param_groups
from monai.utils import ensure_tuple
from tests.util import assert_allclose

TEST_CASE_1 = [{"layer_matches": [lambda x: x.model[-1]], "match_types": "select", "lr_values": [1]}, (1, 100), [5, 21]]

TEST_CASE_2 = [
    {
        "layer_matches": [lambda x: x.model[-1], lambda x: x.model[-2], lambda x: x.model[-3]],
        "match_types": "select",
        "lr_values": [1, 2, 3],
    },
    (1, 2, 3, 100),
    [5, 16, 5, 0],
]

TEST_CASE_3 = [
    {"layer_matches": [lambda x: x.model[2][1].conv[0].conv], "match_types": ["select"], "lr_values": [1]},
    (1, 100),
    [2, 24],
]

TEST_CASE_4 = [
    {
        "layer_matches": [lambda x: x.model[0], lambda x: "2.0.conv" in x[0]],
        "match_types": ["select", "filter"],
        "lr_values": [1, 2],
    },
    (1, 2, 100),
    [5, 4, 17],
]

TEST_CASE_5 = [
    {"layer_matches": [lambda x: x.model[-1]], "match_types": ["select"], "lr_values": [1], "include_others": False},
    (1),
    [5],
]

TEST_CASE_6 = [
    {
        "layer_matches": [lambda x: "weight" in x[0]],
        "match_types": ["filter"],
        "lr_values": [1],
        "include_others": True,
    },
    (1),
    [16, 10],
]


class TestGenerateParamGroups(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4, TEST_CASE_5, TEST_CASE_6])
    def test_lr_values(self, input_param, expected_values, expected_groups):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        net = Unet(
            spatial_dims=3, in_channels=1, out_channels=3, channels=(16, 32, 64), strides=(2, 2), num_res_units=1
        ).to(device)

        params = generate_param_groups(network=net, **input_param)
        optimizer = torch.optim.Adam(params, 100)

        for param_group, value in zip(optimizer.param_groups, ensure_tuple(expected_values)):
            assert_allclose(param_group["lr"], value)

        n = [len(p["params"]) for p in params]
        self.assertListEqual(n, expected_groups)

    def test_wrong(self):
        """overlapped"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        net = Unet(
            spatial_dims=3, in_channels=1, out_channels=3, channels=(16, 32, 64), strides=(2, 2), num_res_units=1
        ).to(device)

        params = generate_param_groups(
            network=net,
            layer_matches=[lambda x: x.model[-1], lambda x: x.model[-1]],
            match_types="select",
            lr_values=0.1,
        )
        with self.assertRaises(ValueError):
            torch.optim.Adam(params, 100)


if __name__ == "__main__":
    unittest.main()
