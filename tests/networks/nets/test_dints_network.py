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

from monai.networks.nets import DiNTS, TopologyInstance, TopologySearch
from monai.networks.nets.dints import Cell
from tests.util import SkipIfBeforePyTorchVersion, skip_if_quick, test_script_save

TEST_CASES_3D = [
    [
        {
            "channel_mul": 0.2,
            "num_blocks": 6,
            "num_depths": 3,
            "device": "cpu",
            "use_downsample": False,
            "spatial_dims": 3,
        },
        {
            "in_channels": 1,
            "num_classes": 3,
            "act_name": "RELU",
            "norm_name": ("INSTANCE", {"affine": True}),
            "use_downsample": False,
            "spatial_dims": 3,
        },
        (3, 1, 32, 32, 16),
        (3, 3, 32, 32, 16),
    ]
]
if torch.cuda.is_available():
    TEST_CASES_3D += [
        [
            {
                "channel_mul": 0.5,
                "num_blocks": 7,
                "num_depths": 4,
                "device": "cuda",
                "use_downsample": True,
                "spatial_dims": 3,
            },
            {
                "in_channels": 2,
                "num_classes": 2,
                "act_name": "PRELU",
                "norm_name": "BATCH",
                "use_downsample": True,
                "spatial_dims": 3,
            },
            (3, 2, 32, 32, 16),
            (3, 2, 32, 32, 16),
        ]
    ]
TEST_CASES_2D = [
    [
        {
            "channel_mul": 1,
            "num_blocks": 7,
            "num_depths": 4,
            "device": "cpu",
            "use_downsample": True,
            "spatial_dims": 2,
        },
        {
            "in_channels": 2,
            "num_classes": 2,
            "act_name": "PRELU",
            "norm_name": "BATCH",
            "use_downsample": True,
            "spatial_dims": 2,
        },
        (2, 2, 32, 16),
        (2, 2, 32, 16),
    ]
]
if torch.cuda.is_available():
    TEST_CASES_2D += [
        [
            {
                "channel_mul": 0.5,
                "num_blocks": 8,
                "num_depths": 4,
                "device": "cuda",
                "use_downsample": False,
                "spatial_dims": 2,
            },
            {
                "in_channels": 1,
                "num_classes": 4,
                "act_name": "RELU",
                "norm_name": ("INSTANCE", {"affine": True}),
                "use_downsample": False,
                "spatial_dims": 2,
            },
            (2, 1, 32, 16),
            (2, 4, 32, 16),
        ]
    ]


@skip_if_quick
class TestDints(unittest.TestCase):
    @parameterized.expand(TEST_CASES_3D + TEST_CASES_2D)
    def test_dints_inference(self, dints_grid_params, dints_params, input_shape, expected_shape):
        grid = TopologySearch(**dints_grid_params)
        dints_params["dints_space"] = grid
        net = DiNTS(**dints_params).to(dints_grid_params["device"])
        result = net(torch.randn(input_shape).to(dints_grid_params["device"]))
        self.assertEqual(result.shape, expected_shape)
        # test functions
        grid.get_ram_cost_usage(in_size=input_shape, full=True)
        grid.get_ram_cost_usage(in_size=input_shape, full=False)
        probs_a, _ = grid.get_prob_a(child=True)
        grid.get_topology_entropy(probs_a)
        grid.decode()
        grid.gen_mtx(depth=4)

    @parameterized.expand(TEST_CASES_3D + TEST_CASES_2D)
    def test_dints_search(self, dints_grid_params, dints_params, input_shape, expected_shape):
        num_blocks = dints_grid_params["num_blocks"]
        num_depths = dints_grid_params["num_depths"]
        # init a Cell to obtain cell operation number
        _cell = Cell(1, 1, 0, spatial_dims=dints_grid_params["spatial_dims"])
        num_cell_ops = len(_cell.OPS)
        # define archtecture codes
        node_a = torch.ones((num_blocks + 1, num_depths))
        arch_code_a = np.ones((num_blocks, 3 * num_depths - 2))
        arch_code_c = np.random.randint(num_cell_ops, size=(num_blocks, 3 * num_depths - 2))
        # initialize with codes
        dints_grid_params["arch_code"] = [arch_code_a, arch_code_c]
        grid = TopologyInstance(**dints_grid_params)
        # set as deploy stage
        dints_params["dints_space"] = grid
        dints_params["node_a"] = node_a
        net = DiNTS(**dints_params).to(dints_grid_params["device"])
        result = net(torch.randn(input_shape).to(dints_grid_params["device"]))
        self.assertEqual(result.shape, expected_shape)
        self.assertTrue(isinstance(net.weight_parameters(), list))


@SkipIfBeforePyTorchVersion((1, 9))
class TestDintsTS(unittest.TestCase):
    @parameterized.expand(TEST_CASES_3D + TEST_CASES_2D)
    def test_script(self, dints_grid_params, dints_params, input_shape, _):
        grid = TopologyInstance(**dints_grid_params)
        dints_grid_params["device"] = "cpu"
        dints_params["dints_space"] = grid
        net = DiNTS(**dints_params).to(dints_grid_params["device"])
        test_script_save(net, torch.randn(input_shape).to(dints_grid_params["device"]))


if __name__ == "__main__":
    unittest.main()
