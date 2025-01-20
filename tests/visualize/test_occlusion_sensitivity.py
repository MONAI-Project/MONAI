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
from typing import Any

import torch
from parameterized import parameterized

from monai.networks.nets import DenseNet, DenseNet121
from monai.visualize import OcclusionSensitivity


class DenseNetAdjoint(DenseNet121):

    def __call__(self, x, adjoint_info):
        if adjoint_info != 42:
            raise ValueError
        return super().__call__(x)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
out_channels_2d = 4
out_channels_3d = 3
model_2d = DenseNet121(spatial_dims=2, in_channels=1, out_channels=out_channels_2d).to(device)
model_2d_2c = DenseNet121(spatial_dims=2, in_channels=2, out_channels=out_channels_2d).to(device)
model_3d = DenseNet(
    spatial_dims=3, in_channels=1, out_channels=out_channels_3d, init_features=2, growth_rate=2, block_config=(6,)
).to(device)
model_2d_adjoint = DenseNetAdjoint(spatial_dims=2, in_channels=1, out_channels=out_channels_2d).to(device)
model_2d.eval()
model_2d_2c.eval()
model_3d.eval()
model_2d_adjoint.eval()

TESTS: list[Any] = []
TESTS_FAIL: list[Any] = []

# 2D w/ bounding box with all modes
for mode in ("gaussian", "mean_patch", "mean_img"):
    TESTS.append(
        [
            {"nn_module": model_2d, "mode": mode},
            {"x": torch.rand(1, 1, 48, 64).to(device), "b_box": [2, 40, 1, 62]},
            (1, out_channels_2d, 38, 61),
            (1, 1, 38, 61),
        ]
    )
# 3D w/ bounding box
TESTS.append(
    [
        {"nn_module": model_3d, "n_batch": 10, "mask_size": (16, 15, 14)},
        {"x": torch.rand(1, 1, 64, 32, 16).to(device), "b_box": [2, 43, -1, -1, -1, -1]},
        (1, out_channels_3d, 41, 32, 16),
        (1, 1, 41, 32, 16),
    ]
)
TESTS.append(
    [
        {"nn_module": model_3d, "n_batch": 10},
        {"x": torch.rand(1, 1, 6, 7, 8).to(device), "b_box": [1, 3, -1, -1, -1, -1]},
        (1, out_channels_3d, 2, 7, 8),
        (1, 1, 2, 7, 8),
    ]
)
TESTS.append(
    [
        {"nn_module": model_2d_2c},
        {"x": torch.rand(1, 2, 48, 64).to(device)},
        (1, out_channels_2d, 48, 64),
        (1, 1, 48, 64),
    ]
)
# 2D w/ bounding box and adjoint
TESTS.append(
    [
        {"nn_module": model_2d_adjoint},
        {"x": torch.rand(1, 1, 48, 64).to(device), "b_box": [2, 40, 1, 62], "adjoint_info": 42},
        (1, out_channels_2d, 38, 61),
        (1, 1, 38, 61),
    ]
)
# 2D should fail: bbox makes image too small
TESTS_FAIL.append(
    [{"nn_module": model_2d, "n_batch": 10, "mask_size": 200}, {"x": torch.rand(1, 1, 48, 64).to(device)}, ValueError]
)
# 2D should fail: batch > 1
TESTS_FAIL.append(
    [{"nn_module": model_2d, "n_batch": 10, "mask_size": 100}, {"x": torch.rand(2, 1, 48, 64).to(device)}, ValueError]
)
# 2D should fail: unknown mode
TESTS_FAIL.append(
    [{"nn_module": model_2d, "mode": "test"}, {"x": torch.rand(1, 1, 48, 64).to(device)}, NotImplementedError]
)


class TestComputeOcclusionSensitivity(unittest.TestCase):

    @parameterized.expand(TESTS)
    def test_shape(self, init_data, call_data, map_expected_shape, most_prob_expected_shape):
        occ_sens = OcclusionSensitivity(**init_data)
        m, most_prob = occ_sens(**call_data)
        self.assertTupleEqual(m.shape, map_expected_shape)
        self.assertTupleEqual(most_prob.shape, most_prob_expected_shape)
        # most probable class should be of type int, and should have min>=0, max<num_classes
        self.assertEqual(most_prob.dtype, torch.int64)
        self.assertGreaterEqual(most_prob.min(), 0)
        self.assertLess(most_prob.max(), m.shape[-1])

    @parameterized.expand(TESTS_FAIL)
    def test_fail(self, init_data, call_data, error_type):
        with self.assertRaises(error_type):
            occ_sens = OcclusionSensitivity(**init_data)
            occ_sens(**call_data)


if __name__ == "__main__":
    unittest.main()
