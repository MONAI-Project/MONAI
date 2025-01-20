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

from monai.apps.vista3d.inferer import point_based_window_inferer
from monai.networks import eval_mode
from monai.networks.nets.vista3d import vista3d132
from monai.utils import optional_import
from tests.util import SkipIfBeforePyTorchVersion, skip_if_quick

device = "cuda" if torch.cuda.is_available() else "cpu"

_, has_tqdm = optional_import("tqdm")

TEST_CASES = [
    [
        {"encoder_embed_dim": 48, "in_channels": 1},
        (1, 1, 64, 64, 64),
        {
            "roi_size": [32, 32, 32],
            "point_coords": torch.tensor([[[1, 2, 3], [1, 2, 3]]], device=device),
            "point_labels": torch.tensor([[1, 0]], device=device),
        },
    ],
    [
        {"encoder_embed_dim": 48, "in_channels": 1},
        (1, 1, 64, 64, 64),
        {
            "roi_size": [32, 32, 32],
            "point_coords": torch.tensor([[[1, 2, 3], [1, 2, 3]]], device=device),
            "point_labels": torch.tensor([[1, 0]], device=device),
            "class_vector": torch.tensor([1], device=device),
        },
    ],
    [
        {"encoder_embed_dim": 48, "in_channels": 1},
        (1, 1, 64, 64, 64),
        {
            "roi_size": [32, 32, 32],
            "point_coords": torch.tensor([[[1, 2, 3], [1, 2, 3]]], device=device),
            "point_labels": torch.tensor([[1, 0]], device=device),
            "class_vector": torch.tensor([1], device=device),
            "point_start": 1,
        },
    ],
]


@SkipIfBeforePyTorchVersion((1, 11))
@skip_if_quick
class TestPointBasedWindowInferer(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_vista3d(self, vista3d_params, inputs_shape, inferer_params):
        vista3d = vista3d132(**vista3d_params).to(device)
        with eval_mode(vista3d):
            inferer_params["predictor"] = vista3d
            inferer_params["inputs"] = torch.randn(*inputs_shape).to(device)
            stitched_output = point_based_window_inferer(**inferer_params)
            self.assertEqual(stitched_output.shape, inputs_shape)


if __name__ == "__main__":
    unittest.main()
