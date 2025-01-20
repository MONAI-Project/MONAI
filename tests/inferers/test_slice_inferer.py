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

from monai.inferers import SliceInferer
from monai.networks.nets import UNet

TEST_CASES = ["0", "1", "2"]


class TestSliceInferer(unittest.TestCase):

    @parameterized.expand(TEST_CASES)
    def test_shape(self, spatial_dim):
        spatial_dim = int(spatial_dim)

        model = UNet(
            spatial_dims=2, in_channels=1, out_channels=1, channels=(4, 8, 16), strides=(2, 2), num_res_units=2
        )

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()

        # Initialize a dummy 3D tensor volume with shape (N,C,D,H,W)
        input_volume = torch.ones(1, 1, 64, 256, 256, device=device)

        # Remove spatial dim to slide across from the roi_size
        roi_size = list(input_volume.shape[2:])
        roi_size.pop(spatial_dim)

        # Initialize and run inferer
        inferer = SliceInferer(roi_size=roi_size, spatial_dim=spatial_dim, sw_batch_size=1, cval=-1)
        result = inferer(input_volume, model)

        self.assertTupleEqual(result.shape, input_volume.shape)

        # test that the inferer can be run multiple times
        result = inferer(input_volume, model)


if __name__ == "__main__":
    unittest.main()
