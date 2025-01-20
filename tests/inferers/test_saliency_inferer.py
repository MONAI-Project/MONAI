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

from monai.inferers import SaliencyInferer
from monai.networks.nets import DenseNet
from monai.visualize.visualizer import default_upsampler

TEST_CASE_1 = ["CAM"]

TEST_CASE_2 = ["GradCAM"]

TEST_CASE_3 = ["GradCAMpp"]


class TestSaliencyInferer(unittest.TestCase):

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3])
    def test_shape(self, cam_name):
        model = DenseNet(
            spatial_dims=3, in_channels=1, out_channels=3, init_features=2, growth_rate=2, block_config=(6,)
        )
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()

        image = torch.rand((2, 1, 6, 6, 6), device=device)
        target_layer = "class_layers.relu"
        fc_layer = "class_layers.out"
        if cam_name == "CAM":
            inferer = SaliencyInferer(cam_name, target_layer, None, fc_layer, upsampler=default_upsampler)
            result = inferer(inputs=image, network=model, layer_idx=-1)
        else:
            inferer = SaliencyInferer(cam_name, target_layer, None, upsampler=default_upsampler)
            result = inferer(image, model, -1, retain_graph=False)

        self.assertTupleEqual(result.shape, (2, 1, 6, 6, 6))


if __name__ == "__main__":
    unittest.main()
