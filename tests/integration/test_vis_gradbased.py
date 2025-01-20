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

from monai.networks.nets import DenseNet, DenseNet121, SEResNet50
from monai.visualize import GuidedBackpropGrad, GuidedBackpropSmoothGrad, SmoothGrad, VanillaGrad


class DenseNetAdjoint(DenseNet121):

    def __call__(self, x, adjoint_info):
        if adjoint_info != 42:
            raise ValueError
        return super().__call__(x)


DENSENET2D = DenseNet121(spatial_dims=2, in_channels=1, out_channels=3)
DENSENET3D = DenseNet(spatial_dims=3, in_channels=1, out_channels=3, init_features=2, growth_rate=2, block_config=(6,))
SENET2D = SEResNet50(spatial_dims=2, in_channels=3, num_classes=4)
SENET3D = SEResNet50(spatial_dims=3, in_channels=3, num_classes=4)
DENSENET2DADJOINT = DenseNetAdjoint(spatial_dims=2, in_channels=1, out_channels=3)

TESTS = []
for type in (VanillaGrad, SmoothGrad, GuidedBackpropGrad, GuidedBackpropSmoothGrad):
    # 2D densenet
    TESTS.append([type, DENSENET2D, (1, 1, 48, 64)])
    # 3D densenet
    TESTS.append([type, DENSENET3D, (1, 1, 6, 6, 6)])
    # 2D senet
    TESTS.append([type, SENET2D, (1, 3, 64, 64)])
    # 3D senet
    TESTS.append([type, SENET3D, (1, 3, 8, 8, 48)])
    # 2D densenet - adjoint
    TESTS.append([type, DENSENET2DADJOINT, (1, 1, 48, 64)])


class TestGradientClassActivationMap(unittest.TestCase):

    @parameterized.expand(TESTS)
    def test_shape(self, vis_type, model, shape):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # optionally test for adjoint info
        kwargs = {"adjoint_info": 42} if isinstance(model, DenseNetAdjoint) else {}

        model.to(device)
        model.eval()
        vis = vis_type(model)
        x = torch.rand(shape, device=device)
        result = vis(x, **kwargs)
        self.assertTupleEqual(result.shape, x.shape)


if __name__ == "__main__":
    unittest.main()
