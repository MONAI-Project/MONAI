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
import torch.nn as nn

import monai.networks.nets.attentionunet as att
from tests.util import skip_if_no_cuda, skip_if_quick


def get_net_parameters(net: nn.Module) -> int:
    """Returns the total number of parameters in a Module."""
    return sum(param.numel() for param in net.parameters())


class TestAttentionUnet(unittest.TestCase):
    def test_attention_block(self):
        for dims in [2, 3]:
            block = att.AttentionBlock(dims, f_int=2, f_g=6, f_l=6)
            shape = (4, 6) + (30,) * dims
            x = torch.rand(*shape, dtype=torch.float32)
            output = block(x, x)
            self.assertEqual(output.shape, x.shape)

            block = att.AttentionBlock(dims, f_int=2, f_g=3, f_l=6)
            xshape = (4, 6) + (30,) * dims
            x = torch.rand(*xshape, dtype=torch.float32)
            gshape = (4, 3) + (30,) * dims
            g = torch.rand(*gshape, dtype=torch.float32)
            output = block(g, x)
            self.assertEqual(output.shape, x.shape)

    @skip_if_quick
    def test_attentionunet(self):
        for dims in [2, 3]:
            shape = (3, 1) + (92,) * dims
            input = torch.rand(*shape)
            model = att.AttentionUnet(
                spatial_dims=dims, in_channels=1, out_channels=2, channels=(3, 4, 5), up_kernel_size=5, strides=(1, 2)
            )
            output = model(input)
            self.assertEqual(output.shape[2:], input.shape[2:])
            self.assertEqual(output.shape[0], input.shape[0])
            self.assertEqual(output.shape[1], 2)

    def test_attentionunet_kernel_size(self):
        args_dict = {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 2,
            "channels": (3, 4, 5),
            "up_kernel_size": 5,
            "strides": (1, 2),
        }
        model_a = att.AttentionUnet(**args_dict, kernel_size=5)
        model_b = att.AttentionUnet(**args_dict, kernel_size=7)
        self.assertEqual(get_net_parameters(model_a), 3534)
        self.assertEqual(get_net_parameters(model_b), 5574)

    @skip_if_no_cuda
    def test_attentionunet_gpu(self):
        for dims in [2, 3]:
            shape = (3, 1) + (92,) * dims
            input = torch.rand(*shape).to("cuda:0")
            model = att.AttentionUnet(
                spatial_dims=dims, in_channels=1, out_channels=2, channels=(3, 4, 5), strides=(2, 2)
            ).to("cuda:0")
            with torch.no_grad():
                output = model(input)
                self.assertEqual(output.shape[2:], input.shape[2:])
                self.assertEqual(output.shape[0], input.shape[0])
                self.assertEqual(output.shape[1], 2)


if __name__ == "__main__":
    unittest.main()
