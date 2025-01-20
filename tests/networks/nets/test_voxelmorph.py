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

from monai.networks import eval_mode
from monai.networks.nets import VoxelMorph, VoxelMorphUNet
from tests.util import test_script_save

device = "cuda" if torch.cuda.is_available() else "cpu"

TEST_CASE_0 = [  # single channel 3D, batch 1,
    {
        "spatial_dims": 3,
        "in_channels": 2,
        "unet_out_channels": 32,
        "channels": (16, 32, 32, 32, 32, 32),
        "final_conv_channels": (16, 16),
    },
    (1, 2, 96, 96, 48),
    (1, 3, 96, 96, 48),
]

TEST_CASE_1 = [  # single channel 3D, batch 1,
    # using strided convolution for downsampling instead of maxpooling
    {
        "spatial_dims": 3,
        "in_channels": 2,
        "unet_out_channels": 32,
        "channels": (16, 32, 32, 32, 32, 32),
        "final_conv_channels": (16, 16),
        "use_maxpool": False,
    },
    (1, 2, 96, 96, 48),
    (1, 3, 96, 96, 48),
]

TEST_CASE_2 = [  # single channel 3D, batch 1,
    # using strided convolution for downsampling instead of maxpooling,
    # explicitly specify leakyrelu with a different negative slope for final convolutions
    {
        "spatial_dims": 3,
        "in_channels": 2,
        "unet_out_channels": 32,
        "channels": (16, 32, 32, 32, 32, 32),
        "final_conv_channels": (16, 16),
        "final_conv_act": ("leakyrelu", {"negative_slope": 0.1, "inplace": True}),
        "use_maxpool": False,
    },
    (1, 2, 96, 96, 48),
    (1, 3, 96, 96, 48),
]

TEST_CASE_3 = [  # single channel 3D, batch 1,
    # using strided convolution for downsampling instead of maxpooling,
    # explicitly specify leakyrelu with a different negative slope for both unet and final convolutions.
    {
        "spatial_dims": 3,
        "in_channels": 2,
        "unet_out_channels": 32,
        "channels": (16, 32, 32, 32, 32, 32),
        "final_conv_channels": (16, 16),
        "final_conv_act": ("leakyrelu", {"negative_slope": 0.1, "inplace": True}),
        "act": ("leakyrelu", {"negative_slope": 0.1, "inplace": True}),
        "use_maxpool": False,
    },
    (1, 2, 96, 96, 48),
    (1, 3, 96, 96, 48),
]

TEST_CASE_4 = [  # 2-channel 3D, batch 1,
    # i.e., possible use case where the input contains both modalities (e.g., T1 and T2)
    {
        "spatial_dims": 3,
        "in_channels": 4,
        "unet_out_channels": 32,
        "channels": (16, 32, 32, 32, 32, 32),
        "final_conv_channels": (16, 16),
    },
    (1, 4, 96, 96, 48),
    (1, 3, 96, 96, 48),
]

TEST_CASE_5 = [  # single channel 3D, batch 2,
    {
        "spatial_dims": 3,
        "in_channels": 2,
        "unet_out_channels": 32,
        "channels": (16, 32, 32, 32, 32, 32),
        "final_conv_channels": (16, 16),
    },
    (2, 2, 96, 96, 48),
    (2, 3, 96, 96, 48),
]

TEST_CASE_6 = [  # single channel 2D, batch 2,
    {
        "spatial_dims": 2,
        "in_channels": 2,
        "unet_out_channels": 32,
        "channels": (16, 32, 32, 32, 32, 32),
        "final_conv_channels": (16, 16),
    },
    (2, 2, 96, 96),
    (2, 2, 96, 96),
]

TEST_CASE_7 = [  # single channel 3D, batch 1,
    # one additional level in the UNet with 32 channels in both down and up branch.
    {
        "spatial_dims": 3,
        "in_channels": 2,
        "unet_out_channels": 32,
        "channels": (16, 32, 32, 32, 32, 32, 32, 32),
        "final_conv_channels": (16, 16),
    },
    (1, 2, 96, 96, 48),
    (1, 3, 96, 96, 48),
]

TEST_CASE_8 = [  # single channel 3D, batch 1,
    # one additional level in the UNet with 32 channels in both down and up branch.
    # and removed one of the two final convolution blocks.
    {
        "spatial_dims": 3,
        "in_channels": 2,
        "unet_out_channels": 32,
        "channels": (16, 32, 32, 32, 32, 32, 32, 32),
        "final_conv_channels": (16,),
    },
    (1, 2, 96, 96, 48),
    (1, 3, 96, 96, 48),
]

TEST_CASE_9 = [  # single channel 3D, batch 1,
    # only one level in the UNet
    {
        "spatial_dims": 3,
        "in_channels": 2,
        "unet_out_channels": 32,
        "channels": (16, 32),
        "final_conv_channels": (16, 16),
    },
    (1, 2, 96, 96, 48),
    (1, 3, 96, 96, 48),
]

CASES = [
    TEST_CASE_0,
    TEST_CASE_1,
    TEST_CASE_2,
    TEST_CASE_3,
    TEST_CASE_4,
    TEST_CASE_5,
    TEST_CASE_6,
    TEST_CASE_7,
    TEST_CASE_8,
    TEST_CASE_9,
]

ILL_CASE_0 = [  # spatial_dims = 1
    {
        "spatial_dims": 1,
        "in_channels": 2,
        "unet_out_channels": 32,
        "channels": (16, 32, 32, 32, 32, 32),
        "final_conv_channels": (16, 16),
    }
]

ILL_CASE_1 = [  # in_channels = 3 (not divisible by 2)
    {
        "spatial_dims": 3,
        "in_channels": 3,
        "unet_out_channels": 32,
        "channels": (16, 32, 32, 32, 32, 32),
        "final_conv_channels": (16, 16),
    }
]

ILL_CASE_2 = [  # len(channels) = 0
    {"spatial_dims": 3, "in_channels": 2, "unet_out_channels": 32, "channels": (), "final_conv_channels": (16, 16)}
]

ILL_CASE_3 = [  # channels not in pairs
    {
        "spatial_dims": 3,
        "in_channels": 2,
        "unet_out_channels": 32,
        "channels": (16, 32, 32, 32, 32),
        "final_conv_channels": (16, 16),
    }
]

ILL_CASE_4 = [  # len(kernel_size) = 3, spatial_dims = 2
    {
        "spatial_dims": 2,
        "in_channels": 2,
        "unet_out_channels": 32,
        "channels": (16, 32, 32, 32, 32, 32),
        "final_conv_channels": (16, 16),
        "kernel_size": (3, 3, 3),
    }
]

ILL_CASE_5 = [  # len(up_kernel_size) = 2, spatial_dims = 3
    {
        "spatial_dims": 3,
        "in_channels": 2,
        "unet_out_channels": 32,
        "channels": (16, 32, 32, 32, 32, 32),
        "final_conv_channels": (16, 16),
        "up_kernel_size": (3, 3),
    }
]

ILL_CASES = [ILL_CASE_0, ILL_CASE_1, ILL_CASE_2, ILL_CASE_3, ILL_CASE_4, ILL_CASE_5]

ILL_CASES_IN_SHAPE_0 = [  # moving and fixed image shape not match
    {"spatial_dims": 3},
    (1, 2, 96, 96, 48),
    (1, 3, 96, 96, 48),
]

ILL_CASES_IN_SHAPE_1 = [  # spatial_dims = 2, ddf has 3 channels
    {"spatial_dims": 2},
    (1, 1, 96, 96, 96),
    (1, 1, 96, 96, 96),
]

ILL_CASES_IN_SHAPE = [ILL_CASES_IN_SHAPE_0, ILL_CASES_IN_SHAPE_1]


class TestVOXELMORPH(unittest.TestCase):
    @parameterized.expand(CASES)
    def test_shape(self, input_param, input_shape, expected_shape):
        net = VoxelMorphUNet(**input_param).to(device)
        with eval_mode(net):
            result = net.forward(torch.randn(input_shape).to(device))
            self.assertEqual(result.shape, expected_shape)

    def test_script(self):
        net = VoxelMorphUNet(
            spatial_dims=2,
            in_channels=2,
            unet_out_channels=32,
            channels=(16, 32, 32, 32, 32, 32),
            final_conv_channels=(16, 16),
        )
        test_data = torch.randn(1, 2, 96, 96)
        test_script_save(net, test_data)

    @parameterized.expand(ILL_CASES)
    def test_ill_input_hyper_params(self, input_param):
        with self.assertRaises(ValueError):
            _ = VoxelMorphUNet(**input_param)

    @parameterized.expand(ILL_CASES_IN_SHAPE)
    def test_ill_input_shape(self, input_param, moving_shape, fixed_shape):
        with self.assertRaises((ValueError, RuntimeError)):
            net = VoxelMorph(**input_param).to(device)
            with eval_mode(net):
                _ = net.forward(torch.randn(moving_shape).to(device), torch.randn(fixed_shape).to(device))


if __name__ == "__main__":
    unittest.main()
