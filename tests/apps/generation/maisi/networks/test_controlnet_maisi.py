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
from unittest import skipUnless

import torch
from parameterized import parameterized

from monai.apps.generation.maisi.networks.controlnet_maisi import ControlNetMaisi
from monai.networks import eval_mode
from monai.utils import optional_import
from tests.util import SkipIfBeforePyTorchVersion

_, has_einops = optional_import("einops")

TEST_CASES = [
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "num_res_blocks": 1,
            "num_channels": (8, 8, 8),
            "attention_levels": (False, False, True),
            "num_head_channels": 8,
            "norm_num_groups": 8,
            "conditioning_embedding_in_channels": 1,
            "conditioning_embedding_num_channels": (8, 8),
            "use_checkpointing": False,
        },
        6,
        (1, 8, 4, 4),
    ],
    [
        {
            "spatial_dims": 3,
            "in_channels": 1,
            "num_res_blocks": 1,
            "num_channels": (8, 8, 8),
            "attention_levels": (False, False, True),
            "num_head_channels": 8,
            "norm_num_groups": 8,
            "conditioning_embedding_in_channels": 1,
            "conditioning_embedding_num_channels": (8, 8),
            "use_checkpointing": True,
        },
        6,
        (1, 8, 4, 4, 4),
    ],
]

TEST_CASES_CONDITIONAL = [
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "num_res_blocks": 1,
            "num_channels": (8, 8, 8),
            "attention_levels": (False, False, True),
            "num_head_channels": 8,
            "norm_num_groups": 8,
            "conditioning_embedding_in_channels": 1,
            "conditioning_embedding_num_channels": (8, 8),
            "use_checkpointing": False,
            "with_conditioning": True,
            "cross_attention_dim": 2,
        },
        6,
        (1, 8, 4, 4),
    ],
    [
        {
            "spatial_dims": 3,
            "in_channels": 1,
            "num_res_blocks": 1,
            "num_channels": (8, 8, 8),
            "attention_levels": (False, False, True),
            "num_head_channels": 8,
            "norm_num_groups": 8,
            "conditioning_embedding_in_channels": 1,
            "conditioning_embedding_num_channels": (8, 8),
            "use_checkpointing": True,
            "with_conditioning": True,
            "cross_attention_dim": 2,
        },
        6,
        (1, 8, 4, 4, 4),
    ],
]

TEST_CASES_ERROR = [
    [
        {"spatial_dims": 2, "in_channels": 1, "with_conditioning": True, "cross_attention_dim": None},
        "ControlNet expects dimension of the cross-attention conditioning (cross_attention_dim) "
        "to be specified when with_conditioning=True.",
    ],
    [
        {"spatial_dims": 2, "in_channels": 1, "with_conditioning": False, "cross_attention_dim": 2},
        "ControlNet expects with_conditioning=True when specifying the cross_attention_dim.",
    ],
    [
        {"spatial_dims": 2, "in_channels": 1, "num_channels": (8, 16), "norm_num_groups": 16},
        f"ControlNet expects all channels to be a multiple of norm_num_groups, but got"
        f" channels={(8, 16)} and norm_num_groups={16}",
    ],
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "num_channels": (8, 16),
            "attention_levels": (True,),
            "norm_num_groups": 8,
        },
        f"ControlNet expects channels to have the same length as attention_levels, but got "
        f"channels={(8, 16)} and attention_levels={(True,)}",
    ],
]


@SkipIfBeforePyTorchVersion((2, 0))
class TestControlNet(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    @skipUnless(has_einops, "Requires einops")
    def test_shape_unconditioned_models(self, input_param, expected_num_down_blocks_residuals, expected_shape):
        net = ControlNetMaisi(**input_param)
        with eval_mode(net):
            x = torch.rand((1, 1, 16, 16)) if input_param["spatial_dims"] == 2 else torch.rand((1, 1, 16, 16, 16))
            timesteps = torch.randint(0, 1000, (1,)).long()
            controlnet_cond = (
                torch.rand((1, 1, 32, 32)) if input_param["spatial_dims"] == 2 else torch.rand((1, 1, 32, 32, 32))
            )
            result = net.forward(x, timesteps, controlnet_cond)
            self.assertEqual(len(result[0]), expected_num_down_blocks_residuals)
            self.assertEqual(result[1].shape, expected_shape)

    @parameterized.expand(TEST_CASES_CONDITIONAL)
    @skipUnless(has_einops, "Requires einops")
    def test_shape_conditioned_models(self, input_param, expected_num_down_blocks_residuals, expected_shape):
        net = ControlNetMaisi(**input_param)
        with eval_mode(net):
            x = torch.rand((1, 1, 16, 16)) if input_param["spatial_dims"] == 2 else torch.rand((1, 1, 16, 16, 16))
            timesteps = torch.randint(0, 1000, (1,)).long()
            controlnet_cond = (
                torch.rand((1, 1, 32, 32)) if input_param["spatial_dims"] == 2 else torch.rand((1, 1, 32, 32, 32))
            )
            context = torch.randn((1, 1, input_param["cross_attention_dim"]))
            result = net.forward(x, timesteps, controlnet_cond, context=context)
            self.assertEqual(len(result[0]), expected_num_down_blocks_residuals)
            self.assertEqual(result[1].shape, expected_shape)

    @parameterized.expand(TEST_CASES_ERROR)
    def test_error_input(self, input_param, expected_error):
        with self.assertRaises(ValueError) as context:  # output shape too small
            _ = ControlNetMaisi(**input_param)
        runtime_error = context.exception
        self.assertEqual(str(runtime_error), expected_error)


if __name__ == "__main__":
    unittest.main()
