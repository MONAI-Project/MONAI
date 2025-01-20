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

import os
import tempfile
import unittest
from unittest import skipUnless

import torch
from parameterized import parameterized

from monai.apps import download_url
from monai.networks import eval_mode
from monai.networks.nets.controlnet import ControlNet
from monai.utils import optional_import
from tests.util import skip_if_downloading_fails, testing_data_config

_, has_einops = optional_import("einops")
UNCOND_CASES_2D = [
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "num_res_blocks": 1,
            "channels": (8, 8, 8),
            "attention_levels": (False, False, False),
            "norm_num_groups": 8,
        },
        (1, 8, 4, 4),
    ],
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "num_res_blocks": 1,
            "channels": (8, 8, 8),
            "attention_levels": (False, False, False),
            "norm_num_groups": 8,
            "resblock_updown": True,
        },
        (1, 8, 4, 4),
    ],
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "num_res_blocks": 1,
            "channels": (4, 4, 4),
            "attention_levels": (False, False, True),
            "num_head_channels": 4,
            "norm_num_groups": 4,
        },
        (1, 4, 4, 4),
    ],
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "num_res_blocks": 1,
            "channels": (8, 8, 8),
            "attention_levels": (False, False, True),
            "num_head_channels": 8,
            "norm_num_groups": 8,
            "resblock_updown": True,
        },
        (1, 8, 4, 4),
    ],
]

UNCOND_CASES_3D = [
    [
        {
            "spatial_dims": 3,
            "in_channels": 1,
            "num_res_blocks": 1,
            "channels": (8, 8, 8),
            "attention_levels": (False, False, False),
            "norm_num_groups": 8,
        },
        (1, 8, 4, 4, 4),
    ],
    [
        {
            "spatial_dims": 3,
            "in_channels": 1,
            "num_res_blocks": 1,
            "channels": (4, 4, 4),
            "num_head_channels": 4,
            "attention_levels": (False, False, False),
            "norm_num_groups": 4,
            "resblock_updown": True,
        },
        (1, 4, 4, 4, 4),
    ],
]

COND_CASES_2D = [
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "num_res_blocks": 1,
            "channels": (8, 8, 8),
            "attention_levels": (False, False, False),
            "norm_num_groups": 8,
            "with_conditioning": True,
            "transformer_num_layers": 1,
            "cross_attention_dim": 3,
        },
        (1, 8, 4, 4),
    ],
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "num_res_blocks": 1,
            "channels": (8, 8, 8),
            "attention_levels": (False, False, False),
            "norm_num_groups": 8,
            "with_conditioning": True,
            "transformer_num_layers": 1,
            "cross_attention_dim": 3,
            "resblock_updown": True,
        },
        (1, 8, 4, 4),
    ],
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "num_res_blocks": 1,
            "channels": (8, 8, 8),
            "attention_levels": (False, False, False),
            "norm_num_groups": 8,
            "with_conditioning": True,
            "transformer_num_layers": 1,
            "cross_attention_dim": 3,
            "upcast_attention": True,
        },
        (1, 8, 4, 4),
    ],
]


class TestControlNet(unittest.TestCase):
    @parameterized.expand(UNCOND_CASES_2D + UNCOND_CASES_3D)
    @skipUnless(has_einops, "Requires einops")
    def test_shape_unconditioned_models(self, input_param, expected_output_shape):
        input_param["conditioning_embedding_in_channels"] = input_param["in_channels"]
        input_param["conditioning_embedding_num_channels"] = (input_param["channels"][0],)
        net = ControlNet(**input_param)
        with eval_mode(net):
            x = torch.rand((1, 1) + (16,) * input_param["spatial_dims"])
            timesteps = torch.randint(0, 1000, (1,)).long()
            controlnet_cond = torch.rand((1, 1) + (16,) * input_param["spatial_dims"])
            result = net.forward(x, timesteps=timesteps, controlnet_cond=controlnet_cond)
            self.assertEqual(len(result[0]), 2 * len(input_param["channels"]))
            self.assertEqual(result[1].shape, expected_output_shape)

    @parameterized.expand(COND_CASES_2D)
    @skipUnless(has_einops, "Requires einops")
    def test_shape_conditioned_models(self, input_param, expected_output_shape):
        input_param["conditioning_embedding_in_channels"] = input_param["in_channels"]
        input_param["conditioning_embedding_num_channels"] = (input_param["channels"][0],)
        net = ControlNet(**input_param)
        with eval_mode(net):
            x = torch.rand((1, 1) + (16,) * input_param["spatial_dims"])
            timesteps = torch.randint(0, 1000, (1,)).long()
            controlnet_cond = torch.rand((1, 1) + (16,) * input_param["spatial_dims"])
            result = net.forward(x, timesteps=timesteps, controlnet_cond=controlnet_cond, context=torch.rand((1, 1, 3)))
            self.assertEqual(len(result[0]), 2 * len(input_param["channels"]))
            self.assertEqual(result[1].shape, expected_output_shape)

    @skipUnless(has_einops, "Requires einops")
    def test_compatibility_with_monai_generative(self):
        # test loading weights from a model saved in MONAI Generative, version 0.2.3
        with skip_if_downloading_fails():
            net = ControlNet(
                spatial_dims=2,
                in_channels=1,
                num_res_blocks=1,
                channels=(8, 8, 8),
                attention_levels=(False, False, True),
                norm_num_groups=8,
                with_conditioning=True,
                transformer_num_layers=1,
                cross_attention_dim=3,
                resblock_updown=True,
            )

            tmpdir = tempfile.mkdtemp()
            key = "controlnet_monai_generative_weights"
            url = testing_data_config("models", key, "url")
            hash_type = testing_data_config("models", key, "hash_type")
            hash_val = testing_data_config("models", key, "hash_val")
            filename = "controlnet_monai_generative_weights.pt"

            weight_path = os.path.join(tmpdir, filename)
            download_url(url=url, filepath=weight_path, hash_val=hash_val, hash_type=hash_type)

            net.load_old_state_dict(torch.load(weight_path), verbose=False)


if __name__ == "__main__":
    unittest.main()
