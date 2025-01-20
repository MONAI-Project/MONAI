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

from monai.networks import eval_mode
from monai.networks.nets import SPADEDiffusionModelUNet
from monai.utils import optional_import

einops, has_einops = optional_import("einops")
UNCOND_CASES_2D = [
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": 1,
            "channels": (8, 8, 8),
            "attention_levels": (False, False, False),
            "norm_num_groups": 8,
            "label_nc": 3,
        }
    ],
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": (1, 1, 2),
            "channels": (8, 8, 8),
            "attention_levels": (False, False, False),
            "norm_num_groups": 8,
            "label_nc": 3,
        }
    ],
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": 1,
            "channels": (8, 8, 8),
            "attention_levels": (False, False, False),
            "norm_num_groups": 8,
            "resblock_updown": True,
            "label_nc": 3,
        }
    ],
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": 1,
            "channels": (8, 8, 8),
            "attention_levels": (False, False, True),
            "num_head_channels": 8,
            "norm_num_groups": 8,
            "label_nc": 3,
        }
    ],
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": 1,
            "channels": (8, 8, 8),
            "attention_levels": (False, False, True),
            "num_head_channels": 8,
            "norm_num_groups": 8,
            "resblock_updown": True,
            "label_nc": 3,
        }
    ],
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": 1,
            "channels": (8, 8, 8),
            "attention_levels": (False, False, True),
            "num_head_channels": 4,
            "norm_num_groups": 8,
            "label_nc": 3,
        }
    ],
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": 1,
            "channels": (8, 8, 8),
            "attention_levels": (False, True, True),
            "num_head_channels": (0, 2, 4),
            "norm_num_groups": 8,
            "label_nc": 3,
        }
    ],
]

UNCOND_CASES_3D = [
    [
        {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": 1,
            "channels": (8, 8, 8),
            "attention_levels": (False, False, False),
            "norm_num_groups": 8,
            "label_nc": 3,
            "spade_intermediate_channels": 256,
        }
    ],
    [
        {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": 1,
            "channels": (8, 8, 8),
            "attention_levels": (False, False, False),
            "norm_num_groups": 8,
            "label_nc": 3,
        }
    ],
    [
        {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": 1,
            "channels": (8, 8, 8),
            "attention_levels": (False, False, False),
            "norm_num_groups": 8,
            "resblock_updown": True,
            "label_nc": 3,
        }
    ],
    [
        {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": 1,
            "channels": (8, 8, 8),
            "attention_levels": (False, False, True),
            "num_head_channels": 8,
            "norm_num_groups": 8,
            "label_nc": 3,
        }
    ],
    [
        {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": 1,
            "channels": (8, 8, 8),
            "attention_levels": (False, False, True),
            "num_head_channels": 8,
            "norm_num_groups": 8,
            "resblock_updown": True,
            "label_nc": 3,
        }
    ],
    [
        {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": 1,
            "channels": (8, 8, 8),
            "attention_levels": (False, False, True),
            "num_head_channels": 4,
            "norm_num_groups": 8,
            "label_nc": 3,
        }
    ],
    [
        {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": 1,
            "channels": (8, 8, 8),
            "attention_levels": (False, False, True),
            "num_head_channels": (0, 0, 4),
            "norm_num_groups": 8,
            "label_nc": 3,
        }
    ],
]

COND_CASES_2D = [
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": 1,
            "channels": (8, 8, 8),
            "attention_levels": (False, False, True),
            "num_head_channels": 4,
            "norm_num_groups": 8,
            "with_conditioning": True,
            "transformer_num_layers": 1,
            "cross_attention_dim": 3,
            "label_nc": 3,
        }
    ],
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": 1,
            "channels": (8, 8, 8),
            "attention_levels": (False, False, True),
            "num_head_channels": 4,
            "norm_num_groups": 8,
            "with_conditioning": True,
            "transformer_num_layers": 1,
            "cross_attention_dim": 3,
            "resblock_updown": True,
            "label_nc": 3,
        }
    ],
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": 1,
            "channels": (8, 8, 8),
            "attention_levels": (False, False, True),
            "num_head_channels": 4,
            "norm_num_groups": 8,
            "with_conditioning": True,
            "transformer_num_layers": 1,
            "cross_attention_dim": 3,
            "upcast_attention": True,
            "label_nc": 3,
        }
    ],
]


class TestSPADEDiffusionModelUNet2D(unittest.TestCase):
    @parameterized.expand(UNCOND_CASES_2D)
    @skipUnless(has_einops, "Requires einops")
    def test_shape_unconditioned_models(self, input_param):
        net = SPADEDiffusionModelUNet(**input_param)
        with eval_mode(net):
            result = net.forward(
                torch.rand((1, 1, 16, 16)),
                torch.randint(0, 1000, (1,)).long(),
                torch.rand((1, input_param["label_nc"], 16, 16)),
            )
            self.assertEqual(result.shape, (1, 1, 16, 16))

    @skipUnless(has_einops, "Requires einops")
    def test_timestep_with_wrong_shape(self):
        net = SPADEDiffusionModelUNet(
            spatial_dims=2,
            label_nc=3,
            in_channels=1,
            out_channels=1,
            num_res_blocks=1,
            channels=(8, 8, 8),
            attention_levels=(False, False, False),
            norm_num_groups=8,
        )
        with self.assertRaises(ValueError):
            with eval_mode(net):
                net.forward(
                    torch.rand((1, 1, 16, 16)), torch.randint(0, 1000, (1, 1)).long(), torch.rand((1, 3, 16, 16))
                )

    @skipUnless(has_einops, "Requires einops")
    def test_label_with_wrong_shape(self):
        net = SPADEDiffusionModelUNet(
            spatial_dims=2,
            label_nc=3,
            in_channels=1,
            out_channels=1,
            num_res_blocks=1,
            channels=(8, 8, 8),
            attention_levels=(False, False, False),
            norm_num_groups=8,
        )
        with self.assertRaises(RuntimeError):
            with eval_mode(net):
                net.forward(torch.rand((1, 1, 16, 16)), torch.randint(0, 1000, (1,)).long(), torch.rand((1, 6, 16, 16)))

    @skipUnless(has_einops, "Requires einops")
    def test_shape_with_different_in_channel_out_channel(self):
        in_channels = 6
        out_channels = 3
        net = SPADEDiffusionModelUNet(
            spatial_dims=2,
            label_nc=3,
            in_channels=in_channels,
            out_channels=out_channels,
            num_res_blocks=1,
            channels=(8, 8, 8),
            attention_levels=(False, False, False),
            norm_num_groups=8,
        )
        with eval_mode(net):
            result = net.forward(
                torch.rand((1, in_channels, 16, 16)), torch.randint(0, 1000, (1,)).long(), torch.rand((1, 3, 16, 16))
            )
            self.assertEqual(result.shape, (1, out_channels, 16, 16))

    def test_model_channels_not_multiple_of_norm_num_group(self):
        with self.assertRaises(ValueError):
            SPADEDiffusionModelUNet(
                spatial_dims=2,
                label_nc=3,
                in_channels=1,
                out_channels=1,
                num_res_blocks=1,
                channels=(8, 8, 12),
                attention_levels=(False, False, False),
                norm_num_groups=8,
            )

    def test_attention_levels_with_different_length_num_head_channels(self):
        with self.assertRaises(ValueError):
            SPADEDiffusionModelUNet(
                spatial_dims=2,
                label_nc=3,
                in_channels=1,
                out_channels=1,
                num_res_blocks=1,
                channels=(8, 8, 8),
                attention_levels=(False, False, False),
                num_head_channels=(0, 2),
                norm_num_groups=8,
            )

    def test_num_res_blocks_with_different_length_channels(self):
        with self.assertRaises(ValueError):
            SPADEDiffusionModelUNet(
                spatial_dims=2,
                label_nc=3,
                in_channels=1,
                out_channels=1,
                num_res_blocks=(1, 1),
                channels=(8, 8, 8),
                attention_levels=(False, False, False),
                norm_num_groups=8,
            )

    @skipUnless(has_einops, "Requires einops")
    def test_shape_conditioned_models(self):
        net = SPADEDiffusionModelUNet(
            spatial_dims=2,
            label_nc=3,
            in_channels=1,
            out_channels=1,
            num_res_blocks=1,
            channels=(8, 8, 8),
            attention_levels=(False, False, True),
            with_conditioning=True,
            transformer_num_layers=1,
            cross_attention_dim=3,
            norm_num_groups=8,
            num_head_channels=8,
        )
        with eval_mode(net):
            result = net.forward(
                x=torch.rand((1, 1, 16, 32)),
                timesteps=torch.randint(0, 1000, (1,)).long(),
                seg=torch.rand((1, 3, 16, 32)),
                context=torch.rand((1, 1, 3)),
            )
            self.assertEqual(result.shape, (1, 1, 16, 32))

    @skipUnless(has_einops, "Requires einops")
    def test_with_conditioning_cross_attention_dim_none(self):
        with self.assertRaises(ValueError):
            SPADEDiffusionModelUNet(
                spatial_dims=2,
                label_nc=3,
                in_channels=1,
                out_channels=1,
                num_res_blocks=1,
                channels=(8, 8, 8),
                attention_levels=(False, False, True),
                with_conditioning=True,
                transformer_num_layers=1,
                cross_attention_dim=None,
                norm_num_groups=8,
            )

    @skipUnless(has_einops, "Requires einops")
    def test_context_with_conditioning_none(self):
        net = SPADEDiffusionModelUNet(
            spatial_dims=2,
            label_nc=3,
            in_channels=1,
            out_channels=1,
            num_res_blocks=1,
            channels=(8, 8, 8),
            attention_levels=(False, False, True),
            with_conditioning=False,
            transformer_num_layers=1,
            norm_num_groups=8,
        )

        with self.assertRaises(ValueError):
            with eval_mode(net):
                net.forward(
                    x=torch.rand((1, 1, 16, 32)),
                    timesteps=torch.randint(0, 1000, (1,)).long(),
                    seg=torch.rand((1, 3, 16, 32)),
                    context=torch.rand((1, 1, 3)),
                )

    @skipUnless(has_einops, "Requires einops")
    def test_shape_conditioned_models_class_conditioning(self):
        net = SPADEDiffusionModelUNet(
            spatial_dims=2,
            label_nc=3,
            in_channels=1,
            out_channels=1,
            num_res_blocks=1,
            channels=(8, 8, 8),
            attention_levels=(False, False, True),
            norm_num_groups=8,
            num_head_channels=8,
            num_class_embeds=2,
        )
        with eval_mode(net):
            result = net.forward(
                x=torch.rand((1, 1, 16, 32)),
                timesteps=torch.randint(0, 1000, (1,)).long(),
                seg=torch.rand((1, 3, 16, 32)),
                class_labels=torch.randint(0, 2, (1,)).long(),
            )
            self.assertEqual(result.shape, (1, 1, 16, 32))

    @skipUnless(has_einops, "Requires einops")
    def test_conditioned_models_no_class_labels(self):
        net = SPADEDiffusionModelUNet(
            spatial_dims=2,
            label_nc=3,
            in_channels=1,
            out_channels=1,
            num_res_blocks=1,
            channels=(8, 8, 8),
            attention_levels=(False, False, True),
            norm_num_groups=8,
            num_head_channels=8,
            num_class_embeds=2,
        )

        with self.assertRaises(ValueError):
            net.forward(
                x=torch.rand((1, 1, 16, 32)),
                timesteps=torch.randint(0, 1000, (1,)).long(),
                seg=torch.rand((1, 3, 16, 32)),
            )

    def test_model_channels_not_same_size_of_attention_levels(self):
        with self.assertRaises(ValueError):
            SPADEDiffusionModelUNet(
                spatial_dims=2,
                label_nc=3,
                in_channels=1,
                out_channels=1,
                num_res_blocks=1,
                channels=(8, 8, 8),
                attention_levels=(False, False),
                norm_num_groups=8,
                num_head_channels=8,
                num_class_embeds=2,
            )

    @parameterized.expand(COND_CASES_2D)
    @skipUnless(has_einops, "Requires einops")
    def test_conditioned_2d_models_shape(self, input_param):
        net = SPADEDiffusionModelUNet(**input_param)
        with eval_mode(net):
            result = net.forward(
                torch.rand((1, 1, 16, 16)),
                torch.randint(0, 1000, (1,)).long(),
                torch.rand((1, input_param["label_nc"], 16, 16)),
                torch.rand((1, 1, 3)),
            )
            self.assertEqual(result.shape, (1, 1, 16, 16))


class TestDiffusionModelUNet3D(unittest.TestCase):
    @parameterized.expand(UNCOND_CASES_3D)
    @skipUnless(has_einops, "Requires einops")
    def test_shape_unconditioned_models(self, input_param):
        net = SPADEDiffusionModelUNet(**input_param)
        with eval_mode(net):
            result = net.forward(
                torch.rand((1, 1, 16, 16, 16)),
                torch.randint(0, 1000, (1,)).long(),
                torch.rand((1, input_param["label_nc"], 16, 16, 16)),
            )
            self.assertEqual(result.shape, (1, 1, 16, 16, 16))

    @skipUnless(has_einops, "Requires einops")
    def test_shape_with_different_in_channel_out_channel(self):
        in_channels = 6
        out_channels = 3
        net = SPADEDiffusionModelUNet(
            spatial_dims=3,
            label_nc=3,
            in_channels=in_channels,
            out_channels=out_channels,
            num_res_blocks=1,
            channels=(8, 8, 8),
            attention_levels=(False, False, True),
            norm_num_groups=4,
        )
        with eval_mode(net):
            result = net.forward(
                torch.rand((1, in_channels, 16, 16, 16)),
                torch.randint(0, 1000, (1,)).long(),
                torch.rand((1, 3, 16, 16, 16)),
            )
            self.assertEqual(result.shape, (1, out_channels, 16, 16, 16))

    @skipUnless(has_einops, "Requires einops")
    def test_shape_conditioned_models(self):
        net = SPADEDiffusionModelUNet(
            spatial_dims=3,
            label_nc=3,
            in_channels=1,
            out_channels=1,
            num_res_blocks=1,
            channels=(16, 16, 16),
            attention_levels=(False, False, True),
            norm_num_groups=16,
            with_conditioning=True,
            transformer_num_layers=1,
            cross_attention_dim=3,
        )
        with eval_mode(net):
            result = net.forward(
                x=torch.rand((1, 1, 16, 16, 16)),
                timesteps=torch.randint(0, 1000, (1,)).long(),
                seg=torch.rand((1, 3, 16, 16, 16)),
                context=torch.rand((1, 1, 3)),
            )
            self.assertEqual(result.shape, (1, 1, 16, 16, 16))


if __name__ == "__main__":
    unittest.main()
