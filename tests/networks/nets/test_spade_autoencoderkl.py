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
from monai.networks.nets import SPADEAutoencoderKL
from monai.utils import optional_import

einops, has_einops = optional_import("einops")

CASES_NO_ATTENTION = [
    [
        {
            "spatial_dims": 2,
            "label_nc": 3,
            "in_channels": 1,
            "out_channels": 1,
            "channels": (4, 4, 4),
            "latent_channels": 4,
            "attention_levels": (False, False, False),
            "num_res_blocks": 1,
            "norm_num_groups": 4,
            "with_encoder_nonlocal_attn": False,
            "with_decoder_nonlocal_attn": False,
        },
        (1, 1, 16, 16),
        (1, 3, 16, 16),
        (1, 1, 16, 16),
        (1, 4, 4, 4),
    ],
    [
        {
            "spatial_dims": 3,
            "label_nc": 3,
            "in_channels": 1,
            "out_channels": 1,
            "channels": (4, 4, 4),
            "latent_channels": 4,
            "attention_levels": (False, False, False),
            "num_res_blocks": 1,
            "norm_num_groups": 4,
            "with_encoder_nonlocal_attn": False,
            "with_decoder_nonlocal_attn": False,
        },
        (1, 1, 16, 16, 16),
        (1, 3, 16, 16, 16),
        (1, 1, 16, 16, 16),
        (1, 4, 4, 4, 4),
    ],
]

CASES_ATTENTION = [
    [
        {
            "spatial_dims": 2,
            "label_nc": 3,
            "in_channels": 1,
            "out_channels": 1,
            "channels": (4, 4, 4),
            "latent_channels": 4,
            "attention_levels": (False, False, False),
            "num_res_blocks": 1,
            "norm_num_groups": 4,
        },
        (1, 1, 16, 16),
        (1, 3, 16, 16),
        (1, 1, 16, 16),
        (1, 4, 4, 4),
    ],
    [
        {
            "spatial_dims": 2,
            "label_nc": 3,
            "in_channels": 1,
            "out_channels": 1,
            "channels": (4, 4, 4),
            "latent_channels": 4,
            "attention_levels": (False, False, False),
            "num_res_blocks": (1, 1, 2),
            "norm_num_groups": 4,
        },
        (1, 1, 16, 16),
        (1, 3, 16, 16),
        (1, 1, 16, 16),
        (1, 4, 4, 4),
    ],
    [
        {
            "spatial_dims": 2,
            "label_nc": 3,
            "in_channels": 1,
            "out_channels": 1,
            "channels": (4, 4, 4),
            "latent_channels": 4,
            "attention_levels": (False, False, False),
            "num_res_blocks": 1,
            "norm_num_groups": 4,
        },
        (1, 1, 16, 16),
        (1, 3, 16, 16),
        (1, 1, 16, 16),
        (1, 4, 4, 4),
    ],
    [
        {
            "spatial_dims": 2,
            "label_nc": 3,
            "in_channels": 1,
            "out_channels": 1,
            "channels": (4, 4, 4),
            "latent_channels": 4,
            "attention_levels": (False, False, True),
            "num_res_blocks": 1,
            "norm_num_groups": 4,
        },
        (1, 1, 16, 16),
        (1, 3, 16, 16),
        (1, 1, 16, 16),
        (1, 4, 4, 4),
    ],
    [
        {
            "spatial_dims": 2,
            "label_nc": 3,
            "in_channels": 1,
            "out_channels": 1,
            "channels": (4, 4, 4),
            "latent_channels": 4,
            "attention_levels": (False, False, False),
            "num_res_blocks": 1,
            "norm_num_groups": 4,
            "with_encoder_nonlocal_attn": False,
        },
        (1, 1, 16, 16),
        (1, 3, 16, 16),
        (1, 1, 16, 16),
        (1, 4, 4, 4),
    ],
    [
        {
            "spatial_dims": 3,
            "label_nc": 3,
            "in_channels": 1,
            "out_channels": 1,
            "channels": (4, 4, 4),
            "latent_channels": 4,
            "attention_levels": (False, False, True),
            "num_res_blocks": 1,
            "norm_num_groups": 4,
        },
        (1, 1, 16, 16, 16),
        (1, 3, 16, 16, 16),
        (1, 1, 16, 16, 16),
        (1, 4, 4, 4, 4),
    ],
    [
        {
            "spatial_dims": 2,
            "label_nc": 3,
            "in_channels": 1,
            "out_channels": 1,
            "channels": (4, 4, 4),
            "latent_channels": 4,
            "attention_levels": (False, False, True),
            "num_res_blocks": 1,
            "norm_num_groups": 4,
            "spade_intermediate_channels": 32,
        },
        (1, 1, 16, 16),
        (1, 3, 16, 16),
        (1, 1, 16, 16),
        (1, 4, 4, 4),
    ],
]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if has_einops:
    CASES = CASES_ATTENTION + CASES_NO_ATTENTION
else:
    CASES = CASES_NO_ATTENTION


class TestSPADEAutoEncoderKL(unittest.TestCase):
    @parameterized.expand(CASES)
    def test_shape(self, input_param, input_shape, input_seg, expected_shape, expected_latent_shape):
        net = SPADEAutoencoderKL(**input_param).to(device)
        with eval_mode(net):
            result = net.forward(torch.randn(input_shape).to(device), torch.randn(input_seg).to(device))
            self.assertEqual(result[0].shape, expected_shape)
            self.assertEqual(result[1].shape, expected_latent_shape)

    @skipUnless(has_einops, "Requires einops")
    def test_model_channels_not_multiple_of_norm_num_group(self):
        with self.assertRaises(ValueError):
            SPADEAutoencoderKL(
                spatial_dims=2,
                label_nc=3,
                in_channels=1,
                out_channels=1,
                channels=(24, 24, 24),
                attention_levels=(False, False, False),
                latent_channels=8,
                num_res_blocks=1,
                norm_num_groups=16,
            )

    @skipUnless(has_einops, "Requires einops")
    def test_model_channels_not_same_size_of_attention_levels(self):
        with self.assertRaises(ValueError):
            SPADEAutoencoderKL(
                spatial_dims=2,
                label_nc=3,
                in_channels=1,
                out_channels=1,
                channels=(24, 24, 24),
                attention_levels=(False, False),
                latent_channels=8,
                num_res_blocks=1,
                norm_num_groups=16,
            )

    @skipUnless(has_einops, "Requires einops")
    def test_model_channels_not_same_size_of_num_res_blocks(self):
        with self.assertRaises(ValueError):
            SPADEAutoencoderKL(
                spatial_dims=2,
                label_nc=3,
                in_channels=1,
                out_channels=1,
                channels=(24, 24, 24),
                attention_levels=(False, False, False),
                latent_channels=8,
                num_res_blocks=(8, 8),
                norm_num_groups=16,
            )

    def test_shape_encode(self):
        input_param, input_shape, _, _, expected_latent_shape = CASES[0]
        net = SPADEAutoencoderKL(**input_param).to(device)
        with eval_mode(net):
            result = net.encode(torch.randn(input_shape).to(device))
            self.assertEqual(result[0].shape, expected_latent_shape)
            self.assertEqual(result[1].shape, expected_latent_shape)

    def test_shape_sampling(self):
        input_param, _, _, _, expected_latent_shape = CASES[0]
        net = SPADEAutoencoderKL(**input_param).to(device)
        with eval_mode(net):
            result = net.sampling(
                torch.randn(expected_latent_shape).to(device), torch.randn(expected_latent_shape).to(device)
            )
            self.assertEqual(result.shape, expected_latent_shape)

    def test_shape_decode(self):
        input_param, _, input_seg_shape, expected_input_shape, latent_shape = CASES[0]
        net = SPADEAutoencoderKL(**input_param).to(device)
        with eval_mode(net):
            result = net.decode(torch.randn(latent_shape).to(device), torch.randn(input_seg_shape).to(device))
            self.assertEqual(result.shape, expected_input_shape)

    @skipUnless(has_einops, "Requires einops")
    def test_wrong_shape_decode(self):
        net = SPADEAutoencoderKL(
            spatial_dims=2,
            label_nc=3,
            in_channels=1,
            out_channels=1,
            channels=(4, 4, 4),
            latent_channels=4,
            attention_levels=(False, False, False),
            num_res_blocks=1,
            norm_num_groups=4,
        )
        with self.assertRaises(RuntimeError):
            _ = net.decode(torch.randn((1, 1, 16, 16)).to(device), torch.randn((1, 6, 16, 16)).to(device))


if __name__ == "__main__":
    unittest.main()
