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

from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi
from monai.networks import eval_mode
from monai.utils import optional_import
from tests.util import SkipIfBeforePyTorchVersion

tqdm, has_tqdm = optional_import("tqdm", name="tqdm")
_, has_einops = optional_import("einops")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CASES_NO_ATTENTION = [
    [
        {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 1,
            "num_channels": (4, 4, 4),
            "latent_channels": 4,
            "attention_levels": (False, False, False),
            "num_res_blocks": (1, 1, 1),
            "norm_num_groups": 4,
            "with_encoder_nonlocal_attn": False,
            "with_decoder_nonlocal_attn": False,
            "num_splits": 2,
            "print_info": False,
        },
        (1, 1, 32, 32, 32),
        (1, 1, 32, 32, 32),
        (1, 4, 8, 8, 8),
    ]
]

CASES_ATTENTION = [
    [
        {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 1,
            "num_channels": (4, 4, 4),
            "latent_channels": 4,
            "attention_levels": (False, False, True),
            "num_res_blocks": (1, 1, 1),
            "norm_num_groups": 4,
            "with_encoder_nonlocal_attn": True,
            "with_decoder_nonlocal_attn": True,
            "num_splits": 2,
            "print_info": False,
        },
        (1, 1, 32, 32, 32),
        (1, 1, 32, 32, 32),
        (1, 4, 8, 8, 8),
    ]
]

if has_einops:
    CASES = CASES_NO_ATTENTION + CASES_ATTENTION
else:
    CASES = CASES_NO_ATTENTION


class TestAutoencoderKlMaisi(unittest.TestCase):
    @parameterized.expand(CASES)
    def test_shape(self, input_param, input_shape, expected_shape, expected_latent_shape):
        net = AutoencoderKlMaisi(**input_param).to(device)
        with eval_mode(net):
            result = net.forward(torch.randn(input_shape).to(device))
            self.assertEqual(result[0].shape, expected_shape)
            self.assertEqual(result[1].shape, expected_latent_shape)
            self.assertEqual(result[2].shape, expected_latent_shape)

    @parameterized.expand(CASES)
    @SkipIfBeforePyTorchVersion((1, 11))
    def test_shape_with_convtranspose_and_checkpointing(
        self, input_param, input_shape, expected_shape, expected_latent_shape
    ):
        input_param = input_param.copy()
        input_param.update({"use_checkpointing": True, "use_convtranspose": True})
        net = AutoencoderKlMaisi(**input_param).to(device)
        with eval_mode(net):
            result = net.forward(torch.randn(input_shape).to(device))
            self.assertEqual(result[0].shape, expected_shape)
            self.assertEqual(result[1].shape, expected_latent_shape)
            self.assertEqual(result[2].shape, expected_latent_shape)

    def test_model_channels_not_multiple_of_norm_num_group(self):
        with self.assertRaises(ValueError):
            AutoencoderKlMaisi(
                spatial_dims=3,
                in_channels=1,
                out_channels=1,
                num_channels=(24, 24, 24),
                attention_levels=(False, False, False),
                latent_channels=8,
                num_res_blocks=(1, 1, 1),
                norm_num_groups=16,
                num_splits=2,
                print_info=False,
            )

    def test_model_num_channels_not_same_size_of_attention_levels(self):
        with self.assertRaises(ValueError):
            AutoencoderKlMaisi(
                spatial_dims=3,
                in_channels=1,
                out_channels=1,
                num_channels=(24, 24, 24),
                attention_levels=(False, False),
                latent_channels=8,
                num_res_blocks=(1, 1, 1),
                norm_num_groups=16,
                num_splits=2,
                print_info=False,
            )

    def test_model_num_channels_not_same_size_of_num_res_blocks(self):
        with self.assertRaises(ValueError):
            AutoencoderKlMaisi(
                spatial_dims=3,
                in_channels=1,
                out_channels=1,
                num_channels=(24, 24),
                attention_levels=(False, False, False),
                latent_channels=8,
                num_res_blocks=(8, 8, 8),
                norm_num_groups=16,
                num_splits=2,
                print_info=False,
            )

    def test_shape_reconstruction(self):
        input_param, input_shape, expected_shape, _ = CASES[0]
        net = AutoencoderKlMaisi(**input_param).to(device)
        with eval_mode(net):
            result = net.reconstruct(torch.randn(input_shape).to(device))
            self.assertEqual(result.shape, expected_shape)

    @SkipIfBeforePyTorchVersion((1, 11))
    def test_shape_reconstruction_with_convtranspose_and_checkpointing(self):
        input_param, input_shape, expected_shape, _ = CASES[0]
        input_param = input_param.copy()
        input_param.update({"use_checkpointing": True, "use_convtranspose": True})
        net = AutoencoderKlMaisi(**input_param).to(device)
        with eval_mode(net):
            result = net.reconstruct(torch.randn(input_shape).to(device))
            self.assertEqual(result.shape, expected_shape)

    def test_shape_encode(self):
        input_param, input_shape, _, expected_latent_shape = CASES[0]
        net = AutoencoderKlMaisi(**input_param).to(device)
        with eval_mode(net):
            result = net.encode(torch.randn(input_shape).to(device))
            self.assertEqual(result[0].shape, expected_latent_shape)
            self.assertEqual(result[1].shape, expected_latent_shape)

    @SkipIfBeforePyTorchVersion((1, 11))
    def test_shape_encode_with_convtranspose_and_checkpointing(self):
        input_param, input_shape, _, expected_latent_shape = CASES[0]
        input_param = input_param.copy()
        input_param.update({"use_checkpointing": True, "use_convtranspose": True})
        net = AutoencoderKlMaisi(**input_param).to(device)
        with eval_mode(net):
            result = net.encode(torch.randn(input_shape).to(device))
            self.assertEqual(result[0].shape, expected_latent_shape)
            self.assertEqual(result[1].shape, expected_latent_shape)

    def test_shape_sampling(self):
        input_param, _, _, expected_latent_shape = CASES[0]
        net = AutoencoderKlMaisi(**input_param).to(device)
        with eval_mode(net):
            result = net.sampling(
                torch.randn(expected_latent_shape).to(device), torch.randn(expected_latent_shape).to(device)
            )
            self.assertEqual(result.shape, expected_latent_shape)

    @SkipIfBeforePyTorchVersion((1, 11))
    def test_shape_sampling_convtranspose_and_checkpointing(self):
        input_param, _, _, expected_latent_shape = CASES[0]
        input_param = input_param.copy()
        input_param.update({"use_checkpointing": True, "use_convtranspose": True})
        net = AutoencoderKlMaisi(**input_param).to(device)
        with eval_mode(net):
            result = net.sampling(
                torch.randn(expected_latent_shape).to(device), torch.randn(expected_latent_shape).to(device)
            )
            self.assertEqual(result.shape, expected_latent_shape)

    def test_shape_decode(self):
        input_param, expected_input_shape, _, latent_shape = CASES[0]
        net = AutoencoderKlMaisi(**input_param).to(device)
        with eval_mode(net):
            result = net.decode(torch.randn(latent_shape).to(device))
            self.assertEqual(result.shape, expected_input_shape)

    @SkipIfBeforePyTorchVersion((1, 11))
    def test_shape_decode_convtranspose_and_checkpointing(self):
        input_param, expected_input_shape, _, latent_shape = CASES[0]
        input_param = input_param.copy()
        input_param.update({"use_checkpointing": True, "use_convtranspose": True})
        net = AutoencoderKlMaisi(**input_param).to(device)
        with eval_mode(net):
            result = net.decode(torch.randn(latent_shape).to(device))
            self.assertEqual(result.shape, expected_input_shape)


if __name__ == "__main__":
    unittest.main()
