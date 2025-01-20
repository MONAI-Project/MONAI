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

from monai.inferers import VQVAETransformerInferer
from monai.networks.nets import VQVAE, DecoderOnlyTransformer
from monai.utils import optional_import
from monai.utils.ordering import Ordering, OrderingType

einops, has_einops = optional_import("einops")
TEST_CASES = [
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "channels": (8, 8),
            "num_res_channels": (8, 8),
            "downsample_parameters": ((2, 4, 1, 1),) * 2,
            "upsample_parameters": ((2, 4, 1, 1, 0),) * 2,
            "num_res_layers": 1,
            "num_embeddings": 16,
            "embedding_dim": 8,
        },
        {
            "num_tokens": 16 + 1,
            "max_seq_len": 4,
            "attn_layers_dim": 4,
            "attn_layers_depth": 2,
            "attn_layers_heads": 1,
            "with_cross_attention": False,
        },
        {"ordering_type": OrderingType.RASTER_SCAN.value, "spatial_dims": 2, "dimensions": (2, 2, 2)},
        (2, 1, 8, 8),
        (2, 4, 17),
        (2, 2, 2),
    ],
    [
        {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 1,
            "channels": (8, 8),
            "num_res_channels": (8, 8),
            "downsample_parameters": ((2, 4, 1, 1),) * 2,
            "upsample_parameters": ((2, 4, 1, 1, 0),) * 2,
            "num_res_layers": 1,
            "num_embeddings": 16,
            "embedding_dim": 8,
        },
        {
            "num_tokens": 16 + 1,
            "max_seq_len": 8,
            "attn_layers_dim": 4,
            "attn_layers_depth": 2,
            "attn_layers_heads": 1,
            "with_cross_attention": False,
        },
        {"ordering_type": OrderingType.RASTER_SCAN.value, "spatial_dims": 3, "dimensions": (2, 2, 2, 2)},
        (2, 1, 8, 8, 8),
        (2, 8, 17),
        (2, 2, 2, 2),
    ],
]


class TestVQVAETransformerInferer(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    @skipUnless(has_einops, "Requires einops")
    def test_prediction_shape(
        self, stage_1_params, stage_2_params, ordering_params, input_shape, logits_shape, latent_shape
    ):
        stage_1 = VQVAE(**stage_1_params)
        stage_2 = DecoderOnlyTransformer(**stage_2_params)
        ordering = Ordering(**ordering_params)

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        stage_1.to(device)
        stage_2.to(device)
        stage_1.eval()
        stage_2.eval()

        input = torch.randn(input_shape).to(device)

        inferer = VQVAETransformerInferer()
        prediction = inferer(inputs=input, vqvae_model=stage_1, transformer_model=stage_2, ordering=ordering)
        self.assertEqual(prediction.shape, logits_shape)

    @parameterized.expand(TEST_CASES)
    @skipUnless(has_einops, "Requires einops")
    def test_prediction_shape_shorter_sequence(
        self, stage_1_params, stage_2_params, ordering_params, input_shape, logits_shape, latent_shape
    ):
        stage_1 = VQVAE(**stage_1_params)
        max_seq_len = 3
        stage_2_params_shorter = dict(stage_2_params)
        stage_2_params_shorter["max_seq_len"] = max_seq_len
        stage_2 = DecoderOnlyTransformer(**stage_2_params_shorter)
        ordering = Ordering(**ordering_params)

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        stage_1.to(device)
        stage_2.to(device)
        stage_1.eval()
        stage_2.eval()

        input = torch.randn(input_shape).to(device)

        inferer = VQVAETransformerInferer()
        prediction = inferer(inputs=input, vqvae_model=stage_1, transformer_model=stage_2, ordering=ordering)
        cropped_logits_shape = (logits_shape[0], max_seq_len, logits_shape[2])
        self.assertEqual(prediction.shape, cropped_logits_shape)

    @skipUnless(has_einops, "Requires einops")
    def test_sample(self):

        stage_1 = VQVAE(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(8, 8),
            num_res_channels=(8, 8),
            downsample_parameters=((2, 4, 1, 1),) * 2,
            upsample_parameters=((2, 4, 1, 1, 0),) * 2,
            num_res_layers=1,
            num_embeddings=16,
            embedding_dim=8,
        )
        stage_2 = DecoderOnlyTransformer(
            num_tokens=16 + 1,
            max_seq_len=4,
            attn_layers_dim=4,
            attn_layers_depth=2,
            attn_layers_heads=1,
            with_cross_attention=False,
        )
        ordering = Ordering(ordering_type=OrderingType.RASTER_SCAN.value, spatial_dims=2, dimensions=(2, 2, 2))

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        stage_1.to(device)
        stage_2.to(device)
        stage_1.eval()
        stage_2.eval()

        inferer = VQVAETransformerInferer()

        starting_token = 16  # from stage_1 num_embeddings

        sample = inferer.sample(
            latent_spatial_dim=(2, 2),
            starting_tokens=starting_token * torch.ones((2, 1), device=device),
            vqvae_model=stage_1,
            transformer_model=stage_2,
            ordering=ordering,
        )
        self.assertEqual(sample.shape, (2, 1, 8, 8))

    @skipUnless(has_einops, "Requires einops")
    def test_sample_shorter_sequence(self):
        stage_1 = VQVAE(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(8, 8),
            num_res_channels=(8, 8),
            downsample_parameters=((2, 4, 1, 1),) * 2,
            upsample_parameters=((2, 4, 1, 1, 0),) * 2,
            num_res_layers=1,
            num_embeddings=16,
            embedding_dim=8,
        )
        stage_2 = DecoderOnlyTransformer(
            num_tokens=16 + 1,
            max_seq_len=2,
            attn_layers_dim=4,
            attn_layers_depth=2,
            attn_layers_heads=1,
            with_cross_attention=False,
        )
        ordering = Ordering(ordering_type=OrderingType.RASTER_SCAN.value, spatial_dims=2, dimensions=(2, 2, 2))

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        stage_1.to(device)
        stage_2.to(device)
        stage_1.eval()
        stage_2.eval()

        inferer = VQVAETransformerInferer()

        starting_token = 16  # from stage_1 num_embeddings

        sample = inferer.sample(
            latent_spatial_dim=(2, 2),
            starting_tokens=starting_token * torch.ones((2, 1), device=device),
            vqvae_model=stage_1,
            transformer_model=stage_2,
            ordering=ordering,
        )
        self.assertEqual(sample.shape, (2, 1, 8, 8))

    @parameterized.expand(TEST_CASES)
    @skipUnless(has_einops, "Requires einops")
    def test_get_likelihood(
        self, stage_1_params, stage_2_params, ordering_params, input_shape, logits_shape, latent_shape
    ):
        stage_1 = VQVAE(**stage_1_params)
        stage_2 = DecoderOnlyTransformer(**stage_2_params)
        ordering = Ordering(**ordering_params)

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        stage_1.to(device)
        stage_2.to(device)
        stage_1.eval()
        stage_2.eval()

        input = torch.randn(input_shape).to(device)

        inferer = VQVAETransformerInferer()
        likelihood = inferer.get_likelihood(
            inputs=input, vqvae_model=stage_1, transformer_model=stage_2, ordering=ordering
        )
        self.assertEqual(likelihood.shape, latent_shape)

    @parameterized.expand(TEST_CASES)
    @skipUnless(has_einops, "Requires einops")
    def test_get_likelihood_shorter_sequence(
        self, stage_1_params, stage_2_params, ordering_params, input_shape, logits_shape, latent_shape
    ):
        stage_1 = VQVAE(**stage_1_params)
        max_seq_len = 3
        stage_2_params_shorter = dict(stage_2_params)
        stage_2_params_shorter["max_seq_len"] = max_seq_len
        stage_2 = DecoderOnlyTransformer(**stage_2_params_shorter)
        ordering = Ordering(**ordering_params)

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        stage_1.to(device)
        stage_2.to(device)
        stage_1.eval()
        stage_2.eval()

        input = torch.randn(input_shape).to(device)

        inferer = VQVAETransformerInferer()
        likelihood = inferer.get_likelihood(
            inputs=input, vqvae_model=stage_1, transformer_model=stage_2, ordering=ordering
        )
        self.assertEqual(likelihood.shape, latent_shape)

    @parameterized.expand(TEST_CASES)
    @skipUnless(has_einops, "Requires einops")
    def test_get_likelihood_resampling(
        self, stage_1_params, stage_2_params, ordering_params, input_shape, logits_shape, latent_shape
    ):
        stage_1 = VQVAE(**stage_1_params)
        stage_2 = DecoderOnlyTransformer(**stage_2_params)
        ordering = Ordering(**ordering_params)

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        stage_1.to(device)
        stage_2.to(device)
        stage_1.eval()
        stage_2.eval()

        input = torch.randn(input_shape).to(device)

        inferer = VQVAETransformerInferer()
        likelihood = inferer.get_likelihood(
            inputs=input,
            vqvae_model=stage_1,
            transformer_model=stage_2,
            ordering=ordering,
            resample_latent_likelihoods=True,
            resample_interpolation_mode="nearest",
        )
        self.assertEqual(likelihood.shape, input_shape)


if __name__ == "__main__":
    unittest.main()
