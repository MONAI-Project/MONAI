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

from monai.inferers import ControlNetDiffusionInferer, ControlNetLatentDiffusionInferer
from monai.networks.nets import (
    VQVAE,
    AutoencoderKL,
    ControlNet,
    DiffusionModelUNet,
    SPADEAutoencoderKL,
    SPADEDiffusionModelUNet,
)
from monai.networks.schedulers import DDIMScheduler, DDPMScheduler
from monai.utils import optional_import

_, has_scipy = optional_import("scipy")
_, has_einops = optional_import("einops")


CNDM_TEST_CASES = [
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "channels": [8],
            "norm_num_groups": 8,
            "attention_levels": [True],
            "num_res_blocks": 1,
            "num_head_channels": 8,
        },
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "channels": [8],
            "attention_levels": [True],
            "norm_num_groups": 8,
            "num_res_blocks": 1,
            "num_head_channels": 8,
            "conditioning_embedding_num_channels": [16],
            "conditioning_embedding_in_channels": 1,
        },
        (2, 1, 8, 8),
    ],
    [
        {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 1,
            "channels": [8],
            "norm_num_groups": 8,
            "attention_levels": [True],
            "num_res_blocks": 1,
            "num_head_channels": 8,
        },
        {
            "spatial_dims": 3,
            "in_channels": 1,
            "channels": [8],
            "attention_levels": [True],
            "num_res_blocks": 1,
            "norm_num_groups": 8,
            "num_head_channels": 8,
            "conditioning_embedding_num_channels": [16],
            "conditioning_embedding_in_channels": 1,
        },
        (2, 1, 8, 8, 8),
    ],
]
LATENT_CNDM_TEST_CASES = [
    [
        "AutoencoderKL",
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "channels": (4, 4),
            "latent_channels": 3,
            "attention_levels": [False, False],
            "num_res_blocks": 1,
            "with_encoder_nonlocal_attn": False,
            "with_decoder_nonlocal_attn": False,
            "norm_num_groups": 4,
        },
        "DiffusionModelUNet",
        {
            "spatial_dims": 2,
            "in_channels": 3,
            "out_channels": 3,
            "channels": [4, 4],
            "norm_num_groups": 4,
            "attention_levels": [False, False],
            "num_res_blocks": 1,
            "num_head_channels": 4,
        },
        {
            "spatial_dims": 2,
            "in_channels": 3,
            "channels": [4, 4],
            "attention_levels": [False, False],
            "num_res_blocks": 1,
            "norm_num_groups": 4,
            "num_head_channels": 4,
            "conditioning_embedding_num_channels": [16],
            "conditioning_embedding_in_channels": 1,
        },
        (1, 1, 8, 8),
        (1, 3, 4, 4),
    ],
    [
        "VQVAE",
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "channels": [4, 4],
            "num_res_layers": 1,
            "num_res_channels": [4, 4],
            "downsample_parameters": ((2, 4, 1, 1), (2, 4, 1, 1)),
            "upsample_parameters": ((2, 4, 1, 1, 0), (2, 4, 1, 1, 0)),
            "num_embeddings": 16,
            "embedding_dim": 3,
        },
        "DiffusionModelUNet",
        {
            "spatial_dims": 2,
            "in_channels": 3,
            "out_channels": 3,
            "channels": [8, 8],
            "norm_num_groups": 8,
            "attention_levels": [False, False],
            "num_res_blocks": 1,
            "num_head_channels": 8,
        },
        {
            "spatial_dims": 2,
            "in_channels": 3,
            "channels": [8, 8],
            "attention_levels": [False, False],
            "num_res_blocks": 1,
            "norm_num_groups": 8,
            "num_head_channels": 8,
            "conditioning_embedding_num_channels": [16],
            "conditioning_embedding_in_channels": 1,
        },
        (1, 1, 16, 16),
        (1, 3, 4, 4),
    ],
    [
        "VQVAE",
        {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 1,
            "channels": [4, 4],
            "num_res_layers": 1,
            "num_res_channels": [4, 4],
            "downsample_parameters": ((2, 4, 1, 1), (2, 4, 1, 1)),
            "upsample_parameters": ((2, 4, 1, 1, 0), (2, 4, 1, 1, 0)),
            "num_embeddings": 16,
            "embedding_dim": 3,
        },
        "DiffusionModelUNet",
        {
            "spatial_dims": 3,
            "in_channels": 3,
            "out_channels": 3,
            "channels": [8, 8],
            "norm_num_groups": 8,
            "attention_levels": [False, False],
            "num_res_blocks": 1,
            "num_head_channels": 8,
        },
        {
            "spatial_dims": 3,
            "in_channels": 3,
            "channels": [8, 8],
            "attention_levels": [False, False],
            "num_res_blocks": 1,
            "norm_num_groups": 8,
            "num_head_channels": 8,
            "conditioning_embedding_num_channels": [16],
            "conditioning_embedding_in_channels": 1,
        },
        (1, 1, 16, 16, 16),
        (1, 3, 4, 4, 4),
    ],
]
LATENT_CNDM_TEST_CASES_DIFF_SHAPES = [
    [
        "AutoencoderKL",
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "channels": (4, 4),
            "latent_channels": 3,
            "attention_levels": [False, False],
            "num_res_blocks": 1,
            "with_encoder_nonlocal_attn": False,
            "with_decoder_nonlocal_attn": False,
            "norm_num_groups": 4,
        },
        "DiffusionModelUNet",
        {
            "spatial_dims": 2,
            "in_channels": 3,
            "out_channels": 3,
            "channels": [4, 4],
            "norm_num_groups": 4,
            "attention_levels": [False, False],
            "num_res_blocks": 1,
            "num_head_channels": 4,
        },
        {
            "spatial_dims": 2,
            "in_channels": 3,
            "channels": [4, 4],
            "attention_levels": [False, False],
            "num_res_blocks": 1,
            "norm_num_groups": 4,
            "num_head_channels": 4,
            "conditioning_embedding_num_channels": [16],
            "conditioning_embedding_in_channels": 1,
        },
        (1, 1, 12, 12),
        (1, 3, 8, 8),
    ],
    [
        "VQVAE",
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "channels": [4, 4],
            "num_res_layers": 1,
            "num_res_channels": [4, 4],
            "downsample_parameters": ((2, 4, 1, 1), (2, 4, 1, 1)),
            "upsample_parameters": ((2, 4, 1, 1, 0), (2, 4, 1, 1, 0)),
            "num_embeddings": 16,
            "embedding_dim": 3,
        },
        "DiffusionModelUNet",
        {
            "spatial_dims": 2,
            "in_channels": 3,
            "out_channels": 3,
            "channels": [8, 8],
            "norm_num_groups": 8,
            "attention_levels": [False, False],
            "num_res_blocks": 1,
            "num_head_channels": 8,
        },
        {
            "spatial_dims": 2,
            "in_channels": 3,
            "channels": [8, 8],
            "attention_levels": [False, False],
            "num_res_blocks": 1,
            "norm_num_groups": 8,
            "num_head_channels": 8,
            "conditioning_embedding_num_channels": [16],
            "conditioning_embedding_in_channels": 1,
        },
        (1, 1, 12, 12),
        (1, 3, 8, 8),
    ],
    [
        "VQVAE",
        {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 1,
            "channels": [4, 4],
            "num_res_layers": 1,
            "num_res_channels": [4, 4],
            "downsample_parameters": ((2, 4, 1, 1), (2, 4, 1, 1)),
            "upsample_parameters": ((2, 4, 1, 1, 0), (2, 4, 1, 1, 0)),
            "num_embeddings": 16,
            "embedding_dim": 3,
        },
        "DiffusionModelUNet",
        {
            "spatial_dims": 3,
            "in_channels": 3,
            "out_channels": 3,
            "channels": [8, 8],
            "norm_num_groups": 8,
            "attention_levels": [False, False],
            "num_res_blocks": 1,
            "num_head_channels": 8,
        },
        {
            "spatial_dims": 3,
            "in_channels": 3,
            "channels": [8, 8],
            "attention_levels": [False, False],
            "num_res_blocks": 1,
            "norm_num_groups": 8,
            "num_head_channels": 8,
            "conditioning_embedding_num_channels": [16],
            "conditioning_embedding_in_channels": 1,
        },
        (1, 1, 12, 12, 12),
        (1, 3, 8, 8, 8),
    ],
    [
        "SPADEAutoencoderKL",
        {
            "spatial_dims": 2,
            "label_nc": 3,
            "in_channels": 1,
            "out_channels": 1,
            "channels": (4, 4),
            "latent_channels": 3,
            "attention_levels": [False, False],
            "num_res_blocks": 1,
            "with_encoder_nonlocal_attn": False,
            "with_decoder_nonlocal_attn": False,
            "norm_num_groups": 4,
        },
        "DiffusionModelUNet",
        {
            "spatial_dims": 2,
            "in_channels": 3,
            "out_channels": 3,
            "channels": [4, 4],
            "norm_num_groups": 4,
            "attention_levels": [False, False],
            "num_res_blocks": 1,
            "num_head_channels": 4,
        },
        {
            "spatial_dims": 2,
            "in_channels": 3,
            "channels": [4, 4],
            "attention_levels": [False, False],
            "num_res_blocks": 1,
            "norm_num_groups": 4,
            "num_head_channels": 4,
            "conditioning_embedding_num_channels": [16],
            "conditioning_embedding_in_channels": 1,
        },
        (1, 1, 8, 8),
        (1, 3, 4, 4),
    ],
    [
        "AutoencoderKL",
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "channels": (4, 4),
            "latent_channels": 3,
            "attention_levels": [False, False],
            "num_res_blocks": 1,
            "with_encoder_nonlocal_attn": False,
            "with_decoder_nonlocal_attn": False,
            "norm_num_groups": 4,
        },
        "SPADEDiffusionModelUNet",
        {
            "spatial_dims": 2,
            "label_nc": 3,
            "in_channels": 3,
            "out_channels": 3,
            "channels": [4, 4],
            "norm_num_groups": 4,
            "attention_levels": [False, False],
            "num_res_blocks": 1,
            "num_head_channels": 4,
        },
        {
            "spatial_dims": 2,
            "in_channels": 3,
            "channels": [4, 4],
            "attention_levels": [False, False],
            "num_res_blocks": 1,
            "norm_num_groups": 4,
            "num_head_channels": 4,
            "conditioning_embedding_num_channels": [16],
            "conditioning_embedding_in_channels": 1,
        },
        (1, 1, 8, 8),
        (1, 3, 4, 4),
    ],
    [
        "SPADEAutoencoderKL",
        {
            "spatial_dims": 2,
            "label_nc": 3,
            "in_channels": 1,
            "out_channels": 1,
            "channels": (4, 4),
            "latent_channels": 3,
            "attention_levels": [False, False],
            "num_res_blocks": 1,
            "with_encoder_nonlocal_attn": False,
            "with_decoder_nonlocal_attn": False,
            "norm_num_groups": 4,
        },
        "SPADEDiffusionModelUNet",
        {
            "spatial_dims": 2,
            "label_nc": 3,
            "in_channels": 3,
            "out_channels": 3,
            "channels": [4, 4],
            "norm_num_groups": 4,
            "attention_levels": [False, False],
            "num_res_blocks": 1,
            "num_head_channels": 4,
        },
        {
            "spatial_dims": 2,
            "in_channels": 3,
            "channels": [4, 4],
            "attention_levels": [False, False],
            "num_res_blocks": 1,
            "norm_num_groups": 4,
            "num_head_channels": 4,
            "conditioning_embedding_num_channels": [16],
            "conditioning_embedding_in_channels": 1,
        },
        (1, 1, 8, 8),
        (1, 3, 4, 4),
    ],
]


class ControlNetTestDiffusionSamplingInferer(unittest.TestCase):
    @parameterized.expand(CNDM_TEST_CASES)
    @skipUnless(has_einops, "Requires einops")
    def test_call(self, model_params, controlnet_params, input_shape):
        model = DiffusionModelUNet(**model_params)
        controlnet = ControlNet(**controlnet_params)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        controlnet.to(device)
        controlnet.eval()
        input = torch.randn(input_shape).to(device)
        mask = torch.randn(input_shape).to(device)
        noise = torch.randn(input_shape).to(device)
        scheduler = DDPMScheduler(num_train_timesteps=10)
        inferer = ControlNetDiffusionInferer(scheduler=scheduler)
        scheduler.set_timesteps(num_inference_steps=10)
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (input_shape[0],), device=input.device).long()
        sample = inferer(
            inputs=input, noise=noise, diffusion_model=model, controlnet=controlnet, timesteps=timesteps, cn_cond=mask
        )
        self.assertEqual(sample.shape, input_shape)

    @parameterized.expand(CNDM_TEST_CASES)
    @skipUnless(has_einops, "Requires einops")
    def test_sample_intermediates(self, model_params, controlnet_params, input_shape):
        model = DiffusionModelUNet(**model_params)
        controlnet = ControlNet(**controlnet_params)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        controlnet.to(device)
        controlnet.eval()
        noise = torch.randn(input_shape).to(device)
        mask = torch.randn(input_shape).to(device)
        scheduler = DDPMScheduler(num_train_timesteps=10)
        inferer = ControlNetDiffusionInferer(scheduler=scheduler)
        scheduler.set_timesteps(num_inference_steps=10)
        sample, intermediates = inferer.sample(
            input_noise=noise,
            diffusion_model=model,
            scheduler=scheduler,
            controlnet=controlnet,
            cn_cond=mask,
            save_intermediates=True,
            intermediate_steps=1,
        )
        self.assertEqual(len(intermediates), 10)

    @parameterized.expand(CNDM_TEST_CASES)
    @skipUnless(has_einops, "Requires einops")
    def test_ddpm_sampler(self, model_params, controlnet_params, input_shape):
        model = DiffusionModelUNet(**model_params)
        controlnet = ControlNet(**controlnet_params)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        controlnet.to(device)
        controlnet.eval()
        mask = torch.randn(input_shape).to(device)
        noise = torch.randn(input_shape).to(device)
        scheduler = DDPMScheduler(num_train_timesteps=1000)
        inferer = ControlNetDiffusionInferer(scheduler=scheduler)
        scheduler.set_timesteps(num_inference_steps=10)
        sample, intermediates = inferer.sample(
            input_noise=noise,
            diffusion_model=model,
            scheduler=scheduler,
            controlnet=controlnet,
            cn_cond=mask,
            save_intermediates=True,
            intermediate_steps=1,
        )
        self.assertEqual(len(intermediates), 10)

    @parameterized.expand(CNDM_TEST_CASES)
    @skipUnless(has_einops, "Requires einops")
    def test_ddim_sampler(self, model_params, controlnet_params, input_shape):
        model = DiffusionModelUNet(**model_params)
        controlnet = ControlNet(**controlnet_params)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        controlnet.to(device)
        controlnet.eval()
        mask = torch.randn(input_shape).to(device)
        noise = torch.randn(input_shape).to(device)
        scheduler = DDIMScheduler(num_train_timesteps=1000)
        inferer = ControlNetDiffusionInferer(scheduler=scheduler)
        scheduler.set_timesteps(num_inference_steps=10)
        sample, intermediates = inferer.sample(
            input_noise=noise,
            diffusion_model=model,
            scheduler=scheduler,
            controlnet=controlnet,
            cn_cond=mask,
            save_intermediates=True,
            intermediate_steps=1,
        )
        self.assertEqual(len(intermediates), 10)

    @parameterized.expand(CNDM_TEST_CASES)
    @skipUnless(has_einops, "Requires einops")
    def test_sampler_conditioned(self, model_params, controlnet_params, input_shape):
        model_params["with_conditioning"] = True
        model_params["cross_attention_dim"] = 3
        model = DiffusionModelUNet(**model_params)
        controlnet = ControlNet(**controlnet_params)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        controlnet.to(device)
        controlnet.eval()
        mask = torch.randn(input_shape).to(device)
        noise = torch.randn(input_shape).to(device)
        scheduler = DDIMScheduler(num_train_timesteps=1000)
        inferer = ControlNetDiffusionInferer(scheduler=scheduler)
        scheduler.set_timesteps(num_inference_steps=10)
        conditioning = torch.randn([input_shape[0], 1, 3]).to(device)
        sample, intermediates = inferer.sample(
            input_noise=noise,
            diffusion_model=model,
            controlnet=controlnet,
            cn_cond=mask,
            scheduler=scheduler,
            save_intermediates=True,
            intermediate_steps=1,
            conditioning=conditioning,
        )
        self.assertEqual(len(intermediates), 10)

    @parameterized.expand(CNDM_TEST_CASES)
    @skipUnless(has_einops, "Requires einops")
    def test_get_likelihood(self, model_params, controlnet_params, input_shape):
        model = DiffusionModelUNet(**model_params)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        controlnet = ControlNet(**controlnet_params)
        controlnet.to(device)
        controlnet.eval()
        input = torch.randn(input_shape).to(device)
        mask = torch.randn(input_shape).to(device)
        scheduler = DDPMScheduler(num_train_timesteps=10)
        inferer = ControlNetDiffusionInferer(scheduler=scheduler)
        scheduler.set_timesteps(num_inference_steps=10)
        likelihood, intermediates = inferer.get_likelihood(
            inputs=input,
            diffusion_model=model,
            scheduler=scheduler,
            controlnet=controlnet,
            cn_cond=mask,
            save_intermediates=True,
        )
        self.assertEqual(intermediates[0].shape, input.shape)
        self.assertEqual(likelihood.shape[0], input.shape[0])

    @unittest.skipUnless(has_scipy, "Requires scipy library.")
    def test_normal_cdf(self):
        from scipy.stats import norm

        scheduler = DDPMScheduler(num_train_timesteps=10)
        inferer = ControlNetDiffusionInferer(scheduler=scheduler)
        x = torch.linspace(-10, 10, 20)
        cdf_approx = inferer._approx_standard_normal_cdf(x)
        cdf_true = norm.cdf(x)
        torch.testing.assert_allclose(cdf_approx, cdf_true, atol=1e-3, rtol=1e-5)

    @parameterized.expand(CNDM_TEST_CASES)
    @skipUnless(has_einops, "Requires einops")
    def test_sampler_conditioned_concat(self, model_params, controlnet_params, input_shape):
        # copy the model_params dict to prevent from modifying test cases
        model_params = model_params.copy()
        n_concat_channel = 2
        model_params["in_channels"] = model_params["in_channels"] + n_concat_channel
        model_params["cross_attention_dim"] = None
        model_params["with_conditioning"] = False
        model = DiffusionModelUNet(**model_params)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        controlnet = ControlNet(**controlnet_params)
        controlnet.to(device)
        controlnet.eval()
        noise = torch.randn(input_shape).to(device)
        mask = torch.randn(input_shape).to(device)
        conditioning_shape = list(input_shape)
        conditioning_shape[1] = n_concat_channel
        conditioning = torch.randn(conditioning_shape).to(device)
        scheduler = DDIMScheduler(num_train_timesteps=1000)
        inferer = ControlNetDiffusionInferer(scheduler=scheduler)
        scheduler.set_timesteps(num_inference_steps=10)
        sample, intermediates = inferer.sample(
            input_noise=noise,
            diffusion_model=model,
            controlnet=controlnet,
            cn_cond=mask,
            scheduler=scheduler,
            save_intermediates=True,
            intermediate_steps=1,
            conditioning=conditioning,
            mode="concat",
        )
        self.assertEqual(len(intermediates), 10)


class LatentControlNetTestDiffusionSamplingInferer(unittest.TestCase):
    @parameterized.expand(LATENT_CNDM_TEST_CASES)
    @skipUnless(has_einops, "Requires einops")
    def test_prediction_shape(
        self,
        ae_model_type,
        autoencoder_params,
        dm_model_type,
        stage_2_params,
        controlnet_params,
        input_shape,
        latent_shape,
    ):
        stage_1 = None

        if ae_model_type == "AutoencoderKL":
            stage_1 = AutoencoderKL(**autoencoder_params)
        if ae_model_type == "VQVAE":
            stage_1 = VQVAE(**autoencoder_params)
        if dm_model_type == "SPADEDiffusionModelUNet":
            stage_2 = SPADEDiffusionModelUNet(**stage_2_params)
        else:
            stage_2 = DiffusionModelUNet(**stage_2_params)
        controlnet = ControlNet(**controlnet_params)

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        stage_1.to(device)
        stage_2.to(device)
        controlnet.to(device)
        stage_1.eval()
        stage_2.eval()
        controlnet.eval()

        input = torch.randn(input_shape).to(device)
        mask = torch.randn(input_shape).to(device)
        noise = torch.randn(latent_shape).to(device)
        scheduler = DDPMScheduler(num_train_timesteps=10)
        inferer = ControlNetLatentDiffusionInferer(scheduler=scheduler, scale_factor=1.0)
        scheduler.set_timesteps(num_inference_steps=10)
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (input_shape[0],), device=input.device).long()

        if dm_model_type == "SPADEDiffusionModelUNet":
            input_shape_seg = list(input_shape)
            if "label_nc" in stage_2_params.keys():
                input_shape_seg[1] = stage_2_params["label_nc"]
            else:
                input_shape_seg[1] = autoencoder_params["label_nc"]
            input_seg = torch.randn(input_shape_seg).to(device)
            prediction = inferer(
                inputs=input,
                autoencoder_model=stage_1,
                diffusion_model=stage_2,
                controlnet=controlnet,
                cn_cond=mask,
                seg=input_seg,
                noise=noise,
                timesteps=timesteps,
            )
        else:
            prediction = inferer(
                inputs=input,
                autoencoder_model=stage_1,
                diffusion_model=stage_2,
                noise=noise,
                timesteps=timesteps,
                controlnet=controlnet,
                cn_cond=mask,
            )
        self.assertEqual(prediction.shape, latent_shape)

    @parameterized.expand(LATENT_CNDM_TEST_CASES)
    @skipUnless(has_einops, "Requires einops")
    def test_sample_shape(
        self,
        ae_model_type,
        autoencoder_params,
        dm_model_type,
        stage_2_params,
        controlnet_params,
        input_shape,
        latent_shape,
    ):
        stage_1 = None

        if ae_model_type == "AutoencoderKL":
            stage_1 = AutoencoderKL(**autoencoder_params)
        if ae_model_type == "VQVAE":
            stage_1 = VQVAE(**autoencoder_params)
        if dm_model_type == "SPADEDiffusionModelUNet":
            stage_2 = SPADEDiffusionModelUNet(**stage_2_params)
        else:
            stage_2 = DiffusionModelUNet(**stage_2_params)
        controlnet = ControlNet(**controlnet_params)

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        stage_1.to(device)
        stage_2.to(device)
        controlnet.to(device)
        stage_1.eval()
        stage_2.eval()
        controlnet.eval()

        noise = torch.randn(latent_shape).to(device)
        mask = torch.randn(input_shape).to(device)
        scheduler = DDPMScheduler(num_train_timesteps=10)
        inferer = ControlNetLatentDiffusionInferer(scheduler=scheduler, scale_factor=1.0)
        scheduler.set_timesteps(num_inference_steps=10)

        if ae_model_type == "SPADEAutoencoderKL" or dm_model_type == "SPADEDiffusionModelUNet":
            input_shape_seg = list(input_shape)
            if "label_nc" in stage_2_params.keys():
                input_shape_seg[1] = stage_2_params["label_nc"]
            else:
                input_shape_seg[1] = autoencoder_params["label_nc"]
            input_seg = torch.randn(input_shape_seg).to(device)
            sample = inferer.sample(
                input_noise=noise,
                autoencoder_model=stage_1,
                diffusion_model=stage_2,
                controlnet=controlnet,
                cn_cond=mask,
                scheduler=scheduler,
                seg=input_seg,
            )
        else:
            sample = inferer.sample(
                input_noise=noise,
                autoencoder_model=stage_1,
                diffusion_model=stage_2,
                scheduler=scheduler,
                controlnet=controlnet,
                cn_cond=mask,
            )
        self.assertEqual(sample.shape, input_shape)

    @parameterized.expand(LATENT_CNDM_TEST_CASES)
    @skipUnless(has_einops, "Requires einops")
    def test_sample_intermediates(
        self,
        ae_model_type,
        autoencoder_params,
        dm_model_type,
        stage_2_params,
        controlnet_params,
        input_shape,
        latent_shape,
    ):
        stage_1 = None

        if ae_model_type == "AutoencoderKL":
            stage_1 = AutoencoderKL(**autoencoder_params)
        if ae_model_type == "VQVAE":
            stage_1 = VQVAE(**autoencoder_params)
        if ae_model_type == "SPADEAutoencoderKL":
            stage_1 = SPADEAutoencoderKL(**autoencoder_params)
        if dm_model_type == "SPADEDiffusionModelUNet":
            stage_2 = SPADEDiffusionModelUNet(**stage_2_params)
        else:
            stage_2 = DiffusionModelUNet(**stage_2_params)
        controlnet = ControlNet(**controlnet_params)

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        stage_1.to(device)
        stage_2.to(device)
        controlnet.to(device)
        stage_1.eval()
        stage_2.eval()
        controlnet.eval()

        noise = torch.randn(latent_shape).to(device)
        mask = torch.randn(input_shape).to(device)
        scheduler = DDPMScheduler(num_train_timesteps=10)
        inferer = ControlNetLatentDiffusionInferer(scheduler=scheduler, scale_factor=1.0)
        scheduler.set_timesteps(num_inference_steps=10)

        if ae_model_type == "SPADEAutoencoderKL" or dm_model_type == "SPADEDiffusionModelUNet":
            input_shape_seg = list(input_shape)
            if "label_nc" in stage_2_params.keys():
                input_shape_seg[1] = stage_2_params["label_nc"]
            else:
                input_shape_seg[1] = autoencoder_params["label_nc"]
            input_seg = torch.randn(input_shape_seg).to(device)
            sample = inferer.sample(
                input_noise=noise,
                autoencoder_model=stage_1,
                diffusion_model=stage_2,
                scheduler=scheduler,
                seg=input_seg,
                controlnet=controlnet,
                cn_cond=mask,
            )

            # TODO: this isn't correct, should the above produce intermediates as well?
            # This test has always passed so is this branch not being used?
            intermediates = None
        else:
            sample, intermediates = inferer.sample(
                input_noise=noise,
                autoencoder_model=stage_1,
                diffusion_model=stage_2,
                scheduler=scheduler,
                save_intermediates=True,
                intermediate_steps=1,
                controlnet=controlnet,
                cn_cond=mask,
            )

        self.assertEqual(len(intermediates), 10)
        self.assertEqual(intermediates[0].shape, input_shape)

    @parameterized.expand(LATENT_CNDM_TEST_CASES)
    @skipUnless(has_einops, "Requires einops")
    def test_get_likelihoods(
        self,
        ae_model_type,
        autoencoder_params,
        dm_model_type,
        stage_2_params,
        controlnet_params,
        input_shape,
        latent_shape,
    ):
        stage_1 = None

        if ae_model_type == "AutoencoderKL":
            stage_1 = AutoencoderKL(**autoencoder_params)
        if ae_model_type == "VQVAE":
            stage_1 = VQVAE(**autoencoder_params)
        if ae_model_type == "SPADEAutoencoderKL":
            stage_1 = SPADEAutoencoderKL(**autoencoder_params)
        if dm_model_type == "SPADEDiffusionModelUNet":
            stage_2 = SPADEDiffusionModelUNet(**stage_2_params)
        else:
            stage_2 = DiffusionModelUNet(**stage_2_params)
        controlnet = ControlNet(**controlnet_params)

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        stage_1.to(device)
        stage_2.to(device)
        controlnet.to(device)
        stage_1.eval()
        stage_2.eval()
        controlnet.eval()

        input = torch.randn(input_shape).to(device)
        mask = torch.randn(input_shape).to(device)
        scheduler = DDPMScheduler(num_train_timesteps=10)
        inferer = ControlNetLatentDiffusionInferer(scheduler=scheduler, scale_factor=1.0)
        scheduler.set_timesteps(num_inference_steps=10)

        if dm_model_type == "SPADEDiffusionModelUNet":
            input_shape_seg = list(input_shape)
            if "label_nc" in stage_2_params.keys():
                input_shape_seg[1] = stage_2_params["label_nc"]
            else:
                input_shape_seg[1] = autoencoder_params["label_nc"]
            input_seg = torch.randn(input_shape_seg).to(device)
            sample, intermediates = inferer.get_likelihood(
                inputs=input,
                autoencoder_model=stage_1,
                diffusion_model=stage_2,
                controlnet=controlnet,
                cn_cond=mask,
                scheduler=scheduler,
                save_intermediates=True,
                seg=input_seg,
            )
        else:
            sample, intermediates = inferer.get_likelihood(
                inputs=input,
                autoencoder_model=stage_1,
                diffusion_model=stage_2,
                scheduler=scheduler,
                controlnet=controlnet,
                cn_cond=mask,
                save_intermediates=True,
            )
        self.assertEqual(len(intermediates), 10)
        self.assertEqual(intermediates[0].shape, latent_shape)

    @parameterized.expand(LATENT_CNDM_TEST_CASES)
    @skipUnless(has_einops, "Requires einops")
    def test_resample_likelihoods(
        self,
        ae_model_type,
        autoencoder_params,
        dm_model_type,
        stage_2_params,
        controlnet_params,
        input_shape,
        latent_shape,
    ):
        stage_1 = None

        if ae_model_type == "AutoencoderKL":
            stage_1 = AutoencoderKL(**autoencoder_params)
        if ae_model_type == "VQVAE":
            stage_1 = VQVAE(**autoencoder_params)
        if ae_model_type == "SPADEAutoencoderKL":
            stage_1 = SPADEAutoencoderKL(**autoencoder_params)
        if dm_model_type == "SPADEDiffusionModelUNet":
            stage_2 = SPADEDiffusionModelUNet(**stage_2_params)
        else:
            stage_2 = DiffusionModelUNet(**stage_2_params)
        controlnet = ControlNet(**controlnet_params)

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        stage_1.to(device)
        stage_2.to(device)
        controlnet.to(device)
        stage_1.eval()
        stage_2.eval()
        controlnet.eval()

        input = torch.randn(input_shape).to(device)
        mask = torch.randn(input_shape).to(device)
        scheduler = DDPMScheduler(num_train_timesteps=10)
        inferer = ControlNetLatentDiffusionInferer(scheduler=scheduler, scale_factor=1.0)
        scheduler.set_timesteps(num_inference_steps=10)

        if dm_model_type == "SPADEDiffusionModelUNet":
            input_shape_seg = list(input_shape)
            if "label_nc" in stage_2_params.keys():
                input_shape_seg[1] = stage_2_params["label_nc"]
            else:
                input_shape_seg[1] = autoencoder_params["label_nc"]
            input_seg = torch.randn(input_shape_seg).to(device)
            sample, intermediates = inferer.get_likelihood(
                inputs=input,
                autoencoder_model=stage_1,
                diffusion_model=stage_2,
                scheduler=scheduler,
                controlnet=controlnet,
                cn_cond=mask,
                save_intermediates=True,
                resample_latent_likelihoods=True,
                seg=input_seg,
            )
        else:
            sample, intermediates = inferer.get_likelihood(
                inputs=input,
                autoencoder_model=stage_1,
                diffusion_model=stage_2,
                scheduler=scheduler,
                controlnet=controlnet,
                cn_cond=mask,
                save_intermediates=True,
                resample_latent_likelihoods=True,
            )
        self.assertEqual(len(intermediates), 10)
        self.assertEqual(intermediates[0].shape[2:], input_shape[2:])

    @parameterized.expand(LATENT_CNDM_TEST_CASES)
    @skipUnless(has_einops, "Requires einops")
    def test_prediction_shape_conditioned_concat(
        self,
        ae_model_type,
        autoencoder_params,
        dm_model_type,
        stage_2_params,
        controlnet_params,
        input_shape,
        latent_shape,
    ):
        stage_1 = None

        if ae_model_type == "AutoencoderKL":
            stage_1 = AutoencoderKL(**autoencoder_params)
        if ae_model_type == "VQVAE":
            stage_1 = VQVAE(**autoencoder_params)
        if ae_model_type == "SPADEAutoencoderKL":
            stage_1 = SPADEAutoencoderKL(**autoencoder_params)
        stage_2_params = stage_2_params.copy()
        n_concat_channel = 3
        stage_2_params["in_channels"] = stage_2_params["in_channels"] + n_concat_channel
        if dm_model_type == "SPADEDiffusionModelUNet":
            stage_2 = SPADEDiffusionModelUNet(**stage_2_params)
        else:
            stage_2 = DiffusionModelUNet(**stage_2_params)
        controlnet = ControlNet(**controlnet_params)

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        stage_1.to(device)
        stage_2.to(device)
        controlnet.to(device)
        stage_1.eval()
        stage_2.eval()
        controlnet.eval()

        input = torch.randn(input_shape).to(device)
        mask = torch.randn(input_shape).to(device)
        noise = torch.randn(latent_shape).to(device)
        conditioning_shape = list(latent_shape)
        conditioning_shape[1] = n_concat_channel
        conditioning = torch.randn(conditioning_shape).to(device)

        scheduler = DDPMScheduler(num_train_timesteps=10)
        inferer = ControlNetLatentDiffusionInferer(scheduler=scheduler, scale_factor=1.0)
        scheduler.set_timesteps(num_inference_steps=10)

        timesteps = torch.randint(0, scheduler.num_train_timesteps, (input_shape[0],), device=input.device).long()

        if dm_model_type == "SPADEDiffusionModelUNet":
            input_shape_seg = list(input_shape)
            if "label_nc" in stage_2_params.keys():
                input_shape_seg[1] = stage_2_params["label_nc"]
            else:
                input_shape_seg[1] = autoencoder_params["label_nc"]
            input_seg = torch.randn(input_shape_seg).to(device)
            prediction = inferer(
                inputs=input,
                autoencoder_model=stage_1,
                diffusion_model=stage_2,
                noise=noise,
                controlnet=controlnet,
                cn_cond=mask,
                timesteps=timesteps,
                condition=conditioning,
                mode="concat",
                seg=input_seg,
            )
        else:
            prediction = inferer(
                inputs=input,
                autoencoder_model=stage_1,
                diffusion_model=stage_2,
                noise=noise,
                controlnet=controlnet,
                cn_cond=mask,
                timesteps=timesteps,
                condition=conditioning,
                mode="concat",
            )
        self.assertEqual(prediction.shape, latent_shape)

    @parameterized.expand(LATENT_CNDM_TEST_CASES)
    @skipUnless(has_einops, "Requires einops")
    def test_sample_shape_conditioned_concat(
        self,
        ae_model_type,
        autoencoder_params,
        dm_model_type,
        stage_2_params,
        controlnet_params,
        input_shape,
        latent_shape,
    ):
        stage_1 = None

        if ae_model_type == "AutoencoderKL":
            stage_1 = AutoencoderKL(**autoencoder_params)
        if ae_model_type == "VQVAE":
            stage_1 = VQVAE(**autoencoder_params)
        if ae_model_type == "SPADEAutoencoderKL":
            stage_1 = SPADEAutoencoderKL(**autoencoder_params)
        stage_2_params = stage_2_params.copy()
        n_concat_channel = 3
        stage_2_params["in_channels"] = stage_2_params["in_channels"] + n_concat_channel
        if dm_model_type == "SPADEDiffusionModelUNet":
            stage_2 = SPADEDiffusionModelUNet(**stage_2_params)
        else:
            stage_2 = DiffusionModelUNet(**stage_2_params)
        controlnet = ControlNet(**controlnet_params)

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        stage_1.to(device)
        stage_2.to(device)
        controlnet.to(device)
        stage_1.eval()
        stage_2.eval()
        controlnet.eval()

        noise = torch.randn(latent_shape).to(device)
        mask = torch.randn(input_shape).to(device)
        conditioning_shape = list(latent_shape)
        conditioning_shape[1] = n_concat_channel
        conditioning = torch.randn(conditioning_shape).to(device)

        scheduler = DDPMScheduler(num_train_timesteps=10)
        inferer = ControlNetLatentDiffusionInferer(scheduler=scheduler, scale_factor=1.0)
        scheduler.set_timesteps(num_inference_steps=10)

        if dm_model_type == "SPADEDiffusionModelUNet":
            input_shape_seg = list(input_shape)
            if "label_nc" in stage_2_params.keys():
                input_shape_seg[1] = stage_2_params["label_nc"]
            else:
                input_shape_seg[1] = autoencoder_params["label_nc"]
            input_seg = torch.randn(input_shape_seg).to(device)
            sample = inferer.sample(
                input_noise=noise,
                autoencoder_model=stage_1,
                diffusion_model=stage_2,
                controlnet=controlnet,
                cn_cond=mask,
                scheduler=scheduler,
                conditioning=conditioning,
                mode="concat",
                seg=input_seg,
            )
        else:
            sample = inferer.sample(
                input_noise=noise,
                autoencoder_model=stage_1,
                diffusion_model=stage_2,
                controlnet=controlnet,
                cn_cond=mask,
                scheduler=scheduler,
                conditioning=conditioning,
                mode="concat",
            )
        self.assertEqual(sample.shape, input_shape)

    @parameterized.expand(LATENT_CNDM_TEST_CASES_DIFF_SHAPES)
    @skipUnless(has_einops, "Requires einops")
    def test_sample_shape_different_latents(
        self,
        ae_model_type,
        autoencoder_params,
        dm_model_type,
        stage_2_params,
        controlnet_params,
        input_shape,
        latent_shape,
    ):
        stage_1 = None

        if ae_model_type == "AutoencoderKL":
            stage_1 = AutoencoderKL(**autoencoder_params)
        if ae_model_type == "VQVAE":
            stage_1 = VQVAE(**autoencoder_params)
        if ae_model_type == "SPADEAutoencoderKL":
            stage_1 = SPADEAutoencoderKL(**autoencoder_params)
        if dm_model_type == "SPADEDiffusionModelUNet":
            stage_2 = SPADEDiffusionModelUNet(**stage_2_params)
        else:
            stage_2 = DiffusionModelUNet(**stage_2_params)
        controlnet = ControlNet(**controlnet_params)

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        stage_1.to(device)
        stage_2.to(device)
        controlnet.to(device)
        stage_1.eval()
        stage_2.eval()
        controlnet.eval()

        input = torch.randn(input_shape).to(device)
        noise = torch.randn(latent_shape).to(device)
        mask = torch.randn(input_shape).to(device)
        scheduler = DDPMScheduler(num_train_timesteps=10)
        # We infer the VAE shape
        autoencoder_latent_shape = [i // (2 ** (len(autoencoder_params["channels"]) - 1)) for i in input_shape[2:]]
        inferer = ControlNetLatentDiffusionInferer(
            scheduler=scheduler,
            scale_factor=1.0,
            ldm_latent_shape=list(latent_shape[2:]),
            autoencoder_latent_shape=autoencoder_latent_shape,
        )
        scheduler.set_timesteps(num_inference_steps=10)

        timesteps = torch.randint(0, scheduler.num_train_timesteps, (input_shape[0],), device=input.device).long()

        if dm_model_type == "SPADEDiffusionModelUNet":
            input_shape_seg = list(input_shape)
            if "label_nc" in stage_2_params.keys():
                input_shape_seg[1] = stage_2_params["label_nc"]
            else:
                input_shape_seg[1] = autoencoder_params["label_nc"]
            input_seg = torch.randn(input_shape_seg).to(device)
            prediction = inferer(
                inputs=input,
                autoencoder_model=stage_1,
                diffusion_model=stage_2,
                controlnet=controlnet,
                cn_cond=mask,
                noise=noise,
                timesteps=timesteps,
                seg=input_seg,
            )
        else:
            prediction = inferer(
                inputs=input,
                autoencoder_model=stage_1,
                diffusion_model=stage_2,
                noise=noise,
                controlnet=controlnet,
                cn_cond=mask,
                timesteps=timesteps,
            )
        self.assertEqual(prediction.shape, latent_shape)

    @skipUnless(has_einops, "Requires einops")
    def test_incompatible_spade_setup(self):
        stage_1 = SPADEAutoencoderKL(
            spatial_dims=2,
            label_nc=6,
            in_channels=1,
            out_channels=1,
            channels=(4, 4),
            latent_channels=3,
            attention_levels=[False, False],
            num_res_blocks=1,
            with_encoder_nonlocal_attn=False,
            with_decoder_nonlocal_attn=False,
            norm_num_groups=4,
        )
        stage_2 = SPADEDiffusionModelUNet(
            spatial_dims=2,
            label_nc=3,
            in_channels=3,
            out_channels=3,
            channels=[4, 4],
            norm_num_groups=4,
            attention_levels=[False, False],
            num_res_blocks=1,
            num_head_channels=4,
        )
        controlnet = ControlNet(
            spatial_dims=2,
            in_channels=1,
            channels=[4, 4],
            norm_num_groups=4,
            attention_levels=[False, False],
            num_res_blocks=1,
            num_head_channels=4,
            conditioning_embedding_num_channels=[16],
        )

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        stage_1.to(device)
        stage_2.to(device)
        controlnet.to(device)
        controlnet.to(device)
        stage_1.eval()
        stage_2.eval()
        controlnet.eval()
        noise = torch.randn((1, 3, 4, 4)).to(device)
        mask = torch.randn((1, 1, 4, 4)).to(device)
        input_seg = torch.randn((1, 3, 8, 8)).to(device)
        scheduler = DDPMScheduler(num_train_timesteps=10)
        inferer = ControlNetLatentDiffusionInferer(scheduler=scheduler, scale_factor=1.0)
        scheduler.set_timesteps(num_inference_steps=10)

        with self.assertRaises(ValueError):
            _ = inferer.sample(
                input_noise=noise,
                autoencoder_model=stage_1,
                diffusion_model=stage_2,
                scheduler=scheduler,
                controlnet=controlnet,
                cn_cond=mask,
                seg=input_seg,
            )


if __name__ == "__main__":
    unittest.main()
