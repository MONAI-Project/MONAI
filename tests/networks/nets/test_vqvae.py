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
from monai.networks.nets.vqvae import VQVAE
from tests.util import SkipIfBeforePyTorchVersion

TEST_CASES = [
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "channels": (4, 4),
            "num_res_layers": 1,
            "num_res_channels": (4, 4),
            "downsample_parameters": ((2, 4, 1, 1),) * 2,
            "upsample_parameters": ((2, 4, 1, 1, 0),) * 2,
            "num_embeddings": 8,
            "embedding_dim": 8,
        },
        (1, 1, 8, 8),
        (1, 1, 8, 8),
    ],
    [
        {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 1,
            "channels": (4, 4),
            "num_res_layers": 1,
            "num_res_channels": 4,
            "downsample_parameters": ((2, 4, 1, 1),) * 2,
            "upsample_parameters": ((2, 4, 1, 1, 0),) * 2,
            "num_embeddings": 8,
            "embedding_dim": 8,
        },
        (1, 1, 8, 8, 8),
        (1, 1, 8, 8, 8),
    ],
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "channels": (4, 4),
            "num_res_layers": 1,
            "num_res_channels": (4, 4),
            "downsample_parameters": (2, 4, 1, 1),
            "upsample_parameters": ((2, 4, 1, 1, 0),) * 2,
            "num_embeddings": 8,
            "embedding_dim": 8,
        },
        (1, 1, 8, 8),
        (1, 1, 8, 8),
    ],
    [
        {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 1,
            "channels": (4, 4),
            "num_res_layers": 1,
            "num_res_channels": (4, 4),
            "downsample_parameters": ((2, 4, 1, 1),) * 2,
            "upsample_parameters": (2, 4, 1, 1, 0),
            "num_embeddings": 8,
            "embedding_dim": 8,
        },
        (1, 1, 8, 8, 8),
        (1, 1, 8, 8, 8),
    ],
]

TEST_LATENT_SHAPE = {
    "spatial_dims": 2,
    "in_channels": 1,
    "out_channels": 1,
    "downsample_parameters": ((2, 4, 1, 1),) * 2,
    "upsample_parameters": ((2, 4, 1, 1, 0),) * 2,
    "num_res_layers": 1,
    "channels": (8, 8),
    "num_res_channels": (8, 8),
    "num_embeddings": 16,
    "embedding_dim": 8,
}


class TestVQVAE(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_shape(self, input_param, input_shape, expected_shape):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        net = VQVAE(**input_param).to(device)

        with eval_mode(net):
            result, _ = net(torch.randn(input_shape).to(device))

        self.assertEqual(result.shape, expected_shape)

    @parameterized.expand(TEST_CASES)
    @SkipIfBeforePyTorchVersion((1, 11))
    def test_shape_with_checkpoint(self, input_param, input_shape, expected_shape):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        input_param = input_param.copy()
        input_param.update({"use_checkpointing": True})

        net = VQVAE(**input_param).to(device)

        with eval_mode(net):
            result, _ = net(torch.randn(input_shape).to(device))

        self.assertEqual(result.shape, expected_shape)

    # Removed this test case since TorchScript currently does not support activation checkpoint.
    # def test_script(self):
    #     net = VQVAE(
    #         spatial_dims=2,
    #         in_channels=1,
    #         out_channels=1,
    #         downsample_parameters=((2, 4, 1, 1),) * 2,
    #         upsample_parameters=((2, 4, 1, 1, 0),) * 2,
    #         num_res_layers=1,
    #         channels=(8, 8),
    #         num_res_channels=(8, 8),
    #         num_embeddings=16,
    #         embedding_dim=8,
    #         ddp_sync=False,
    #     )
    #     test_data = torch.randn(1, 1, 16, 16)
    #     test_script_save(net, test_data)

    def test_channels_not_same_size_of_num_res_channels(self):
        with self.assertRaises(ValueError):
            VQVAE(
                spatial_dims=2,
                in_channels=1,
                out_channels=1,
                channels=(16, 16),
                num_res_channels=(16, 16, 16),
                downsample_parameters=((2, 4, 1, 1),) * 2,
                upsample_parameters=((2, 4, 1, 1, 0),) * 2,
            )

    def test_channels_not_same_size_of_downsample_parameters(self):
        with self.assertRaises(ValueError):
            VQVAE(
                spatial_dims=2,
                in_channels=1,
                out_channels=1,
                channels=(16, 16),
                num_res_channels=(16, 16),
                downsample_parameters=((2, 4, 1, 1),) * 3,
                upsample_parameters=((2, 4, 1, 1, 0),) * 2,
            )

    def test_channels_not_same_size_of_upsample_parameters(self):
        with self.assertRaises(ValueError):
            VQVAE(
                spatial_dims=2,
                in_channels=1,
                out_channels=1,
                channels=(16, 16),
                num_res_channels=(16, 16),
                downsample_parameters=((2, 4, 1, 1),) * 2,
                upsample_parameters=((2, 4, 1, 1, 0),) * 3,
            )

    def test_downsample_parameters_not_sequence_or_int(self):
        with self.assertRaises(ValueError):
            VQVAE(
                spatial_dims=2,
                in_channels=1,
                out_channels=1,
                channels=(16, 16),
                num_res_channels=(16, 16),
                downsample_parameters=(("test", 4, 1, 1),) * 2,
                upsample_parameters=((2, 4, 1, 1, 0),) * 2,
            )

    def test_upsample_parameters_not_sequence_or_int(self):
        with self.assertRaises(ValueError):
            VQVAE(
                spatial_dims=2,
                in_channels=1,
                out_channels=1,
                channels=(16, 16),
                num_res_channels=(16, 16),
                downsample_parameters=((2, 4, 1, 1),) * 2,
                upsample_parameters=(("test", 4, 1, 1, 0),) * 2,
            )

    def test_downsample_parameter_length_different_4(self):
        with self.assertRaises(ValueError):
            VQVAE(
                spatial_dims=2,
                in_channels=1,
                out_channels=1,
                channels=(16, 16),
                num_res_channels=(16, 16),
                downsample_parameters=((2, 4, 1),) * 3,
                upsample_parameters=((2, 4, 1, 1, 0),) * 2,
            )

    def test_upsample_parameter_length_different_5(self):
        with self.assertRaises(ValueError):
            VQVAE(
                spatial_dims=2,
                in_channels=1,
                out_channels=1,
                channels=(16, 16),
                num_res_channels=(16, 16, 16),
                downsample_parameters=((2, 4, 1, 1),) * 2,
                upsample_parameters=((2, 4, 1, 1, 0, 1),) * 3,
            )

    def test_encode_shape(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        net = VQVAE(**TEST_LATENT_SHAPE).to(device)

        with eval_mode(net):
            latent = net.encode(torch.randn(1, 1, 32, 32).to(device))

        self.assertEqual(latent.shape, (1, 8, 8, 8))

    def test_index_quantize_shape(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        net = VQVAE(**TEST_LATENT_SHAPE).to(device)

        with eval_mode(net):
            latent = net.index_quantize(torch.randn(1, 1, 32, 32).to(device))

        self.assertEqual(latent.shape, (1, 8, 8))

    def test_decode_shape(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        net = VQVAE(**TEST_LATENT_SHAPE).to(device)

        with eval_mode(net):
            latent = net.decode(torch.randn(1, 8, 8, 8).to(device))

        self.assertEqual(latent.shape, (1, 1, 32, 32))

    def test_decode_samples_shape(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        net = VQVAE(**TEST_LATENT_SHAPE).to(device)

        with eval_mode(net):
            latent = net.decode_samples(torch.randint(low=0, high=16, size=(1, 8, 8)).to(device))

        self.assertEqual(latent.shape, (1, 1, 32, 32))


if __name__ == "__main__":
    unittest.main()
