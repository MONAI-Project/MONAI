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
from monai.networks.nets.masked_autoencoder_vit import MaskedAutoEncoderViT
from tests.util import skip_if_quick

TEST_CASE_MaskedAutoEncoderViT = []
for masking_ratio in [0.5]:
    for dropout_rate in [0.6]:
        for in_channels in [4]:
            for hidden_size in [768]:
                for img_size in [96, 128]:
                    for patch_size in [16]:
                        for num_heads in [12]:
                            for mlp_dim in [3072]:
                                for num_layers in [4]:
                                    for decoder_hidden_size in [384]:
                                        for decoder_mlp_dim in [512]:
                                            for decoder_num_layers in [4]:
                                                for decoder_num_heads in [16]:
                                                    for pos_embed_type in ["sincos", "learnable"]:
                                                        for proj_type in ["conv", "perceptron"]:
                                                            for nd in (2, 3):
                                                                test_case = [
                                                                    {
                                                                        "in_channels": in_channels,
                                                                        "img_size": (img_size,) * nd,
                                                                        "patch_size": (patch_size,) * nd,
                                                                        "hidden_size": hidden_size,
                                                                        "mlp_dim": mlp_dim,
                                                                        "num_layers": num_layers,
                                                                        "decoder_hidden_size": decoder_hidden_size,
                                                                        "decoder_mlp_dim": decoder_mlp_dim,
                                                                        "decoder_num_layers": decoder_num_layers,
                                                                        "decoder_num_heads": decoder_num_heads,
                                                                        "pos_embed_type": pos_embed_type,
                                                                        "masking_ratio": masking_ratio,
                                                                        "decoder_pos_embed_type": pos_embed_type,
                                                                        "num_heads": num_heads,
                                                                        "proj_type": proj_type,
                                                                        "dropout_rate": dropout_rate,
                                                                    },
                                                                    (2, in_channels, *([img_size] * nd)),
                                                                    (
                                                                        2,
                                                                        (img_size // patch_size) ** nd,
                                                                        in_channels * (patch_size**nd),
                                                                    ),
                                                                ]
                                                                if nd == 2:
                                                                    test_case[0]["spatial_dims"] = 2  # type: ignore
                                                                TEST_CASE_MaskedAutoEncoderViT.append(test_case)

TEST_CASE_ill_args = [
    [{"in_channels": 1, "img_size": (128, 128, 128), "patch_size": (16, 16, 16), "dropout_rate": 5.0}],
    [{"in_channels": 1, "img_size": (128, 128, 128), "patch_size": (64, 64, 64), "pos_embed_type": "sin"}],
    [{"in_channels": 1, "img_size": (128, 128, 128), "patch_size": (64, 64, 64), "decoder_pos_embed_type": "sin"}],
    [{"in_channels": 1, "img_size": (32, 32, 32), "patch_size": (64, 64, 64)}],
    [{"in_channels": 1, "img_size": (128, 128, 128), "patch_size": (64, 64, 64), "num_layers": 12, "num_heads": 14}],
    [{"in_channels": 1, "img_size": (97, 97, 97), "patch_size": (16, 16, 16)}],
    [{"in_channels": 1, "img_size": (128, 128, 128), "patch_size": (64, 64, 64), "masking_ratio": 1.1}],
    [{"in_channels": 1, "img_size": (128, 128, 128), "patch_size": (64, 64, 64), "masking_ratio": -0.1}],
]


@skip_if_quick
class TestMaskedAutoencoderViT(unittest.TestCase):
    @parameterized.expand(TEST_CASE_MaskedAutoEncoderViT)
    def test_shape(self, input_param, input_shape, expected_shape):
        net = MaskedAutoEncoderViT(**input_param)
        with eval_mode(net):
            result, _ = net(torch.randn(input_shape))
            self.assertEqual(result.shape, expected_shape)

    def test_frozen_pos_embedding(self):
        net = MaskedAutoEncoderViT(in_channels=1, img_size=(96, 96, 96), patch_size=(16, 16, 16))

        self.assertEqual(net.decoder_pos_embedding.requires_grad, False)

    @parameterized.expand(TEST_CASE_ill_args)
    def test_ill_arg(self, input_param):
        with self.assertRaises(ValueError):
            MaskedAutoEncoderViT(**input_param)

    def test_access_attn_matrix(self):
        # input format
        in_channels = 1
        img_size = (96, 96, 96)
        patch_size = (16, 16, 16)
        in_shape = (1, in_channels, img_size[0], img_size[1], img_size[2])

        # no data in the matrix
        no_matrix_acess_blk = MaskedAutoEncoderViT(in_channels=in_channels, img_size=img_size, patch_size=patch_size)
        no_matrix_acess_blk(torch.randn(in_shape))
        assert isinstance(no_matrix_acess_blk.blocks[0].attn.att_mat, torch.Tensor)
        # no of elements is zero
        assert no_matrix_acess_blk.blocks[0].attn.att_mat.nelement() == 0

        # be able to acess the attention matrix
        matrix_acess_blk = MaskedAutoEncoderViT(
            in_channels=in_channels, img_size=img_size, patch_size=patch_size, save_attn=True
        )
        matrix_acess_blk(torch.randn(in_shape))

        assert matrix_acess_blk.blocks[0].attn.att_mat.shape == (in_shape[0], 12, 55, 55)

    def test_masking_ratio(self):
        # input format
        in_channels = 1
        img_size = (96, 96, 96)
        patch_size = (16, 16, 16)
        in_shape = (1, in_channels, img_size[0], img_size[1], img_size[2])

        # masking ratio 0.25
        masking_ratio_blk = MaskedAutoEncoderViT(
            in_channels=in_channels, img_size=img_size, patch_size=patch_size, masking_ratio=0.25, save_attn=True
        )
        masking_ratio_blk(torch.randn(in_shape))
        desired_num_tokens = int(
            (img_size[0] // patch_size[0])
            * (img_size[1] // patch_size[1])
            * (img_size[2] // patch_size[2])
            * (1 - 0.25)
        )
        assert masking_ratio_blk.blocks[0].attn.att_mat.shape[-1] - 1 == desired_num_tokens

        # masking ratio 0.33
        masking_ratio_blk = MaskedAutoEncoderViT(
            in_channels=in_channels, img_size=img_size, patch_size=patch_size, masking_ratio=0.33, save_attn=True
        )
        masking_ratio_blk(torch.randn(in_shape))
        desired_num_tokens = int(
            (img_size[0] // patch_size[0])
            * (img_size[1] // patch_size[1])
            * (img_size[2] // patch_size[2])
            * (1 - 0.33)
        )

        assert masking_ratio_blk.blocks[0].attn.att_mat.shape[-1] - 1 == desired_num_tokens


if __name__ == "__main__":
    unittest.main()
