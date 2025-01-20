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

import numpy as np
import torch
from parameterized import parameterized

from monai.networks import eval_mode
from monai.networks.blocks.crossattention import CrossAttentionBlock
from monai.networks.layers.factories import RelPosEmbedding
from monai.utils import optional_import
from tests.util import SkipIfBeforePyTorchVersion, assert_allclose

einops, has_einops = optional_import("einops")

TEST_CASE_CABLOCK = []
for dropout_rate in np.linspace(0, 1, 4):
    for hidden_size in [360, 480, 600, 768]:
        for num_heads in [4, 6, 8, 12]:
            for rel_pos_embedding in [None, RelPosEmbedding.DECOMPOSED]:
                for input_size in [(16, 32), (8, 8, 8)]:
                    for flash_attn in [True, False]:
                        test_case = [
                            {
                                "hidden_size": hidden_size,
                                "num_heads": num_heads,
                                "dropout_rate": dropout_rate,
                                "rel_pos_embedding": rel_pos_embedding if not flash_attn else None,
                                "input_size": input_size,
                                "use_flash_attention": flash_attn,
                            },
                            (2, 512, hidden_size),
                            (2, 512, hidden_size),
                        ]
                        TEST_CASE_CABLOCK.append(test_case)


class TestResBlock(unittest.TestCase):
    @parameterized.expand(TEST_CASE_CABLOCK)
    @skipUnless(has_einops, "Requires einops")
    @SkipIfBeforePyTorchVersion((2, 0))
    def test_shape(self, input_param, input_shape, expected_shape):
        # Without flash attention
        net = CrossAttentionBlock(**input_param)
        with eval_mode(net):
            result = net(torch.randn(input_shape), context=torch.randn(2, 512, input_param["hidden_size"]))
            self.assertEqual(result.shape, expected_shape)

    def test_ill_arg(self):
        with self.assertRaises(ValueError):
            CrossAttentionBlock(hidden_size=128, num_heads=12, dropout_rate=6.0)

        with self.assertRaises(ValueError):
            CrossAttentionBlock(hidden_size=620, num_heads=8, dropout_rate=0.4)

    @SkipIfBeforePyTorchVersion((2, 0))
    def test_save_attn_with_flash_attention(self):
        with self.assertRaises(ValueError):
            CrossAttentionBlock(
                hidden_size=128, num_heads=3, dropout_rate=0.1, use_flash_attention=True, save_attn=True
            )

    @SkipIfBeforePyTorchVersion((2, 0))
    def test_rel_pos_embedding_with_flash_attention(self):
        with self.assertRaises(ValueError):
            CrossAttentionBlock(
                hidden_size=128,
                num_heads=3,
                dropout_rate=0.1,
                use_flash_attention=True,
                save_attn=False,
                rel_pos_embedding=RelPosEmbedding.DECOMPOSED,
            )

    @skipUnless(has_einops, "Requires einops")
    def test_attention_dim_not_multiple_of_heads(self):
        with self.assertRaises(ValueError):
            CrossAttentionBlock(hidden_size=128, num_heads=3, dropout_rate=0.1)

    @skipUnless(has_einops, "Requires einops")
    def test_inner_dim_different(self):
        CrossAttentionBlock(hidden_size=128, num_heads=4, dropout_rate=0.1, dim_head=30)

    def test_causal_no_sequence_length(self):
        with self.assertRaises(ValueError):
            CrossAttentionBlock(hidden_size=128, num_heads=4, dropout_rate=0.1, causal=True)

    @skipUnless(has_einops, "Requires einops")
    @SkipIfBeforePyTorchVersion((2, 0))
    def test_causal_flash_attention(self):
        block = CrossAttentionBlock(
            hidden_size=128,
            num_heads=1,
            dropout_rate=0.1,
            causal=True,
            sequence_length=16,
            save_attn=False,
            use_flash_attention=True,
        )
        input_shape = (1, 16, 128)
        # Check it runs correctly
        block(torch.randn(input_shape))

    @skipUnless(has_einops, "Requires einops")
    def test_causal(self):
        block = CrossAttentionBlock(
            hidden_size=128, num_heads=1, dropout_rate=0.1, causal=True, sequence_length=16, save_attn=True
        )
        input_shape = (1, 16, 128)
        block(torch.randn(input_shape))
        # check upper triangular part of the attention matrix is zero
        assert torch.triu(block.att_mat, diagonal=1).sum() == 0

    @skipUnless(has_einops, "Requires einops")
    def test_context_input(self):
        block = CrossAttentionBlock(
            hidden_size=128, num_heads=1, dropout_rate=0.1, causal=True, sequence_length=16, context_input_size=12
        )
        input_shape = (1, 16, 128)
        block(torch.randn(input_shape), context=torch.randn(1, 3, 12))

    @skipUnless(has_einops, "Requires einops")
    def test_context_wrong_input_size(self):
        block = CrossAttentionBlock(
            hidden_size=128, num_heads=1, dropout_rate=0.1, causal=True, sequence_length=16, context_input_size=12
        )
        input_shape = (1, 16, 128)
        with self.assertRaises(RuntimeError):
            block(torch.randn(input_shape), context=torch.randn(1, 3, 24))

    @skipUnless(has_einops, "Requires einops")
    def test_access_attn_matrix(self):
        # input format
        hidden_size = 128
        num_heads = 2
        dropout_rate = 0
        input_shape = (2, 256, hidden_size)

        # be  not able to access the matrix
        no_matrix_acess_blk = CrossAttentionBlock(
            hidden_size=hidden_size, num_heads=num_heads, dropout_rate=dropout_rate
        )
        no_matrix_acess_blk(torch.randn(input_shape))
        assert isinstance(no_matrix_acess_blk.att_mat, torch.Tensor)
        # no of elements is zero
        assert no_matrix_acess_blk.att_mat.nelement() == 0

        # be able to acess the attention matrix.
        matrix_acess_blk = CrossAttentionBlock(
            hidden_size=hidden_size, num_heads=num_heads, dropout_rate=dropout_rate, save_attn=True
        )
        matrix_acess_blk(torch.randn(input_shape))
        assert matrix_acess_blk.att_mat.shape == (input_shape[0], input_shape[0], input_shape[1], input_shape[1])

    @parameterized.expand([[True], [False]])
    @skipUnless(has_einops, "Requires einops")
    @SkipIfBeforePyTorchVersion((2, 0))
    def test_flash_attention(self, causal):
        input_param = {"hidden_size": 128, "num_heads": 1, "causal": causal, "sequence_length": 16 if causal else None}
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        block_w_flash_attention = CrossAttentionBlock(**input_param, use_flash_attention=True).to(device)
        block_wo_flash_attention = CrossAttentionBlock(**input_param, use_flash_attention=False).to(device)
        block_wo_flash_attention.load_state_dict(block_w_flash_attention.state_dict())
        test_data = torch.randn(1, 16, 128).to(device)

        out_1 = block_w_flash_attention(test_data)
        out_2 = block_wo_flash_attention(test_data)
        assert_allclose(out_1, out_2, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
