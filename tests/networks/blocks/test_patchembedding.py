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
import torch.nn as nn
from parameterized import parameterized

from monai.networks import eval_mode
from monai.networks.blocks.patchembedding import PatchEmbed, PatchEmbeddingBlock
from monai.utils import optional_import
from tests.util import SkipIfBeforePyTorchVersion

einops, has_einops = optional_import("einops")

TEST_CASE_PATCHEMBEDDINGBLOCK = []
for dropout_rate in (0.5,):
    for in_channels in [1, 4]:
        for hidden_size in [96, 288]:
            for img_size in [32, 64]:
                for patch_size in [8, 16]:
                    for num_heads in [8, 12]:
                        for proj_type in ["conv", "perceptron"]:
                            for pos_embed_type in ["none", "learnable", "sincos"]:
                                # for classification in (False, True):  # TODO: add classification tests
                                for nd in (2, 3):
                                    test_case = [
                                        {
                                            "in_channels": in_channels,
                                            "img_size": (img_size,) * nd,
                                            "patch_size": (patch_size,) * nd,
                                            "hidden_size": hidden_size,
                                            "num_heads": num_heads,
                                            "proj_type": proj_type,
                                            "pos_embed_type": pos_embed_type,
                                            "dropout_rate": dropout_rate,
                                        },
                                        (2, in_channels, *([img_size] * nd)),
                                        (2, (img_size // patch_size) ** nd, hidden_size),
                                    ]
                                    if nd == 2:
                                        test_case[0]["spatial_dims"] = 2  # type: ignore
                                    TEST_CASE_PATCHEMBEDDINGBLOCK.append(test_case)

TEST_CASE_PATCHEMBED = []
for patch_size in [2]:
    for in_chans in [1, 4]:
        for img_size in [96]:
            for embed_dim in [6, 12]:
                for norm_layer in [nn.LayerNorm]:
                    for nd in [2, 3]:
                        test_case = [
                            {
                                "patch_size": (patch_size,) * nd,
                                "in_chans": in_chans,
                                "embed_dim": embed_dim,
                                "norm_layer": norm_layer,
                                "spatial_dims": nd,
                            },
                            (2, in_chans, *([img_size] * nd)),
                            (2, embed_dim, *([img_size // patch_size] * nd)),
                        ]
                        TEST_CASE_PATCHEMBED.append(test_case)


@SkipIfBeforePyTorchVersion((1, 11, 1))
class TestPatchEmbeddingBlock(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(4)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    @parameterized.expand(TEST_CASE_PATCHEMBEDDINGBLOCK)
    @skipUnless(has_einops, "Requires einops")
    def test_shape(self, input_param, input_shape, expected_shape):
        net = PatchEmbeddingBlock(**input_param)
        with eval_mode(net):
            result = net(torch.randn(input_shape))
            self.assertEqual(result.shape, expected_shape)

    def test_sincos_pos_embed(self):
        net = PatchEmbeddingBlock(
            in_channels=1,
            img_size=(32, 32, 32),
            patch_size=(8, 8, 8),
            hidden_size=96,
            num_heads=8,
            pos_embed_type="sincos",
            dropout_rate=0.5,
        )

        self.assertEqual(net.position_embeddings.requires_grad, False)

    def test_learnable_pos_embed(self):
        net = PatchEmbeddingBlock(
            in_channels=1,
            img_size=(32, 32, 32),
            patch_size=(8, 8, 8),
            hidden_size=96,
            num_heads=8,
            pos_embed_type="learnable",
            dropout_rate=0.5,
        )

        self.assertEqual(net.position_embeddings.requires_grad, True)

    def test_ill_arg(self):
        with self.assertRaises(ValueError):
            PatchEmbeddingBlock(
                in_channels=1,
                img_size=(128, 128, 128),
                patch_size=(16, 16, 16),
                hidden_size=128,
                num_heads=12,
                proj_type="conv",
                pos_embed_type="sincos",
                dropout_rate=5.0,
            )

        with self.assertRaises(ValueError):
            PatchEmbeddingBlock(
                in_channels=1,
                img_size=(32, 32, 32),
                patch_size=(64, 64, 64),
                hidden_size=512,
                num_heads=8,
                proj_type="perceptron",
                pos_embed_type="sincos",
                dropout_rate=0.3,
            )

        with self.assertRaises(ValueError):
            PatchEmbeddingBlock(
                in_channels=1,
                img_size=(96, 96, 96),
                patch_size=(8, 8, 8),
                hidden_size=512,
                num_heads=14,
                proj_type="conv",
                dropout_rate=0.3,
            )

        with self.assertRaises(ValueError):
            PatchEmbeddingBlock(
                in_channels=1,
                img_size=(97, 97, 97),
                patch_size=(4, 4, 4),
                hidden_size=768,
                num_heads=8,
                proj_type="perceptron",
                dropout_rate=0.3,
            )
        with self.assertRaises(ValueError):
            PatchEmbeddingBlock(
                in_channels=1,
                img_size=(97, 97, 97),
                patch_size=(4, 4, 4),
                hidden_size=768,
                num_heads=8,
                proj_type="perceptron",
                dropout_rate=0.3,
            )

        with self.assertRaises(ValueError):
            PatchEmbeddingBlock(
                in_channels=4,
                img_size=(96, 96, 96),
                patch_size=(16, 16, 16),
                hidden_size=768,
                num_heads=12,
                proj_type="perc",
                dropout_rate=0.3,
            )


class TestPatchEmbed(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(4)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    @parameterized.expand(TEST_CASE_PATCHEMBED)
    @skipUnless(has_einops, "Requires einops")
    def test_shape(self, input_param, input_shape, expected_shape):
        net = PatchEmbed(**input_param)
        with eval_mode(net):
            result = net(torch.randn(input_shape))
            self.assertEqual(result.shape, expected_shape)

    def test_ill_arg(self):
        with self.assertRaises(ValueError):
            PatchEmbed(patch_size=(2, 2, 2), in_chans=1, embed_dim=24, norm_layer=nn.LayerNorm, spatial_dims=5)


if __name__ == "__main__":
    unittest.main()
