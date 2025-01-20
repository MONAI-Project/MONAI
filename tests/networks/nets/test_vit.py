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
from monai.networks.nets.vit import ViT
from tests.util import SkipIfBeforePyTorchVersion, skip_if_quick, test_script_save

TEST_CASE_Vit = []
for dropout_rate in [0.6]:
    for in_channels in [4]:
        for hidden_size in [768]:
            for img_size in [96, 128]:
                for patch_size in [16]:
                    for num_heads in [12]:
                        for mlp_dim in [3072]:
                            for num_layers in [4]:
                                for num_classes in [8]:
                                    for proj_type in ["conv", "perceptron"]:
                                        for classification in [False, True]:
                                            for nd in (2, 3):
                                                test_case = [
                                                    {
                                                        "in_channels": in_channels,
                                                        "img_size": (img_size,) * nd,
                                                        "patch_size": (patch_size,) * nd,
                                                        "hidden_size": hidden_size,
                                                        "mlp_dim": mlp_dim,
                                                        "num_layers": num_layers,
                                                        "num_heads": num_heads,
                                                        "proj_type": proj_type,
                                                        "classification": classification,
                                                        "num_classes": num_classes,
                                                        "dropout_rate": dropout_rate,
                                                    },
                                                    (2, in_channels, *([img_size] * nd)),
                                                    (2, (img_size // patch_size) ** nd, hidden_size),
                                                ]
                                                if nd == 2:
                                                    test_case[0]["spatial_dims"] = 2  # type: ignore
                                                    if classification:
                                                        test_case[0]["post_activation"] = False  # type: ignore
                                                if test_case[0]["classification"]:  # type: ignore
                                                    test_case[2] = (2, test_case[0]["num_classes"])  # type: ignore
                                                TEST_CASE_Vit.append(test_case)


@skip_if_quick
class TestViT(unittest.TestCase):
    @parameterized.expand(TEST_CASE_Vit)
    def test_shape(self, input_param, input_shape, expected_shape):
        net = ViT(**input_param)
        with eval_mode(net):
            result, _ = net(torch.randn(input_shape))
            self.assertEqual(result.shape, expected_shape)

    @parameterized.expand(
        [
            (1, (128, 128, 128), (16, 16, 16), 128, 3072, 12, 12, "conv", False, 5.0),
            (1, (32, 32, 32), (64, 64, 64), 512, 3072, 12, 8, "perceptron", False, 0.3),
            (1, (96, 96, 96), (8, 8, 8), 512, 3072, 12, 14, "conv", False, 0.3),
            (1, (97, 97, 97), (4, 4, 4), 768, 3072, 12, 8, "perceptron", True, 0.3),
            (4, (96, 96, 96), (16, 16, 16), 768, 3072, 12, 12, "perc", False, 0.3),
        ]
    )
    def test_ill_arg(
        self,
        in_channels,
        img_size,
        patch_size,
        hidden_size,
        mlp_dim,
        num_layers,
        num_heads,
        proj_type,
        classification,
        dropout_rate,
    ):
        with self.assertRaises(ValueError):
            ViT(
                in_channels=in_channels,
                img_size=img_size,
                patch_size=patch_size,
                hidden_size=hidden_size,
                mlp_dim=mlp_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                proj_type=proj_type,
                classification=classification,
                dropout_rate=dropout_rate,
            )

    @parameterized.expand(TEST_CASE_Vit[:1])
    @SkipIfBeforePyTorchVersion((2, 0))
    def test_script(self, input_param, input_shape, _):
        net = ViT(**(input_param))
        net.eval()
        with torch.no_grad():
            torch.jit.script(net)

        test_data = torch.randn(input_shape)
        test_script_save(net, test_data)

    def test_access_attn_matrix(self):
        # input format
        in_channels = 1
        img_size = (96, 96, 96)
        patch_size = (16, 16, 16)
        in_shape = (1, in_channels, img_size[0], img_size[1], img_size[2])

        # no data in the matrix
        no_matrix_acess_blk = ViT(in_channels=in_channels, img_size=img_size, patch_size=patch_size)
        no_matrix_acess_blk(torch.randn(in_shape))
        assert isinstance(no_matrix_acess_blk.blocks[0].attn.att_mat, torch.Tensor)
        # no of elements is zero
        assert no_matrix_acess_blk.blocks[0].attn.att_mat.nelement() == 0

        # be able to acess the attention matrix
        matrix_acess_blk = ViT(in_channels=in_channels, img_size=img_size, patch_size=patch_size, save_attn=True)
        matrix_acess_blk(torch.randn(in_shape))
        assert matrix_acess_blk.blocks[0].attn.att_mat.shape == (in_shape[0], 12, 216, 216)


if __name__ == "__main__":
    unittest.main()
