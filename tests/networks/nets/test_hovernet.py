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

from monai.networks import eval_mode, train_mode
from monai.networks.nets import HoVerNet
from monai.networks.nets.hovernet import _DenseLayerDecoder
from tests.util import test_script_save

device = "cuda" if torch.cuda.is_available() else "cpu"

TEST_CASE_0 = [  # fast mode
    {"out_classes": 5, "mode": HoVerNet.Mode.FAST},
    (1, 3, 256, 256),
    {HoVerNet.Branch.NP: (1, 2, 164, 164), HoVerNet.Branch.NC: (1, 5, 164, 164), HoVerNet.Branch.HV: (1, 2, 164, 164)},
]

TEST_CASE_1 = [  # original mode
    {"out_classes": 6, "mode": HoVerNet.Mode.ORIGINAL},
    (1, 3, 270, 270),
    {HoVerNet.Branch.NP: (1, 2, 80, 80), HoVerNet.Branch.NC: (1, 6, 80, 80), HoVerNet.Branch.HV: (1, 2, 80, 80)},
]

TEST_CASE_2 = [  # dropout
    {"mode": HoVerNet.Mode.FAST, "dropout_prob": 0.5, "out_classes": 3},
    (1, 3, 256, 256),
    {HoVerNet.Branch.NP: (1, 2, 164, 164), HoVerNet.Branch.NC: (1, 3, 164, 164), HoVerNet.Branch.HV: (1, 2, 164, 164)},
]

TEST_CASE_3 = [  # np_out_channels
    {"mode": HoVerNet.Mode.FAST, "np_out_channels": 3, "out_classes": 2},
    (1, 3, 256, 256),
    {HoVerNet.Branch.NP: (1, 3, 164, 164), HoVerNet.Branch.NC: (1, 2, 164, 164), HoVerNet.Branch.HV: (1, 2, 164, 164)},
]

CASES = [TEST_CASE_0, TEST_CASE_1, TEST_CASE_2, TEST_CASE_3]

ILL_CASES = [
    [{"out_classes": 6, "mode": 3}],
    [{"out_classes": 6, "mode": "Wrong"}],
    [{"out_classes": 1000, "mode": HoVerNet.Mode.ORIGINAL}],
    [{"out_classes": 1, "mode": HoVerNet.Mode.ORIGINAL}],
    [{"out_classes": 6, "mode": HoVerNet.Mode.ORIGINAL, "dropout_prob": 100}],
]


def check_branch(branch, mode):
    if mode == HoVerNet.Mode.ORIGINAL:
        ksize = 5
    else:
        ksize = 3

    if branch.decoderblock1.conva.kernel_size != (ksize, ksize):
        return True
    if branch.decoderblock1.convf.kernel_size != (1, 1):
        return True
    for block in branch.decoderblock1:
        if isinstance(block, _DenseLayerDecoder):
            if block.layers.conv1.kernel_size != (1, 1) or block.layers.conv2.kernel_size != (ksize, ksize):
                return True

    if branch.decoderblock2.conva.kernel_size != (ksize, ksize):
        return True
    if branch.decoderblock2.convf.kernel_size != (1, 1):
        return True

    for block in branch.decoderblock2:
        if isinstance(block, _DenseLayerDecoder):
            if block.layers.conv1.kernel_size != (1, 1) or block.layers.conv2.kernel_size != (ksize, ksize):
                return True

    return False


def check_output(out_block, mode):
    if mode == HoVerNet.Mode.ORIGINAL:
        ksize = 5
    else:
        ksize = 3

    if out_block.decoderblock3.conva.kernel_size != (ksize, ksize) or out_block.decoderblock3.conva.stride != (1, 1):
        return True
    if out_block.decoderblock4.conv.kernel_size != (1, 1) or out_block.decoderblock4.conv.stride != (1, 1):
        return True

    return False


def check_kernels(net, mode):
    # Check the Encoder blocks
    for layer_num, res_block in enumerate(net.res_blocks):
        for inner_num, layer in enumerate(res_block.layers):
            if layer_num > 0 and inner_num == 0:
                sz = 2
            else:
                sz = 1

            if (
                layer.layers.conv1.kernel_size != (1, 1)
                or layer.layers.conv2.kernel_size != (3, 3)
                or layer.layers.conv3.kernel_size != (1, 1)
            ):
                return True

            if (
                layer.layers.conv1.stride != (1, 1)
                or layer.layers.conv2.stride != (sz, sz)
                or layer.layers.conv3.stride != (1, 1)
            ):
                return True

        sz2 = 1
        if layer_num > 0:
            sz2 = 2
        if res_block.shortcut.kernel_size != (1, 1) or res_block.shortcut.stride != (sz2, sz2):
            return True

    if net.bottleneck.conv_bottleneck.kernel_size != (1, 1) or net.bottleneck.conv_bottleneck.stride != (1, 1):
        return True

    # Check HV Branch
    if check_branch(net.horizontal_vertical.decoder_blocks, mode):
        return True
    if check_output(net.horizontal_vertical.output_features, mode):
        return True

    # Check NP Branch
    if check_branch(net.nucleus_prediction.decoder_blocks, mode):
        return True
    if check_output(net.nucleus_prediction.output_features, mode):
        return True

    # Check NC Branch
    if check_branch(net.type_prediction.decoder_blocks, mode):
        return True
    if check_output(net.type_prediction.output_features, mode):
        return True

    return False


class TestHoverNet(unittest.TestCase):
    @parameterized.expand(CASES)
    def test_shape(self, input_param, input_shape, expected_shapes):
        input_param["decoder_padding"] = False
        net = HoVerNet(**input_param).to(device)
        with eval_mode(net):
            result = net.forward(torch.randn(input_shape).to(device))
            for item in result:
                self.assertEqual(result[item].shape, expected_shapes[item])

    @parameterized.expand(CASES)
    def test_decoder_padding_shape(self, input_param, input_shape, expected_shapes):
        if input_param["mode"] == HoVerNet.Mode.FAST:
            input_param["decoder_padding"] = True
            net = HoVerNet(**input_param).to(device)
            with eval_mode(net):
                result = net.forward(torch.randn(input_shape).to(device))
                for item in result:
                    expected_shape = expected_shapes[item]
                    padding_expected_shape = list(expected_shape)
                    padding_expected_shape[2:] = input_shape[2:]
                    self.assertEqual(result[item].shape, tuple(padding_expected_shape))
        else:
            pass

    def test_script(self):
        for padding_flag in [True, False]:
            net = HoVerNet(mode=HoVerNet.Mode.FAST, decoder_padding=padding_flag)
        test_data = torch.randn(1, 3, 256, 256)
        test_script_save(net, test_data)

    def test_ill_input_shape(self):
        net = HoVerNet(mode=HoVerNet.Mode.FAST)
        with eval_mode(net):
            with self.assertRaises(ValueError):
                net.forward(torch.randn(1, 3, 270, 260))

    def test_kernels_strides(self):
        net = HoVerNet(mode=HoVerNet.Mode.FAST, out_classes=2)
        with eval_mode(net):
            self.assertEqual(check_kernels(net, HoVerNet.Mode.FAST), False)

        net = HoVerNet(mode=HoVerNet.Mode.ORIGINAL, out_classes=2)
        with eval_mode(net):
            self.assertEqual(check_kernels(net, HoVerNet.Mode.ORIGINAL), False)

    def test_freeze_encoder(self):
        net = HoVerNet(mode=HoVerNet.Mode.FAST, freeze_encoder=True)
        with train_mode(net):
            for _, param in net.res_blocks[1:].named_parameters():
                self.assertFalse(param.requires_grad)
            for name, param in net.res_blocks[0].named_parameters():
                if param.requires_grad is True:
                    self.assertTrue("bna_block" or "shortcut" in name)

    @parameterized.expand(ILL_CASES)
    def test_ill_input_hyper_params(self, input_param):
        with self.assertRaises(ValueError):
            _ = HoVerNet(**input_param)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
