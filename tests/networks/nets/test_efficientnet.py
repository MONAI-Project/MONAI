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

import os
import unittest
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import skipUnless

import torch
from parameterized import parameterized

from monai.networks import eval_mode
from monai.networks.nets import (
    BlockArgs,
    EfficientNetBN,
    EfficientNetBNFeatures,
    drop_connect,
    get_efficientnet_image_size,
)
from monai.utils import optional_import
from tests.util import skip_if_downloading_fails, skip_if_quick, test_pretrained_networks, test_script_save

TESTS_PATH = Path(__file__).parents[1]

if TYPE_CHECKING:
    import torchvision

    has_torchvision = True
else:
    torchvision, has_torchvision = optional_import("torchvision")

if TYPE_CHECKING:
    import PIL

    has_pil = True
else:
    PIL, has_pil = optional_import("PIL")


def get_model_names():
    return [f"efficientnet-b{d}" for d in range(8)]


def get_expected_model_shape(model_name):
    model_input_shapes = {
        "efficientnet-b0": 224,
        "efficientnet-b1": 240,
        "efficientnet-b2": 260,
        "efficientnet-b3": 300,
        "efficientnet-b4": 380,
        "efficientnet-b5": 456,
        "efficientnet-b6": 528,
        "efficientnet-b7": 600,
    }
    return model_input_shapes[model_name]


def get_block_args():
    # test string list
    return [
        "r1_k3_s11_e1_i32_o16_se0.25",
        "r2_k3_s22_e6_i16_o24_se0.25",
        "r2_k5_s22_e6_i24_o40_se0.25",
        "r3_k3_s22_e6_i40_o80_se0.25",
        "r3_k5_s11_e6_i80_o112_se0.25",
        "r4_k5_s22_e6_i112_o192_se0.25",
        "r1_k3_s11_e6_i192_o320_se0.25",
        "r1_k3_s11_e1_i32_o16_se0.25_noskip",
        "r2_k3_s22_e6_i16_o24_se0.25_noskip",
        "r2_k5_s22_e6_i24_o40_se0.25_noskip",
        "r3_k3_s22_e6_i40_o80_se0.25_noskip",
        "r3_k5_s11_e6_i80_o112_se0.25_noskip",
        "r4_k5_s22_e6_i112_o192_se0.25_noskip",
        "r1_k3_s11_e6_i192_o320_se0.25_noskip",
    ]


def make_shape_cases(
    models,
    spatial_dims,
    batches,
    pretrained,
    in_channels=3,
    num_classes=1000,
    norm=("batch", {"eps": 1e-3, "momentum": 0.01}),
):
    ret_tests = []
    for spatial_dim in spatial_dims:  # selected spatial_dims
        for batch in batches:  # check single batch as well as multiple batch input
            for model in models:  # selected models
                for is_pretrained in pretrained:  # pretrained or not pretrained
                    kwargs = {
                        "model_name": model,
                        "pretrained": is_pretrained,
                        "progress": False,
                        "spatial_dims": spatial_dim,
                        "in_channels": in_channels,
                        "num_classes": num_classes,
                        "norm": norm,
                    }
                    ret_tests.append(
                        [
                            kwargs,
                            (batch, in_channels) + (get_expected_model_shape(model),) * spatial_dim,
                            (batch, num_classes),
                        ]
                    )
    return ret_tests


# create list of selected models to speed up redundant tests
# only test the models B0, B3, B7
SEL_MODELS = [get_model_names()[i] for i in [0, 3, 7]]

# pretrained=False cases
# 1D models are cheap so do test for all models in 1D
CASES_1D = make_shape_cases(
    models=get_model_names(), spatial_dims=[1], batches=[1, 4], pretrained=[False], in_channels=3, num_classes=1000
)

# 2D and 3D models are expensive so use selected models
CASES_2D = make_shape_cases(
    models=SEL_MODELS,
    spatial_dims=[2],
    batches=[1, 4],
    pretrained=[False],
    in_channels=3,
    num_classes=1000,
    norm="instance",
)
CASES_3D = make_shape_cases(
    models=[SEL_MODELS[0]],
    spatial_dims=[3],
    batches=[1],
    pretrained=[False],
    in_channels=3,
    num_classes=1000,
    norm="batch",
)

# pretrained=True cases
# tabby kitty test with pretrained model
# needs 'testing_data/kitty_test.jpg'
# image from: https://commons.wikimedia.org/wiki/File:Tabby_cat_with_blue_eyes-3336579.jpg
CASES_KITTY_TRAINED = [
    (
        {
            "model_name": "efficientnet-b0",
            "pretrained": True,
            "progress": False,
            "spatial_dims": 2,
            "in_channels": 3,
            "num_classes": 1000,
            "norm": ("batch", {"eps": 1e-3, "momentum": 0.01}),
            "adv_prop": False,
        },
        os.path.join(TESTS_PATH, "testing_data", "kitty_test.jpg"),
        282,  # ~ tiger cat
    ),
    (
        {
            "model_name": "efficientnet-b3",
            "pretrained": True,
            "progress": False,
            "spatial_dims": 2,
            "in_channels": 3,
            "num_classes": 1000,
        },
        os.path.join(TESTS_PATH, "testing_data", "kitty_test.jpg"),
        282,  # ~ tiger cat
    ),
    (
        {
            "model_name": "efficientnet-b7",
            "pretrained": True,
            "progress": False,
            "spatial_dims": 2,
            "in_channels": 3,
            "num_classes": 1000,
        },
        os.path.join(TESTS_PATH, "testing_data", "kitty_test.jpg"),
        282,  # ~ tiger cat
    ),
]

# varying num_classes and in_channels
CASES_VARIATIONS = []

# change num_classes test
# 10 classes
# 2D
CASES_VARIATIONS.extend(
    make_shape_cases(
        models=SEL_MODELS, spatial_dims=[2], batches=[1], pretrained=[False, True], in_channels=3, num_classes=10
    )
)
# 3D
CASES_VARIATIONS.extend(
    make_shape_cases(
        models=[SEL_MODELS[0]], spatial_dims=[3], batches=[1], pretrained=[False], in_channels=3, num_classes=10
    )
)

# change in_channels test
# 1 channel
# 2D
CASES_VARIATIONS.extend(
    make_shape_cases(
        models=SEL_MODELS, spatial_dims=[2], batches=[1], pretrained=[False, True], in_channels=1, num_classes=1000
    )
)
# 8 channel
# 2D
CASES_VARIATIONS.extend(
    make_shape_cases(
        models=SEL_MODELS, spatial_dims=[2], batches=[1], pretrained=[False, True], in_channels=8, num_classes=1000
    )
)
# 3D
CASES_VARIATIONS.extend(
    make_shape_cases(
        models=[SEL_MODELS[0]], spatial_dims=[3], batches=[1], pretrained=[False], in_channels=1, num_classes=1000
    )
)

CASE_EXTRACT_FEATURES = [
    (
        {
            "model_name": "efficientnet-b8",
            "pretrained": True,
            "progress": False,
            "spatial_dims": 2,
            "in_channels": 2,
            "adv_prop": True,
        },
        [1, 2, 224, 224],
        ([1, 32, 112, 112], [1, 56, 56, 56], [1, 88, 28, 28], [1, 248, 14, 14], [1, 704, 7, 7]),
    )
]


class TestEFFICIENTNET(unittest.TestCase):
    @parameterized.expand(CASES_1D + CASES_2D + CASES_3D + CASES_VARIATIONS)
    def test_shape(self, input_param, input_shape, expected_shape):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        with skip_if_downloading_fails():
            net = EfficientNetBN(**input_param).to(device)

        # run inference with random tensor
        with eval_mode(net):
            result = net(torch.randn(input_shape).to(device))

        # check output shape
        self.assertEqual(result.shape, expected_shape)

    @parameterized.expand(CASES_1D + CASES_2D)
    def test_non_default_shapes(self, input_param, input_shape, expected_shape):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        with skip_if_downloading_fails():
            net = EfficientNetBN(**input_param).to(device)

        # override input shape with different variations
        num_dims = len(input_shape) - 2
        non_default_sizes = [128, 256, 512]
        for candidate_size in non_default_sizes:
            input_shape = input_shape[0:2] + (candidate_size,) * num_dims
            # run inference with random tensor
            with eval_mode(net):
                result = net(torch.randn(input_shape).to(device))

            # check output shape
            self.assertEqual(result.shape, expected_shape)

    @parameterized.expand(CASES_KITTY_TRAINED)
    @skip_if_quick
    @skipUnless(has_torchvision, "Requires `torchvision` package.")
    @skipUnless(has_pil, "Requires `pillow` package.")
    def test_kitty_pretrained(self, input_param, image_path, expected_label):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # open image
        image_size = get_efficientnet_image_size(input_param["model_name"])
        img = PIL.Image.open(image_path)

        # define ImageNet transforms
        tfms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(image_size),
                torchvision.transforms.CenterCrop(image_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        # preprocess and prepare image tensor
        img = tfms(img).unsqueeze(0).to(device)

        # initialize a pretrained model
        net = test_pretrained_networks(EfficientNetBN, input_param, device)

        # run inference
        with eval_mode(net):
            result = net(img)
        pred_label = torch.argmax(result, dim=-1)

        # check output label
        self.assertEqual(pred_label, expected_label)

    def test_drop_connect_layer(self):
        p_list = [float(d + 1) / 10.0 for d in range(9)]

        # testing 1D, 2D and 3D shape
        for rand_tensor_shape in [(512, 16, 4), (384, 16, 4, 4), (256, 16, 4, 4, 4)]:
            # test validation mode, out tensor == in tensor
            training = False
            for p in p_list:
                in_tensor = torch.rand(rand_tensor_shape) + 0.1
                out_tensor = drop_connect(in_tensor, p, training=training)
                self.assertTrue(torch.equal(out_tensor, in_tensor))

            # test training mode, sum((out tensor * (1.0 - p)) != in tensor)/out_tensor.size() == p
            # use tolerance of 0.175 to account for rounding errors due to finite set in/out
            tol = 0.175
            training = True
            for p in p_list:
                in_tensor = torch.rand(rand_tensor_shape) + 0.1
                out_tensor = drop_connect(in_tensor, p, training=training)

                p_calculated = 1.0 - torch.sum(torch.isclose(in_tensor, out_tensor * (1.0 - p))) / float(
                    in_tensor.numel()
                )
                p_calculated = p_calculated.cpu().numpy()

                self.assertTrue(abs(p_calculated - p) < tol)

    def test_block_args_decode(self):
        blocks_args_str = get_block_args()

        # convert strings to BlockArgs
        blocks_args = [BlockArgs.from_string(s) for s in blocks_args_str]
        # convert BlockArgs back to string
        blocks_args_str_convert = [s.to_string() for s in blocks_args]

        # check if converted strings match original
        [self.assertEqual(original, converted) for original, converted in zip(blocks_args_str, blocks_args_str_convert)]

    def test_ill_arg(self):
        with self.assertRaises(ValueError):
            # wrong spatial_dims
            EfficientNetBN(model_name="efficientnet-b0", spatial_dims=4)
            # wrong model_name
            EfficientNetBN(model_name="efficientnet-b10", spatial_dims=3)

    def test_func_get_efficientnet_input_shape(self):
        for model in get_model_names():
            result_shape = get_efficientnet_image_size(model_name=model)
            expected_shape = get_expected_model_shape(model)
            self.assertEqual(result_shape, expected_shape)

    def test_script(self):
        with skip_if_downloading_fails():
            net = EfficientNetBN(model_name="efficientnet-b0", spatial_dims=2, in_channels=3, num_classes=1000)
        net.set_swish(memory_efficient=False)  # at the moment custom memory efficient swish is not exportable with jit
        test_data = torch.randn(1, 3, 224, 224)
        test_script_save(net, test_data)


class TestExtractFeatures(unittest.TestCase):
    @parameterized.expand(CASE_EXTRACT_FEATURES)
    def test_shape(self, input_param, input_shape, expected_shapes):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        with skip_if_downloading_fails():
            net = EfficientNetBNFeatures(**input_param).to(device)

        # run inference with random tensor
        with eval_mode(net):
            features = net(torch.randn(input_shape).to(device))

        # check output shape
        self.assertEqual(len(features), len(expected_shapes))
        for feature, expected_shape in zip(features, expected_shapes):
            self.assertEqual(feature.shape, torch.Size(expected_shape))


if __name__ == "__main__":
    unittest.main()
