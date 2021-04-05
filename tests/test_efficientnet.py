# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
from typing import TYPE_CHECKING
from unittest import skipUnless

import torch
from parameterized import parameterized

from monai.networks import eval_mode
from monai.networks.nets import EfficientNetBN, get_efficientnet_image_size
from monai.utils import optional_import
from tests.utils import test_script_save

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
    return ["efficientnet-b{}".format(d) for d in range(8)]


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


def make_shape_cases(models, spatial_dims, batches, pretrained, in_channels=3, num_classes=1000):
    ret_test = []
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
                    }
                    ret_test.append(
                        [
                            kwargs,
                            (
                                batch,
                                in_channels,
                            )
                            + (get_expected_model_shape(model),) * spatial_dim,
                            (batch, num_classes),
                        ]
                    )
    return ret_test


# create list of selected models to speed up redundant tests
# only test the models B0, B3
SEL_MODELS = [get_model_names()[i] for i in [0, 3, 7]]

# pretrained=False cases
# 1D models are cheap so do test for all models in 1D
CASES_1D = make_shape_cases(
    models=get_model_names(), spatial_dims=[1], batches=[1, 4], pretrained=[False], in_channels=3, num_classes=1000
)

# 2D and 3D models are expensive so use selected models
CASES_2D = make_shape_cases(
    models=SEL_MODELS, spatial_dims=[2], batches=[1, 4], pretrained=[False], in_channels=3, num_classes=1000
)
CASES_3D = make_shape_cases(
    models=[SEL_MODELS[0]], spatial_dims=[3], batches=[1], pretrained=[False], in_channels=3, num_classes=1000
)

# pretrained=True cases
# tabby kitty test with pretrained model
# needs 'testing_data/kitty_test.jpg'
CASES_KITTY_TRAINED = [
    (
        {
            "model_name": "efficientnet-b0",
            "pretrained": True,
            "progress": False,
            "spatial_dims": 2,
            "in_channels": 3,
            "num_classes": 1000,
        },
        "testing_data/kitty_test.jpg",
        285,  # ~ Egyptian cat
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
        "testing_data/kitty_test.jpg",
        285,  # ~ Egyptian cat
    ),
]

# varying num_classes and in_channels
CASES_VARITAIONS = []

# change num_classes test
# 10 classes
# 2D
CASES_VARITAIONS.extend(
    make_shape_cases(
        models=SEL_MODELS, spatial_dims=[2], batches=[1], pretrained=[False, True], in_channels=3, num_classes=10
    )
)
# 3D
# CASES_VARITAIONS.extend(
#     make_shape_cases(
#         models=[SEL_MODELS[0]], spatial_dims=[3], batches=[1], pretrained=[False], in_channels=3, num_classes=10
#         )
# )

# change in_channels test
# 1 channel
# 2D
CASES_VARITAIONS.extend(
    make_shape_cases(
        models=SEL_MODELS, spatial_dims=[2], batches=[1], pretrained=[False, True], in_channels=1, num_classes=1000
    )
)
# 8 channel
# 2D
CASES_VARITAIONS.extend(
    make_shape_cases(
        models=SEL_MODELS, spatial_dims=[2], batches=[1], pretrained=[False, True], in_channels=8, num_classes=1000
    )
)
# 3D
# CASES_VARITAIONS.extend(
#     make_shape_cases(
#         models=[SEL_MODELS[0]], spatial_dims=[3], batches=[1], pretrained=[False], in_channels=1, num_classes=1000
#         )
# )


class TestEFFICIENTNET(unittest.TestCase):
    @parameterized.expand(CASES_1D + CASES_2D + CASES_3D + CASES_VARITAIONS)
    def test_shape(self, input_param, input_shape, expected_shape):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(input_param)
        net = EfficientNetBN(**input_param).to(device)
        with eval_mode(net):
            result = net(torch.randn(input_shape).to(device))
        self.assertEqual(result.shape, expected_shape)

    @parameterized.expand(CASES_KITTY_TRAINED)
    @skipUnless(has_torchvision, "Requires `torchvision` package.")
    @skipUnless(has_pil, "Requires `pillow` package.")
    def test_kitty_pretrained(self, input_param, image_path, expected_label):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Open image
        image_size = get_efficientnet_image_size(input_param["model_name"])
        img = PIL.Image.open("testdata/cat.jpeg")
        tfms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(image_size),
                torchvision.transforms.CenterCrop(image_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        img = tfms(img).unsqueeze(0).to(device)
        net = EfficientNetBN(**input_param).to(device)
        with eval_mode(net):
            result = net(img)
        pred_label = torch.argmax(result, dim=-1)
        self.assertEqual(pred_label, expected_label)

    def test_ill_arg(self):
        with self.assertRaises(AssertionError):
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
        net = EfficientNetBN(model_name="efficientnet-b0", spatial_dims=2, in_channels=3, num_classes=1000)
        net.set_swish(memory_efficient=False)  # at the moment custom memory efficient swish is not exportable with jit
        test_data = torch.randn(1, 3, 224, 224)
        test_script_save(net, test_data)


if __name__ == "__main__":
    unittest.main()
