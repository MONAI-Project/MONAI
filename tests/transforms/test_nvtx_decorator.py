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

import numpy as np
import torch
from parameterized import parameterized

from monai.transforms import (
    Compose,
    CuCIM,
    Flip,
    Flipd,
    OneOf,
    RandAdjustContrast,
    RandCuCIM,
    RandFlip,
    RandomizableTrait,
    Rotate90,
    ToCupy,
    ToNumpy,
    TorchVision,
    ToTensor,
    ToTensord,
)
from monai.utils import Range, optional_import
from tests.util import HAS_CUPY

_, has_nvtx = optional_import("torch._C._nvtx", descriptor="NVTX is not installed. Are you sure you have a CUDA build?")
_, has_tvt = optional_import("torchvision.transforms")
_, has_cut = optional_import("cucim.core.operations.expose.transform")

TEST_CASE_ARRAY_0 = [np.random.randn(3, 3)]
TEST_CASE_ARRAY_1 = [np.random.randn(3, 10, 10)]

TEST_CASE_DICT_0 = [{"image": np.random.randn(3, 3)}]
TEST_CASE_DICT_1 = [{"image": np.random.randn(3, 10, 10)}]

TEST_CASE_TORCH_0 = [torch.randn(3, 3)]
TEST_CASE_TORCH_1 = [torch.randn(3, 10, 10)]

TEST_CASE_WRAPPER = [np.random.randn(3, 10, 10)]

TEST_CASE_RECURSIVE_0 = [
    torch.randn(3, 3),
    Compose([ToNumpy(), Flip(), RandAdjustContrast(prob=0.0), RandFlip(prob=1.0), ToTensor()]),
]
TEST_CASE_RECURSIVE_1 = [
    torch.randn(3, 3),
    Compose([ToNumpy(), Flip(), Compose([RandAdjustContrast(prob=0.0), RandFlip(prob=1.0)]), ToTensor()]),
]
TEST_CASE_RECURSIVE_2 = [
    torch.randn(3, 3),
    Compose([ToNumpy(), Flip(), OneOf([RandAdjustContrast(prob=0.0), RandFlip(prob=1.0)], weights=[0, 1]), ToTensor()]),
]
TEST_CASE_RECURSIVE_LIST = [
    torch.randn(3, 3),
    [ToNumpy(), Flip(), RandAdjustContrast(prob=0.0), RandFlip(prob=1.0), ToTensor()],
]


@unittest.skipUnless(has_nvtx, "Required torch._C._nvtx for NVTX Range!")
class TestNVTXRangeDecorator(unittest.TestCase):
    @parameterized.expand([TEST_CASE_ARRAY_0, TEST_CASE_ARRAY_1])
    def test_tranform_array(self, input):
        transforms = Compose([Range("random flip")(Flip()), Range()(ToTensor())])
        # Apply transforms
        output = transforms(input)

        # Decorate with NVTX Range
        transforms1 = Range()(transforms)
        transforms2 = Range("Transforms2")(transforms)
        transforms3 = Range(name="Transforms3", methods="__call__")(transforms)

        # Apply transforms with Range
        output1 = transforms1(input)
        output2 = transforms2(input)
        output3 = transforms3(input)

        # Check the outputs
        self.assertIsInstance(output, torch.Tensor)
        self.assertIsInstance(output1, torch.Tensor)
        self.assertIsInstance(output2, torch.Tensor)
        self.assertIsInstance(output3, torch.Tensor)
        np.testing.assert_equal(output.numpy(), output1.numpy())
        np.testing.assert_equal(output.numpy(), output2.numpy())
        np.testing.assert_equal(output.numpy(), output3.numpy())

    @parameterized.expand([TEST_CASE_DICT_0, TEST_CASE_DICT_1])
    def test_tranform_dict(self, input):
        transforms = Compose([Range("random flip dict")(Flipd(keys="image")), Range()(ToTensord("image"))])
        # Apply transforms
        output = transforms(input)["image"]

        # Decorate with NVTX Range
        transforms1 = Range()(transforms)
        transforms2 = Range("Transforms2")(transforms)
        transforms3 = Range(name="Transforms3", methods="__call__")(transforms)

        # Apply transforms with Range
        output1 = transforms1(input)["image"]
        output2 = transforms2(input)["image"]
        output3 = transforms3(input)["image"]

        # Check the outputs
        self.assertIsInstance(output, torch.Tensor)
        self.assertIsInstance(output1, torch.Tensor)
        self.assertIsInstance(output2, torch.Tensor)
        self.assertIsInstance(output3, torch.Tensor)
        np.testing.assert_equal(output.numpy(), output1.numpy())
        np.testing.assert_equal(output.numpy(), output2.numpy())
        np.testing.assert_equal(output.numpy(), output3.numpy())

    @parameterized.expand([TEST_CASE_WRAPPER])
    @unittest.skipUnless(HAS_CUPY, "Requires CuPy.")
    @unittest.skipUnless(has_cut, "Requires cuCIM transforms.")
    @unittest.skipUnless(has_tvt, "Requires torchvision transforms.")
    def test_wrapper_tranforms(self, input):
        transform_list = [
            ToTensor(),
            TorchVision(name="RandomHorizontalFlip", p=1.0),
            ToCupy(),
            CuCIM(name="image_flip", spatial_axis=-1),
            RandCuCIM(name="rand_image_rotate_90", prob=1.0, max_k=1, spatial_axis=(-2, -1)),
        ]

        transforms = Compose(transform_list)
        transforms_range = Compose([Range()(t) for t in transform_list])

        # Apply transforms
        output = transforms(input)

        # Apply transforms with Range
        output_r = transforms_range(input)

        # Check the outputs
        np.testing.assert_equal(output.get(), output_r.get())

    @parameterized.expand([TEST_CASE_RECURSIVE_0, TEST_CASE_RECURSIVE_1, TEST_CASE_RECURSIVE_2])
    def test_recursive_tranforms(self, input, transforms):
        transforms_range = Range(name="Recursive Compose", recursive=True)(transforms)

        # Apply transforms
        output = transforms(input)

        # Apply transforms with Range
        output_r = transforms_range(input)

        # Check the outputs
        self.assertEqual(transforms.map_items, transforms_range.map_items)
        self.assertEqual(transforms.unpack_items, transforms_range.unpack_items)
        np.testing.assert_equal(output.numpy(), output_r.numpy())

    @parameterized.expand([TEST_CASE_RECURSIVE_LIST])
    def test_recursive_list_tranforms(self, input, transform_list):
        transforms_list_range = Range(recursive=True)(transform_list)

        # Apply transforms
        output = Compose(transform_list)(input)

        # Apply transforms with Range
        output_r = Compose(transforms_list_range)(input)

        # Check the outputs
        np.testing.assert_equal(output.numpy(), output_r.numpy())

    @parameterized.expand([TEST_CASE_ARRAY_1])
    def test_tranform_randomized(self, input):
        # Compose deterministic and randomized transforms
        transforms = Compose(
            [
                Range("flip")(Flip()),
                Rotate90(),
                Range()(RandAdjustContrast(prob=0.0)),
                Range("random flip")(RandFlip(prob=1.0)),
                ToTensor(),
            ]
        )
        # Apply transforms
        output = transforms(input)

        # Decorate with NVTX Range
        transforms1 = Range()(transforms)
        transforms2 = Range("Transforms2")(transforms)
        transforms3 = Range(name="Transforms3", methods="__call__")(transforms)

        # Apply transforms with Range
        output1 = transforms1(input)
        output2 = transforms2(input)
        output3 = transforms3(input)

        # Check if the outputs are equal
        self.assertIsInstance(output, torch.Tensor)
        self.assertIsInstance(output1, torch.Tensor)
        self.assertIsInstance(output2, torch.Tensor)
        self.assertIsInstance(output3, torch.Tensor)
        np.testing.assert_equal(output.numpy(), output1.numpy())
        np.testing.assert_equal(output.numpy(), output2.numpy())
        np.testing.assert_equal(output.numpy(), output3.numpy())

        # Check if the first randomized is RandAdjustContrast
        for tran in transforms.transforms:
            if isinstance(tran, RandomizableTrait):
                self.assertIsInstance(tran, RandAdjustContrast)
                break

    @parameterized.expand([TEST_CASE_TORCH_0, TEST_CASE_TORCH_1])
    def test_network(self, input):
        # Create a network
        model = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Sigmoid())

        # Forward
        output = model(input)

        # Decorate with NVTX Range
        model1 = Range()(model)
        model2 = Range("Model2")(model)
        model3 = Range(name="Model3", methods="forward")(model)

        # Forward with Range
        output1 = model1(input)
        output2 = model2(input)
        output3 = model3(input)

        # Check the outputs
        self.assertIsInstance(output, torch.Tensor)
        self.assertIsInstance(output1, torch.Tensor)
        self.assertIsInstance(output2, torch.Tensor)
        self.assertIsInstance(output3, torch.Tensor)
        np.testing.assert_equal(output.numpy(), output1.numpy())
        np.testing.assert_equal(output.numpy(), output2.numpy())
        np.testing.assert_equal(output.numpy(), output3.numpy())

    @parameterized.expand([TEST_CASE_TORCH_0, TEST_CASE_TORCH_1])
    def test_loss(self, input):
        # Create a network and loss
        model = torch.nn.Sigmoid()
        loss = torch.nn.BCELoss()
        pred = model(input)
        target = torch.empty_like(input).random_(2)

        # Loss evaluation
        output = loss(pred, target)

        # Decorate with NVTX Range
        loss1 = Range()(loss)
        loss2 = Range("Loss2")(loss)
        loss3 = Range(name="Loss3", methods="forward")(loss)

        # Loss evaluation with Range
        output1 = loss1(pred, target)
        output2 = loss2(pred, target)
        output3 = loss3(pred, target)

        # Check the outputs
        self.assertIsInstance(output, torch.Tensor)
        self.assertIsInstance(output1, torch.Tensor)
        self.assertIsInstance(output2, torch.Tensor)
        self.assertIsInstance(output3, torch.Tensor)
        np.testing.assert_equal(output.numpy(), output1.numpy())
        np.testing.assert_equal(output.numpy(), output2.numpy())
        np.testing.assert_equal(output.numpy(), output3.numpy())

    def test_context_manager(self):
        model = torch.nn.Sigmoid()
        loss = torch.nn.BCELoss()

        with Range():
            input = torch.randn(3, requires_grad=True)
            target = torch.empty(3).random_(2)

        with Range("Model"):
            output = loss(model(input), target)
            output.backward()


if __name__ == "__main__":
    unittest.main()
