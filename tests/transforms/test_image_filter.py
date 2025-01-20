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

from monai.data.meta_tensor import MetaTensor
from monai.networks.layers.simplelayers import GaussianFilter
from monai.transforms import ImageFilter, ImageFilterd, RandImageFilter, RandImageFilterd

EXPECTED_FILTERS = {
    "mean": torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]]).float(),
    "laplace": torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]).float(),
    "elliptical": torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]]).float(),
    "sharpen": torch.tensor([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]).float(),
}

SUPPORTED_FILTERS = ["mean", "laplace", "elliptical", "sobel", "sharpen", "median", "gauss", "savitzky_golay"]
SAMPLE_IMAGE_2D = torch.randn(1, 10, 10)
SAMPLE_IMAGE_3D = torch.randn(1, 10, 10, 10)
SAMPLE_DICT = {"image_2d": SAMPLE_IMAGE_2D, "image_3d": SAMPLE_IMAGE_3D}

# Sobel filter uses reflect pad as default which is not implemented for 3d in torch 1.8.1 or 1.9.1
ADDITIONAL_ARGUMENTS = {"order": 1, "sigma": 1, "padding_mode": "zeros"}


class TestModule(torch.nn.Module):
    __test__ = False  # indicate to pytest that this class is not intended for collection

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x + 1


class TestNotAModuleOrTransform:
    pass


class TestImageFilter(unittest.TestCase):

    @parameterized.expand(SUPPORTED_FILTERS)
    def test_init_from_string(self, filter_name):
        "Test init from string"
        _ = ImageFilter(filter_name, 3, **ADDITIONAL_ARGUMENTS)

    def test_init_raises(self):
        with self.assertRaises(Exception) as context:
            _ = ImageFilter("mean")
            self.assertTrue("`filter_size` must be specified when specifying filters by string." in str(context.output))
        with self.assertRaises(Exception) as context:
            _ = ImageFilter("mean")
            self.assertTrue("`filter_size` should be a single uneven integer." in str(context.output))
        with self.assertRaises(Exception) as context:
            _ = ImageFilter("gauss", 3)
            self.assertTrue("`filter='gauss', requires the additonal keyword argument `sigma`" in str(context.output))
        with self.assertRaises(Exception) as context:
            _ = ImageFilter("savitzky_golay", 3)
            self.assertTrue(
                "`filter='savitzky_golay', requires the additonal keyword argument `order`" in str(context.output)
            )

    def test_init_from_array(self):
        "Test init with custom filter and assert wrong filter shape throws an error"
        _ = ImageFilter(torch.ones(3, 3))
        _ = ImageFilter(torch.ones(3, 3, 3))
        _ = ImageFilter(np.ones((3, 3)))
        _ = ImageFilter(np.ones((3, 3, 3)))

        with self.assertRaises(Exception) as context:
            _ = ImageFilter(torch.ones(3, 3, 3, 3))
            self.assertTrue("Only 1D, 2D, and 3D filters are supported." in str(context.output))

    def test_init_from_module(self):
        filter = ImageFilter(TestModule())
        out = filter(torch.zeros(1, 3, 3, 3))
        torch.testing.assert_allclose(torch.ones(1, 3, 3, 3), out)

    def test_init_from_transform(self):
        _ = ImageFilter(GaussianFilter(3, sigma=2))

    def test_init_from_wrong_type_fails(self):
        with self.assertRaises(Exception) as context:
            _ = ImageFilter(TestNotAModuleOrTransform())
            self.assertTrue("<class 'type'> is not supported." in str(context.output))

    @parameterized.expand(EXPECTED_FILTERS.keys())
    def test_2d_filter_correctness(self, filter_name):
        "Test correctness of filters (2d only)"
        tfm = ImageFilter(filter_name, 3, **ADDITIONAL_ARGUMENTS)
        filter = tfm._get_filter_from_string(filter_name, size=3, ndim=2).filter.squeeze()
        torch.testing.assert_allclose(filter, EXPECTED_FILTERS[filter_name])

    @parameterized.expand(SUPPORTED_FILTERS)
    def test_call_2d(self, filter_name):
        "Text function `__call__` for 2d images"
        filter = ImageFilter(filter_name, 3, **ADDITIONAL_ARGUMENTS)
        out_tensor = filter(SAMPLE_IMAGE_2D)
        self.assertEqual(out_tensor.shape[1:], SAMPLE_IMAGE_2D.shape[1:])

    @parameterized.expand(SUPPORTED_FILTERS)
    def test_call_3d(self, filter_name):
        "Text function `__call__` for 3d images"
        filter = ImageFilter(filter_name, 3, **ADDITIONAL_ARGUMENTS)
        out_tensor = filter(SAMPLE_IMAGE_3D)
        self.assertEqual(out_tensor.shape[1:], SAMPLE_IMAGE_3D.shape[1:])

    def test_pass_applied_operations(self):
        "Test that applied operations are passed through"
        applied_operations = ["op1", "op2"]
        image = MetaTensor(SAMPLE_IMAGE_2D, applied_operations=applied_operations)
        filter = ImageFilter(SUPPORTED_FILTERS[0], 3, **ADDITIONAL_ARGUMENTS)
        out_tensor = filter(image)
        self.assertEqual(out_tensor.applied_operations, applied_operations)

    def test_pass_empty_metadata_dict(self):
        "Test that applied operations are passed through"
        image = MetaTensor(SAMPLE_IMAGE_2D, meta={})
        filter = ImageFilter(SUPPORTED_FILTERS[0], 3, **ADDITIONAL_ARGUMENTS)
        out_tensor = filter(image)
        self.assertTrue(isinstance(out_tensor, MetaTensor))

    def test_gaussian_filter_without_filter_size(self):
        "Test Gaussian filter without specifying filter_size"
        filter = ImageFilter("gauss", sigma=2)
        out_tensor = filter(SAMPLE_IMAGE_2D)
        self.assertEqual(out_tensor.shape[1:], SAMPLE_IMAGE_2D.shape[1:])


class TestImageFilterDict(unittest.TestCase):

    @parameterized.expand(SUPPORTED_FILTERS)
    def test_init_from_string_dict(self, filter_name):
        "Test init from string and assert an error is thrown if no size is passed"
        _ = ImageFilterd("image", filter_name, 3, **ADDITIONAL_ARGUMENTS)
        with self.assertRaises(Exception) as _:
            _ = ImageFilterd(self.image_key, filter_name)

    def test_init_from_array_dict(self):
        "Test init with custom filter and assert wrong filter shape throws an error"
        _ = ImageFilterd("image", torch.ones(3, 3))
        with self.assertRaises(Exception) as _:
            _ = ImageFilterd(self.image_key, torch.ones(3, 3, 3, 3))

    @parameterized.expand(SUPPORTED_FILTERS)
    def test_call_2d(self, filter_name):
        "Text function `__call__` for 2d images"
        filter = ImageFilterd("image_2d", filter_name, 3, **ADDITIONAL_ARGUMENTS)
        out_tensor = filter(SAMPLE_DICT)
        self.assertEqual(out_tensor["image_2d"].shape[1:], SAMPLE_IMAGE_2D.shape[1:])

    @parameterized.expand(SUPPORTED_FILTERS)
    def test_call_3d(self, filter_name):
        "Text function `__call__` for 3d images"
        filter = ImageFilterd("image_3d", filter_name, 3, **ADDITIONAL_ARGUMENTS)
        out_tensor = filter(SAMPLE_DICT)
        self.assertEqual(out_tensor["image_3d"].shape[1:], SAMPLE_IMAGE_3D.shape[1:])


class TestRandImageFilter(unittest.TestCase):

    @parameterized.expand(SUPPORTED_FILTERS)
    def test_init_from_string(self, filter_name):
        "Test init from string and assert an error is thrown if no size is passed"
        _ = RandImageFilter(filter_name, 3, **ADDITIONAL_ARGUMENTS)
        with self.assertRaises(Exception) as _:
            _ = RandImageFilter(filter_name)

    def test_init_from_array(self):
        "Test init with custom filter and assert wrong filter shape throws an error"
        _ = RandImageFilter(torch.ones(3, 3))
        with self.assertRaises(Exception) as _:
            _ = RandImageFilter(torch.ones(3, 3, 3, 3))

    @parameterized.expand(SUPPORTED_FILTERS)
    def test_call_2d_prob_1(self, filter_name):
        "Text function `__call__` for 2d images"
        filter = RandImageFilter(filter_name, 3, 1, **ADDITIONAL_ARGUMENTS)
        out_tensor = filter(SAMPLE_IMAGE_2D)
        self.assertEqual(out_tensor.shape[1:], SAMPLE_IMAGE_2D.shape[1:])

    @parameterized.expand(SUPPORTED_FILTERS)
    def test_call_3d_prob_1(self, filter_name):
        "Text function `__call__` for 3d images"
        filter = RandImageFilter(filter_name, 3, 1, **ADDITIONAL_ARGUMENTS)
        out_tensor = filter(SAMPLE_IMAGE_3D)
        self.assertEqual(out_tensor.shape[1:], SAMPLE_IMAGE_3D.shape[1:])

    @parameterized.expand(SUPPORTED_FILTERS)
    def test_call_2d_prob_0(self, filter_name):
        "Text function `__call__` for 2d images"
        filter = RandImageFilter(filter_name, 3, 0, **ADDITIONAL_ARGUMENTS)
        out_tensor = filter(SAMPLE_IMAGE_2D)
        torch.testing.assert_allclose(out_tensor, SAMPLE_IMAGE_2D)

    @parameterized.expand(SUPPORTED_FILTERS)
    def test_call_3d_prob_0(self, filter_name):
        "Text function `__call__` for 3d images"
        filter = RandImageFilter(filter_name, 3, 0, **ADDITIONAL_ARGUMENTS)
        out_tensor = filter(SAMPLE_IMAGE_3D)
        torch.testing.assert_allclose(out_tensor, SAMPLE_IMAGE_3D)


class TestRandImageFilterDict(unittest.TestCase):

    @parameterized.expand(SUPPORTED_FILTERS)
    def test_init_from_string_dict(self, filter_name):
        "Test init from string and assert an error is thrown if no size is passed"
        _ = RandImageFilterd("image", filter_name, 3, **ADDITIONAL_ARGUMENTS)
        with self.assertRaises(Exception) as _:
            _ = RandImageFilterd("image", filter_name)

    def test_init_from_array_dict(self):
        "Test init with custom filter and assert wrong filter shape throws an error"
        _ = RandImageFilterd("image", torch.ones(3, 3))
        with self.assertRaises(Exception) as _:
            _ = RandImageFilterd("image", torch.ones(3, 3, 3, 3))

    @parameterized.expand(SUPPORTED_FILTERS)
    def test_call_2d_prob_1(self, filter_name):
        filter = RandImageFilterd("image_2d", filter_name, 3, 1.0, **ADDITIONAL_ARGUMENTS)
        out_tensor = filter(SAMPLE_DICT)
        self.assertEqual(out_tensor["image_2d"].shape[1:], SAMPLE_IMAGE_2D.shape[1:])

    @parameterized.expand(SUPPORTED_FILTERS)
    def test_call_3d_prob_1(self, filter_name):
        filter = RandImageFilterd("image_3d", filter_name, 3, 1.0, **ADDITIONAL_ARGUMENTS)
        out_tensor = filter(SAMPLE_DICT)
        self.assertEqual(out_tensor["image_3d"].shape[1:], SAMPLE_IMAGE_3D.shape[1:])

    @parameterized.expand(SUPPORTED_FILTERS)
    def test_call_2d_prob_0(self, filter_name):
        filter = RandImageFilterd("image_2d", filter_name, 3, 0.0, **ADDITIONAL_ARGUMENTS)
        out_tensor = filter(SAMPLE_DICT)
        torch.testing.assert_allclose(out_tensor["image_2d"].shape[1:], SAMPLE_IMAGE_2D.shape[1:])

    @parameterized.expand(SUPPORTED_FILTERS)
    def test_call_3d_prob_0(self, filter_name):
        filter = RandImageFilterd("image_3d", filter_name, 3, 0.0, **ADDITIONAL_ARGUMENTS)
        out_tensor = filter(SAMPLE_DICT)
        torch.testing.assert_allclose(out_tensor["image_3d"].shape[1:], SAMPLE_IMAGE_3D.shape[1:])


if __name__ == "__main__":
    unittest.main()
