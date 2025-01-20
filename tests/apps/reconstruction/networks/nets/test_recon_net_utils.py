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

from monai.apps.reconstruction.networks.nets.utils import (
    complex_normalize,
    divisible_pad_t,
    inverse_divisible_pad_t,
    reshape_batch_channel_to_channel_dim,
    reshape_channel_complex_to_last_dim,
    reshape_channel_to_batch_dim,
    reshape_complex_to_channel_dim,
    sensitivity_map_expand,
    sensitivity_map_reduce,
)
from tests.util import assert_allclose

# no need for checking devices, these functions don't change device format
# reshape test case
im_2d, im_3d = torch.ones([3, 4, 50, 70, 2]), torch.ones([3, 4, 50, 70, 80, 2])
TEST_RESHAPE = [(im_2d,), (im_3d,)]

# normalize test case
im_2d, im_3d = (torch.randint(0, 3, [3, 4, 50, 70]).float(), torch.randint(0, 3, [3, 4, 50, 70, 80]).float())
TEST_NORMALIZE = [(im_2d,), (im_3d,)]

# pad test case
im_2d, im_3d = torch.ones([3, 4, 50, 70]), torch.ones([3, 4, 50, 70, 80])
TEST_PAD = [(im_2d,), (im_3d,)]

# test case for sensitivity map expansion/reduction
ksp_2d, ksp_3d = torch.ones([3, 4, 50, 70, 2]), torch.ones([3, 4, 50, 70, 80, 2])
sens_2d, sens_3d = torch.ones([3, 4, 50, 70, 2]), torch.ones([3, 4, 50, 70, 80, 2])
TEST_SENS = [(ksp_2d, sens_2d), (ksp_3d, sens_3d)]


class TestReconNetUtils(unittest.TestCase):
    @parameterized.expand(TEST_RESHAPE)
    def test_reshape_channel_complex(self, test_data):
        result = reshape_complex_to_channel_dim(test_data)
        result = reshape_channel_complex_to_last_dim(result)
        self.assertEqual(result.shape, test_data.shape)

        result, batch_size = reshape_channel_to_batch_dim(test_data)
        result = reshape_batch_channel_to_channel_dim(result, batch_size)
        self.assertEqual(result.shape, test_data.shape)

    @parameterized.expand(TEST_NORMALIZE)
    def test_complex_normalize(self, test_data):
        result, mean, std = complex_normalize(test_data)
        result = result * std + mean
        self.assertLess((((result - test_data) ** 2).mean() ** 0.5).item(), 1e-5)

    @parameterized.expand(TEST_PAD)
    def test_pad(self, test_data):
        result, padding_sizes = divisible_pad_t(test_data, k=16)
        result = inverse_divisible_pad_t(result, padding_sizes)
        assert_allclose(result, test_data)

    @parameterized.expand(TEST_SENS)
    def test_sens_expand_reduce(self, test_data, sens):
        result = sensitivity_map_reduce(test_data, sens)
        result = sensitivity_map_expand(result, sens)
        self.assertEqual(result.shape, test_data.shape)


if __name__ == "__main__":
    unittest.main()
