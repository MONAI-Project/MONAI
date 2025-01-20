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

from monai.utils import unsqueeze_left, unsqueeze_right

RIGHT_CASES = [
    (np.random.rand(3, 4).astype(np.float32), 5, (3, 4, 1, 1, 1)),
    (torch.rand(3, 4).type(torch.float32), 5, (3, 4, 1, 1, 1)),
    (np.random.rand(3, 4).astype(np.float64), 5, (3, 4, 1, 1, 1)),
    (torch.rand(3, 4).type(torch.float64), 5, (3, 4, 1, 1, 1)),
    (np.random.rand(3, 4).astype(np.int32), 5, (3, 4, 1, 1, 1)),
    (torch.rand(3, 4).type(torch.int32), 5, (3, 4, 1, 1, 1)),
]

LEFT_CASES = [
    (np.random.rand(3, 4).astype(np.float32), 5, (1, 1, 1, 3, 4)),
    (torch.rand(3, 4).type(torch.float32), 5, (1, 1, 1, 3, 4)),
    (np.random.rand(3, 4).astype(np.float64), 5, (1, 1, 1, 3, 4)),
    (torch.rand(3, 4).type(torch.float64), 5, (1, 1, 1, 3, 4)),
    (np.random.rand(3, 4).astype(np.int32), 5, (1, 1, 1, 3, 4)),
    (torch.rand(3, 4).type(torch.int32), 5, (1, 1, 1, 3, 4)),
]
ALL_CASES = [
    (np.random.rand(3, 4), 2, (3, 4)),
    (np.random.rand(3, 4), 0, (3, 4)),
    (np.random.rand(3, 4), -1, (3, 4)),
    (np.array(3), 4, (1, 1, 1, 1)),
    (np.array(3), 0, ()),
    (np.random.rand(3, 4).astype(np.int32), 2, (3, 4)),
    (np.random.rand(3, 4).astype(np.int32), 0, (3, 4)),
    (np.random.rand(3, 4).astype(np.int32), -1, (3, 4)),
    (np.array(3).astype(np.int32), 4, (1, 1, 1, 1)),
    (np.array(3).astype(np.int32), 0, ()),
    (torch.rand(3, 4), 2, (3, 4)),
    (torch.rand(3, 4), 0, (3, 4)),
    (torch.rand(3, 4), -1, (3, 4)),
    (torch.tensor(3), 4, (1, 1, 1, 1)),
    (torch.tensor(3), 0, ()),
    (torch.rand(3, 4).type(torch.int32), 2, (3, 4)),
    (torch.rand(3, 4).type(torch.int32), 0, (3, 4)),
    (torch.rand(3, 4).type(torch.int32), -1, (3, 4)),
    (torch.tensor(3).type(torch.int32), 4, (1, 1, 1, 1)),
    (torch.tensor(3).type(torch.int32), 0, ()),
]


class TestUnsqueeze(unittest.TestCase):

    @parameterized.expand(RIGHT_CASES + ALL_CASES)
    def test_unsqueeze_right(self, arr, ndim, shape):
        self.assertEqual(unsqueeze_right(arr, ndim).shape, shape)

    @parameterized.expand(LEFT_CASES + ALL_CASES)
    def test_unsqueeze_left(self, arr, ndim, shape):
        self.assertEqual(unsqueeze_left(arr, ndim).shape, shape)
