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
import torch.nn.functional as F
from parameterized import parameterized

from monai.transforms.utils import get_unique_labels
from monai.transforms.utils_pytorch_numpy_unification import moveaxis
from tests.util import TEST_NDARRAYS

grid_raw = [[0, 0, 0], [0, 0, 1], [2, 2, 3], [5, 5, 6], [3, 6, 2], [5, 6, 6]]
grid = torch.Tensor(grid_raw).unsqueeze(0).to(torch.int64)
grid_onehot = moveaxis(F.one_hot(grid)[0], -1, 0)

TESTS = []
for p in TEST_NDARRAYS:
    for o_h in (False, True):
        im = grid_onehot if o_h else grid
        TESTS.append([dict(img=p(im), is_onehot=o_h), {0, 1, 2, 3, 5, 6}])
        TESTS.append([dict(img=p(im), is_onehot=o_h, discard=0), {1, 2, 3, 5, 6}])
        TESTS.append([dict(img=p(im), is_onehot=o_h, discard=[1, 2]), {0, 3, 5, 6}])


class TestGetUniqueLabels(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_correct_results(self, args, expected):
        result = get_unique_labels(**args)
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
