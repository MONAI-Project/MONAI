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

from monai.transforms import MeanEnsemble
from tests.util import TEST_NDARRAYS, assert_allclose

TESTS = []
for p in TEST_NDARRAYS:
    TESTS.append([{"weights": None}, [p(torch.ones(2, 2, 2)), p(torch.ones(2, 2, 2)) + 2], p(torch.ones(2, 2, 2)) + 1])

    TESTS.append(
        [{"weights": None}, p(torch.stack([torch.ones(2, 2, 2), torch.ones(2, 2, 2) + 2])), p(torch.ones(2, 2, 2)) + 1]
    )

    TESTS.append(
        [{"weights": [1, 3]}, [p(torch.ones(2, 2, 2)), p(torch.ones(2, 2, 2)) + 2], p(torch.ones(2, 2, 2)) * 2.5]
    )

    TESTS.append(
        [
            {"weights": [[1, 3], [3, 1]]},
            [p(torch.ones(2, 2, 2)), p(torch.ones(2, 2, 2)) + 2],
            p(torch.ones(2, 2, 2) * torch.tensor([2.5, 1.5]).reshape(2, 1, 1)),
        ]
    )

    TESTS.append(
        [
            {"weights": np.array([[1, 3], [3, 1]])},
            [p(torch.ones(2, 2, 2)), p(torch.ones(2, 2, 2)) + 2],
            p(torch.ones(2, 2, 2) * torch.tensor([2.5, 1.5]).reshape(2, 1, 1)),
        ]
    )

    TESTS.append(
        [
            {"weights": torch.tensor([[[1, 3]], [[3, 1]]])},
            [p(torch.ones(2, 2, 2, 2)), p(torch.ones(2, 2, 2, 2)) + 2],
            p(torch.ones(2, 2, 2, 2) * torch.tensor([2.5, 1.5]).reshape(1, 2, 1, 1)),
        ]
    )


class TestMeanEnsemble(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_value(self, input_param, img, expected_value):
        result = MeanEnsemble(**input_param)(img)
        assert_allclose(result, expected_value)

    def test_cuda_value(self):
        img = torch.stack([torch.ones(2, 2, 2, 2), torch.ones(2, 2, 2, 2) + 2])
        expected_value = torch.ones(2, 2, 2, 2) * torch.tensor([2.5, 1.5]).reshape(1, 2, 1, 1)
        if torch.cuda.is_available():
            img = img.to(torch.device("cuda:0"))
            expected_value = expected_value.to(torch.device("cuda:0"))
        result = MeanEnsemble(torch.tensor([[[1, 3]], [[3, 1]]]))(img)
        assert_allclose(result, expected_value)


if __name__ == "__main__":
    unittest.main()
