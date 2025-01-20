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

from monai.transforms import VoteEnsemble
from tests.util import TEST_NDARRAYS, assert_allclose

TESTS = []
for p in TEST_NDARRAYS:
    # shape: [2, 1, 1]
    TESTS.append(
        [
            {"num_classes": None},
            [p(torch.tensor([[[1]], [[0]]])), p(torch.tensor([[[1]], [[0]]])), p(torch.tensor([[[0]], [[1]]]))],
            p(torch.tensor([[[1.0]], [[0.0]]])),
        ]
    )

    # shape: [1, 2, 1, 1]
    TESTS.append(
        [
            {"num_classes": None},
            p(
                torch.stack(
                    [torch.tensor([[[[1]], [[0]]]]), torch.tensor([[[[1]], [[0]]]]), torch.tensor([[[[0]], [[1]]]])]
                )
            ),
            p(torch.tensor([[[[1.0]], [[0.0]]]])),
        ]
    )

    # shape: [1, 2, 1]
    TESTS.append(
        [
            {"num_classes": 3},
            [p(torch.tensor([[[0], [2]]])), p(torch.tensor([[[0], [2]]])), p(torch.tensor([[[1], [1]]]))],
            p(torch.tensor([[[0], [2]]])),
        ]
    )

    # shape: [1, 2, 1]
    TESTS.append(
        [
            {"num_classes": 5},
            [p(torch.tensor([[[0], [2]]])), p(torch.tensor([[[0], [2]]])), p(torch.tensor([[[1], [1]]]))],
            p(torch.tensor([[[0], [2]]])),
        ]
    )

    # shape: [1]
    TESTS.append(
        [{"num_classes": 3}, [p(torch.tensor([2])), p(torch.tensor([2])), p(torch.tensor([1]))], p(torch.tensor([2]))]
    )

    # shape: 1
    TESTS.append([{"num_classes": 3}, [p(torch.tensor(2)), p(torch.tensor(2)), p(torch.tensor(1))], p(torch.tensor(2))])


class TestVoteEnsemble(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_value(self, input_param, img, expected_value):
        result = VoteEnsemble(**input_param)(img)
        if isinstance(img, torch.Tensor):
            self.assertIsInstance(result, torch.Tensor)
            self.assertEqual(result.device, img.device)
        assert_allclose(result, expected_value)


if __name__ == "__main__":
    unittest.main()
