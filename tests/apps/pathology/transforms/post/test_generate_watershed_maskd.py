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

from monai.apps.pathology.transforms.post.dictionary import GenerateWatershedMaskd
from monai.utils import min_version, optional_import
from tests.util import TEST_NDARRAYS

_, has_scipy = optional_import("scipy", "1.8.1", min_version)

EXCEPTION_TESTS = []
TESTS = []

np.random.RandomState(123)

for p in TEST_NDARRAYS:
    EXCEPTION_TESTS.append([{"keys": "img", "activation": "incorrect"}, ValueError])
    EXCEPTION_TESTS.append([{"keys": "img", "activation": 1}, ValueError])

    TESTS.append(
        [
            {"keys": "img", "mask_key": "mask", "activation": "softmax", "min_object_size": 0},
            p(
                [
                    [[0.5022, 0.3403, 0.9997], [0.8793, 0.5514, 0.2697], [0.6134, 0.6389, 0.0680]],
                    [[0.5000, 0.3400, 0.9900], [0.8900, 0.5600, 0.2700], [0.6100, 0.6300, 0.0600]],
                ]
            ),
            (1, 3, 3),
            [0, 1],
        ]
    )

    TESTS.append(
        [
            {"keys": "img", "mask_key": "mask", "activation": "sigmoid", "threshold": 0.5, "min_object_size": 0},
            p([[[0.5022, 0.3403, 0.9997], [0.8793, 0.5514, 0.2697], [-0.1134, -0.0389, -0.0680]]]),
            (1, 3, 3),
            [0, 1],
        ]
    )


@unittest.skipUnless(has_scipy, "Requires scipy library.")
class TestGenerateWatershedMaskd(unittest.TestCase):
    @parameterized.expand(EXCEPTION_TESTS)
    def test_value(self, arguments, exception_type):
        with self.assertRaises(exception_type):
            GenerateWatershedMaskd(**arguments)

    @parameterized.expand(TESTS)
    def test_value2(self, arguments, image, expected_shape, expected_value):
        result = GenerateWatershedMaskd(**arguments)({"img": image})
        self.assertEqual(result["mask"].shape, expected_shape)

        if isinstance(result["mask"], torch.Tensor):
            result["mask"] = result["mask"].cpu().numpy()
        self.assertEqual(np.unique(result["mask"]).tolist(), expected_value)


if __name__ == "__main__":
    unittest.main()
