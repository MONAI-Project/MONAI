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
from unittest.case import skipUnless

import torch
from parameterized import parameterized

from monai.apps.vista3d.transforms import VistaPostTransformd, VistaPreTransformd
from monai.utils import min_version
from monai.utils.module import optional_import

measure, has_measure = optional_import("skimage.measure", "0.14.2", min_version)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


TEST_VISTA_PRETRANSFORM = [
    [
        {"label_prompt": [1], "points": [[0, 0, 0]], "point_labels": [1]},
        {"label_prompt": [1], "points": [[0, 0, 0]], "point_labels": [3]},
    ],
    [
        {"label_prompt": [2], "points": [[0, 0, 0]], "point_labels": [0]},
        {"label_prompt": [2], "points": [[0, 0, 0]], "point_labels": [2]},
    ],
    [
        {"label_prompt": [3], "points": [[0, 0, 0]], "point_labels": [0]},
        {"label_prompt": [4, 5], "points": [[0, 0, 0]], "point_labels": [0]},
    ],
    [
        {"label_prompt": [6], "points": [[0, 0, 0]], "point_labels": [0]},
        {"label_prompt": [7, 8], "points": [[0, 0, 0]], "point_labels": [0]},
    ],
]


pred1 = torch.zeros([2, 64, 64, 64])
pred1[0, :10, :10, :10] = 1
pred1[1, 20:30, 20:30, 20:30] = 1
output1 = torch.zeros([1, 64, 64, 64])
output1[:, :10, :10, :10] = 2
output1[:, 20:30, 20:30, 20:30] = 3

# -1 is needed since pred should be before sigmoid.
pred2 = torch.zeros([1, 64, 64, 64]) - 1
pred2[:, :10, :10, :10] = 1
pred2[:, 20:30, 20:30, 20:30] = 1
output2 = torch.zeros([1, 64, 64, 64])
output2[:, 20:30, 20:30, 20:30] = 1

TEST_VISTA_POSTTRANSFORM = [
    [{"pred": pred1.to(device), "label_prompt": torch.tensor([2, 3]).to(device)}, output1.to(device)],
    [
        {
            "pred": pred2.to(device),
            "points": torch.tensor([[25, 25, 25]]).to(device),
            "point_labels": torch.tensor([1]).to(device),
        },
        output2.to(device),
    ],
]


class TestVistaPreTransformd(unittest.TestCase):
    @parameterized.expand(TEST_VISTA_PRETRANSFORM)
    def test_result(self, input_data, expected):
        transform = VistaPreTransformd(keys="image", subclass={"3": [4, 5], "6": [7, 8]}, special_index=[1, 2])
        result = transform(input_data)
        self.assertEqual(result, expected)


@skipUnless(has_measure, "skimage.measure required")
class TestVistaPostTransformd(unittest.TestCase):
    @parameterized.expand(TEST_VISTA_POSTTRANSFORM)
    def test_result(self, input_data, expected):
        transform = VistaPostTransformd(keys="pred")
        result = transform(input_data)
        self.assertEqual((result["pred"] == expected).all(), True)


if __name__ == "__main__":
    unittest.main()
