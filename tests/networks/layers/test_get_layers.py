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

from parameterized import parameterized

from monai.networks.layers import get_act_layer, get_dropout_layer, get_norm_layer

TEST_CASE_NORM = [
    [{"name": ("group", {"num_groups": 1})}, "GroupNorm(1, 1, eps=1e-05, affine=True)"],
    [
        {"name": "instance", "spatial_dims": 2},
        "InstanceNorm2d(1, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)",
    ],
]

TEST_CASE_ACT = [
    [{"name": "swish"}, "Swish()"],
    [{"name": ("prelu", {"num_parameters": 1, "init": 0.25})}, "PReLU(num_parameters=1)"],
]

TEST_CASE_DROPOUT = [
    [{"name": "dropout"}, "Dropout(p=0.5, inplace=False)"],
    [{"name": ("alphadropout", {"p": 0.25})}, "AlphaDropout(p=0.25, inplace=False)"],
]


class TestGetLayers(unittest.TestCase):

    @parameterized.expand(TEST_CASE_NORM)
    def test_norm_layer(self, input_param, expected):
        layer = get_norm_layer(**input_param)
        self.assertEqual(f"{layer}", expected)

    @parameterized.expand(TEST_CASE_ACT)
    def test_act_layer(self, input_param, expected):
        layer = get_act_layer(**input_param)
        self.assertEqual(f"{layer}", expected)

    @parameterized.expand(TEST_CASE_DROPOUT)
    def test_dropout_layer(self, input_param, expected):
        layer = get_dropout_layer(**input_param)
        self.assertEqual(f"{layer}", expected)


class TestSuggestion(unittest.TestCase):

    def test_suggested(self):
        with self.assertRaisesRegex(ValueError, "did you mean 'GROUP'?"):
            get_norm_layer(name="grop", spatial_dims=2)


if __name__ == "__main__":
    unittest.main()
