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

from monai.apps.vista3d.sampler import sample_prompt_pairs

label = torch.zeros([1, 1, 64, 64, 64])
label[:, :, :10, :10, :10] = 1
label[:, :, 20:30, 20:30, 20:30] = 2
label[:, :, 30:40, 30:40, 30:40] = 3
label1 = torch.zeros([1, 1, 64, 64, 64])

TEST_VISTA_SAMPLE_PROMPT = [
    [
        {
            "labels": label,
            "label_set": [0, 1, 2, 3, 4],
            "max_prompt": 5,
            "max_foreprompt": 4,
            "max_backprompt": 1,
            "drop_label_prob": 0,
            "drop_point_prob": 0,
        },
        [4, 4, 4, 4],
    ],
    [
        {
            "labels": label,
            "label_set": [0, 1],
            "max_prompt": 5,
            "max_foreprompt": 4,
            "max_backprompt": 1,
            "drop_label_prob": 0,
            "drop_point_prob": 1,
        },
        [2, None, None, 2],
    ],
    [
        {
            "labels": label,
            "label_set": [0, 1, 2, 3, 4],
            "max_prompt": 5,
            "max_foreprompt": 4,
            "max_backprompt": 1,
            "drop_label_prob": 1,
            "drop_point_prob": 0,
        },
        [None, 3, 3, 3],
    ],
    [
        {
            "labels": label1,
            "label_set": [0, 1],
            "max_prompt": 5,
            "max_foreprompt": 4,
            "max_backprompt": 1,
            "drop_label_prob": 0,
            "drop_point_prob": 1,
        },
        [1, None, None, 1],
    ],
    [
        {
            "labels": label1,
            "label_set": [0, 1],
            "max_prompt": 5,
            "max_foreprompt": 4,
            "max_backprompt": 0,
            "drop_label_prob": 0,
            "drop_point_prob": 1,
        },
        [None, None, None, None],
    ],
]


class TestGeneratePrompt(unittest.TestCase):
    @parameterized.expand(TEST_VISTA_SAMPLE_PROMPT)
    def test_result(self, input_data, expected):
        output = sample_prompt_pairs(**input_data)
        result = [i.shape[0] if i is not None else None for i in output]
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
