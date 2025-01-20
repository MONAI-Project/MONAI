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

from monai.apps.pathology.utils import PathologyProbNMS

probs_map_2d = np.random.rand(100, 100).clip(0, 0.5)
probs_map_2d[33, 33] = 0.7
probs_map_2d[66, 66] = 0.9
expected_2d = [[0.9, 133, 133], [0.7, 67, 67]]
TEST_CASES_2D = [
    {"spatial_dims": 2, "prob_threshold": 0.5, "box_size": [10, 10]},
    {"resolution_level": 1},
    probs_map_2d,
    expected_2d,
]

probs_map_3d = torch.rand([50, 50, 50]).uniform_(0, 0.5)
probs_map_3d[25, 25, 25] = 0.7
probs_map_3d[45, 45, 45] = 0.9
expected_3d = [[0.9, 91, 91, 91], [0.7, 51, 51, 51]]
TEST_CASES_3D = [
    {"spatial_dims": 3, "prob_threshold": 0.5, "box_size": (10, 10, 10)},
    {"resolution_level": 1},
    probs_map_3d,
    expected_3d,
]


class TestPathologyProbNMS(unittest.TestCase):

    @parameterized.expand([TEST_CASES_2D, TEST_CASES_3D])
    def test_output(self, class_args, call_args, probs_map, expected):
        nms = PathologyProbNMS(**class_args)
        output = nms(probs_map, **call_args)
        np.testing.assert_allclose(output, expected)


if __name__ == "__main__":
    unittest.main()
