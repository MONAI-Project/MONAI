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

from monai.transforms import convert_to_contiguous
from tests.util import assert_allclose


class TestToContiguous(unittest.TestCase):
    def test_contiguous_dict(self):
        tochange = np.moveaxis(np.zeros((2, 3, 4)), 0, -1)
        test_dict = {"test_key": [[1]], 0: np.array(0), 1: np.array([0]), "nested": {"nested": [tochange]}}
        output = convert_to_contiguous(test_dict)
        self.assertEqual(output["test_key"], [[1]])
        assert_allclose(output[0], np.array(0))
        assert_allclose(output[1], np.array([0]))
        self.assertTrue(output["nested"]["nested"][0].flags.c_contiguous)

    def test_contiguous_seq(self):
        tochange = torch.zeros(2, 3, 4).transpose(0, 1)
        test_seq = [[[1]], np.array(0), np.array([0]), torch.tensor(1.0), [[tochange]], "test_string", (1, 2, 3)]
        output = convert_to_contiguous(test_seq)
        self.assertEqual(output[0], [[1]])
        assert_allclose(output[1], np.array(0))
        assert_allclose(output[2], np.array([0]))
        assert_allclose(output[3], torch.tensor(1.0))
        self.assertTrue(output[4][0][0].is_contiguous())
        self.assertEqual(output[5], "test_string")
        self.assertEqual(output[6], (1, 2, 3))


if __name__ == "__main__":
    unittest.main()
