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

from monai.metrics import Cumulative
from tests.util import assert_allclose


class TestCumulative(unittest.TestCase):
    def test_single(self):
        c = Cumulative()
        c.extend([2, 3])
        c.append(1)
        assert_allclose(c.get_buffer(), torch.tensor([2, 3, 1]))

    def test_multi(self):
        c = Cumulative()
        c.extend([2, 3], [4, 6])
        c.append(1)
        assert_allclose(c.get_buffer()[0], torch.tensor([2, 3, 1]))
        assert_allclose(c.get_buffer()[1], torch.tensor([4, 6]))

        c.reset()
        c.append()
        c.extend()
        self.assertEqual(c.get_buffer(), [])
        c.get_buffer().append(1)
        self.assertEqual(c.get_buffer(), [])  # no in-place update for the buffer

        c.reset()

    def test_ill(self):
        c = Cumulative()
        with self.assertRaises(TypeError):
            c.extend(None)
        with self.assertRaises(TypeError):
            c.extend([])
        with self.assertRaises(TypeError):
            c.extend(1)
        with self.assertRaises(TypeError):
            c.append([])
            c.append([1, 2])
            c.get_buffer()
        with self.assertRaises(TypeError):
            c.append(None)
            c.get_buffer()


if __name__ == "__main__":
    unittest.main()
