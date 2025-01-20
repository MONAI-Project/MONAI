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

from monai.transforms.transform import Randomizable


class RandTest(Randomizable):

    def randomize(self, data=None):
        pass


class TestRandomizable(unittest.TestCase):

    def test_default(self):
        inst = RandTest()
        r1 = inst.R.rand()
        self.assertTrue(isinstance(inst.R, np.random.RandomState))
        inst.set_random_state()
        r2 = inst.R.rand()
        self.assertNotAlmostEqual(r1, r2)

    def test_seed(self):
        inst = RandTest()
        inst.set_random_state(seed=123)
        self.assertAlmostEqual(inst.R.rand(), 0.69646918)
        inst.set_random_state(123)
        self.assertAlmostEqual(inst.R.rand(), 0.69646918)

    def test_state(self):
        inst = RandTest()
        inst_r = np.random.RandomState(123)
        inst.set_random_state(state=inst_r)
        self.assertAlmostEqual(inst.R.rand(), 0.69646918)


if __name__ == "__main__":
    unittest.main()
