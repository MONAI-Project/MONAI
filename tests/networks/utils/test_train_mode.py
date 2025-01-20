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

from monai.networks.utils import train_mode


class TestEvalMode(unittest.TestCase):

    def test_eval_mode(self):
        t = torch.rand(1, 1, 4, 4)
        p = torch.nn.Conv2d(1, 1, 3)
        p.eval()
        self.assertFalse(p.training)  # False
        with train_mode(p):
            self.assertTrue(p.training)  # True
            p(t).sum().backward()


if __name__ == "__main__":
    unittest.main()
