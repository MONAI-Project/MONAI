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

from monai.transforms import CastToType, Pad
from monai.utils import NumpyPadMode, PytorchPadMode
from tests.util import SkipIfBeforePyTorchVersion


@SkipIfBeforePyTorchVersion((1, 10, 1))
class TestPadMode(unittest.TestCase):
    def test_pad(self):
        expected_shapes = {3: (1, 15, 10), 4: (1, 10, 6, 7)}
        for t in (float, int, np.uint8, np.int16, np.float32, bool):
            for d in ("cuda:0", "cpu") if torch.cuda.is_available() else ("cpu",):
                for s in ((1, 10, 10), (1, 5, 6, 7)):
                    for m in list(PytorchPadMode) + list(NumpyPadMode):
                        a = torch.rand(s)
                        to_pad = [(0, 0), (2, 3)] if len(s) == 3 else [(0, 0), (2, 3), (0, 0), (0, 0)]
                        out = Pad(to_pad=to_pad, mode=m)(CastToType(dtype=t)(a).to(d))
                        self.assertEqual(out.shape, expected_shapes[len(s)])


if __name__ == "__main__":
    unittest.main()
