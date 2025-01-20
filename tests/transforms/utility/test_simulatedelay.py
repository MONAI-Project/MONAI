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

import time
import unittest

import numpy as np
from parameterized import parameterized

from monai.transforms.utility.array import SimulateDelay
from tests.util import NumpyImageTestCase2D


class TestSimulateDelay(NumpyImageTestCase2D):
    @parameterized.expand([(0.45,), (1,)])
    def test_value(self, delay_test_time: float):
        resize = SimulateDelay(delay_time=delay_test_time)
        start: float = time.time()
        _ = resize(self.imt[0])
        stop: float = time.time()
        measured_approximate: float = stop - start
        np.testing.assert_allclose(delay_test_time, measured_approximate, rtol=0.5)


if __name__ == "__main__":
    unittest.main()
