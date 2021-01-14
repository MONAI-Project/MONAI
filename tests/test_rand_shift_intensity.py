# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np

from monai.transforms import RandShiftIntensity
from tests.utils import NumpyImageTestCase2D


class TestRandShiftIntensity(NumpyImageTestCase2D):
    def test_value(self):
        shifter = RandShiftIntensity(offsets=1.0, prob=1.0)
        shifter.set_random_state(seed=0)
        result = shifter(self.imt)
        np.random.seed(0)
        expected = self.imt + np.random.uniform(low=-1.0, high=1.0)
        np.testing.assert_allclose(result, expected)


if __name__ == "__main__":
    unittest.main()
