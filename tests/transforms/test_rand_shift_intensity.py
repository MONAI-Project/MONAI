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
from parameterized import parameterized

from monai.transforms import RandShiftIntensity
from tests.util import TEST_NDARRAYS, NumpyImageTestCase2D, assert_allclose


class TestRandShiftIntensity(NumpyImageTestCase2D):
    @parameterized.expand([[p] for p in TEST_NDARRAYS])
    def test_value(self, p):
        shifter = RandShiftIntensity(offsets=1.0, prob=1.0)
        shifter.set_random_state(seed=0)
        im = p(self.imt)
        result = shifter(im, factor=1.0)
        np.random.seed(0)
        # simulate the randomize() of transform
        np.random.random()
        expected = self.imt + np.random.uniform(low=-1.0, high=1.0)
        assert_allclose(result, expected, type_test="tensor")

    @parameterized.expand([[p] for p in TEST_NDARRAYS])
    def test_channel_wise(self, p):
        scaler = RandShiftIntensity(offsets=3.0, channel_wise=True, prob=1.0)
        scaler.set_random_state(seed=0)
        im = p(self.imt)
        result = scaler(im)
        np.random.seed(0)
        # simulate the randomize() of transform
        np.random.random()
        channel_num = self.imt.shape[0]
        factor = [np.random.uniform(low=-3.0, high=3.0) for _ in range(channel_num)]
        expected = p(np.stack([np.asarray((self.imt[i]) + factor[i]) for i in range(channel_num)]).astype(np.float32))
        assert_allclose(result, expected, atol=0, rtol=1e-5, type_test=False)


if __name__ == "__main__":
    unittest.main()
