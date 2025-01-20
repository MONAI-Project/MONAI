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

from monai.transforms import RandScaleIntensity
from tests.util import TEST_NDARRAYS, NumpyImageTestCase2D, assert_allclose


class TestRandScaleIntensity(NumpyImageTestCase2D):
    @parameterized.expand([[p] for p in TEST_NDARRAYS])
    def test_value(self, p):
        scaler = RandScaleIntensity(factors=0.5, prob=1.0)
        scaler.set_random_state(seed=0)
        im = p(self.imt)
        result = scaler(im)
        np.random.seed(0)
        # simulate the randomize() of transform
        np.random.random()
        expected = p((self.imt * (1 + np.random.uniform(low=-0.5, high=0.5))).astype(np.float32))
        assert_allclose(result, p(expected), rtol=1e-7, atol=0, type_test="tensor")

    @parameterized.expand([[p] for p in TEST_NDARRAYS])
    def test_channel_wise(self, p):
        scaler = RandScaleIntensity(factors=0.5, channel_wise=True, prob=1.0)
        scaler.set_random_state(seed=0)
        im = p(self.imt)
        result = scaler(im)
        np.random.seed(0)
        # simulate the randomize() of transform
        np.random.random()
        channel_num = self.imt.shape[0]
        factor = [np.random.uniform(low=-0.5, high=0.5) for _ in range(channel_num)]
        expected = p(
            np.stack([np.asarray((self.imt[i]) * (1 + factor[i])) for i in range(channel_num)]).astype(np.float32)
        )
        assert_allclose(result, expected, atol=0, rtol=1e-5, type_test=False)


if __name__ == "__main__":
    unittest.main()
