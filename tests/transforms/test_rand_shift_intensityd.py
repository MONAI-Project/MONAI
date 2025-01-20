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

from monai.transforms import IntensityStatsd, RandShiftIntensityd
from monai.utils.enums import PostFix
from tests.util import TEST_NDARRAYS, NumpyImageTestCase2D, assert_allclose


class TestRandShiftIntensityd(NumpyImageTestCase2D):
    def test_value(self):
        key = "img"
        for p in TEST_NDARRAYS:
            shifter = RandShiftIntensityd(keys=[key], offsets=1.0, prob=1.0)
            shifter.set_random_state(seed=0)
            result = shifter({key: p(self.imt)})
            np.random.seed(0)
            # simulate the randomize() of transform
            np.random.random()
            expected = self.imt + np.random.uniform(low=-1.0, high=1.0)
            assert_allclose(result[key], p(expected), type_test="tensor")

    def test_factor(self):
        key = "img"
        stats = IntensityStatsd(keys=key, ops="max", key_prefix="orig")
        shifter = RandShiftIntensityd(keys=[key], offsets=1.0, factor_key=["orig_max"], prob=1.0)
        data = {key: self.imt, PostFix.meta(key): {"affine": None}}
        shifter.set_random_state(seed=0)
        result = shifter(stats(data))
        np.random.seed(0)
        # simulate the randomize() of transform
        np.random.random()
        expected = self.imt + np.random.uniform(low=-1.0, high=1.0) * np.nanmax(self.imt)
        np.testing.assert_allclose(result[key], expected)

    def test_channel_wise(self):
        key = "img"
        for p in TEST_NDARRAYS:
            scaler = RandShiftIntensityd(keys=[key], offsets=3.0, prob=1.0, channel_wise=True)
            scaler.set_random_state(seed=0)
            result = scaler({key: p(self.imt)})
            np.random.seed(0)
            # simulate the randomize function of transform
            np.random.random()
            channel_num = self.imt.shape[0]
            factor = [np.random.uniform(low=-3.0, high=3.0) for _ in range(channel_num)]
            expected = p(
                np.stack([np.asarray((self.imt[i]) + factor[i]) for i in range(channel_num)]).astype(np.float32)
            )
            assert_allclose(result[key], p(expected), type_test="tensor")


if __name__ == "__main__":
    unittest.main()
