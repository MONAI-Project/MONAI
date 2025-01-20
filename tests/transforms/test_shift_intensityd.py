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

from monai.transforms import IntensityStatsd, ShiftIntensityd
from monai.utils.enums import PostFix
from tests.util import TEST_NDARRAYS, NumpyImageTestCase2D, assert_allclose


class TestShiftIntensityd(NumpyImageTestCase2D):
    def test_value(self):
        key = "img"
        for p in TEST_NDARRAYS:
            shifter = ShiftIntensityd(keys=[key], offset=1.0)
            result = shifter({key: p(self.imt)})
            expected = self.imt + 1.0
            assert_allclose(result[key], p(expected), type_test="tensor")

    def test_factor(self):
        key = "img"
        stats = IntensityStatsd(keys=key, ops="max", key_prefix="orig")
        shifter = ShiftIntensityd(keys=[key], offset=1.0, factor_key=["orig_max"])
        data = {key: self.imt, PostFix.meta(key): {"affine": None}}

        result = shifter(stats(data))
        expected = self.imt + 1.0 * np.nanmax(self.imt)
        np.testing.assert_allclose(result[key], expected)


if __name__ == "__main__":
    unittest.main()
