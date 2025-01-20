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

import monai.transforms as mt
from monai.data import CacheDataset
from tests.util import TEST_NDARRAYS, NumpyImageTestCase2D, assert_allclose


class T(mt.Transform):
    def __call__(self, x):
        return x * 2


class TestIdentity(NumpyImageTestCase2D):
    def test_identity(self):
        for p in TEST_NDARRAYS:
            img = p(self.imt)
            identity = mt.RandIdentity()
            assert_allclose(img, identity(img))

    def test_caching(self, init=1, expect=4, expect_pre_cache=2):
        # check that we get the correct result (two lots of T so should get 4)
        x = init
        transforms = mt.Compose([T(), mt.RandIdentity(), T()])
        self.assertEqual(transforms(x), expect)

        # check we get correct result with CacheDataset
        x = [init]
        ds = CacheDataset(x, transforms)
        self.assertEqual(ds[0], expect)

        # check that the cached value is correct
        self.assertEqual(ds._cache[0], expect_pre_cache)


if __name__ == "__main__":
    unittest.main()
