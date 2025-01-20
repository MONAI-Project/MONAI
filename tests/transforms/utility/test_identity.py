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

from monai.transforms.utility.array import Identity
from tests.util import TEST_NDARRAYS, NumpyImageTestCase2D, assert_allclose


class TestIdentity(NumpyImageTestCase2D):
    def test_identity(self):
        for p in TEST_NDARRAYS:
            img = p(self.imt)
            identity = Identity()
            assert_allclose(img, identity(img))


if __name__ == "__main__":
    unittest.main()
