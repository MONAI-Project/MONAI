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

from monai.utils import optional_import
from tests.util import skip_if_no_cpp_extension

b, _ = optional_import("monai._C", name="BoundType")
p, _ = optional_import("monai._C", name="InterpolationType")


@skip_if_no_cpp_extension
class TestEnumBoundInterp(unittest.TestCase):
    def test_bound(self):
        self.assertEqual(str(b.replicate), "BoundType.replicate")
        self.assertEqual(str(b.nearest), "BoundType.replicate")
        self.assertEqual(str(b.dct1), "BoundType.dct1")
        self.assertEqual(str(b.mirror), "BoundType.dct1")
        self.assertEqual(str(b.dct2), "BoundType.dct2")
        self.assertEqual(str(b.reflect), "BoundType.dct2")
        self.assertEqual(str(b.dst1), "BoundType.dst1")
        self.assertEqual(str(b.antimirror), "BoundType.dst1")
        self.assertEqual(str(b.dst2), "BoundType.dst2")
        self.assertEqual(str(b.antireflect), "BoundType.dst2")
        self.assertEqual(str(b.dft), "BoundType.dft")
        self.assertEqual(str(b.wrap), "BoundType.dft")
        self.assertEqual(str(b.zero), "BoundType.zero")

        self.assertEqual(int(b.replicate), 0)
        self.assertEqual(int(b.nearest), 0)
        self.assertEqual(int(b.dct1), 1)
        self.assertEqual(int(b.mirror), 1)
        self.assertEqual(int(b.dct2), 2)
        self.assertEqual(int(b.reflect), 2)
        self.assertEqual(int(b.dst1), 3)
        self.assertEqual(int(b.antimirror), 3)
        self.assertEqual(int(b.dst2), 4)
        self.assertEqual(int(b.antireflect), 4)
        self.assertEqual(int(b.dft), 5)
        self.assertEqual(int(b.wrap), 5)
        self.assertEqual(int(b.zero), 7)

    def test_interp(self):
        self.assertEqual(str(p.nearest), "InterpolationType.nearest")
        self.assertEqual(str(p.linear), "InterpolationType.linear")
        self.assertEqual(str(p.quadratic), "InterpolationType.quadratic")
        self.assertEqual(str(p.cubic), "InterpolationType.cubic")
        self.assertEqual(str(p.fourth), "InterpolationType.fourth")
        self.assertEqual(str(p.fifth), "InterpolationType.fifth")
        self.assertEqual(str(p.sixth), "InterpolationType.sixth")
        self.assertEqual(str(p.seventh), "InterpolationType.seventh")

        self.assertEqual(int(p.nearest), 0)
        self.assertEqual(int(p.linear), 1)
        self.assertEqual(int(p.quadratic), 2)
        self.assertEqual(int(p.cubic), 3)
        self.assertEqual(int(p.fourth), 4)
        self.assertEqual(int(p.fifth), 5)
        self.assertEqual(int(p.sixth), 6)
        self.assertEqual(int(p.seventh), 7)


if __name__ == "__main__":
    unittest.main()
