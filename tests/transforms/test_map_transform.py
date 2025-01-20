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

from parameterized import parameterized

from monai.transforms import MapTransform

TEST_CASES = [["item", ("item",)], [None, (None,)], [["item1", "item2"], ("item1", "item2")]]

TEST_ILL_CASES = [[ValueError, []], [ValueError, ()], [TypeError, [[]]]]


class MapTest(MapTransform):

    def __call__(self, data):
        pass


class TestRandomizable(unittest.TestCase):

    @parameterized.expand(TEST_CASES)
    def test_keys(self, keys, expected):
        transform = MapTest(keys=keys)
        self.assertEqual(transform.keys, expected)

    @parameterized.expand(TEST_ILL_CASES)
    def test_wrong_keys(self, exception, keys):
        with self.assertRaisesRegex(exception, ""):
            MapTest(keys=keys)


if __name__ == "__main__":
    unittest.main()
