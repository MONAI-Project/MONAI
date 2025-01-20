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

from monai.transforms.inverse import TraceableTransform


class _TraceTest(TraceableTransform):

    def __call__(self, data):
        self.push_transform(data)
        return data

    def pop(self, data):
        self.pop_transform(data)
        return data


class TestTraceable(unittest.TestCase):

    def test_default(self):
        expected_key = "_transforms"
        a = _TraceTest()
        for x in a.transform_info_keys():
            self.assertIn(x, a.get_transform_info())
        self.assertEqual(a.trace_key(), expected_key)

        data = {"image": "test"}
        data = a(data)  # adds to the stack
        self.assertIsInstance(data[expected_key], list)
        self.assertEqual(data[expected_key][0]["class"], "_TraceTest")

        data = a(data)  # adds to the stack
        self.assertEqual(len(data[expected_key]), 2)
        self.assertEqual(data[expected_key][-1]["class"], "_TraceTest")

        with self.assertRaises(IndexError):
            a.pop({"test": "test"})  # no stack in the data
        data = a.pop(data)
        data = a.pop(data)
        self.assertEqual(data[expected_key], [])

        with self.assertRaises(IndexError):  # no more items
            a.pop(data)


if __name__ == "__main__":
    unittest.main()
