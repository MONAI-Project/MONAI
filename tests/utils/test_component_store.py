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

from monai.utils import ComponentStore


class TestComponentStore(unittest.TestCase):

    def setUp(self):
        self.cs = ComponentStore("TestStore", "I am a test store, please ignore")

    def test_empty(self):
        self.assertEqual(len(self.cs), 0)
        self.assertEqual(list(self.cs), [])

    def test_add(self):
        test_obj = object()

        self.assertFalse("test_obj" in self.cs)

        self.cs.add("test_obj", "Test object", test_obj)

        self.assertTrue("test_obj" in self.cs)

        self.assertEqual(len(self.cs), 1)
        self.assertEqual(list(self.cs), [("test_obj", test_obj)])

        self.assertEqual(self.cs.test_obj, test_obj)
        self.assertEqual(self.cs["test_obj"], test_obj)

    def test_add2(self):
        test_obj1 = object()
        test_obj2 = object()

        self.cs.add("test_obj1", "Test object", test_obj1)
        self.cs.add("test_obj2", "Test object", test_obj2)

        self.assertEqual(len(self.cs), 2)
        self.assertIn("test_obj1", self.cs)
        self.assertIn("test_obj2", self.cs)

    def test_add_def(self):
        self.assertNotIn("test_func", self.cs)

        @self.cs.add_def("test_func", "Test function")
        def test_func():
            return 123

        self.assertIn("test_func", self.cs)

        self.assertEqual(len(self.cs), 1)
        self.assertEqual(list(self.cs), [("test_func", test_func)])

        self.assertEqual(self.cs.test_func, test_func)
        self.assertEqual(self.cs["test_func"], test_func)

        # try adding the same function again
        self.cs.add_def("test_func", "Test function but with new description")(test_func)

        self.assertEqual(len(self.cs), 1)
        self.assertEqual(self.cs.test_func, test_func)
