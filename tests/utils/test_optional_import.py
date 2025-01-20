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

from monai.utils import OptionalImportError, exact_version, optional_import


class TestOptionalImport(unittest.TestCase):

    @parameterized.expand(["not_a_module", "torch.randint"])
    def test_default(self, import_module):
        my_module, flag = optional_import(import_module)
        self.assertFalse(flag)
        with self.assertRaises(OptionalImportError):
            my_module.test

    def test_import_valid(self):
        my_module, flag = optional_import("torch")
        self.assertTrue(flag)
        print(my_module.randint(1, 2, (1, 2)))

    def test_import_wrong_number(self):
        my_module, flag = optional_import("torch", "42")
        with self.assertRaisesRegex(OptionalImportError, "version"):
            my_module.nn
        self.assertFalse(flag)
        with self.assertRaisesRegex(OptionalImportError, "version"):
            my_module.randint(1, 2, (1, 2))
        with self.assertRaisesRegex(ValueError, "invalid literal"):
            my_module, flag = optional_import("torch", "test")  # version should be number.number
            my_module.nn
            self.assertTrue(flag)
            print(my_module.randint(1, 2, (1, 2)))

    @parameterized.expand(["0", "0.0.0.1", "1.1.0"])
    def test_import_good_number(self, version_number):
        my_module, flag = optional_import("torch", version_number)
        my_module.nn
        self.assertTrue(flag)
        print(my_module.randint(1, 2, (1, 2)))

    def test_import_exact(self):
        my_module, flag = optional_import("torch", "0", exact_version)
        with self.assertRaisesRegex(OptionalImportError, "exact_version"):
            my_module.nn
        self.assertFalse(flag)
        with self.assertRaisesRegex(OptionalImportError, "exact_version"):
            my_module.randint(1, 2, (1, 2))

    def test_import_method(self):
        nn, flag = optional_import("torch", "1.1", name="nn")
        self.assertTrue(flag)
        print(nn.functional)

    def test_additional(self):
        test_args = {"a": "test", "b": "test"}

        def versioning(module, ver, a):
            self.assertEqual(a, test_args)
            return True

        nn, flag = optional_import("torch", "1.1", version_checker=versioning, name="nn", version_args=test_args)
        self.assertTrue(flag)


if __name__ == "__main__":
    unittest.main()
