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

from monai.utils import OptionalImportError, min_version, require_pkg


class TestRequirePkg(unittest.TestCase):

    def test_class(self):

        @require_pkg(pkg_name="torch", version="1.4", version_checker=min_version)
        class TestClass:
            pass

        TestClass()

    def test_function(self):

        @require_pkg(pkg_name="torch", version="1.4", version_checker=min_version)
        def test_func(x):
            return x

        test_func(x=None)

    def test_warning(self):

        @require_pkg(pkg_name="test123", raise_error=False)
        def test_func(x):
            return x

        test_func(x=None)

    def test_class_exception(self):
        with self.assertRaises(OptionalImportError):

            @require_pkg(pkg_name="test123")
            class TestClass:
                pass

            TestClass()

    def test_class_version_exception(self):
        with self.assertRaises(OptionalImportError):

            @require_pkg(pkg_name="torch", version="10000", version_checker=min_version)
            class TestClass:
                pass

            TestClass()

    def test_func_exception(self):
        with self.assertRaises(OptionalImportError):

            @require_pkg(pkg_name="test123")
            def test_func(x):
                return x

            test_func(x=None)

    def test_func_versions_exception(self):
        with self.assertRaises(OptionalImportError):

            @require_pkg(pkg_name="torch", version="10000", version_checker=min_version)
            def test_func(x):
                return x

            test_func(x=None)


if __name__ == "__main__":
    unittest.main()
