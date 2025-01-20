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
import warnings

from monai.utils import DeprecatedError, deprecated, deprecated_arg, deprecated_arg_default


class TestDeprecatedRC(unittest.TestCase):

    def setUp(self):
        self.test_version_rc = "0.6.0rc1"
        self.test_version = "0.6.0"
        self.next_version = "0.7.0"

    def test_warning(self):
        """Test deprecated decorator with `since` and `removed` set for an RC version"""

        @deprecated(since=self.test_version, removed=self.next_version, version_val=self.test_version_rc)
        def foo2():
            pass

        foo2()  # should not raise any warnings

    def test_warning_milestone(self):
        """Test deprecated decorator with `since` and `removed` set for a milestone version"""

        @deprecated(since=self.test_version, removed=self.next_version, version_val=self.test_version)
        def foo2():
            pass

        self.assertWarns(FutureWarning, foo2)

    def test_warning_last(self):
        """Test deprecated decorator with `since` and `removed` set, for the last version"""

        @deprecated(since=self.test_version, removed=self.next_version, version_val=self.next_version)
        def foo3():
            pass

        self.assertRaises(DeprecatedError, foo3)

    def test_warning_beyond(self):
        """Test deprecated decorator with `since` and `removed` set, beyond the last version"""

        @deprecated(since=self.test_version_rc, removed=self.test_version, version_val=self.next_version)
        def foo3():
            pass

        self.assertRaises(DeprecatedError, foo3)


class TestDeprecated(unittest.TestCase):

    def setUp(self):
        self.test_version = "0.5.3+96.g1fa03c2.dirty"
        self.prev_version = "0.4.3+96.g1fa03c2.dirty"
        self.next_version = "0.6.3+96.g1fa03c2.dirty"

    def test_warning1(self):
        """Test deprecated decorator with just `since` set."""

        @deprecated(since=self.prev_version, version_val=self.test_version)
        def foo1():
            pass

        self.assertWarns(FutureWarning, foo1)

    def test_warning2(self):
        """Test deprecated decorator with `since` and `removed` set."""

        @deprecated(since=self.prev_version, removed=self.next_version, version_val=self.test_version)
        def foo2():
            pass

        self.assertWarns(FutureWarning, foo2)

    def test_except1(self):
        """Test deprecated decorator raises exception with no versions set."""

        @deprecated(version_val=self.test_version)
        def foo3():
            pass

        self.assertRaises(DeprecatedError, foo3)

    def test_except2(self):
        """Test deprecated decorator raises exception with `removed` set in the past."""

        @deprecated(removed=self.prev_version, version_val=self.test_version)
        def foo4():
            pass

        self.assertRaises(DeprecatedError, foo4)

    def test_class_warning1(self):
        """Test deprecated decorator with just `since` set."""

        @deprecated(since=self.prev_version, version_val=self.test_version)
        class Foo1:
            pass

        self.assertWarns(FutureWarning, Foo1)

    def test_class_warning2(self):
        """Test deprecated decorator with `since` and `removed` set."""

        @deprecated(since=self.prev_version, removed=self.next_version, version_val=self.test_version)
        class Foo2:
            pass

        self.assertWarns(FutureWarning, Foo2)

    def test_class_except1(self):
        """Test deprecated decorator raises exception with no versions set."""

        @deprecated(version_val=self.test_version)
        class Foo3:
            pass

        self.assertRaises(DeprecatedError, Foo3)

    def test_class_except2(self):
        """Test deprecated decorator raises exception with `removed` set in the past."""

        @deprecated(removed=self.prev_version, version_val=self.test_version)
        class Foo4:
            pass

        self.assertRaises(DeprecatedError, Foo4)

    def test_meth_warning1(self):
        """Test deprecated decorator with just `since` set."""

        class Foo5:

            @deprecated(since=self.prev_version, version_val=self.test_version)
            def meth1(self):
                pass

        self.assertWarns(FutureWarning, lambda: Foo5().meth1())

    def test_meth_except1(self):
        """Test deprecated decorator with just `since` set."""

        class Foo6:

            @deprecated(version_val=self.test_version)
            def meth1(self):
                pass

        self.assertRaises(DeprecatedError, lambda: Foo6().meth1())

    def test_arg_warn1(self):
        """Test deprecated_arg decorator with just `since` set."""

        @deprecated_arg("b", since=self.prev_version, version_val=self.test_version)
        def afoo1(a, b=None):
            pass

        afoo1(1)  # ok when no b provided

        self.assertWarns(FutureWarning, lambda: afoo1(1, 2))

    def test_arg_warn2(self):
        """Test deprecated_arg decorator with just `since` set."""

        @deprecated_arg("b", since=self.prev_version, version_val=self.test_version)
        def afoo2(a, **kw):
            pass

        afoo2(1)  # ok when no b provided

        self.assertWarns(FutureWarning, lambda: afoo2(1, b=2))

    def test_arg_except1(self):
        """Test deprecated_arg decorator raises exception with no versions set."""

        @deprecated_arg("b", version_val=self.test_version)
        def afoo3(a, b=None):
            pass

        self.assertRaises(DeprecatedError, lambda: afoo3(1, b=2))

    def test_arg_except2(self):
        """Test deprecated_arg decorator raises exception with `removed` set in the past."""

        @deprecated_arg("b", removed=self.prev_version, version_val=self.test_version)
        def afoo4(a, b=None):
            pass

        self.assertRaises(DeprecatedError, lambda: afoo4(1, b=2))

    def test_2arg_warn1(self):
        """Test deprecated_arg decorator applied twice with just `since` set."""

        @deprecated_arg("b", since=self.prev_version, version_val=self.test_version)
        @deprecated_arg("c", since=self.prev_version, version_val=self.test_version)
        def afoo5(a, b=None, c=None):
            pass

        afoo5(1)  # ok when no b or c provided

        self.assertWarns(FutureWarning, lambda: afoo5(1, 2))
        self.assertWarns(FutureWarning, lambda: afoo5(1, 2, 3))

    def test_future(self):
        """Test deprecated decorator with `since` set to a future version."""

        @deprecated(since=self.next_version, version_val=self.test_version)
        def future1():
            pass

        with self.assertWarns(FutureWarning) as aw:
            future1()
            warnings.warn("fake warning", FutureWarning)

        self.assertEqual(aw.warning.args[0], "fake warning")

    def test_arg_except2_unknown(self):
        """
        Test deprecated_arg decorator raises exception with `removed` set in the past.
        with unknown version
        """

        @deprecated_arg("b", removed=self.prev_version, version_val="0+untagged.1.g3131155")
        def afoo4(a, b=None):
            pass

        afoo4(1, b=2)

    def test_arg_except3_unknown(self):
        """
        Test deprecated_arg decorator raises exception with `removed` set in the past.
        with unknown version and kwargs
        """

        @deprecated_arg("b", removed=self.prev_version, version_val="0+untagged.1.g3131155")
        def afoo4(a, b=None, **kwargs):
            pass

        afoo4(1, b=2)
        afoo4(1, b=2, c=3)

    def test_replacement_arg(self):
        """
        Test deprecated arg being replaced.
        """

        @deprecated_arg("b", new_name="a", since=self.prev_version, version_val=self.test_version)
        def afoo4(a, b=None):
            return a

        self.assertEqual(afoo4(b=2), 2)
        self.assertEqual(afoo4(1, b=2), 1)  # new name is in use
        self.assertEqual(afoo4(a=1, b=2), 1)  # prefers the new arg

    def test_replacement_arg1(self):
        """
        Test deprecated arg being replaced with kwargs.
        """

        @deprecated_arg("b", new_name="a", since=self.prev_version, version_val=self.test_version)
        def afoo4(a, *args, **kwargs):
            return a

        self.assertEqual(afoo4(b=2), 2)
        self.assertEqual(afoo4(1, b=2, c=3), 1)  # new name is in use
        self.assertEqual(afoo4(a=1, b=2, c=3), 1)  # prefers the new arg

    def test_replacement_arg2(self):
        """
        Test deprecated arg (with a default value) being replaced.
        """

        @deprecated_arg("b", new_name="a", since=self.prev_version, version_val=self.test_version)
        def afoo4(a, b=None, **kwargs):
            return a, kwargs

        self.assertEqual(afoo4(b=2, c=3), (2, {"c": 3}))
        self.assertEqual(afoo4(1, b=2, c=3), (1, {"c": 3}))  # new name is in use
        self.assertEqual(afoo4(a=1, b=2, c=3), (1, {"c": 3}))  # prefers the new arg
        self.assertEqual(afoo4(1, 2, c=3), (1, {"c": 3}))  # prefers the new positional arg

    def test_deprecated_arg_default_explicit_default(self):
        """
        Test deprecated arg default, where the default is explicitly set (no warning).
        """

        @deprecated_arg_default(
            "b", old_default="a", new_default="b", since=self.prev_version, version_val=self.test_version
        )
        def foo(a, b="a"):
            return a, b

        with self.assertWarns(FutureWarning) as aw:
            self.assertEqual(foo("a", "a"), ("a", "a"))
            self.assertEqual(foo("a", "b"), ("a", "b"))
            self.assertEqual(foo("a", "c"), ("a", "c"))
            warnings.warn("fake warning", FutureWarning)

        self.assertEqual(aw.warning.args[0], "fake warning")

    def test_deprecated_arg_default_version_less_than_since(self):
        """
        Test deprecated arg default, where the current version is less than `since` (no warning).
        """

        @deprecated_arg_default(
            "b", old_default="a", new_default="b", since=self.test_version, version_val=self.prev_version
        )
        def foo(a, b="a"):
            return a, b

        with self.assertWarns(FutureWarning) as aw:
            self.assertEqual(foo("a"), ("a", "a"))
            self.assertEqual(foo("a", "a"), ("a", "a"))
            warnings.warn("fake warning", FutureWarning)

        self.assertEqual(aw.warning.args[0], "fake warning")

    def test_deprecated_arg_default_warning_deprecated(self):
        """
        Test deprecated arg default, where the default is used.
        """

        @deprecated_arg_default(
            "b", old_default="a", new_default="b", since=self.prev_version, version_val=self.test_version
        )
        def foo(a, b="a"):
            return a, b

        self.assertWarns(FutureWarning, lambda: foo("a"))

    def test_deprecated_arg_default_warning_replaced(self):
        """
        Test deprecated arg default, where the default is used.
        """

        @deprecated_arg_default(
            "b",
            old_default="a",
            new_default="b",
            since=self.prev_version,
            replaced=self.prev_version,
            version_val=self.test_version,
        )
        def foo(a, b="a"):
            return a, b

        self.assertWarns(FutureWarning, lambda: foo("a"))

    def test_deprecated_arg_default_warning_with_none_as_placeholder(self):
        """
        Test deprecated arg default, where the default is used.
        """

        @deprecated_arg_default(
            "b", old_default="a", new_default="b", since=self.prev_version, version_val=self.test_version
        )
        def foo(a, b=None):
            if b is None:
                b = "a"
            return a, b

        self.assertWarns(FutureWarning, lambda: foo("a"))

        @deprecated_arg_default(
            "b", old_default="a", new_default="b", since=self.prev_version, version_val=self.test_version
        )
        def foo2(a, b=None):
            if b is None:
                b = "b"
            return a, b

        self.assertWarns(FutureWarning, lambda: foo2("a"))

    def test_deprecated_arg_default_errors(self):
        """
        Test deprecated arg default, where the decorator is wrongly used.
        """

        # since > replaced
        def since_grater_than_replaced():

            @deprecated_arg_default(
                "b",
                old_default="a",
                new_default="b",
                since=self.test_version,
                replaced=self.prev_version,
                version_val=self.test_version,
            )
            def foo(a, b=None):
                return a, b

        self.assertRaises(ValueError, since_grater_than_replaced)

        # argname doesnt exist
        def argname_doesnt_exist():

            @deprecated_arg_default(
                "other", old_default="a", new_default="b", since=self.test_version, version_val=self.test_version
            )
            def foo(a, b=None):
                return a, b

        self.assertRaises(ValueError, argname_doesnt_exist)

        # argname has no default
        def argname_has_no_default():

            @deprecated_arg_default(
                "a",
                old_default="a",
                new_default="b",
                since=self.prev_version,
                replaced=self.test_version,
                version_val=self.test_version,
            )
            def foo(a):
                return a

        self.assertRaises(ValueError, argname_has_no_default)

        # new default is used but version < replaced
        def argname_was_replaced_before_specified_version():

            @deprecated_arg_default(
                "a",
                old_default="a",
                new_default="b",
                since=self.prev_version,
                replaced=self.next_version,
                version_val=self.test_version,
            )
            def foo(a, b="b"):
                return a, b

        self.assertRaises(ValueError, argname_was_replaced_before_specified_version)


if __name__ == "__main__":
    unittest.main()
