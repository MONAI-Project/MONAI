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

import os
import unittest

from parameterized import parameterized

from monai.utils.misc import MONAIEnvVars, check_kwargs_exist_in_class_init, run_cmd, to_tuple_of_dictionaries

TO_TUPLE_OF_DICTIONARIES_TEST_CASES = [
    ({}, tuple(), tuple()),
    ({}, ("x",), ({},)),
    ({}, ("x", "y"), ({}, {})),
    ({"a": 1}, tuple(), tuple()),
    ({"a": 1}, ("x",), ({"a": 1},)),
    ({"a": (1,)}, ("x",), ({"a": 1},)),
    ({"a": (1,)}, ("x", "y"), ValueError()),
    ({"a": 1}, ("x", "y"), ({"a": 1}, {"a": 1})),
    ({"a": (1, 2)}, tuple(), tuple()),
    ({"a": (1, 2)}, ("x", "y"), ({"a": 1}, {"a": 2})),
    ({"a": (1, 2, 3)}, ("x", "y"), ValueError()),
    ({"b": (2,), "a": 1}, tuple(), tuple()),
    ({"b": (2,), "a": 1}, ("x",), ({"b": 2, "a": 1},)),
    ({"b": (2,), "a": 1}, ("x", "y"), ValueError()),
    ({"b": (3, 2), "a": 1}, tuple(), tuple()),
    ({"b": (3, 2), "a": 1}, ("x",), ValueError()),
    ({"b": (3, 2), "a": 1}, ("x", "y"), ({"b": 3, "a": 1}, {"b": 2, "a": 1})),
]


class MiscClass:

    def __init__(self, arg1, arg2, kwargs1=None, kwargs2=None):
        pass


class TestToTupleOfDictionaries(unittest.TestCase):

    @parameterized.expand(TO_TUPLE_OF_DICTIONARIES_TEST_CASES)
    def test_to_tuple_of_dictionaries(self, dictionary, keys, expected):
        self._test_to_tuple_of_dictionaries(dictionary, keys, expected)

    def _test_to_tuple_of_dictionaries(self, dictionary, keys, expected):
        if isinstance(expected, Exception):
            with self.assertRaises(type(expected)):
                to_tuple_of_dictionaries(dictionary, keys)
            print(type(expected))
        else:
            actual = to_tuple_of_dictionaries(dictionary, keys)
            print(actual, expected)
            self.assertTupleEqual(actual, expected)


class TestMiscKwargs(unittest.TestCase):

    def test_kwargs(self):
        present, extra_args = self._custom_user_function(MiscClass, 1, kwargs1="value1", kwargs2="value2")
        self.assertEqual(present, True)
        self.assertEqual(extra_args, set())
        present, extra_args = self._custom_user_function(MiscClass, 1, kwargs1="value1", kwargs3="value3")
        self.assertEqual(present, False)
        self.assertEqual(extra_args, {"kwargs3"})

    def _custom_user_function(self, cls, *args, **kwargs):
        return check_kwargs_exist_in_class_init(cls, kwargs)


class TestCommandRunner(unittest.TestCase):

    def setUp(self):
        self.orig_flag = str(MONAIEnvVars.debug())

    def tearDown(self):
        if self.orig_flag is not None:
            os.environ["MONAI_DEBUG"] = self.orig_flag
        else:
            os.environ.pop("MONAI_DEBUG")

    def test_run_cmd(self):
        cmd1 = "python"
        cmd2 = "-c"
        cmd3 = 'import sys; print("\\tThis is on stderr\\n", file=sys.stderr); sys.exit(1)'
        os.environ["MONAI_DEBUG"] = str(True)
        with self.assertRaises(RuntimeError) as cm:
            run_cmd([cmd1, cmd2, cmd3], check=True)
        self.assertIn("This is on stderr", str(cm.exception))
        self.assertNotIn("\\n", str(cm.exception))
        self.assertNotIn("\\t", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
