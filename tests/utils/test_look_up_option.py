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
from enum import Enum

from parameterized import parameterized

from monai.utils import StrEnum, look_up_option


class _CaseEnum(Enum):
    CONST = "constant"
    EMPTY = "empty"


class _CaseEnum1(Enum):
    CONST = "constant"
    EMPTY = "empty"


class _CaseStrEnum(StrEnum):
    MODE_A = "A"
    MODE_B = "B"


TEST_CASES = (
    ("test", ("test", "test1"), "test"),
    ("test1", {"test1", "test"}, "test1"),
    (2, {1: "test", 2: "valid"}, "valid"),
    (_CaseEnum.EMPTY, _CaseEnum, _CaseEnum.EMPTY),
    ("empty", _CaseEnum, _CaseEnum.EMPTY),
)


class TestLookUpOption(unittest.TestCase):

    @parameterized.expand(TEST_CASES)
    def test_look_up(self, input_str, supported, expected):
        output = look_up_option(input_str, supported)
        self.assertEqual(output, expected)

    def test_default(self):
        output = look_up_option("not here", {"a", "b"}, default=None)
        self.assertEqual(output, None)

    def test_str_enum(self):
        output = look_up_option("C", {"A", "B"}, default=None)
        self.assertIsNone(output)
        self.assertEqual(list(_CaseStrEnum), ["A", "B"])
        self.assertEqual(_CaseStrEnum.MODE_A, "A")
        self.assertEqual(str(_CaseStrEnum.MODE_A), "A")
        self.assertEqual(look_up_option("A", _CaseStrEnum), "A")

    def test_no_found(self):
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            look_up_option("not here", {"a", "b"})
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            look_up_option("not here", ["a", "b"])
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            look_up_option("not here", {"a": 1, "b": 2})
        with self.assertRaisesRegex(ValueError, "did you mean"):
            look_up_option(3, {1: "a", 2: "b", "c": 3})
        with self.assertRaisesRegex(ValueError, "did.*empty"):
            look_up_option("empy", _CaseEnum)
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            look_up_option(_CaseEnum1.EMPTY, _CaseEnum)
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            look_up_option(None, _CaseEnum)
        with self.assertRaisesRegex(ValueError, "No"):
            look_up_option(None, None)
        with self.assertRaisesRegex(ValueError, "No"):
            look_up_option("test", None)


if __name__ == "__main__":
    unittest.main()
