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

import itertools
import unittest

from parameterized import parameterized

from monai.utils import version_geq, version_leq


# from pkg_resources
def _pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


# from pkg_resources
torture = """
    0.80.1-3 0.80.1-2 0.80.1-1 0.79.9999+0.80.0pre4-1
    0.79.9999+0.80.0pre2-3 0.79.9999+0.80.0pre2-2
    0.77.2-1 0.77.1-1 0.77.0-1
    """

TEST_CASES = (
    ("1.6.0", "1.6.0"),
    ("1.6.0a0+9907a3e", "1.6.0"),
    ("0+unknown", "0.6"),
    ("ab", "abc"),
    ("0.6rc1", "0.6"),
    ("0.6", "0.7"),
    ("1.2.a", "1.2a"),
    ("1.2-rc1", "1.2rc1"),
    ("0.4", "0.4.0"),
    ("0.4.0.0", "0.4.0"),
    ("0.4.0-0", "0.4-0"),
    ("0post1", "0.0post1"),
    ("0pre1", "0.0c1"),
    ("0.0.0preview1", "0c1"),
    ("0.0c1", "0-rc1"),
    ("1.2a1", "1.2.a.1"),
    ("1.2.a", "1.2a"),
    ("2.1", "2.1.1"),
    ("2a1", "2b0"),
    ("2a1", "2.1"),
    ("2.3a1", "2.3"),
    ("2.1-1", "2.1-2"),
    ("2.1-1", "2.1.1"),
    ("2.1", "2.1post4"),
    ("2.1a0-20040501", "2.1"),
    ("1.1", "02.1"),
    ("3.2", "3.2.post0"),
    ("3.2post1", "3.2post2"),
    ("0.4", "4.0"),
    ("0.0.4", "0.4.0"),
    ("0post1", "0.4post1"),
    ("2.1.0-rc1", "2.1.0"),
    ("2.1dev", "2.1a0"),
    (1.6, "1.6.0"),
    ("1.6.0", 1.6),
    (1.6, 1.7),
) + tuple(_pairwise(reversed(torture.split())))


class TestVersionCompare(unittest.TestCase):

    @parameterized.expand(TEST_CASES)
    def test_compare_leq(self, a, b, expected=True):
        """Test version_leq with `a` and `b`"""
        self.assertEqual(version_leq(a, b), expected)

    @parameterized.expand(TEST_CASES)
    def test_compare_geq(self, a, b, expected=True):
        """Test version_geq with `b` and `a`"""
        self.assertEqual(version_geq(b, a), expected)


if __name__ == "__main__":
    unittest.main()
