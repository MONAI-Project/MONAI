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

from monai.utils import compute_capabilities_after, pytorch_after

TEST_CASES_PT = (
    (1, 5, 9, "1.6.0"),
    (1, 6, 0, "1.6.0"),
    (1, 6, 1, "1.6.0", False),
    (1, 7, 0, "1.6.0", False),
    (2, 6, 0, "1.6.0", False),
    (0, 6, 0, "1.6.0a0+3fd9dcf"),
    (1, 5, 9, "1.6.0a0+3fd9dcf"),
    (1, 6, 0, "1.6.0a0+3fd9dcf", False),
    (1, 6, 1, "1.6.0a0+3fd9dcf", False),
    (2, 6, 0, "1.6.0a0+3fd9dcf", False),
    (1, 6, 0, "1.6.0-rc0+3fd9dcf", False),  # defaults to prerelease
    (1, 6, 0, "1.6.0rc0", False),
    (1, 6, 0, "1.6", True),
    (1, 6, 0, "1", False),
    (1, 6, 0, "1.6.0+cpu", True),
    (1, 6, 1, "1.6.0+cpu", False),
)

TEST_CASES_SM = [
    # (major, minor, sm, expected)
    (6, 1, "6.1", True),
    (6, 1, "6.0", False),
    (6, 0, "8.6", True),
    (7, 0, "8", True),
    (8, 6, "8", False),
]


class TestPytorchVersionCompare(unittest.TestCase):

    @parameterized.expand(TEST_CASES_PT)
    def test_compare(self, a, b, p, current, expected=True):
        """Test pytorch_after with a and b"""
        self.assertEqual(pytorch_after(a, b, p, current), expected)


class TestComputeCapabilitiesAfter(unittest.TestCase):

    @parameterized.expand(TEST_CASES_SM)
    def test_compute_capabilities_after(self, major, minor, sm, expected):
        self.assertEqual(compute_capabilities_after(major, minor, sm), expected)


if __name__ == "__main__":
    unittest.main()
