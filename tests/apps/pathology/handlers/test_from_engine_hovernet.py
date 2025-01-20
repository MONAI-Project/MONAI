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

from monai.apps.pathology.handlers.utils import from_engine_hovernet
from tests.util import assert_allclose

TEST_CASE_0 = [
    [{"A": {"C": 1, "D": 2}, "B": {"C": 2, "D": 2}}, {"A": {"C": 3, "D": 2}, "B": {"C": 4, "D": 2}}],
    ([1, 3], [2, 4]),
]
TEST_CASE_1 = [{"A": {"C": 1, "D": 2}, "B": {"C": 2, "D": 2}}, (1, 2)]

CASES = [TEST_CASE_0, TEST_CASE_1]


class TestFromEngineHovernet(unittest.TestCase):
    @parameterized.expand(CASES)
    def test_results(self, input, expected):
        output = from_engine_hovernet(keys=["A", "B"], nested_key="C")(input)
        assert_allclose(output, expected, type_test=False)


if __name__ == "__main__":
    unittest.main()
