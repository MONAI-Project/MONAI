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

from monai.utils.module import get_package_version


class TestGetVersion(unittest.TestCase):

    def test_default(self):
        output = get_package_version("42foobarnoexist")
        self.assertIn("UNKNOWN", output)

        output = get_package_version("numpy")
        self.assertNotIn("UNKNOWN", output)

    def test_msg(self):
        output = get_package_version("42foobarnoexist", "test")
        self.assertIn("test", output)


if __name__ == "__main__":
    unittest.main()
