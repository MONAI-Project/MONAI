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

from monai.utils.misc import str2bool


class TestStr2Bool(unittest.TestCase):

    def test_str_2_bool(self):
        for i in ("yes", "true", "t", "y", "1", True):
            self.assertTrue(str2bool(i))
        for i in ("no", "false", "f", "n", "0", False):
            self.assertFalse(str2bool(i))
        for bad_value in ("test", 0, 1, 2, None):
            self.assertFalse(str2bool(bad_value, default=False, raise_exc=False))
            self.assertTrue(str2bool(bad_value, default=True, raise_exc=False))
            with self.assertRaises(ValueError):
                self.assertTrue(str2bool(bad_value))


if __name__ == "__main__":
    unittest.main()
