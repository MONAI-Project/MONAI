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

from monai.utils.misc import str2list


class TestStr2List(unittest.TestCase):

    def test_str_2_list(self):
        for i in ("1,2,3", "1, 2, 3", "1,2e-0,3.0", [1, 2, 3]):
            self.assertEqual(str2list(i), [1, 2, 3])
        for i in ("1,2,3", "1,2,3,4.3", [1, 2, 3, 4.001]):
            self.assertNotEqual(str2list(i), [1, 2, 3, 4])
        for bad_value in ((1, 3), int):
            self.assertIsNone(str2list(bad_value, raise_exc=False))
            with self.assertRaises(ValueError):
                self.assertIsNone(str2list(bad_value))


if __name__ == "__main__":
    unittest.main()
