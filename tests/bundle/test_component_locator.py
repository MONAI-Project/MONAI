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
from pydoc import locate

from monai.bundle import ComponentLocator
from monai.utils import optional_import

_, has_ignite = optional_import("ignite")


class TestComponentLocator(unittest.TestCase):

    def test_locate(self):
        locator = ComponentLocator(excludes=None if has_ignite else ["monai.handlers"])
        # test init mapping table and get the module path of component
        self.assertEqual(locator.get_component_module_name("LoadImage"), "monai.transforms.io.array")
        self.assertGreater(len(locator._components_table), 0)
        for _, mods in locator._components_table.items():
            for i in mods:
                self.assertGreater(len(mods), 0)
                # ensure we can locate all the items by `name`
                self.assertIsNotNone(locate(i), msg=f"can not locate target: {i}.")


if __name__ == "__main__":
    unittest.main()
