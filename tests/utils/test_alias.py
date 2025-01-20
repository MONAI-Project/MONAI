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

import glob
import inspect
import os
import unittest
from pathlib import Path

from monai.utils import optional_import

TESTS_PATH = Path(__file__).parents[1]


class TestModuleAlias(unittest.TestCase):
    """check that 'import monai.xx.file_name' returns a module"""

    def test_files(self):
        src_dir = os.path.dirname(TESTS_PATH)
        monai_dir = os.path.join(src_dir, "monai")
        py_files = glob.glob(os.path.join(monai_dir, "**", "*.py"), recursive=True)
        for x in py_files:
            if os.path.basename(x).startswith("_"):
                continue
            mod_name = x[len(src_dir) : -3]  # create relative path
            mod_name = mod_name[1:].replace(mod_name[0], ".")
            mod, cls = mod_name.rsplit(".", 1)
            obj, exist = optional_import(mod, name=cls)
            if exist:
                self.assertTrue(inspect.ismodule(obj), msg=mod_name)


if __name__ == "__main__":
    unittest.main()
