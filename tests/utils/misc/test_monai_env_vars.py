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

from monai.utils.misc import MONAIEnvVars


class TestMONAIEnvVars(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(__class__, cls).setUpClass()
        cls.orig_value = str(MONAIEnvVars.debug())

    @classmethod
    def tearDownClass(cls):
        if cls.orig_value is not None:
            os.environ["MONAI_DEBUG"] = cls.orig_value
        else:
            os.environ.pop("MONAI_DEBUG")
        print("MONAI debug value:", str(MONAIEnvVars.debug()))
        super(__class__, cls).tearDownClass()

    def test_monai_env_vars(self):
        for debug in (False, True):
            os.environ["MONAI_DEBUG"] = str(debug)
            self.assertEqual(os.environ.get("MONAI_DEBUG"), str(debug))
            self.assertEqual(MONAIEnvVars.debug(), debug)


if __name__ == "__main__":
    unittest.main()
