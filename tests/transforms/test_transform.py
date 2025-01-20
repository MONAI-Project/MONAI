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

import monai.transforms as mt
from monai.data import Dataset
from monai.utils.misc import MONAIEnvVars


class FaultyTransform(mt.Transform):

    def __call__(self, _):
        raise RuntimeError


def faulty_lambda(_):
    raise RuntimeError


class TestTransform(unittest.TestCase):

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
        super(__class__, cls).tearDownClass()

    def test_raise(self):
        for transform in (FaultyTransform(), mt.Lambda(faulty_lambda)):
            ds = Dataset([None] * 10, transform)
            for debug in ("False", "True"):
                os.environ["MONAI_DEBUG"] = debug
                try:
                    ds[0]
                except RuntimeError as re:
                    if debug == "False":
                        self.assertTrue("applying transform" in str(re))
                    else:
                        self.assertFalse("applying transform" in str(re))


if __name__ == "__main__":
    unittest.main()
