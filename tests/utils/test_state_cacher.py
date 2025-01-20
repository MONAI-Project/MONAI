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

import pickle
import unittest
from os.path import exists, join
from pathlib import Path
from tempfile import gettempdir

import torch
from parameterized import parameterized

from monai.utils import StateCacher

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

TEST_CASE_0 = [torch.Tensor([1]).to(DEVICE), {"in_memory": True}]
TEST_CASE_1 = [
    torch.Tensor([1]).to(DEVICE),
    {"in_memory": False, "cache_dir": gettempdir(), "pickle_module": None, "pickle_protocol": pickle.HIGHEST_PROTOCOL},
]
TEST_CASE_2 = [torch.Tensor([1]).to(DEVICE), {"in_memory": False, "allow_overwrite": False}]
TEST_CASE_3 = [torch.Tensor([1]).to(DEVICE), {"in_memory": False, "cache_dir": Path(gettempdir())}]

TEST_CASES = [TEST_CASE_0, TEST_CASE_1, TEST_CASE_2, TEST_CASE_3]


class TestStateCacher(unittest.TestCase):

    @parameterized.expand(TEST_CASES)
    def test_state_cacher(self, data_obj, params):
        key = "data_obj"

        state_cacher = StateCacher(**params)
        # store it
        state_cacher.store(key, data_obj, pickle_module=pickle)
        # create clone then modify original
        data_obj_orig = data_obj.clone()
        data_obj += 1
        # Restore and check nothing has changed
        data_obj_restored = state_cacher.retrieve(key)
        self.assertEqual(data_obj_orig, data_obj_restored)

        # If not allow overwrite, check an attempt would raise exception
        if "allow_overwrite" in params and params["allow_overwrite"]:
            with self.assertRaises(RuntimeError):
                state_cacher.store(key, data_obj)

        # If using a cache dir, check file has been deleted et end
        if "cache_dir" in params:
            i = id(state_cacher)
            del state_cacher
            self.assertFalse(exists(join(params["cache_dir"], f"state_{key}_{i}.pt")))


if __name__ == "__main__":
    unittest.main()
