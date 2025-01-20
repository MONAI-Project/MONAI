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
import tempfile
import unittest

import torch
import torch.optim as optim
from parameterized import parameterized

from monai.networks import save_state

TEST_CASE_1 = [torch.nn.PReLU(), ["weight"]]

TEST_CASE_2 = [{"net": torch.nn.PReLU()}, ["net"]]

TEST_CASE_3 = [{"net": torch.nn.PReLU(), "opt": optim.SGD(torch.nn.PReLU().parameters(), lr=0.02)}, ["net", "opt"]]

TEST_CASE_4 = [torch.nn.DataParallel(torch.nn.PReLU()), ["weight"]]

TEST_CASE_5 = [{"net": torch.nn.DataParallel(torch.nn.PReLU())}, ["net"]]

TEST_CASE_6 = [torch.nn.PReLU(), ["weight"], True, True, None, {"pickle_protocol": 2}]

TEST_CASE_7 = [torch.nn.PReLU().state_dict(), ["weight"]]

TEST_CASE_8 = [torch.nn.PReLU(), ["weight"], False]

TEST_CASE_9 = [torch.nn.PReLU(), ["weight"], True, False]

TEST_CASE_10 = [torch.nn.PReLU(), ["weight"], True, True, torch.save]


class TestSaveState(unittest.TestCase):

    @parameterized.expand(
        [
            TEST_CASE_1,
            TEST_CASE_2,
            TEST_CASE_3,
            TEST_CASE_4,
            TEST_CASE_5,
            TEST_CASE_6,
            TEST_CASE_7,
            TEST_CASE_8,
            TEST_CASE_9,
            TEST_CASE_10,
        ]
    )
    def test_file(self, src, expected_keys, create_dir=True, atomic=True, func=None, kwargs=None):
        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "test_ckpt.pt")
            if kwargs is None:
                kwargs = {}
            save_state(src=src, path=path, create_dir=create_dir, atomic=atomic, func=func, **kwargs)
            ckpt = dict(torch.load(path))
            for k in ckpt.keys():
                self.assertIn(k, expected_keys)


if __name__ == "__main__":
    unittest.main()
