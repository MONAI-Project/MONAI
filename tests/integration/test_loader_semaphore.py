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
"""this test should not generate errors or
UserWarning: semaphore_tracker: There appear to be 1 leaked semaphores"""
from __future__ import annotations

import multiprocessing as mp
import unittest

import monai  # noqa


def w():
    pass


def _main():
    ps = mp.Process(target=w)
    ps.start()
    ps.join()


def _run_test():
    try:
        tmp = mp.get_context("spawn")
    except RuntimeError:
        tmp = mp
    p = tmp.Process(target=_main)
    p.start()
    p.join()


class TestImportLock(unittest.TestCase):

    def test_start(self):
        _run_test()


if __name__ == "__main__":
    unittest.main()
