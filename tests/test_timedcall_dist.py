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

import multiprocessing
import sys
import time
import unittest

from tests.util import TimedCall


@TimedCall(seconds=20 if sys.platform == "linux" else 60, force_quit=False)
def case_1_seconds(arg=None):
    time.sleep(1)
    return "good" if not arg else arg


@TimedCall(seconds=0.1, skip_timing=True, force_quit=True)
def case_1_seconds_skip(arg=None):
    time.sleep(1)
    return "good" if not arg else arg


@TimedCall(seconds=0.1, force_quit=True)
def case_1_seconds_timeout(arg=None):
    time.sleep(1)
    return "good" if not arg else arg


@TimedCall(seconds=0.1, force_quit=False)
def case_1_seconds_timeout_warning(arg=None):
    time.sleep(1)
    return "good" if not arg else arg


@TimedCall(seconds=0.1, force_quit=True)
def case_1_seconds_bad(arg=None):
    time.sleep(1)
    assert 0 == 1, "wrong case"


class TestTimedCall(unittest.TestCase):
    def test_good_call(self):
        output = case_1_seconds()
        self.assertEqual(output, "good")

    def test_skip_timing(self):
        output = case_1_seconds_skip("testing")
        self.assertEqual(output, "testing")

    def test_timeout(self):
        with self.assertRaises(multiprocessing.TimeoutError):
            case_1_seconds_timeout()

    def test_timeout_not_force_quit(self):
        with self.assertWarns(Warning):
            with self.assertRaises(multiprocessing.TimeoutError):
                case_1_seconds_timeout_warning()

    def test_timeout_bad(self):
        # timeout before the method's error
        with self.assertRaises(multiprocessing.TimeoutError):
            case_1_seconds_bad()


if __name__ == "__main__":
    unittest.main()
