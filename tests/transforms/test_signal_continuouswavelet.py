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
from pathlib import Path
from unittest import skipUnless

import numpy as np
from parameterized import parameterized

from monai.transforms import SignalContinuousWavelet
from monai.utils import optional_import

_, has_pywt = optional_import("pywt")
TESTS_PATH = Path(__file__).parents[1]
TEST_SIGNAL = os.path.join(TESTS_PATH, "testing_data", "signal.npy")
VALID_CASES = [("mexh", 150, 500)]
EXPECTED_RESULTS = [(6, 150, 2000)]


@skipUnless(has_pywt, "pywt required")
class TestSignalContinousWavelet(unittest.TestCase):
    @parameterized.expand(VALID_CASES)
    def test_correct_parameters_multi_channels(self, type, length, frequency):
        self.assertIsInstance(SignalContinuousWavelet(type, length, frequency), SignalContinuousWavelet)
        sig = np.load(TEST_SIGNAL)
        cwt = SignalContinuousWavelet(type, length, frequency)
        cwtsignal = cwt(sig)
        self.assertEqual(cwtsignal.shape, EXPECTED_RESULTS[0])


if __name__ == "__main__":
    unittest.main()
