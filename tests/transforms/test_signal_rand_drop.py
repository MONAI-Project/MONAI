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

import numpy as np
from parameterized import parameterized

from monai.transforms import SignalRandDrop
from monai.utils.type_conversion import convert_to_tensor

TESTS_PATH = Path(__file__).parents[1]
TEST_SIGNAL = os.path.join(TESTS_PATH, "testing_data", "signal.npy")
VALID_CASES = [([0.0, 1.0],), ([0.01, 0.1],)]


class TestSignalRandDropNumpy(unittest.TestCase):
    @parameterized.expand(VALID_CASES)
    def test_correct_parameters_multi_channels(self, boundaries):
        self.assertIsInstance(SignalRandDrop(boundaries), SignalRandDrop)
        sig = np.load(TEST_SIGNAL)
        droped = SignalRandDrop(boundaries)
        dropedsignal = droped(sig)
        self.assertEqual(dropedsignal.shape[1], sig.shape[1])


class TestSignalRandDropTorch(unittest.TestCase):
    @parameterized.expand(VALID_CASES)
    def test_correct_parameters_multi_channels(self, boundaries):
        self.assertIsInstance(SignalRandDrop(boundaries), SignalRandDrop)
        sig = convert_to_tensor(np.load(TEST_SIGNAL))
        droped = SignalRandDrop(boundaries)
        dropedsignal = droped(sig)
        self.assertEqual(dropedsignal.shape[1], sig.shape[1])


if __name__ == "__main__":
    unittest.main()
