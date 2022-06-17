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

import os
import unittest

import numpy as np

from monai.transforms import SignalNormalize

TEST_SIGNAL = os.path.join(os.path.dirname(__file__), "testing_data", "signal.npy")


class TestSignalRandDrop(unittest.TestCase):
    def test_correct_parameters_multi_channels(self):
        self.assertIsInstance(SignalNormalize(), SignalNormalize)
        sig = np.load(TEST_SIGNAL)
        normalized = SignalNormalize()
        normalizedsignal = normalized(sig)
        self.assertAlmostEqual(round(normalizedsignal.max()), 1.0)
        self.assertAlmostEqual(round(normalizedsignal.min()), 0.0)

    def test_correct_parameters_mono_channels(self):
        self.assertIsInstance(SignalNormalize(), SignalNormalize)
        sig = np.load(TEST_SIGNAL)[0, :]
        normalized = SignalNormalize()
        normalizedsignal = normalized(sig)
        self.assertAlmostEqual(round(normalizedsignal.max()), 1.0)
        self.assertAlmostEqual(round(normalizedsignal.min()), 0.0)


if __name__ == "__main__":
    unittest.main()
