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

import unittest

import numpy as np

from monai.transforms import Compose, SpatialPad, SpatialPadd, allow_missing_keys_mode


class TestWithAllowMissingKeysMode(unittest.TestCase):

    def setUp(self):
        self.data = {"image": np.arange(16, dtype=float).reshape(1, 4, 4)}

    def test_map_transform(self):
        for amk in [True, False]:
            t = SpatialPadd(["image", "label"], 10, allow_missing_keys=amk)
            with allow_missing_keys_mode(t):
                # check state is True
                self.assertTrue(t.allow_missing_keys)
                # and that transform works even though key is missing
                _ = t(self.data)
            # check it has returned to original state
            self.assertEqual(t.allow_missing_keys, amk)
            if not amk:
                # should fail because amks==False and key is missing
                with self.assertRaises(KeyError):
                    _ = t(self.data)

    def test_compose(self):
        amks = [True, False, True]
        t = Compose([SpatialPadd(["image", "label"], 10, allow_missing_keys=amk) for amk in amks])
        with allow_missing_keys_mode(t):
            # check states are all True
            for _t in t.transforms:
                self.assertTrue(_t.allow_missing_keys)
            # and that transform works even though key is missing
            _ = t(self.data)
        # check they've returned to original state
        for _t, amk in zip(t.transforms, amks):
            self.assertEqual(_t.allow_missing_keys, amk)
        # should fail because not all amks==True and key is missing
        with self.assertRaises((KeyError, RuntimeError)):
            _ = t(self.data)

    def test_array_transform(self):
        for t in [SpatialPad(10), Compose([SpatialPad(10)])]:
            with self.assertRaises(TypeError):
                with allow_missing_keys_mode(t):
                    pass

    def test_multiple(self):
        orig_states = [True, False]
        ts = [SpatialPadd(["image", "label"], 10, allow_missing_keys=i) for i in orig_states]
        with allow_missing_keys_mode(ts):
            for t in ts:
                self.assertTrue(t.allow_missing_keys)
                # and that transform works even though key is missing
                _ = t(self.data)
        for t, o_s in zip(ts, orig_states):
            self.assertEqual(t.allow_missing_keys, o_s)


if __name__ == "__main__":
    unittest.main()
