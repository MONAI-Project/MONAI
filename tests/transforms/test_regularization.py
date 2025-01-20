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
import torch

from monai.transforms import CutMix, CutMixd, CutOut, CutOutd, MixUp, MixUpd
from tests.util import assert_allclose


class TestMixup(unittest.TestCase):
    def test_mixup(self):
        for dims in [2, 3]:
            shape = (6, 3) + (32,) * dims
            sample = torch.rand(*shape, dtype=torch.float32)
            mixup = MixUp(6, 1.0)
            mixup.set_random_state(seed=0)
            output = mixup(sample)
            np.random.seed(0)
            # simulate the randomize() of transform
            np.random.random()
            weight = torch.from_numpy(np.random.beta(1.0, 1.0, 6)).type(torch.float32)
            perm = np.random.permutation(6)
            self.assertEqual(output.shape, sample.shape)
            mixweight = weight[(Ellipsis,) + (None,) * (dims + 1)]
            expected = mixweight * sample + (1 - mixweight) * sample[perm, ...]
            assert_allclose(output, expected, type_test=False, atol=1e-7)

        with self.assertRaises(ValueError):
            MixUp(6, -0.5)

        mixup = MixUp(6, 0.5)
        for dims in [2, 3]:
            with self.assertRaises(ValueError):
                shape = (5, 3) + (32,) * dims
                sample = torch.rand(*shape, dtype=torch.float32)
                mixup(sample)

    def test_mixupd(self):
        for dims in [2, 3]:
            shape = (6, 3) + (32,) * dims
            t = torch.rand(*shape, dtype=torch.float32)
            sample = {"a": t, "b": t}
            mixup = MixUpd(["a", "b"], 6)
            mixup.set_random_state(seed=0)
            output = mixup(sample)
            np.random.seed(0)
            # simulate the randomize() of transform
            np.random.random()
            weight = torch.from_numpy(np.random.beta(1.0, 1.0, 6)).type(torch.float32)
            perm = np.random.permutation(6)
            self.assertEqual(output["a"].shape, sample["a"].shape)
            mixweight = weight[(Ellipsis,) + (None,) * (dims + 1)]
            expected = mixweight * sample["a"] + (1 - mixweight) * sample["a"][perm, ...]
            assert_allclose(output["a"], expected, type_test=False, atol=1e-7)
            assert_allclose(output["a"], output["b"], type_test=False, atol=1e-7)
            # self.assertTrue(torch.allclose(output["a"], output["b"]))

        with self.assertRaises(ValueError):
            MixUpd(["k1", "k2"], 6, -0.5)


class TestCutMix(unittest.TestCase):
    def test_cutmix(self):
        for dims in [2, 3]:
            shape = (6, 3) + (32,) * dims
            sample = torch.rand(*shape, dtype=torch.float32)
            cutmix = CutMix(6, 1.0)
            cutmix.set_random_state(seed=0)
            output = cutmix(sample)
            self.assertEqual(output.shape, sample.shape)
            self.assertTrue(any(not torch.allclose(sample, cutmix(sample)) for _ in range(10)))

    def test_cutmixd(self):
        for dims in [2, 3]:
            shape = (6, 3) + (32,) * dims
            t = torch.rand(*shape, dtype=torch.float32)
            label = torch.randint(0, 1, shape)
            sample = {"a": t, "b": t, "lbl1": label, "lbl2": label}
            cutmix = CutMixd(["a", "b"], 6, label_keys=("lbl1", "lbl2"))
            cutmix.set_random_state(seed=123)
            output = cutmix(sample)
            # but mixing of labels is not affected by it
            self.assertTrue(torch.allclose(output["lbl1"], output["lbl2"]))


class TestCutOut(unittest.TestCase):
    def test_cutout(self):
        for dims in [2, 3]:
            shape = (6, 3) + (32,) * dims
            sample = torch.rand(*shape, dtype=torch.float32)
            cutout = CutOut(6, 1.0)
            cutout.set_random_state(seed=123)
            output = cutout(sample)
            np.random.seed(123)
            # simulate the randomize() of transform
            np.random.random()
            weight = torch.from_numpy(np.random.beta(1.0, 1.0, 6)).type(torch.float32)
            perm = np.random.permutation(6)
            coords = [torch.from_numpy(np.random.randint(0, d, size=(1,))) for d in sample.shape[2:]]
            assert_allclose(weight, cutout._params[0])
            assert_allclose(perm, cutout._params[1])
            self.assertSequenceEqual(coords, cutout._params[2])
            self.assertEqual(output.shape, sample.shape)

    def test_cutoutd(self):
        for dims in [2, 3]:
            shape = (6, 3) + (32,) * dims
            t = torch.rand(*shape, dtype=torch.float32)
            sample = {"a": t, "b": t}
            cutout = CutOutd(["a", "b"], 6, 1.0)
            cutout.set_random_state(seed=123)
            output = cutout(sample)
            np.random.seed(123)
            # simulate the randomize() of transform
            np.random.random()
            weight = torch.from_numpy(np.random.beta(1.0, 1.0, 6)).type(torch.float32)
            perm = np.random.permutation(6)
            coords = [torch.from_numpy(np.random.randint(0, d, size=(1,))) for d in t.shape[2:]]
            assert_allclose(weight, cutout.cutout._params[0])
            assert_allclose(perm, cutout.cutout._params[1])
            self.assertSequenceEqual(coords, cutout.cutout._params[2])
            self.assertEqual(output["a"].shape, sample["a"].shape)


if __name__ == "__main__":
    unittest.main()
