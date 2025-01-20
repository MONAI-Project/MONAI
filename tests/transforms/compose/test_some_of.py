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
from copy import deepcopy

import numpy as np
import torch
from parameterized import parameterized

import monai.transforms.intensity.array as ia
import monai.transforms.spatial.array as sa
import monai.transforms.spatial.dictionary as sd
from monai.data import MetaTensor
from monai.transforms import TraceableTransform, Transform
from monai.transforms.compose import Compose, SomeOf
from monai.utils import set_determinism
from monai.utils.enums import TraceKeys
from tests.integration.test_one_of import NonInv
from tests.transforms.test_random_order import InvC, InvD


class A(Transform):

    def __call__(self, x):
        return 2 * x


class B(Transform):

    def __call__(self, x):
        return 3 * x


class C(Transform):

    def __call__(self, x):
        return 5 * x


class D(Transform):

    def __call__(self, x):
        return 7 * x


KEYS = ["x", "y"]
TEST_COMPOUND = [
    (SomeOf((A(), B(), C()), num_transforms=3), 2 * 3 * 5),
    (Compose((SomeOf((A(), B(), C()), num_transforms=3), D())), 2 * 3 * 5 * 7),
    (SomeOf((A(), B(), C(), Compose(D())), num_transforms=4), 2 * 3 * 5 * 7),
    (SomeOf(()), 1),
    (SomeOf(None), 1),
]

# Modified from RandomOrder
TEST_INVERSES = [
    (SomeOf((InvC(KEYS), InvD(KEYS))), True, True),
    (Compose((SomeOf((InvC(KEYS), InvD(KEYS))), SomeOf((InvD(KEYS), InvC(KEYS))))), True, False),
    (SomeOf((SomeOf((InvC(KEYS), InvD(KEYS))), SomeOf((InvD(KEYS), InvC(KEYS))))), True, False),
    (SomeOf((Compose((InvC(KEYS), InvD(KEYS))), Compose((InvD(KEYS), InvC(KEYS))))), True, False),
    (SomeOf((NonInv(KEYS), NonInv(KEYS))), False, False),
    (SomeOf(()), False, False),
]


class TestSomeOf(unittest.TestCase):

    def setUp(self):
        set_determinism(seed=0)

    def tearDown(self):
        set_determinism(None)

    def update_transform_count(self, counts, output):
        op_count = 0

        if output % 2 == 0:
            counts[0] += 1
            op_count += 1
        if output % 3 == 0:
            counts[1] += 1
            op_count += 1
        if output % 5 == 0:
            counts[2] += 1
            op_count += 1

        return op_count

    def test_fixed(self):
        iterations = 10000
        num_transforms = 3
        transform_counts = 3 * [0]
        subset_size_counts = 4 * [0]

        s = SomeOf((A(), B(), C()), num_transforms=num_transforms)

        for _ in range(iterations):
            output = s(1)
            subset_size = self.update_transform_count(transform_counts, output)
            subset_size_counts[subset_size] += 1

        for i in range(3):
            self.assertEqual(transform_counts[i], iterations)

        for i in range(3):
            self.assertEqual(subset_size_counts[i], 0)

        self.assertEqual(subset_size_counts[3], iterations)

    def test_unfixed(self):
        iterations = 10000
        num_transforms = (0, 3)
        transform_counts = 3 * [0]
        subset_size_counts = 4 * [0]

        s = SomeOf((A(), B(), C()), num_transforms=num_transforms)

        for _ in range(iterations):
            output = s(1)
            subset_size = self.update_transform_count(transform_counts, output)
            subset_size_counts[subset_size] += 1

        for i in range(3):
            self.assertAlmostEqual(transform_counts[i] / iterations, 0.5, delta=0.01)

        for i in range(4):
            self.assertAlmostEqual(subset_size_counts[i] / iterations, 0.25, delta=0.01)

    def test_non_dict_metatensor(self):
        data = MetaTensor(1)
        s = SomeOf([A()], num_transforms=1)
        out = s(data)
        self.assertEqual(out, 2)
        inv = s.inverse(out)  # A() is not invertible, nothing happens
        self.assertEqual(inv, 2)

    @parameterized.expand(TEST_COMPOUND)
    def test_compound_pipeline(self, transform, expected_value):
        output = transform(1)
        self.assertEqual(output, expected_value)

    # Modified from RandomOrder
    @parameterized.expand(TEST_INVERSES)
    def test_inverse(self, transform, invertible, use_metatensor):
        data = {k: (i + 1) * 10.0 if not use_metatensor else MetaTensor((i + 1) * 10.0) for i, k in enumerate(KEYS)}
        fwd_data1 = transform(data)
        # test call twice won't affect inverse
        fwd_data2 = transform(data)

        if invertible:
            for k in KEYS:
                t = (
                    fwd_data1[TraceableTransform.trace_key(k)][-1]
                    if not use_metatensor
                    else fwd_data1[k].applied_operations[-1]
                )
                # make sure the SomeOf applied_order was stored
                self.assertEqual(t[TraceKeys.CLASS_NAME], SomeOf.__name__)

        # call the inverse
        fwd_inv_data1 = transform.inverse(fwd_data1)
        fwd_inv_data2 = transform.inverse(fwd_data2)

        fwd_data = [fwd_data1, fwd_data2]
        fwd_inv_data = [fwd_inv_data1, fwd_inv_data2]
        for i, _fwd_inv_data in enumerate(fwd_inv_data):
            if invertible:
                for k in KEYS:
                    # check transform was removed
                    if not use_metatensor:
                        self.assertTrue(
                            len(_fwd_inv_data[TraceableTransform.trace_key(k)])
                            < len(fwd_data[i][TraceableTransform.trace_key(k)])
                        )
                    # check data is same as original (and different from forward)
                    self.assertEqual(_fwd_inv_data[k], data[k])
                    self.assertNotEqual(_fwd_inv_data[k], fwd_data[i][k])
            else:
                # if not invertible, should not change the data
                self.assertDictEqual(fwd_data[i], _fwd_inv_data)

    def test_bad_inverse_data(self):
        tr = SomeOf((A(), B(), C()), num_transforms=1, weights=(1, 2, 1))
        self.assertRaises(RuntimeError, tr.inverse, [])

    def test_normalize_weights(self):
        tr = SomeOf((A(), B(), C()), num_transforms=1, weights=(1, 2, 1))
        self.assertTupleEqual(tr.weights, (0.25, 0.5, 0.25))

        tr = SomeOf((), num_transforms=1, weights=(1, 2, 1))
        self.assertIsNone(tr.weights)

    def test_no_weights_arg(self):
        tr = SomeOf((A(), B(), C(), D()), num_transforms=1)
        self.assertIsNone(tr.weights)

    def test_bad_weights(self):
        self.assertRaises(ValueError, SomeOf, (A(), B(), C()), num_transforms=1, weights=(1, 2))
        self.assertRaises(ValueError, SomeOf, (A(), B(), C()), num_transforms=1, weights=(0, 0, 0))
        self.assertRaises(ValueError, SomeOf, (A(), B(), C()), num_transforms=1, weights=(-1, 1, 1))

    def test_bad_num_transforms(self):
        self.assertRaises(ValueError, SomeOf, (A(), B(), C()), num_transforms=(-1, 2))
        self.assertRaises(ValueError, SomeOf, (A(), B(), C()), num_transforms="str")
        self.assertRaises(ValueError, SomeOf, (A(), B(), C()), num_transforms=(1, 2, 3))
        self.assertRaises(ValueError, SomeOf, (A(), B(), C()), num_transforms=("a", 1))


TEST_SOMEOF_EXTENDED_TEST_CASES = [
    [None, tuple()],
    [None, (sa.Rotate(np.pi / 8),)],
    [None, (sa.Flip(0), sa.Flip(1), sa.Rotate90(1), sa.Zoom(0.8), ia.NormalizeIntensity())],
    [("a",), (sd.Rotated(("a",), np.pi / 8),)],
]


class TestSomeOfAPITests(unittest.TestCase):

    @staticmethod
    def data_from_keys(keys):
        if keys is None:
            data = torch.unsqueeze(torch.tensor(np.arange(12 * 16).reshape(12, 16)), dim=0)
        else:
            data = {}
            for i_k, k in enumerate(keys):
                data[k] = torch.unsqueeze(torch.tensor(np.arange(12 * 16)).reshape(12, 16) + i_k * 192, dim=0)
        return data

    @parameterized.expand(TEST_SOMEOF_EXTENDED_TEST_CASES)
    def test_execute_change_start_end(self, keys, pipeline):
        data = self.data_from_keys(keys)

        c = SomeOf(deepcopy(pipeline))
        with self.assertRaises(ValueError):
            c(data, start=1)
        with self.assertRaises(ValueError):
            c(data, start=1)

        c = SomeOf(deepcopy(pipeline))
        with self.assertRaises(ValueError):
            c(data, end=1)
        with self.assertRaises(ValueError):
            c(data, end=1)


if __name__ == "__main__":
    unittest.main()
