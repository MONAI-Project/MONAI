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

import logging
import sys
import unittest
from copy import deepcopy
from io import StringIO

import numpy as np
import torch
from parameterized import parameterized

import monai.transforms as mt
from monai.data import DataLoader, Dataset
from monai.transforms.compose import execute_compose
from monai.transforms.transform import Randomizable
from monai.utils import set_determinism


def data_from_keys(keys, h, w):
    if keys is None:
        data = torch.arange(h * w).reshape(1, h, w)
    else:
        data = {}
        for i_k, k in enumerate(keys):
            data[k] = torch.arange(h * w).reshape(1, h, w).mul_(i_k * h * w)
    return data


class _RandXform(Randomizable):

    def randomize(self):
        self.val = self.R.random_sample()

    def __call__(self, __unused):
        self.randomize()
        return self.val


class TestCompose(unittest.TestCase):

    def test_empty_compose(self):
        c = mt.Compose()
        i = 1
        self.assertEqual(c(i), 1)

    def test_non_dict_compose(self):

        def a(i):
            return i + "a"

        def b(i):
            return i + "b"

        c = mt.Compose([a, b, a, b])
        self.assertEqual(c(""), "abab")

    def test_dict_compose(self):

        def a(d):
            d = dict(d)
            d["a"] += 1
            return d

        def b(d):
            d = dict(d)
            d["b"] += 1
            return d

        transforms = [a, b, a, b, a]
        data = {"a": 0, "b": 0}
        expected = {"a": 3, "b": 2}

        self.assertDictEqual(mt.Compose(transforms)(data), expected)
        self.assertDictEqual(execute_compose(data, transforms), expected)

    def test_list_dict_compose(self):

        def a(d):  # transform to handle dict data
            d = dict(d)
            d["a"] += 1
            return d

        def b(d):  # transform to generate a batch list of data
            d = dict(d)
            d["b"] += 1
            d = [d] * 5
            return d

        def c(d):  # transform to handle dict data
            d = dict(d)
            d["c"] += 1
            return d

        transforms = [a, a, b, c, c]
        data = {"a": 0, "b": 0, "c": 0}
        expected = {"a": 2, "b": 1, "c": 2}
        value = mt.Compose(transforms)(data)
        for item in value:
            self.assertDictEqual(item, expected)
        value = execute_compose(data, transforms)
        for item in value:
            self.assertDictEqual(item, expected)

    def test_non_dict_compose_with_unpack(self):

        def a(i, i2):
            return i + "a", i2 + "a2"

        def b(i, i2):
            return i + "b", i2 + "b2"

        transforms = [a, b, a, b]
        data = ("", "")
        expected = ("abab", "a2b2a2b2")
        self.assertEqual(mt.Compose(transforms, map_items=False, unpack_items=True)(data), expected)
        self.assertEqual(execute_compose(data, transforms, map_items=False, unpack_items=True), expected)

    def test_list_non_dict_compose_with_unpack(self):

        def a(i, i2):
            return i + "a", i2 + "a2"

        def b(i, i2):
            return i + "b", i2 + "b2"

        transforms = [a, b, a, b]
        data = [("", ""), ("t", "t")]
        expected = [("abab", "a2b2a2b2"), ("tabab", "ta2b2a2b2")]
        self.assertEqual(mt.Compose(transforms, unpack_items=True)(data), expected)
        self.assertEqual(execute_compose(data, transforms, unpack_items=True), expected)

    def test_list_non_dict_compose_with_unpack_map_2(self):

        def a(i, i2):
            return i + "a", i2 + "a2"

        def b(i, i2):
            return i + "b", i2 + "b2"

        transforms = [a, b, a, b]
        data = [[("", ""), ("", "")], [("t", "t"), ("t", "t")]]
        expected = [[("abab", "a2b2a2b2"), ("abab", "a2b2a2b2")], [("tabab", "ta2b2a2b2"), ("tabab", "ta2b2a2b2")]]
        self.assertEqual(mt.Compose(transforms, map_items=2, unpack_items=True)(data), expected)
        self.assertEqual(execute_compose(data, transforms, map_items=2, unpack_items=True), expected)

    def test_list_dict_compose_no_map(self):

        def a(d):  # transform to handle dict data
            d = dict(d)
            d["a"] += 1
            return d

        def b(d):  # transform to generate a batch list of data
            d = dict(d)
            d["b"] += 1
            d = [d] * 5
            return d

        def c(d):  # transform to handle dict data
            d = [dict(di) for di in d]
            for di in d:
                di["c"] += 1
            return d

        transforms = [a, a, b, c, c]
        data = {"a": 0, "b": 0, "c": 0}
        expected = {"a": 2, "b": 1, "c": 2}
        value = mt.Compose(transforms, map_items=False)(data)
        for item in value:
            self.assertDictEqual(item, expected)
        value = execute_compose(data, transforms, map_items=False)
        for item in value:
            self.assertDictEqual(item, expected)

    def test_random_compose(self):

        class _Acc(Randomizable):
            self.rand = 0.0

            def randomize(self, data=None):
                self.rand = self.R.rand()

            def __call__(self, data):
                self.randomize()
                return self.rand + data

        c = mt.Compose([_Acc(), _Acc()])
        self.assertNotAlmostEqual(c(0), c(0))
        c.set_random_state(123)
        self.assertAlmostEqual(c(1), 1.61381597)
        c.set_random_state(223)
        c.randomize()
        self.assertAlmostEqual(c(1), 1.90734751)

    def test_randomize_warn(self):

        class _RandomClass(Randomizable):

            def randomize(self, foo1, foo2):
                pass

            def __call__(self, data):
                pass

        c = mt.Compose([_RandomClass(), _RandomClass()])
        with self.assertWarns(Warning):
            c.randomize()

    def test_err_msg(self):
        transforms = mt.Compose([abs, mt.EnsureChannelFirst(), round])
        with self.assertRaisesRegex(Exception, "EnsureChannelFirst"):
            transforms(42.1)

    def test_data_loader(self):
        xform_1 = mt.Compose([_RandXform()])
        train_ds = Dataset([1], transform=xform_1)

        xform_1.set_random_state(123)
        out_1 = train_ds[0]
        self.assertAlmostEqual(out_1, 0.2045649)

        set_determinism(seed=123)
        train_loader = DataLoader(train_ds, num_workers=0)
        out_1 = next(iter(train_loader))
        self.assertAlmostEqual(out_1.cpu().item(), 0.0409280)

        if sys.platform != "win32":  # skip multi-worker tests on win32
            train_loader = DataLoader(train_ds, num_workers=1)
            out_1 = next(iter(train_loader))
            self.assertAlmostEqual(out_1.cpu().item(), 0.78663897075)

            train_loader = DataLoader(train_ds, num_workers=2)
            out_1 = next(iter(train_loader))
            self.assertAlmostEqual(out_1.cpu().item(), 0.785907334)
        set_determinism(None)

    def test_data_loader_2(self):
        set_determinism(seed=123)
        xform_2 = mt.Compose([_RandXform(), _RandXform()])
        train_ds = Dataset([1], transform=xform_2)

        out_2 = train_ds[0]
        self.assertAlmostEqual(out_2, 0.4092510)

        train_loader = DataLoader(train_ds, num_workers=0)
        out_2 = next(iter(train_loader))
        self.assertAlmostEqual(out_2.cpu().item(), 0.98921915918)

        if sys.platform != "win32":  # skip multi-worker tests on win32
            train_loader = DataLoader(train_ds, num_workers=1)
            out_2 = next(iter(train_loader))
            self.assertAlmostEqual(out_2.cpu().item(), 0.32985207)

            train_loader = DataLoader(train_ds, num_workers=2)
            out_1 = next(iter(train_loader))
            self.assertAlmostEqual(out_1.cpu().item(), 0.28602141572)
        set_determinism(None)

    def test_flatten_and_len(self):
        x = mt.EnsureChannelFirst(channel_dim="no_channel")
        t1 = mt.Compose([x, x, x, x, mt.Compose([mt.Compose([x, x]), x, x])])

        t2 = t1.flatten()
        for t in t2.transforms:
            self.assertNotIsInstance(t, mt.Compose)

        # test len
        self.assertEqual(len(t1), 8)

    def test_backwards_compatible_imports(self):
        from monai.transforms.transform import MapTransform, RandomizableTransform, Transform  # noqa: F401


TEST_COMPOSE_EXECUTE_TEST_CASES = [
    [None, tuple()],
    [None, (mt.Rotate(np.pi / 8),)],
    [None, (mt.Flip(0), mt.Flip(1), mt.Rotate90(1), mt.Zoom(0.8), mt.NormalizeIntensity())],
    [("a",), (mt.Rotated(("a",), np.pi / 8),)],
]


class TestComposeExecute(unittest.TestCase):

    @parameterized.expand(TEST_COMPOSE_EXECUTE_TEST_CASES)
    def test_compose_execute_equivalence(self, keys, pipeline):
        data = data_from_keys(keys, 12, 16)

        expected = mt.Compose(deepcopy(pipeline))(data)

        for cutoff in range(len(pipeline)):
            c = mt.Compose(deepcopy(pipeline))
            actual = c(c(data, end=cutoff), start=cutoff)
            if isinstance(actual, dict):
                for k in actual.keys():
                    self.assertTrue(torch.allclose(expected[k], actual[k]))
            else:
                self.assertTrue(torch.allclose(expected, actual))

            p = deepcopy(pipeline)
            actual = execute_compose(execute_compose(data, p, start=0, end=cutoff), p, start=cutoff)
            if isinstance(actual, dict):
                for k in actual.keys():
                    self.assertTrue(torch.allclose(expected[k], actual[k]))
            else:
                self.assertTrue(torch.allclose(expected, actual))

    @parameterized.expand(TEST_COMPOSE_EXECUTE_TEST_CASES)
    def test_compose_execute_bad_start_param(self, keys, pipeline):
        data = data_from_keys(keys, 12, 16)

        c = mt.Compose(deepcopy(pipeline))
        with self.assertRaises(ValueError):
            c(data, start=None)
        with self.assertRaises(ValueError):
            c(data, start=None)

        with self.assertRaises(ValueError):
            execute_compose(data, deepcopy(pipeline), start=None)

        c = mt.Compose(deepcopy(pipeline))
        with self.assertRaises(ValueError):
            c(data, start=-1)
        with self.assertRaises(ValueError):
            c(data, start=-1)

        with self.assertRaises(ValueError):
            execute_compose(data, deepcopy(pipeline), start=-1)

    @parameterized.expand(TEST_COMPOSE_EXECUTE_TEST_CASES)
    def test_compose_execute_negative_range(self, keys, pipeline):
        data = data_from_keys(keys, 12, 16)

        with self.assertRaises(ValueError):
            c = mt.Compose(deepcopy(pipeline))
            c(data, start=2, end=1)

        with self.assertRaises(ValueError):
            execute_compose(data, deepcopy(pipeline), start=2, end=1)

    @parameterized.expand(TEST_COMPOSE_EXECUTE_TEST_CASES)
    def test_compose_execute_bad_end_param(self, keys, pipeline):
        data = data_from_keys(keys, 12, 16)

        with self.assertRaises(ValueError):
            c = mt.Compose(deepcopy(pipeline))
            c(data, end=len(pipeline) + 1)

        with self.assertRaises(ValueError):
            execute_compose(data, deepcopy(pipeline), end=len(pipeline) + 1)

    @parameterized.expand(TEST_COMPOSE_EXECUTE_TEST_CASES)
    def test_compose_execute_empty_range(self, keys, pipeline):
        data = data_from_keys(keys, 12, 16)

        c = mt.Compose(deepcopy(pipeline))
        for i in range(len(pipeline)):
            result = c(data, start=i, end=i)
            self.assertIs(data, result)

    @parameterized.expand(TEST_COMPOSE_EXECUTE_TEST_CASES)
    def test_compose_with_logger(self, keys, pipeline):
        data = data_from_keys(keys, 12, 16)

        c = mt.Compose(deepcopy(pipeline), log_stats="a_logger_name")
        c(data)


TEST_COMPOSE_EXECUTE_LOGGING_TEST_CASES = [
    [
        None,
        (mt.Flip(0), mt.Spacing((1.2, 1.2)), mt.Flip(1), mt.Rotate90(1), mt.Zoom(0.8), mt.NormalizeIntensity()),
        False,
        (
            "INFO - Apply pending transforms - lazy: False, pending: 0, "
            "upcoming 'Flip', transform.lazy: False\n"
            "INFO - Apply pending transforms - lazy: False, pending: 0, "
            "upcoming 'Spacing', transform.lazy: False\n"
            "INFO - Apply pending transforms - lazy: False, pending: 0, "
            "upcoming 'Flip', transform.lazy: False\n"
            "INFO - Apply pending transforms - lazy: False, pending: 0, "
            "upcoming 'Rotate90', transform.lazy: False\n"
            "INFO - Apply pending transforms - lazy: False, pending: 0, "
            "upcoming 'Zoom', transform.lazy: False\n"
            "INFO - Apply pending transforms - lazy: False, pending: 0, "
            "upcoming 'NormalizeIntensity', transform is not lazy\n"
        ),
    ],
    [
        None,
        (
            mt.Flip(0, lazy=True),
            mt.Spacing((1.2, 1.2), lazy=True),
            mt.Flip(1, lazy=True),
            mt.Rotate90(1),
            mt.Zoom(0.8, lazy=True),
            mt.NormalizeIntensity(),
        ),
        None,
        (
            "INFO - Accumulate pending transforms - lazy: None, pending: 0, "
            "upcoming 'Flip', transform.lazy: True\n"
            "INFO - Accumulate pending transforms - lazy: None, pending: 1, "
            "upcoming 'Spacing', transform.lazy: True\n"
            "INFO - Accumulate pending transforms - lazy: None, pending: 2, "
            "upcoming 'Flip', transform.lazy: True\n"
            "INFO - Apply pending transforms - lazy: None, pending: 3, "
            "upcoming 'Rotate90', transform.lazy: False\n"
            "INFO - Pending transforms applied: applied_operations: 3\n"
            "INFO - Accumulate pending transforms - lazy: None, pending: 0, "
            "upcoming 'Zoom', transform.lazy: True\n"
            "INFO - Apply pending transforms - lazy: None, pending: 1, "
            "upcoming 'NormalizeIntensity', transform is not lazy\n"
            "INFO - Pending transforms applied: applied_operations: 5\n"
        ),
    ],
    [
        None,
        (mt.Flip(0), mt.Spacing((1.2, 1.2)), mt.Flip(1), mt.Rotate90(1), mt.Zoom(0.8), mt.NormalizeIntensity()),
        True,
        (
            "INFO - Accumulate pending transforms - lazy: True, pending: 0, "
            "upcoming 'Flip', transform.lazy: False (overridden)\n"
            "INFO - Accumulate pending transforms - lazy: True, pending: 1, "
            "upcoming 'Spacing', transform.lazy: False (overridden)\n"
            "INFO - Accumulate pending transforms - lazy: True, pending: 2, "
            "upcoming 'Flip', transform.lazy: False (overridden)\n"
            "INFO - Accumulate pending transforms - lazy: True, pending: 3, "
            "upcoming 'Rotate90', transform.lazy: False (overridden)\n"
            "INFO - Accumulate pending transforms - lazy: True, pending: 4, "
            "upcoming 'Zoom', transform.lazy: False (overridden)\n"
            "INFO - Apply pending transforms - lazy: True, pending: 5, "
            "upcoming 'NormalizeIntensity', transform is not lazy\n"
            "INFO - Pending transforms applied: applied_operations: 5\n"
        ),
    ],
    [
        ("a", "b"),
        (
            mt.Flipd(("a", "b"), 0),
            mt.Spacingd(("a", "b"), 1.2),
            mt.Rotate90d(("a", "b"), 1),
            mt.NormalizeIntensityd(("a",)),
        ),
        True,
        (
            "INFO - Accumulate pending transforms - lazy: True, key: 'a', pending: 0, "
            "upcoming 'Flipd', transform.lazy: False (overridden)\n"
            "INFO - Accumulate pending transforms - lazy: True, key: 'b', pending: 0, "
            "upcoming 'Flipd', transform.lazy: False (overridden)\n"
            "INFO - Accumulate pending transforms - lazy: True, key: 'a', pending: 1, "
            "upcoming 'Spacingd', transform.lazy: False (overridden)\n"
            "INFO - Accumulate pending transforms - lazy: True, key: 'b', pending: 1, "
            "upcoming 'Spacingd', transform.lazy: False (overridden)\n"
            "INFO - Accumulate pending transforms - lazy: True, key: 'a', pending: 2, "
            "upcoming 'Rotate90d', transform.lazy: False (overridden)\n"
            "INFO - Accumulate pending transforms - lazy: True, key: 'b', pending: 2, "
            "upcoming 'Rotate90d', transform.lazy: False (overridden)\n"
            "INFO - Apply pending transforms - lazy: True, key: 'a', pending: 3, "
            "upcoming 'NormalizeIntensityd', transform is not lazy\n"
            "INFO - Pending transforms applied: key: 'a', applied_operations: 3\n"
            "INFO - Pending transforms applied: key: 'b', applied_operations: 3\n"
        ),
    ],
    [
        ("a", "b"),
        (
            mt.Flipd(keys="a", spatial_axis=0),
            mt.Rotate90d(keys="b", k=1, allow_missing_keys=True),
            mt.Zoomd(keys=("a", "b"), zoom=0.8, allow_missing_keys=True),
            mt.Spacingd(keys="a", pixdim=1.2),
        ),
        True,
        (
            "INFO - Accumulate pending transforms - lazy: True, key: 'a', pending: 0, "
            "upcoming 'Flipd', transform.lazy: False (overridden)\n"
            "INFO - Accumulate pending transforms - lazy: True, key: 'b', pending: 0, "
            "upcoming 'Rotate90d', transform.lazy: False (overridden)\n"
            "INFO - Accumulate pending transforms - lazy: True, key: 'a', pending: 1, "
            "upcoming 'Zoomd', transform.lazy: False (overridden)\n"
            "INFO - Accumulate pending transforms - lazy: True, key: 'b', pending: 1, "
            "upcoming 'Zoomd', transform.lazy: False (overridden)\n"
            "INFO - Accumulate pending transforms - lazy: True, key: 'a', pending: 2, "
            "upcoming 'Spacingd', transform.lazy: False (overridden)\n"
            "INFO - Pending transforms applied: key: 'a', applied_operations: 3\n"
            "INFO - Pending transforms applied: key: 'b', applied_operations: 2\n"
        ),
    ],
    [
        None,
        (
            mt.Flip(0),
            mt.Spacing((1.2, 1.2)),
            mt.Flip(1),
            mt.ApplyPending(),
            mt.Rotate90(1),
            mt.Zoom(0.8),
            mt.NormalizeIntensity(),
        ),
        False,
        (
            "INFO - Apply pending transforms - lazy: False, pending: 0, "
            "upcoming 'Flip', transform.lazy: False\n"
            "INFO - Apply pending transforms - lazy: False, pending: 0, "
            "upcoming 'Spacing', transform.lazy: False\n"
            "INFO - Apply pending transforms - lazy: False, pending: 0, "
            "upcoming 'Flip', transform.lazy: False\n"
            "INFO - Apply pending transforms - lazy: False, pending: 0, "
            "upcoming 'ApplyPending', transform is not lazy\n"
            "INFO - Apply pending transforms - lazy: False, pending: 0, "
            "upcoming 'Rotate90', transform.lazy: False\n"
            "INFO - Apply pending transforms - lazy: False, pending: 0, "
            "upcoming 'Zoom', transform.lazy: False\n"
            "INFO - Apply pending transforms - lazy: False, pending: 0, "
            "upcoming 'NormalizeIntensity', transform is not lazy\n"
        ),
    ],
    [
        None,
        (
            mt.Flip(0),
            mt.Spacing((1.2, 1.2)),
            mt.Flip(1),
            mt.ApplyPending(),
            mt.Rotate90(1),
            mt.Zoom(0.8),
            mt.NormalizeIntensity(),
        ),
        True,
        (
            "INFO - Accumulate pending transforms - lazy: True, pending: 0, "
            "upcoming 'Flip', transform.lazy: False (overridden)\n"
            "INFO - Accumulate pending transforms - lazy: True, pending: 1, "
            "upcoming 'Spacing', transform.lazy: False (overridden)\n"
            "INFO - Accumulate pending transforms - lazy: True, pending: 2, "
            "upcoming 'Flip', transform.lazy: False (overridden)\n"
            "INFO - Apply pending transforms - lazy: True, pending: 3, "
            "upcoming 'ApplyPending', transform is not lazy\n"
            "INFO - Pending transforms applied: applied_operations: 3\n"
            "INFO - Accumulate pending transforms - lazy: True, pending: 0, "
            "upcoming 'Rotate90', transform.lazy: False (overridden)\n"
            "INFO - Accumulate pending transforms - lazy: True, pending: 1, "
            "upcoming 'Zoom', transform.lazy: False (overridden)\n"
            "INFO - Apply pending transforms - lazy: True, pending: 2, "
            "upcoming 'NormalizeIntensity', transform is not lazy\n"
            "INFO - Pending transforms applied: applied_operations: 5\n"
        ),
    ],
    [
        ("a", "b"),
        (
            mt.Flipd(keys="a", spatial_axis=0),
            mt.Rotate90d(keys="b", k=1, allow_missing_keys=True),
            mt.ApplyPendingd(keys=("a", "b")),
            mt.Zoomd(keys=("a", "b"), zoom=0.8, allow_missing_keys=True),
            mt.Spacingd(keys="a", pixdim=1.2),
        ),
        True,
        (
            "INFO - Accumulate pending transforms - lazy: True, key: 'a', pending: 0, "
            "upcoming 'Flipd', transform.lazy: False (overridden)\n"
            "INFO - Accumulate pending transforms - lazy: True, key: 'b', pending: 0, "
            "upcoming 'Rotate90d', transform.lazy: False (overridden)\n"
            "INFO - Apply pending transforms - lazy: True, key: 'a', pending: 1, "
            "upcoming 'ApplyPendingd', transform is not lazy\n"
            "INFO - Apply pending transforms - lazy: True, key: 'b', pending: 1, "
            "upcoming 'ApplyPendingd', transform is not lazy\n"
            "INFO - Pending transforms applied: key: 'a', applied_operations: 1\n"
            "INFO - Pending transforms applied: key: 'b', applied_operations: 1\n"
            "INFO - Accumulate pending transforms - lazy: True, key: 'a', pending: 0, "
            "upcoming 'Zoomd', transform.lazy: False (overridden)\n"
            "INFO - Accumulate pending transforms - lazy: True, key: 'b', pending: 0, "
            "upcoming 'Zoomd', transform.lazy: False (overridden)\n"
            "INFO - Accumulate pending transforms - lazy: True, key: 'a', pending: 1, "
            "upcoming 'Spacingd', transform.lazy: False (overridden)\n"
            "INFO - Pending transforms applied: key: 'a', applied_operations: 3\n"
            "INFO - Pending transforms applied: key: 'b', applied_operations: 2\n"
        ),
    ],
]

TEST_COMPOSE_LAZY_ON_CALL_LOGGING_TEST_CASES = [
    [
        mt.Compose,
        (mt.Flip(0), mt.Spacing((1.2, 1.2))),
        True,
        (
            "INFO - Accumulate pending transforms - lazy: True, pending: 0, "
            "upcoming 'Flip', transform.lazy: False (overridden)\n"
            "INFO - Accumulate pending transforms - lazy: True, pending: 1, "
            "upcoming 'Spacing', transform.lazy: False (overridden)\n"
            "INFO - Pending transforms applied: applied_operations: 2\n"
        ),
    ],
    [
        mt.SomeOf,
        (mt.Flip(0),),
        True,
        (
            "INFO - Accumulate pending transforms - lazy: True, pending: 0, "
            "upcoming 'Flip', transform.lazy: False (overridden)\n"
            "INFO - Pending transforms applied: applied_operations: 1\n"
        ),
    ],
    [
        mt.RandomOrder,
        (mt.Flip(0),),
        True,
        (
            "INFO - Accumulate pending transforms - lazy: True, pending: 0, "
            "upcoming 'Flip', transform.lazy: False (overridden)\n"
            "INFO - Pending transforms applied: applied_operations: 1\n"
        ),
    ],
    [
        mt.OneOf,
        (mt.Flip(0),),
        True,
        (
            "INFO - Accumulate pending transforms - lazy: True, pending: 0, "
            "upcoming 'Flip', transform.lazy: False (overridden)\n"
            "INFO - Pending transforms applied: applied_operations: 1\n"
        ),
    ],
    [
        mt.OneOf,
        (mt.Flip(0),),
        False,
        ("INFO - Apply pending transforms - lazy: False, pending: 0, " "upcoming 'Flip', transform.lazy: False\n"),
    ],
]


class TestComposeExecuteWithLogging(unittest.TestCase):
    LOGGER_NAME = "a_logger_name"

    def init_logger(self, name=LOGGER_NAME):
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        formatter = logging.Formatter("%(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        while len(logger.handlers) > 0:
            logger.removeHandler(logger.handlers[-1])
        logger.addHandler(handler)
        return handler, stream

    @parameterized.expand(TEST_COMPOSE_EXECUTE_LOGGING_TEST_CASES)
    def test_compose_with_logging(self, keys, pipeline, lazy, expected):
        handler, stream = self.init_logger(name=self.LOGGER_NAME)

        data = data_from_keys(keys, 12, 16)
        c = mt.Compose(deepcopy(pipeline), lazy=lazy, log_stats=self.LOGGER_NAME)
        c(data)

        handler.flush()
        actual = stream.getvalue()
        self.assertEqual(actual, expected)

    @parameterized.expand(TEST_COMPOSE_LAZY_ON_CALL_LOGGING_TEST_CASES)
    def test_compose_lazy_on_call_with_logging(self, compose_type, pipeline, lazy_on_call, expected):
        handler, stream = self.init_logger(name=self.LOGGER_NAME)

        data = data_from_keys(None, 12, 16)
        c = compose_type(deepcopy(pipeline), log_stats=self.LOGGER_NAME)
        c(data, lazy=lazy_on_call)

        handler.flush()
        actual = stream.getvalue()
        self.assertEqual(actual, expected)


class TestOps:

    @staticmethod
    def concat(value):

        def _inner(data):
            return data + value

        return _inner

    @staticmethod
    def concatd(value):

        def _inner(data):
            return {k: v + value for k, v in data.items()}

        return _inner

    @staticmethod
    def concata(value):

        def _inner(data1, data2):
            return data1 + value, data2 + value

        return _inner


TEST_COMPOSE_EXECUTE_FLAG_TEST_CASES = [
    [{}, ("",), (TestOps.concat("a"), TestOps.concat("b"))],
    [{"unpack_items": True}, ("x", "y"), (TestOps.concat("a"), TestOps.concat("b"))],
    [{"map_items": False}, {"x": "1", "y": "2"}, (TestOps.concatd("a"), TestOps.concatd("b"))],
    [{"unpack_items": True, "map_items": False}, ("x", "y"), (TestOps.concata("a"), TestOps.concata("b"))],
]


class TestComposeExecuteWithFlags(unittest.TestCase):

    @parameterized.expand(TEST_COMPOSE_EXECUTE_FLAG_TEST_CASES)
    def test_compose_execute_equivalence_with_flags(self, flags, data, pipeline):
        expected = mt.Compose(pipeline, **flags)(data)

        for cutoff in range(len(pipeline)):
            c = mt.Compose(deepcopy(pipeline), **flags)
            actual = c(c(data, end=cutoff), start=cutoff)
            if isinstance(actual, dict):
                for k in actual.keys():
                    self.assertEqual(expected[k], actual[k])
            else:
                self.assertEqual(expected, actual)

            p = deepcopy(pipeline)
            actual = execute_compose(execute_compose(data, p, start=0, end=cutoff, **flags), p, start=cutoff, **flags)
            if isinstance(actual, dict):
                for k in actual.keys():
                    self.assertEqual(expected[k], actual[k])
            else:
                self.assertEqual(expected, actual)


class TestComposeCallableInput(unittest.TestCase):

    def test_value_error_when_not_sequence(self):
        data = torch.tensor(np.random.randn(1, 5, 5))

        xform = mt.Compose([mt.Flip(0), mt.Flip(0)])
        res = xform(data)
        np.testing.assert_allclose(data, res, atol=1e-3)

        with self.assertRaises(ValueError):
            mt.Compose(mt.Flip(0), mt.Flip(0))(data)


if __name__ == "__main__":
    unittest.main()
