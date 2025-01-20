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

import itertools
import unittest

from monai.transforms.adaptors import FunctionSignature, adaptor, apply_alias, to_kwargs


class TestAdaptors(unittest.TestCase):

    def test_function_signature(self):

        def foo(image, label=None, *a, **kw):
            pass

        _ = FunctionSignature(foo)

    def test_single_in_single_out(self):

        def foo(image):
            return image * 2

        it = itertools.product(["image", ["image"]], [None, "image", ["image"], {"image": "image"}])
        for i in it:
            d = {"image": 2}
            dres = adaptor(foo, i[0], i[1])(d)
            self.assertEqual(dres["image"], 4)

        d = {"image": 2}
        dres = adaptor(foo, "image")(d)
        self.assertEqual(dres["image"], 4)

        d = {"image": 2}
        dres = adaptor(foo, "image", "image")(d)
        self.assertEqual(dres["image"], 4)

        d = {"image": 2}
        dres = adaptor(foo, "image", {"image": "image"})(d)
        self.assertEqual(dres["image"], 4)

        d = {"img": 2}
        dres = adaptor(foo, "img", {"img": "image"})(d)
        self.assertEqual(dres["img"], 4)

        d = {"img": 2}
        dres = adaptor(foo, ["img"], {"img": "image"})(d)
        self.assertEqual(dres["img"], 4)

    def test_multi_in_single_out(self):

        def foo(image, label):
            return image * label

        it = itertools.product(["image", ["image"]], [None, ["image", "label"], {"image": "image", "label": "label"}])

        for i in it:
            d = {"image": 2, "label": 3}
            dres = adaptor(foo, i[0], i[1])(d)
            self.assertEqual(dres["image"], 6)
            self.assertEqual(dres["label"], 3)

        it = itertools.product(
            ["newimage", ["newimage"]], [None, ["image", "label"], {"image": "image", "label": "label"}]
        )

        for i in it:
            d = {"image": 2, "label": 3}
            dres = adaptor(foo, i[0], i[1])(d)
            self.assertEqual(dres["image"], 2)
            self.assertEqual(dres["label"], 3)
            self.assertEqual(dres["newimage"], 6)

        it = itertools.product(["img", ["img"]], [{"img": "image", "lbl": "label"}])

        for i in it:
            d = {"img": 2, "lbl": 3}
            dres = adaptor(foo, i[0], i[1])(d)
            self.assertEqual(dres["img"], 6)
            self.assertEqual(dres["lbl"], 3)

    def test_default_arg_single_out(self):

        def foo(a, b=2):
            return a * b

        d = {"a": 5}
        dres = adaptor(foo, "c")(d)
        self.assertEqual(dres["c"], 10)

        d = {"b": 5}
        with self.assertRaises(TypeError):
            dres = adaptor(foo, "c")(d)

    def test_multi_out(self):

        def foo(a, b):
            return a * b, a / b

        d = {"a": 3, "b": 4}
        dres = adaptor(foo, ["c", "d"])(d)
        self.assertEqual(dres["c"], 12)
        self.assertEqual(dres["d"], 3 / 4)

    def test_dict_out(self):

        def foo(a):
            return {"a": a * 2}

        d = {"a": 2}
        dres = adaptor(foo, {"a": "a"})(d)
        self.assertEqual(dres["a"], 4)

        d = {"b": 2}
        dres = adaptor(foo, {"a": "b"}, {"b": "a"})(d)
        self.assertEqual(dres["b"], 4)


class TestApplyAlias(unittest.TestCase):

    def test_apply_alias(self):

        def foo(d):
            d["x"] *= 2
            return d

        d = {"a": 1, "b": 3}
        result = apply_alias(foo, {"b": "x"})(d)
        self.assertDictEqual({"a": 1, "b": 6}, result)


class TestToKwargs(unittest.TestCase):

    def test_to_kwargs(self):

        def foo(**kwargs):
            results = {k: v * 2 for k, v in kwargs.items()}
            return results

        def compose_like(fn, data):
            data = fn(data)
            return data

        d = {"a": 1, "b": 2}

        actual = compose_like(to_kwargs(foo), d)
        self.assertDictEqual(actual, {"a": 2, "b": 4})

        with self.assertRaises(TypeError):
            actual = compose_like(foo, d)
