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
from functools import partial
from typing import Callable

import torch
from parameterized import parameterized

import monai
from monai.bundle import ComponentLocator, ConfigComponent, ConfigExpression, ConfigItem
from monai.data import DataLoader, Dataset
from monai.transforms import LoadImaged, RandTorchVisiond
from monai.utils import min_version, optional_import

_, has_tv = optional_import("torchvision", "0.8.0", min_version)

TEST_CASE_1 = [{"lr": 0.001}, 0.0001]

TEST_CASE_2 = [{"_target_": "LoadImaged", "keys": ["image"], "_desc_": "an image reader for 'image'"}, LoadImaged]
# test full module path
TEST_CASE_3 = [{"_target_": "monai.transforms.LoadImaged", "keys": ["image"]}, LoadImaged]
# test `_disabled_`
TEST_CASE_4 = [{"_target_": "LoadImaged", "_disabled_": True, "keys": ["image"]}, dict]
# test `_disabled_` with string
TEST_CASE_5 = [{"_target_": "LoadImaged", "_disabled_": "true", "keys": ["image"]}, dict]
# test non-monai modules and excludes
TEST_CASE_6 = [{"_target_": "torch.optim.Adam", "params": torch.nn.PReLU().parameters(), "lr": 1e-4}, torch.optim.Adam]
TEST_CASE_7 = [{"_target_": "decollate_batch", "detach": True, "pad": True, "_mode_": "callable"}, partial]
# test args contains "name" field
TEST_CASE_8 = [
    {"_target_": "RandTorchVisiond", "keys": "image", "name": "ColorJitter", "brightness": 0.25},
    RandTorchVisiond,
]
# test execute some function in args, test pre-imported global packages `monai`
TEST_CASE_9 = ["collate_fn", "$monai.data.list_data_collate"]
# test lambda function
TEST_CASE_10 = ["collate_fn", "$lambda x: monai.data.list_data_collate(x) + torch.tensor(var)"]
# test regular expression with reference
TEST_CASE_11 = ["collate_fn", "$var + 100"]


class TestConfigItem(unittest.TestCase):

    @parameterized.expand([TEST_CASE_1])
    def test_item(self, test_input, expected):
        item = ConfigItem(config=test_input)
        conf = item.get_config()
        conf["lr"] = 0.0001
        item.update_config(config=conf)
        self.assertEqual(item.get_config()["lr"], expected)

    @parameterized.expand(
        [TEST_CASE_2, TEST_CASE_3, TEST_CASE_4, TEST_CASE_5, TEST_CASE_6, TEST_CASE_7]
        + ([TEST_CASE_8] if has_tv else [])
    )
    def test_component(self, test_input, output_type):
        locator = ComponentLocator(excludes=["metrics"])
        configer = ConfigComponent(id="test", config=test_input, locator=locator)
        ret = configer.instantiate()
        if test_input.get("_disabled_", False):
            # test `_disabled_` works fine
            self.assertEqual(ret, None)
            return
        self.assertTrue(isinstance(ret, output_type))
        if isinstance(ret, LoadImaged):
            self.assertEqual(ret.keys[0], "image")

    @parameterized.expand([TEST_CASE_9, TEST_CASE_10, TEST_CASE_11])
    def test_expression(self, id, test_input):
        configer = ConfigExpression(id=id, config=test_input, globals={"monai": monai, "torch": torch})
        var = 100
        ret = configer.evaluate(globals={"var": var, "monai": monai})  # `{"monai": monai}` to verify the warning
        if isinstance(ret, Callable):
            self.assertTrue(isinstance(ret([torch.tensor(1), torch.tensor(2)]), torch.Tensor))
        else:
            # also test the `locals` for regular expressions
            ret = configer.evaluate(locals={"var": var})
            self.assertEqual(ret, 200)

    def test_lazy_instantiation(self):
        config = {"_target_": "DataLoader", "dataset": Dataset(data=[1, 2]), "batch_size": 2}
        configer = ConfigComponent(config=config, locator=None)
        init_config = configer.get_config()
        # modify config content at runtime
        init_config["batch_size"] = 4
        configer.update_config(config=init_config)

        ret = configer.instantiate()
        self.assertTrue(isinstance(ret, DataLoader))
        self.assertEqual(ret.batch_size, 4)

    @parameterized.expand([("$import json", "json"), ("$import json as j", "j")])
    def test_import(self, stmt, mod_name):
        test_globals = {}
        ConfigExpression(id="", config=stmt, globals=test_globals).evaluate()
        self.assertTrue(callable(test_globals[mod_name].dump))

    @parameterized.expand(
        [
            ("$from json import dump", "dump"),
            ("$from json import dump, dumps", "dump"),
            ("$from json import dump as jd", "jd"),
            ("$from json import dump as jd, dumps as ds", "jd"),
        ]
    )
    def test_import_from(self, stmt, mod_name):
        test_globals = {}
        ConfigExpression(id="", config=stmt, globals=test_globals).evaluate()
        self.assertTrue(callable(test_globals[mod_name]))
        self.assertTrue(ConfigExpression.is_import_statement(ConfigExpression(id="", config=stmt).config))

    @parameterized.expand(
        [("$from json import dump", True), ("$print()", False), ("$import json", True), ("import json", False)]
    )
    def test_is_import_stmt(self, stmt, expected):
        expr = ConfigExpression(id="", config=stmt)
        flag = expr.is_import_statement(expr.config)
        self.assertEqual(flag, expected)

    def test_error_expr(self):
        with self.assertRaisesRegex(RuntimeError, r"1\+\[\]"):
            ConfigExpression(id="", config="$1+[]").evaluate()


if __name__ == "__main__":
    unittest.main()
