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
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock, skipUnless

import numpy as np
from parameterized import parameterized

from monai.bundle import ConfigParser, ReferenceResolver
from monai.bundle.config_item import ConfigItem
from monai.data import DataLoader, Dataset
from monai.transforms import Compose, LoadImaged, RandTorchVisiond
from monai.utils import min_version, optional_import
from tests.util import TimedCall

_, has_tv = optional_import("torchvision", "0.8.0", min_version)
_, has_yaml = optional_import("yaml")


@TimedCall(seconds=100, force_quit=True)
def case_pdb(sarg=None):
    config = {"transform": {"_target_": "Compose", "transforms": [], "_debug_": True}}
    parser = ConfigParser(config=config)
    parser.get_parsed_content()


@TimedCall(seconds=100, force_quit=True)
def case_pdb_inst(sarg=None):
    config = {"transform": {"_target_": "Compose", "transforms": [], "_mode_": "debug"}}
    parser = ConfigParser(config=config)
    return parser.transform


# test the resolved and parsed instances
TEST_CASE_1 = [
    {
        "transform": {
            "_target_": "Compose",
            "transforms": [
                {"_target_": "LoadImaged", "keys": "image"},
                # test relative id in `keys`
                {"_target_": "RandTorchVisiond", "keys": "@##0#keys", "name": "ColorJitter", "brightness": 0.25},
            ],
        },
        "dataset": {"_target_": "Dataset", "data": [1, 2], "transform": "@transform"},
        "dataloader": {
            "_target_": "DataLoader",
            # test relative id in `dataset`
            "dataset": "@##dataset",
            "batch_size": 2,
            "collate_fn": "$monai.data.list_data_collate",
        },
    },
    ["transform", "transform#transforms#0", "transform#transforms#1", "dataset", "dataloader"],
    [Compose, LoadImaged, RandTorchVisiond, Dataset, DataLoader],
]


class TestClass:
    @staticmethod
    def compute(a, b, func=lambda x, y: x + y):
        return func(a, b)

    @classmethod
    def cls_compute(cls, a, b, func=lambda x, y: x + y):
        return cls.compute(a, b, func)

    def __call__(self, a, b):
        return self.compute(a, b)


TEST_CASE_2 = [
    {
        "basic_func": "$lambda x, y: x + y",
        "static_func": "$TestClass.compute",
        "cls_func": "$TestClass.cls_compute",
        "lambda_static_func": "$lambda x, y: TestClass.compute(x, y)",
        "lambda_cls_func": "$lambda x, y: TestClass.cls_compute(x, y)",
        "compute": {"_target_": "tests.bundle.test_config_parser.TestClass.compute", "func": "@basic_func"},
        "cls_compute": {"_target_": "tests.bundle.test_config_parser.TestClass.cls_compute", "func": "@basic_func"},
        "call_compute": {"_target_": "tests.bundle.test_config_parser.TestClass"},
        "error_func": "$TestClass.__call__",
        "<test>": "$lambda x, y: x + y",
    }
]

TEST_CASE_3 = [
    {
        "A": 1,
        "B": "@A",
        "C": "@#A",
        "D": {"key": "@##A", "value1": 2, "value2": "%#value1", "value3": [3, 4, "@#1", "$100 + @#0 + @##value1"]},
    }
]

TEST_CASE_4 = [{"A": 1, "B": "@A", "C": "@D", "E": "$'test' + '@F'"}]

TEST_CASE_5 = [{"training": {"A": 1, "A_B": 2}, "total": "$@training#A + @training#A_B + 1"}, 4]

TEST_CASE_DUPLICATED_KEY_JSON = ["""{"key": {"unique": 1, "duplicate": 0, "duplicate": 4 } }""", "json", 1, [0, 4]]

TEST_CASE_DUPLICATED_KEY_YAML = [
    """key:
    unique: 1
    duplicate: 0
    duplicate: 4""",
    "yaml",
    1,
    [0, 4],
]

TEST_CASE_MERGE_JSON = ["""{"key1": [0], "key2": [0] }""", """{"key1": [1], "+key2": [4] }""", "json", [1], [0, 4]]

TEST_CASE_MERGE_YAML = [
    """
    key1: 0
    key2: [0]
    """,
    """
    key1: 1
    +key2: [4]
    """,
    "yaml",
    1,
    [0, 4],
]


class TestConfigParser(unittest.TestCase):
    def test_config_content(self):
        test_config = {"preprocessing": [{"_target_": "LoadImage"}], "dataset": {"_target_": "Dataset"}}
        parser = ConfigParser(config=test_config)
        # test `get`, `set`, `__getitem__`, `__setitem__`
        self.assertEqual(str(parser.get()), str(test_config))
        parser.set(config=test_config)
        self.assertListEqual(parser["preprocessing"], test_config["preprocessing"])
        parser["dataset"] = {"_target_": "CacheDataset"}
        self.assertEqual(parser["dataset"]["_target_"], "CacheDataset")
        # test nested ids
        parser["dataset#_target_"] = "Dataset"
        self.assertEqual(parser["dataset#_target_"], "Dataset")
        parser.update({"dataset#_target_1": "Dataset1"})
        self.assertEqual(parser["dataset#_target_1"], "Dataset1")
        # test int id
        parser.set(["test1", "test2", "test3"])
        parser[1] = "test4"
        self.assertEqual(parser[1], "test4")

    @parameterized.expand([TEST_CASE_1])
    @skipUnless(has_tv, "Requires torchvision >= 0.8.0.")
    def test_parse(self, config, expected_ids, output_types):
        parser = ConfigParser(config=config, globals={"monai": "monai"})
        # test lazy instantiation with original config content
        parser["transform"]["transforms"][0]["keys"] = "label1"
        trans = parser.get_parsed_content(id="transform#transforms#0")
        self.assertEqual(trans.keys[0], "label1")
        # test re-use the parsed content or not with the `lazy` option
        self.assertEqual(trans, parser.get_parsed_content(id="transform#transforms#0"))
        self.assertEqual(trans, parser.get_parsed_content(id="transform#transforms#0", lazy=True))
        self.assertNotEqual(trans, parser.get_parsed_content(id="transform#transforms#0", lazy=False))
        # test new nested id
        parser.set("fake_key", "transform#other_transforms#keys", True)
        self.assertEqual(parser.get(id="transform#other_transforms#keys"), "fake_key")
        # remove temp fake data
        parser["transform"].pop("other_transforms")
        # test update nested id
        parser["transform#transforms#0#keys"] = "label2"
        self.assertEqual(parser.get_parsed_content(id="transform#transforms#0").keys[0], "label2")

        for id, cls in zip(expected_ids, output_types):
            self.assertTrue(isinstance(parser.get_parsed_content(id), cls))
        # test root content
        root = parser.get_parsed_content(id="")
        for v, cls in zip(root.values(), [Compose, Dataset, DataLoader]):
            self.assertTrue(isinstance(v, cls))
        # test default value
        self.assertEqual(parser.get_parsed_content(id="abc", default=ConfigItem(12345, "abc")), 12345)
        self.assertEqual(parser.get_parsed_content(id="abcd", default=1), 1)

    @parameterized.expand([TEST_CASE_2])
    def test_function(self, config):
        parser = ConfigParser(config=config, globals={"TestClass": TestClass})
        for id in config:
            if id in ("compute", "cls_compute"):
                parser[f"{id}#_mode_"] = "callable"
            func = parser.get_parsed_content(id=id)
            self.assertIn(id, parser.ref_resolver.resolved_content)
            if id == "error_func":
                with self.assertRaises(TypeError):
                    func(1, 2)
                continue
            self.assertEqual(func(1, 2), 3)

    @parameterized.expand([TEST_CASE_3])
    def test_relative_id(self, config):
        parser = ConfigParser(config=config)
        for id in config:
            item = parser.get_parsed_content(id=id)
            if isinstance(item, int):
                self.assertEqual(item, 1)
            if isinstance(item, dict):
                self.assertEqual(str(item), str({"key": 1, "value1": 2, "value2": 2, "value3": [3, 4, 4, 105]}))

    def test_macro_replace(self):
        with tempfile.TemporaryDirectory() as tempdir:
            another_file = os.path.join(tempdir, "another.json")
            ConfigParser.export_config_file(config={"E": 4}, filepath=another_file)
            # test macro with id, relative id, and macro in another file
            config = {"A": {"B": 1, "C": 2}, "D": [3, "%A#B", "%#0", f"%{another_file}#E"]}
            parser = ConfigParser(config=config)
            parser.resolve_macro_and_relative_ids()
            self.assertEqual(str(parser.get()), str({"A": {"B": 1, "C": 2}, "D": [3, 1, 3, 4]}))

    @parameterized.expand([TEST_CASE_4])
    def test_allow_missing_reference(self, config):
        default = ReferenceResolver.allow_missing_reference
        ReferenceResolver.allow_missing_reference = True
        parser = ConfigParser(config=config)

        for id in config:
            item = parser.get_parsed_content(id=id)
            if id in ("A", "B"):
                self.assertEqual(item, 1)
            elif id == "C":
                self.assertEqual(item, "@D")
            elif id == "E":
                self.assertEqual(item, "test@F")

        # restore the default value
        ReferenceResolver.allow_missing_reference = default
        with self.assertRaises(ValueError):
            parser.parse()
            parser.get_parsed_content(id="E")

    def test_list_expressions(self):
        config = {
            "transform": {
                "_target_": "Compose",
                "transforms": [{"_target_": "RandScaleIntensity", "factors": 0.5, "prob": 1.0}],
            },
            "training": ["$monai.utils.set_determinism(seed=123)", "$@transform(np.asarray([1, 2]))"],
        }
        parser = ConfigParser(config=config)
        parser.get_parsed_content("training", lazy=True, instantiate=True, eval_expr=True)
        np.testing.assert_allclose(parser.get_parsed_content("training#1", lazy=True), [0.7942, 1.5885], atol=1e-4)

    def test_contains(self):
        empty_parser = ConfigParser({})
        empty_parser.parse()

        parser = ConfigParser({"value": 1, "entry": "string content", "array": [1, 2]})
        parser.parse()

        with self.subTest("Testing empty parser"):
            self.assertFalse("something" in empty_parser)
        with self.assertRaises(KeyError):
            empty_parser["something"]
        empty_parser["osmething"] = "test"
        with self.assertRaises(KeyError):
            empty_parser["something"]

        with self.subTest("Testing with keys"):
            self.assertTrue("value" in parser)
            self.assertFalse("value1" in parser)
            self.assertTrue("entry" in parser)
            self.assertFalse("entr" in parser)
            self.assertFalse("array#2" in parser)

    def test_lambda_reference(self):
        configs = {
            "patch_size": [8, 8],
            "transform": {"_target_": "Lambda", "func": "$lambda x: x.reshape((1, *@patch_size))"},
        }
        parser = ConfigParser(config=configs)
        trans = parser.get_parsed_content(id="transform")
        result = trans(np.ones(64))
        self.assertTupleEqual(result.shape, (1, 8, 8))

    def test_non_str_target(self):
        configs = {
            "fwd": {"_target_": "$@model.forward", "x": "$torch.rand(1, 3, 256, 256)", "_mode_": "callable"},
            "model": {"_target_": "monai.networks.nets.resnet.resnet18", "pretrained": False, "spatial_dims": 2},
        }
        self.assertTrue(callable(ConfigParser(config=configs).fwd))
        self.assertTupleEqual(tuple(ConfigParser(config=configs).fwd().shape), (1, 400))

    def test_error_instance(self):
        config = {"transform": {"_target_": "Compose", "transforms_wrong_key": []}}
        parser = ConfigParser(config=config)
        with self.assertRaises(RuntimeError):
            parser.get_parsed_content("transform", instantiate=True, eval_expr=True)

    def test_pdb(self):
        with self.assertRaisesRegex(RuntimeError, ".*bdb.BdbQuit.*"):
            case_pdb()
        self.assertEqual(case_pdb_inst(), None)  # pdb.runcall without input is None

    def test_get_via_attributes(self):
        config = {
            "A": {"B": {"C": 1}},
            "my_dims": 2,
            "dims_1": "$@my_dims + 1",
            "patch_size": [8, 8],
            "transform": {"_target_": "Lambda", "func": "$lambda x: x.reshape((1, *@patch_size))"},
        }
        parser = ConfigParser(config=config)
        self.assertEqual(parser.A, {"B": {"C": 1}})
        self.assertEqual(parser.dims_1, 3)

        trans = parser.transform
        result = trans(np.ones(64))
        self.assertTupleEqual(result.shape, (1, 8, 8))

    def test_builtin(self):
        config = {"import statements": "$import math", "calc": {"_target_": "math.isclose", "a": 0.001, "b": 0.001}}
        self.assertEqual(ConfigParser(config).calc, True)

    def test_slicing(self):
        config = {"test": [1, 2, 3, 4], "test1": "$@test[::]", "test2": "$@test[::-1]", "st": "aten::relu"}
        self.assertEqual(ConfigParser(config).test1, [1, 2, 3, 4])
        self.assertEqual(ConfigParser(config).test2, [4, 3, 2, 1])
        self.assertEqual(ConfigParser(config).st, "aten::relu")

    @parameterized.expand([TEST_CASE_5])
    def test_substring_reference(self, config, expected):
        parser = ConfigParser(config=config)
        self.assertEqual(parser.get_parsed_content("total"), expected)

    @parameterized.expand([TEST_CASE_DUPLICATED_KEY_JSON, TEST_CASE_DUPLICATED_KEY_YAML])
    @mock.patch.dict(os.environ, {"MONAI_FAIL_ON_DUPLICATE_CONFIG": "1"})
    @skipUnless(has_yaml, "Requires pyyaml")
    def test_parse_json_raise(self, config_string, extension, _, __):
        with tempfile.TemporaryDirectory() as tempdir:
            config_path = Path(tempdir) / f"config.{extension}"
            config_path.write_text(config_string)
            parser = ConfigParser()

            with self.assertRaises(ValueError) as context:
                parser.read_config(config_path)

            self.assertTrue("Duplicate key: `duplicate`" in str(context.exception))

    @parameterized.expand([TEST_CASE_DUPLICATED_KEY_JSON, TEST_CASE_DUPLICATED_KEY_YAML])
    @skipUnless(has_yaml, "Requires pyyaml")
    def test_parse_json_warn(self, config_string, extension, expected_unique_val, expected_duplicate_vals):
        with tempfile.TemporaryDirectory() as tempdir:
            config_path = Path(tempdir) / f"config.{extension}"
            config_path.write_text(config_string)
            parser = ConfigParser()

            with warnings.catch_warnings(record=True) as w:
                parser.read_config(config_path)
            self.assertEqual(len(w), 1)
            self.assertTrue("Duplicate key: `duplicate`" in str(w[-1].message))

            self.assertEqual(parser.get_parsed_content("key#unique"), expected_unique_val)
            self.assertIn(parser.get_parsed_content("key#duplicate"), expected_duplicate_vals)

    @parameterized.expand([TEST_CASE_MERGE_JSON, TEST_CASE_MERGE_YAML])
    @skipUnless(has_yaml, "Requires pyyaml")
    def test_load_configs(
        self, config_string, config_string2, extension, expected_overridden_val, expected_merged_vals
    ):
        with tempfile.TemporaryDirectory() as tempdir:
            config_path1 = Path(tempdir) / f"config1.{extension}"
            config_path2 = Path(tempdir) / f"config2.{extension}"
            config_path1.write_text(config_string)
            config_path2.write_text(config_string2)

            parser = ConfigParser.load_config_files([config_path1, config_path2])

            self.assertEqual(parser["key1"], expected_overridden_val)
            self.assertEqual(parser["key2"], expected_merged_vals)


if __name__ == "__main__":
    unittest.main()
